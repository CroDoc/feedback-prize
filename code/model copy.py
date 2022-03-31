import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score
from transformers import (AutoConfig, AutoModel,
                          get_polynomial_decay_schedule_with_warmup)


class TextModel(pl.LightningModule):

    def __init__(self, cfg, text_util, config_path=None):
        super().__init__()

        self.cfg = cfg
        self.text_util = text_util
        model_cfg = cfg['model']
        self.num_labels = model_cfg['num_labels']

        self.criterion = eval(model_cfg['loss'])()

        if config_path:
            self.config = torch.load(config_path)
        else:
            self.config = AutoConfig.from_pretrained(model_cfg['model_name'], output_hidden_states=True)

        hidden_dropout_prob = 0.1
        layer_norm_eps = 1e-7

        self.skip_validation = 0
        if 'skip_validation' in cfg:
            self.skip_validation = cfg['skip_validation']

        self.config.update(
            {
                'layer_norm_eps': layer_norm_eps,
                'add_pooling_layer': False,
                'hidden_dropout_prob': hidden_dropout_prob,
            }
        )

        if model_cfg['pretrained']:
            self.backbone = AutoModel.from_pretrained(model_cfg['model_name'], config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)


        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.fc = nn.Linear(self.config.hidden_size, self.num_labels)
        self.cluster_fc = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, x):

        x = self.backbone(**x).last_hidden_state
        x = self.dropout(x)

        x1 = self.dropout1(x)
        x2 = self.dropout2(x)
        x3 = self.dropout3(x)
        x4 = self.dropout4(x)
        x5 = self.dropout5(x)

        x = (x1+x2+x3+x4+x5) / 5.0

        x_cluster = self.cluster_fc(x)

        x = self.fc(x)
        x[:, 0, :] = x_cluster[:, 0, :]

        return x

    def on_epoch_start(self):
        print('\n')

    def on_epoch_end(self):
        if self.skip_validation > 0:
            self.skip_validation -= 1

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        if self.skip_validation > 0:
            return {}

        return self.shared_step(batch)

    def shared_step(self, batch):

        texts, labels = batch

        output = self(texts)

        loss_cluster = self.criterion(output[:, 0, :], labels[:, 0])

        output_view = output.view(-1, self.num_labels)
        labels_view = labels.view(-1)

        active_indexes = np.where(labels_view.cpu().numpy() != -100)[0][1:]

        output_view = output_view[active_indexes]
        labels_view = labels_view[active_indexes]

        loss_fc = self.criterion(output_view, labels_view)

        loss = loss_fc + loss_cluster

        f1 = f1_score(labels_view.detach().cpu().numpy(), output_view.softmax(dim=-1).argmax(dim=-1).detach().cpu().numpy(), average="macro")

        preds = output.softmax(dim=-1).detach().cpu()
        labels = labels.detach().cpu()

        self.log(f"f1", f1, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'labels': labels}

    def predict_step(self, batch, batch_idx):

            texts, _ = batch
            output = self(texts)
            pred = output.softmax(dim=-1).detach().cpu()

            return pred

    def validation_epoch_end(self, outputs):
        if self.skip_validation > 0:
            self.log(f'f1_score', 0.1 / self.skip_validation, on_epoch=True, prog_bar=True)
            return

        preds = []
        labels = []

        for out in outputs:
            pred, label = out['preds'], out['labels']
            preds.extend([x for x in pred])
            labels.extend([x for x in label])

        f1_preds, f1_labels = [], []

        for pred, label in zip(preds, labels):
            active_indexes = np.where(label.numpy() != -100)[0]
            pred = pred[active_indexes].argmax(dim=-1).numpy()
            label = label[active_indexes].numpy()

            f1_preds.extend([p for p in pred])
            f1_labels.extend([l for l in label])

        f1 = f1_score(f1_labels, f1_preds, average="macro")

        metrics = self.text_util.score(preds)

        self.log(f'f1_score', metrics, on_epoch=True, prog_bar=True)
        self.log(f'val_f1', f1, on_epoch=True, prog_bar=True)
        self.log(f'val_loss', out['loss'], on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = eval(self.cfg['optimizer']['name'])(
            self.parameters(), **self.cfg['optimizer']['params']
        )

        if 'scheduler' in self.cfg:

            if self.cfg['scheduler']['name'] == 'poly':

                epoch_steps = self.cfg['dataset_size']
                batch_size = self.cfg['train_loader']['batch_size']

                params = self.cfg['scheduler']['params']

                power = params['power']
                lr_end = params['lr_end']

                warmup_steps = self.cfg['scheduler']['warmup'] * epoch_steps // batch_size
                training_steps = params['epochs'] * epoch_steps // batch_size

                scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)
            else:
                scheduler = eval(self.cfg['scheduler']['name'])(
                    optimizer, **self.cfg['scheduler']['params']
                )

            lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': self.cfg['scheduler']['interval'],
                'frequency': 1,
            }

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

        return optimizer
