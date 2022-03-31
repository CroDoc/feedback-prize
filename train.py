# https://github.com/abhishekkrthakur/long-text-token-classification/

import argparse
import os
from code.data import get_train, make_folds
from code.dataset import CutTextDataModule, CutTextDataset, TextDataModule
from code.model import TextModel
from code.utils import TextUtil, get_cleaned_df

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default=None, action='store', required=True
    )
    parser.add_argument(
        '--yaml', default=None, action='store', required=True
    )

    parser.add_argument(
        '--gpus', default=0, type=int, required=False
    )

    parser.add_argument(
        '--fold', default=None, type=int, required=False
    )

    parser.add_argument(
        '--new_labels', default=False, action='store_true', required=False
    )

    parser.add_argument(
        '--odd', default=False, action='store_true', required=False
    )

    parser.add_argument(
        '--even', default=False, action='store_true', required=False
    )

    parser.add_argument(
        '--clean', default=False, action='store_true', required=False
    )

    parser.add_argument(
        '--cache', default=False, action='store_true', required=False
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

with open(opt.yaml, 'r') as f:
    cfg = yaml.safe_load(f)

for key,value in vars(opt).items():
    if key == 'gpus':
        cfg['trainer']['gpus'] = 1
    else:
        cfg[key] =  value

root_dir = "runs/" + cfg['name']

if opt.fold is not None:
    try:
        os.makedirs(root_dir)
    except:
        pass
else:
    os.makedirs(root_dir)

yaml.dump(cfg, open(root_dir + '/hparams.yml', 'w'))
cfg['trainer']['gpus'] = [opt.gpus]

df, train = get_train(use_cache=opt.cache, folds_name=cfg['folds_name'])

tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_name'])
tokenizer.save_pretrained(root_dir + '/tokenizer/')

for fold in range(cfg['num_folds']):

    if opt.fold is not None and opt.fold != fold:
        continue

    if opt.odd and fold % 2 != 1:
        continue

    if opt.even and fold % 2 != 0:
        continue

    train_df = df[df[cfg['folds_name']] != fold].reset_index(drop=True)
    valid_df = df[df[cfg['folds_name']] == fold].reset_index(drop=True)

    if opt.clean:
        train_df = get_cleaned_df(train_df)
        valid_df = get_cleaned_df(valid_df, validation=True)
    else:
        train_df = get_cleaned_df(train_df, validation=True)
        valid_df = get_cleaned_df(valid_df, validation=True)

    vd = train[train[cfg['folds_name']] == fold].reset_index(drop=True)

    cfg['dataset_size'] = len(train_df)

    if 'cut' in cfg and cfg['cut'] and cfg['stride'] > 0:
        print('CUTTING...')
        datamodule = CutTextDataModule(train_df, valid_df, tokenizer, cfg)
        text_util = TextUtil(datamodule, vd, num_labels=cfg['model']['num_labels'], is_cut=True)
    else:
        print('NOT CUTTING...')
        datamodule = TextDataModule(train_df, valid_df, tokenizer, cfg)
        text_util = TextUtil(datamodule, vd, num_labels=cfg['model']['num_labels'])

    model = TextModel(cfg, text_util)

    torch.save(model.config, root_dir + '/config.pth')

    logger = TensorBoardLogger(
        save_dir='runs', name=cfg['name'] + '/logs', version=f'fold_{fold}'
    )

    earystopping = EarlyStopping(
        monitor = 'f1_score',
        patience = cfg['callbacks']['patience'],
        mode = 'max',
    )

    callback_list = [earystopping]

    if cfg['callbacks']['weights']:
        loss_weights = callbacks.ModelCheckpoint(
            dirpath=root_dir + '/weights',
            filename='fold=' + str(fold) + '-{epoch}-{f1_score:.4f}_weights',
            monitor='f1_score',
            save_weights_only=True,
            save_top_k=1,
            mode='max',
            save_last=False,
        )
        callback_list.append(loss_weights)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg['epoch'],
        callbacks=callback_list,
        **cfg['trainer'],
    )

    trainer.fit(model, datamodule=datamodule)
