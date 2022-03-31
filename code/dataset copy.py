import random
from code.utils import discourse_marker_to_label

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, cfg, validation=False):

        self.tokenizer = tokenizer
        self.max_length = cfg['max_length']
        self.validation = validation

        self.texts = df['text'].values.tolist()
        self.labels = df['labels'].values.tolist()
        self.ids = df['id'].values.tolist()
        self.clusters = df['cluster'].values.tolist()

        if validation == True:
            self.x, self.y, self.offset_mappings = [], [], []
            self.max_length = cfg['max_length_valid']

            for text, label, cluster in zip(self.texts, self.labels, self.clusters):
                x, y, offset_mapping = self.make_item(text, label, cluster)
                self.x.append(x)
                self.y.append(y)
                self.offset_mappings.append(offset_mapping)

    def get_offset_mapping(self, text):

        tokenized = self.tokenizer(
            text,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation=True,
            return_offsets_mapping = True,
        )

        offset_mapping = tokenized['offset_mapping']
        skip_indices = np.where(np.array(tokenized.sequence_ids()) != 0)[0]

        return offset_mapping, skip_indices

    def make_item(self, text, label_spans, cluster):

        tokenized = self.tokenizer(
            text,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation=True,
            return_offsets_mapping = False,
        )

        offset_mapping, skip_indices = self.get_offset_mapping(text)

        label = np.zeros(len(offset_mapping))
        label[skip_indices] = -100

        for discourse, start_span, end_span in label_spans:
            text_labels = [0] * len(text)

            text_labels[start_span:end_span] = [1] * (end_span - start_span)

            target_idx = []

            for map_idx, (offset1, offset2) in enumerate(offset_mapping):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            # truncated
            if len(target_idx) == 0:
                #print('TRUNCATED')
                continue

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = discourse_marker_to_label['B-' + discourse]
            pred_end = discourse_marker_to_label['I-' + discourse]

            label[targets_start] = pred_start
            label[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)

        label[0] = cluster

        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v, dtype=torch.long)

        label = torch.tensor(label, dtype=torch.long)

        return tokenized, label, offset_mapping

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.validation:
            return self.x[idx], self.y[idx]

        x, y, _ = self.make_item(self.texts[idx], self.labels[idx], self.clusters[idx])

        return x, y

class CustomCollator():

    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        text = []
        labels = []

        for item in batch:
            text.append(item[0])
            labels.append(item[1])

        text = self.data_collator(text)

        length = text['input_ids'].size(dim=1)

        for i, label in enumerate(labels):
            labels[i] = torch.nn.functional.pad(label, pad=(0,length-len(labels[i])), value=-100)

        labels = torch.stack(labels)

        return text, labels


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        tokenizer,
        cfg,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage):

        self.train_dataset = TextDataset(self.train_df, self.tokenizer, self.cfg)
        self.valid_dataset = TextDataset(self.valid_df, self.tokenizer, self.cfg, validation=True)

    def train_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.train_dataset, **self.cfg["train_loader"], collate_fn=custom_collator)

    def val_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)

    def predict_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)


class CutTextDataset(Dataset):
    def __init__(self, df, tokenizer, cfg, validation=False):

        self.tokenizer = tokenizer
        self.max_length = cfg['max_length']
        self.validation = validation

        self.texts = df['text'].values.tolist()
        self.labels = df['labels'].values.tolist()
        self.ids = df['id'].values.tolist()
        self.clusters = df['cluster'].values.tolist()
        self.stride = cfg['stride']
        self.mask = 0.0

        if 'mask' in cfg:
            self.mask = cfg['mask']

        if validation == True:
            self.x, self.y, self.offset_mappings, self.text_indexes = [], [], [], []
            self.x_cut, self.y_cut = [], []
            self.max_length = cfg['max_length_valid']

            text_index = 0

            for text, label, cluster in zip(self.texts, self.labels, self.clusters):
                x, y, offset_mapping = self.make_item(text, label, cluster)

                self.x.append(x)
                self.y.append(y)
                self.offset_mappings.append(offset_mapping)

                start = 0
                total_tokens = len(offset_mapping)

                break_bool = False

                while start < total_tokens and not break_bool:

                    if start + self.max_length > total_tokens:
                        start = max(0, total_tokens - self.max_length)
                        break_bool = True

                    x_cut, y_cut, offset_mapping_cut = self.get_cut_item(x, y, offset_mapping, start)

                    self.x_cut.append(x_cut)
                    self.y_cut.append(y_cut)
                    #self.offset_mappings.append(offset_mapping_cut)

                    self.text_indexes.append((text_index, start))

                    start += self.stride

                text_index += 1
        """
        else:
            extra_texts = []
            extra_labels = []
            extra_ids = []

            for text, label, text_id in zip(self.texts, self.labels, self.ids):

                discourse_types = {l[0] for l in label}
                if 'Rebuttal' in discourse_types or 'Counterclaim' in discourse_types:
                    extra_texts.append(text)
                    extra_labels.append(label)
                    extra_ids.append(text_id)

            print('ADDING EXTRA:', len(extra_texts))

            self.texts.extend(extra_texts)
            self.labels.extend(extra_labels)
            self.ids.extend(extra_ids)

        """

    def get_cut_element(self, tokenized_element, start, length, is_list=False):

        new_tokenized_element = tokenized_element[start:start+length]
        if not is_list:
            new_tokenized_element = new_tokenized_element.clone()

        new_tokenized_element[0] = tokenized_element[0]
        new_tokenized_element[-1] = tokenized_element[-1]

        return new_tokenized_element

    def get_cut_item(self, tokenized, labels, offset_mapping, start):

        cut_length = min(self.max_length, len(offset_mapping))

        new_tokenized = {}

        for k in tokenized:
            new_tokenized[k] = self.get_cut_element(tokenized[k], start, cut_length)

        if offset_mapping is not None:
            offset_mapping = self.get_cut_element(offset_mapping, start, cut_length, is_list=True)

        if labels is not None:
            labels = self.get_cut_element(labels, start, cut_length)

        return new_tokenized, labels, offset_mapping

    def make_item(self, text, label_spans, cluster):

        tokenized = self.tokenizer(
            text,
            add_special_tokens = True,
            return_offsets_mapping = True,
        )

        offset_mapping = tokenized['offset_mapping']
        del tokenized['offset_mapping']

        skip_indices = np.where(np.array(tokenized.sequence_ids()) != 0)[0]

        label = np.zeros(len(offset_mapping))
        label[skip_indices] = -100

        for discourse, start_span, end_span in label_spans:
            text_labels = [0] * len(text)

            text_labels[start_span:end_span] = [1] * (end_span - start_span)

            target_idx = []

            for map_idx, (offset1, offset2) in enumerate(offset_mapping):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            # truncated
            if len(target_idx) == 0:
                continue

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = discourse_marker_to_label['B-' + discourse]
            pred_end = discourse_marker_to_label['I-' + discourse]

            label[targets_start] = pred_start
            label[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)

        # cluster/topic
        label[0] = cluster

        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v, dtype=torch.long)

        label = torch.tensor(label, dtype=torch.long)

        return tokenized, label, offset_mapping

    def __len__(self):
        if self.validation:
            return len(self.x_cut)
        else:
            return len(self.texts)

    def add_masking(self, x):
        mask_id = self.tokenizer.mask_token_id

        input_len = len(x['input_ids'])
        random_value = random.random() * self.mask
        indices = random.sample(range(input_len), int(input_len * random_value))

        for idx in indices:
            x['input_ids'][idx] = mask_id

    def __getitem__(self, idx):
        if self.validation:
            return self.x_cut[idx], self.y_cut[idx]

        x, y, offset_mapping = self.make_item(self.texts[idx], self.labels[idx], self.clusters[idx])

        random_value = random.random()

        min_start = 0
        max_start = max(0, len(offset_mapping) - self.max_length)

        if len(offset_mapping) <= self.max_length:
            start = min_start
        #elif len(offset_mapping) > self.max_length and len(offset_mapping) <= 2 * self.max_length:
        #else:
        #    if random_value < 0.5:
        #        start = min_start
        #    else:
        #        start = max_start
        else:
            if random_value < 0.10:
                start = min_start
            elif random_value < 0.30:
                start = max_start
            else:
                start = random.randint(min_start, max_start)

        x, y, _ = self.get_cut_item(x, y, offset_mapping, start)

        if self.mask > 0:
            self.add_masking(x)

        return x, y

class CutTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        tokenizer,
        cfg,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage):

        self.train_dataset = CutTextDataset(self.train_df, self.tokenizer, self.cfg)
        self.valid_dataset = CutTextDataset(self.valid_df, self.tokenizer, self.cfg, validation=True)

    def train_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.train_dataset, **self.cfg["train_loader"], collate_fn=custom_collator)

    def val_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)

    def predict_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)
