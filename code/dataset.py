import random
from code.data import get_word_spans
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
        self.word_spans = df['word_spans'].values.tolist()
        self.ids = df['id'].values.tolist()
        self.mask = 0.0

        if 'mask' in cfg:
            self.mask = cfg['mask']

        if validation == True:
            self.x, self.y, self.offset_mappings = [], [], []
            self.max_length = cfg['max_length_valid']

            for text, label, word_spans in zip(self.texts, self.labels, self.word_spans):
                x, y, offset_mapping = self.make_item(text, label, word_spans)
                self.x.append(x)
                self.y.append(y)
                self.offset_mappings.append(offset_mapping)

    def make_item(self, text, label_spans, word_spans):

        tokenized = self.tokenizer(
            text,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation=True,
            return_offsets_mapping = True,
        )

        offset_mapping = tokenized['offset_mapping']
        del tokenized['offset_mapping']

        skip_indices = np.where(np.array(tokenized.sequence_ids()) != 0)[0]

        label = np.zeros(len(offset_mapping))
        label[skip_indices] = -100

        for discourse, start_span, end_span in label_spans:

            start, end = -1, -1

            for word_start, word_end in word_spans:
                if min(end_span, word_end) - max(start_span, word_start) > 0:
                    start = word_start
                    end = word_end
                    break

            text_labels = [0] * len(text)
            text_labels[start_span:end_span] = [1] * (end_span - start_span)

            target_idx = []

            for idx, (offset1, offset2) in enumerate(offset_mapping):
                if sum(text_labels[offset1:offset2]) > 0:
                    # TODO: CHECK THIS
                    #if len(text[offset1:offset2].split()) > 0:
                    target_idx.append(idx)

            # truncated
            if len(target_idx) == 0:
                continue

            targets_start = target_idx[0]
            targets_end = target_idx[-1] + 1

            if discourse in ['Claim', 'Evidence']:
                pred_start = discourse_marker_to_label['B-' + discourse]
                pred_end = discourse_marker_to_label['I-' + discourse]
            else:
                pred_end = discourse_marker_to_label['X-' + discourse]

            label[targets_start : targets_end] = [pred_end] * (targets_end - targets_start)

            if discourse in ['Claim', 'Evidence']:
                for idx, (offset1, offset2) in enumerate(offset_mapping):
                    if min(offset2, end) - max(offset1, start) > 0:
                        label[idx] = pred_start

        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v, dtype=torch.long)

        label = torch.tensor(label, dtype=torch.long)

        return tokenized, label, offset_mapping

    def __len__(self):
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
            return self.x[idx], self.y[idx]

        x, y, _ = self.make_item(self.texts[idx], self.labels[idx], self.word_spans[idx])

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
        self.word_spans = df['word_spans'].values.tolist()
        self.ids = df['id'].values.tolist()
        self.stride = cfg['stride']
        self.mask = 0.0

        if 'mask' in cfg:
            self.mask = cfg['mask']

        if validation == True:
            self.x, self.y, self.offset_mappings, self.text_indexes = [], [], [], []
            self.x_cut, self.y_cut = [], []
            self.max_length = cfg['max_length_valid']

            text_index = 0

            for text, label, word_spans in zip(self.texts, self.labels, self.word_spans):
                x, y, offset_mapping = self.make_item(text, label, word_spans)

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
        else:
            self.clusters = df['cluster'].values.tolist()
            self.cluster_indexes = {k:[] for k in range(15)}
            self.claim_cuts = {}

            for idx, cluster in enumerate(self.clusters):
                self.cluster_indexes[cluster].append(idx)

                labels = self.labels[idx]
                cuts = []

                first_claim = False
                for i in range(len(labels)):
                    label = labels[i]
                    if label[0] == 'Claim':
                        if first_claim:
                            # label before Claim end, Claim start
                            cuts.append((labels[i-1][2], labels[i][1]))
                        first_claim = False
                    else:
                        first_claim = True

                if len(cuts) == 0:
                    if labels[-1][0] == 'Concluding Statement':
                        cuts.append((labels[-2][2], labels[-1][1]))
                    else:
                        cuts.append((len(self.texts[idx]), len(self.texts[idx])))

                self.claim_cuts[idx] = cuts

    def get_cut_element(self, tokenized_element, start, length, is_list=False):

        new_tokenized_element = tokenized_element[start:start+length]
        if not is_list:
            new_tokenized_element = new_tokenized_element.clone()

        #new_tokenized_element[0] = tokenized_element[0]
        #new_tokenized_element[-1] = tokenized_element[-1]

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

    def make_item(self, text, label_spans, word_spans):

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

            start, end = -1, -1

            for word_start, word_end in word_spans:
                if min(end_span, word_end) - max(start_span, word_start) > 0:
                    start = word_start
                    end = word_end
                    break

            text_labels = [0] * len(text)
            text_labels[start_span:end_span] = [1] * (end_span - start_span)

            target_idx = []

            for idx, (offset1, offset2) in enumerate(offset_mapping):
                if sum(text_labels[offset1:offset2]) > 0:
                    # TODO: CHECK THIS
                    #if len(text[offset1:offset2].split()) > 0:
                    target_idx.append(idx)

            # truncated
            if len(target_idx) == 0:
                continue

            targets_start = target_idx[0]
            targets_end = target_idx[-1] + 1

            if discourse in ['Claim', 'Evidence']:
                pred_start = discourse_marker_to_label['B-' + discourse]
                pred_end = discourse_marker_to_label['I-' + discourse]
            else:
                pred_end = discourse_marker_to_label['X-' + discourse]

            label[targets_start : targets_end] = [pred_end] * (targets_end - targets_start)

            if discourse in ['Claim', 'Evidence']:
                for idx, (offset1, offset2) in enumerate(offset_mapping):
                    if min(offset2, end) - max(offset1, start) > 0:
                        label[idx] = pred_start

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

        """
        cluster = self.clusters[idx]
        text = self.texts[idx]
        labels = self.labels[idx]
        cut = random.choice(self.claim_cuts[idx])[0]

        cut2 = None
        while cut2 is None:

            idx2 = random.choice(self.cluster_indexes[cluster])
            text2 = self.texts[idx2]
            labels2 = self.labels[idx2]
            cut2 = random.choice(self.claim_cuts[idx2])[1]

            if cut2 == len(text2):
                cut2 = None

        while text[cut-1].isspace():
            cut -= 1

        while text2[cut2].isspace() or text2[cut2] == '.' or text2[cut2] == ',':
            cut2 += 1

        text = text[:cut]
        text2 = text2[cut2:]

        if text2[0].isalnum():
            text2 = text2[0].upper() + text2[1:]

        if text[-1].isalnum():
            text = text + '.'

        text = text + random.choice([' ', '\n'])
        cut = len(text)

        new_text = text + text2
        new_labels = []

        for label in labels:
            if label[1] < cut:
                new_labels.append((label[0], label[1], min(label[2], cut)))

        for label in labels2:
            if label[2] >= cut2:
                new_labels.append((label[0], max(label[1], cut2)-cut2+cut, label[2]-cut2+cut))

        new_word_spans = get_word_spans(new_text)
        x, y, offset_mapping = self.make_item(new_text, new_labels, new_word_spans)
        """
        x, y, offset_mapping = self.make_item(self.texts[idx], self.labels[idx], self.word_spans[idx])

        random_value = random.random()

        min_start = 0
        max_start = max(0, len(offset_mapping) - self.max_length)

        if len(offset_mapping) <= self.max_length:
            start = min_start
        elif len(offset_mapping) > self.max_length and len(offset_mapping) <= 2 * self.max_length:
            if random_value < 0.5:
                start = min_start
            else:
                start = max_start
        else:
            if random_value < 0.25:
                start = min_start
            elif random_value < 0.25:
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
