import ast
import glob
import os
import pickle

import pandas as pd
# pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm

#https://www.kaggle.com/abhishek/creating-folds-properly-hopefully-p/notebook

def make_folds(folds, seed):
    df = pd.read_csv('data/train.csv')

    df1 = pd.get_dummies(df, columns=['discourse_type']).groupby(['id'], as_index=False).sum()
    label_col = [c for c in df1.columns if c.startswith('discourse_type_') and c != 'discourse_type_num']
    col = label_col +['id']
    df1 = df1[col]
    df1.loc[:,'fold'] = -1

    mskf  = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(mskf.split(df1, df1[label_col])):
        df1.loc[valid_index, 'fold'] = fold
    df1[['id', 'fold']].to_csv('data/folds_df.csv', index=False)

"""
def set_folds(train_df, folds, seed):

    train = pd.read_csv('data/train.csv')

    dfx = pd.get_dummies(train, columns=['discourse_type']).groupby(['id'], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols]

    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [c for c in dfx.columns if c != 'id']
    dfx_labels = dfx[labels]
    dfx['fold'] = -1

    for fold, (_, val_index) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_index, 'fold'] = fold

    dfx = dfx[['id', 'fold']]

    train_df = train_df.merge(dfx, on='id', how='left')
    train = train.merge(dfx, on='id', how='left')

    return train_df, train
"""

def get_train(use_cache=True, new_labels=False):

    if new_labels:
        CACHE_PATH = 'data/train_new_labels_df.p'
    else:
        CACHE_PATH = 'data/train_old_labels_df.p'

    if use_cache and os.path.exists(CACHE_PATH):
        df, train = pickle.load(open(CACHE_PATH, 'rb'))
        return df, train

    train = pd.read_csv('data/corrected_train.csv')
    folds_df = pd.read_csv('data/folds_df.csv')
    train_ids, train_texts = [], []

    for text_path in glob.glob('data/train/*'):
        train_ids.append(text_path.split('/')[-1].replace('.txt', ''))

        text = open(text_path, 'r').read()

        text = text.replace(u'\xa0', u' ')
        text = text.rstrip()
        text = text.lstrip()

        train_texts.append(text)

    df = pd.DataFrame({'id': train_ids, 'text': train_texts})

    labels = []

    for text_id, text in tqdm(list(zip(df['id'], df['text']))):

        #labels -> span_name, span_start, span_end

        label_df = train[train['id'] == text_id]

        discoures = list(label_df['discourse_type'])

        if new_labels:
            discourse_starts = list(label_df['new_start'].apply(int))
            discourse_ends = list(label_df['new_end'].apply(int))
        else:
            discourse_starts = list(label_df['discourse_start'].apply(int))
            discourse_ends = list(label_df['discourse_end'].apply(int))

        text_labels = []

        for label in zip(discoures, discourse_starts, discourse_ends):
            text_labels.append(label)

        labels.append(text_labels)

    df['labels'] = labels

    df = df.merge(folds_df, on='id')
    train = train.merge(folds_df, on='id')

    pickle.dump((df, train), open(CACHE_PATH, 'wb'))

    return df, train

# str.split() & is_split_into_words=True
def get_train_split(use_cache=True):

    if use_cache and os.path.exists('data/train_df.p'):
        df = pickle.load(open('data/train_df.p', 'rb'))
        return df

    if use_cache and os.path.exists('data/train_df.csv'):
        df = pd.read_csv('data/train_df.csv')
        df['labels'] = df['labels'].apply(ast.literal_eval)
        return df

    train = pd.read_csv('data/corrected_train.csv')
    train_ids, train_texts = [], []

    for text_path in glob.glob('data/train/*'):
        train_ids.append(text_path.split('/')[-1].replace('.txt', ''))
        train_texts.append(open(text_path, 'r').read())

    df = pd.DataFrame({'id': train_ids, 'text': train_texts})

    labels = []

    for text_id, text in tqdm(list(zip(df['id'], df['text']))):
        text_split = text.split()

        text_labels = ['O'] * len(text_split)

        label_df = train[train['id'] == text_id]
        discoures = list(label_df['discourse_type'].apply(lambda s: 'B-' + s))
        discoure_indexes = list(label_df['predictionstring'].apply(lambda s: [int(x) for x in s.split()]))

        for discourse, indexes in zip(discoures, discoure_indexes):

            for idx in indexes:
                text_labels[idx] = discourse
                discourse = 'I' + discourse[1:]

        labels.append(text_labels)

    df['labels'] = labels

    pickle.dump(df, open('data/train_df.p', 'wb'))

    return df
