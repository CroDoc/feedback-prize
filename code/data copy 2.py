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
    df = pd.read_csv('data/train_wth_folds.csv')

    df1 = pd.get_dummies(df, columns=['discourse_type']).groupby(['id'], as_index=False).sum()
    label_col = [c for c in df1.columns if c.startswith('discourse_type_') and c != 'discourse_type_num']
    col = label_col +['id']
    df1 = df1[col]
    df1.loc[:,'fold'] = -1

    mskf  = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(mskf.split(df1, df1[label_col])):
        df1.loc[valid_index, 'fold'] = fold

    return df1[['id', 'fold']]#.to_csv('data/folds_df.csv', index=False)

def get_train(use_cache=False, folds_name=None):

    CACHE_PATH = 'data/train_labels_df.p'

    if use_cache and os.path.exists(CACHE_PATH):
        df, train = pickle.load(open(CACHE_PATH, 'rb'))
        return df, train

    train = pd.read_csv('data/train_wth_folds.csv')

    #folds_df = pd.read_csv('data/folds_df.csv')
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

        discourse_starts = list(label_df['discourse_start'].apply(int))
        discourse_ends = list(label_df['discourse_end'].apply(int))

        word_labels_len = label_df['predictionstring'].apply(lambda x: len(x.split()))

        text_labels = []

        for label in zip(discoures, discourse_starts, discourse_ends, word_labels_len):
            text_labels.append(label)

        labels.append(text_labels)

    df['labels'] = labels

    df = df.merge(train[['id',folds_name, 'cluster']].drop_duplicates(), on='id')
    # train = train.merge(folds_df, on='id')

    pickle.dump((df, train), open(CACHE_PATH, 'wb'))

    return df, train
