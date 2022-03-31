import numpy as np
import pandas as pd

discourse_marker_to_label = {
    'O': 0,
    'B-Claim': 1,
    'I-Claim': 2,
    'B-Evidence': 3,
    'I-Evidence': 4,
    'X-Lead': 5,
    'X-Position': 6,
    'X-Counterclaim': 7,
    'X-Rebuttal': 8,
    'X-Concluding Statement': 9,
}

label_to_discourse_marker = {v: k for k, v in discourse_marker_to_label.items()}

min_thresh = {
    'Lead': 6,
    'Position': 4,
    'Evidence': 16,
    'Claim': 2,
    'Concluding Statement': 11,
    'Counterclaim': 7,
    'Rebuttal': 6,
}

proba_thresh = {
    'Lead': 0.7,
    'Position': 0.6,
    'Evidence': 0.65,
    'Claim': 0.55,
    'Concluding Statement': 0.7,
    'Counterclaim': 0.6,
    'Rebuttal': 0.6,
}

def label_thresh(labels):
    new_labels = []

    for label in labels:
        if label[3] <= min_thresh[label[0]]:
            continue
        new_labels.append(label)

    return new_labels

def label_clean(labels):
    new_labels = []

    for label in labels:
        new_labels.append((label[0], label[1], label[2]))

    return new_labels

def get_cleaned_df(df, validation=False):

    df = df.copy()
    if not validation:
        df['labels'] = df['labels'].apply(label_thresh)

    df['labels'] = df['labels'].apply(label_clean)

    return df

def calc_overlap3(set_pred, set_gt):
    """
    Calculates if the overlap between prediction and
    ground truth is enough fora potential True positive
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter/ len_pred
        return overlap_1 >= 0.5 and overlap_2 >= 0.5
    except:  # at least one of the input is NaN
        return False

def score_feedback_comp_micro3(pred_df, gt_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df['class'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    overlaps = [calc_overlap3(*args) for args in zip(joined.predictionstring_pred,
                                                     joined.predictionstring_gt)]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    # we don't need to compute the match to compute the score
    TP = joined.loc[overlaps]['gt_id'].nunique()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    TPandFP = len(pred_df)
    TPandFN = len(gt_df)

    #calc microf1
    my_f1_score = 2*TP / (TPandFP + TPandFN)
    return my_f1_score

def score_feedback_comp3(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_score = score_feedback_comp_micro3(pred_df, gt_df, discourse_type)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1

class TextUtil():

    def __init__(self, datamodule, valid_df, num_labels, is_cut=False):
        self.datamodule = datamodule
        self.valid_df = valid_df
        self.is_cut = is_cut

        self.is_setup = False
        self.num_labels = num_labels

    def text_to_words(self, text):
        word_start = True

        starts, ends = [], []
        end = -1

        for i in range(len(text)):

            if text[i].isspace():
                if end != -1:
                    ends.append(end)
                    end = -1

                word_start=True
                continue

            if word_start==True:
                starts.append(i)
                word_start=False

            end = i + 1

        if len(starts) > len(ends):
            ends.append(end)

        return text.split(), list(zip(starts, ends))

    def setup(self):
        self.text_words, self.text_word_offsets, self.text_ids, self.text_lenghts = [], [], [], []

        dataset = self.datamodule.valid_dataset

        for text, text_id in zip(dataset.texts, dataset.ids):
            self.text_ids.append(text_id)
            self.text_lenghts.append(len(text))

            row_words, row_word_offsets = self.text_to_words(text)
            self.text_words.append(row_words)
            self.text_word_offsets.append(row_word_offsets)

    def merge_cut_preds(self, model_preds, dataset):

        index = 0
        preds_tmp = []
        text_indexes = dataset.text_indexes

        overlap = dataset.stride // 2

        while index < len(model_preds):

            text_index, _ = text_indexes[index]
            offset_mapping = dataset.offset_mappings[text_index]

            preds = np.zeros((len(offset_mapping), self.num_labels))

            while index < len(model_preds):
                curr_text_index, start = text_indexes[index]

                if curr_text_index != text_index:
                    break

                curr_preds = model_preds[index]

                if start == 0:
                    length = min(len(preds), len(curr_preds))
                    preds[:length] = curr_preds[:length]
                elif start + len(curr_preds) > len(offset_mapping):
                    preds[-len(curr_preds)+overlap:] = curr_preds[overlap:]
                else:
                    preds[start+overlap:start+len(curr_preds)] = curr_preds[overlap:]

                index += 1

            preds_tmp.append(preds)

        return preds_tmp

    def get_word_preds(self, model_preds, offset_mappings):

        text_word_preds = []

        for idx, row_preds in enumerate(model_preds):

            word_preds = np.full((len(self.text_words[idx]), self.num_labels),0, np.float32)
            character_preds = np.full((self.text_lenghts[idx],self.num_labels),0, np.float32)

            for pos,(start,end) in enumerate(offset_mappings[idx]):
                character_preds[start:end] = row_preds[pos]

            for pos,(start,end) in enumerate(self.text_word_offsets[idx]):
                word_preds[pos] = character_preds[start:end].mean(0)

            text_word_preds.append(word_preds)

        return text_word_preds

    def word_probability_to_predict_df(self, text_to_word_probability, id):
        len_word = len(text_to_word_probability)
        word_predict = text_to_word_probability.argmax(-1)
        word_score   = text_to_word_probability.max(-1)
        predict_df = []

        t = 0
        while 1:
            if word_predict[t] not in [
                discourse_marker_to_label['O'],
            ]:
                start = t
                b_marker_label = word_predict[t]
            else:
                t = t+1
                if t== len_word-1:break
                continue

            t = t+1
            if t== len_word-1: break

            if label_to_discourse_marker[b_marker_label][0]=='B':
                i_marker_label = b_marker_label+1
            else:
                i_marker_label = b_marker_label

            while 1:
                if (word_predict[t] != i_marker_label) or (t ==len_word-1):
                    end = t
                    prediction_string = ' '.join([str(i) for i in range(start,end)])
                    discourse_type = label_to_discourse_marker[b_marker_label][2:]
                    discourse_score = word_score[start:end].tolist()
                    predict_df.append((id, discourse_type, prediction_string, str(discourse_score)))
                    break
                else:
                    t = t+1
                    continue
            if t== len_word-1: break

        predict_df = pd.DataFrame(predict_df, columns=['id', 'class', 'predictionstring', 'score'])

        return predict_df

    def do_threshold(self, submit_df, use=['length','probability']):
        df = submit_df.copy()
        df = df.fillna('')

        if 'length' in use:
            df['l'] = df.predictionstring.apply(lambda x: len(x.split()))
            for key, value in min_thresh.items():
                index = df.loc[df['class'] == key].query('l<%d'%value).index
                df.drop(index, inplace=True)

        if 'probability' in use:
            df['s'] = df.score.apply(lambda x: np.mean(eval(x)))
            for key, value in proba_thresh.items():
                index = df.loc[df['class'] == key].query('s<%f'%value).index
                df.drop(index, inplace=True)

        df = df[['id', 'class', 'predictionstring']]
        return df

    def score(self, model_preds):

        if not self.is_setup:
            self.setup()
            self.is_setup = True

        dataset = self.datamodule.valid_dataset

        if self.is_cut:
            model_preds = self.merge_cut_preds(model_preds, dataset)

        offset_mappings = dataset.offset_mappings

        text_word_preds = self.get_word_preds(model_preds, offset_mappings)

        submit_df = []

        for idx, row_word_preds in enumerate(text_word_preds):
            submit_df.append(self.word_probability_to_predict_df(row_word_preds, self.text_ids[idx]))

        submit_df = pd.concat(submit_df).reset_index(drop=True)
        submit_df = self.do_threshold(submit_df, use=['length', 'probability'])

        scr = score_feedback_comp3(submit_df, self.valid_df, return_class_scores=True)
        print(scr)

        return float(scr[0])

