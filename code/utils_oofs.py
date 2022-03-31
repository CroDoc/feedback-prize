import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

output_labels = [
    'O',
    'B-Lead',
    'I-Lead',
    'B-Position',
    'I-Position',
    'B-Claim',
    'I-Claim',
    'B-Counterclaim',
    'I-Counterclaim',
    'B-Rebuttal',
    'I-Rebuttal',
    'B-Evidence',
    'I-Evidence',
    'B-Concluding Statement',
    'I-Concluding Statement',
    ]

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    This code is from Rob Mulla's Kaggle kernel.
    """
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )

    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1

class TextUtil():

    def __init__(self, min_thresh, proba_thresh):
        self.min_thresh = min_thresh
        self.proba_thresh = proba_thresh
        self.labels_df = pd.read_csv('data/train.csv')

        train_ids, train_texts = [], []

        for text_path in glob.glob('data/train/*'):
            train_ids.append(text_path.split('/')[-1].replace('.txt', ''))

            text = open(text_path, 'r').read()

            text = text.replace(u'\xa0', u' ')
            text = text.rstrip()
            text = text.lstrip()

            train_texts.append(text)

        self.text_df = pd.DataFrame({'id': train_ids, 'text': train_texts})

    def score(self, preds_iter):

        final_preds = []
        final_scores = []
        valid_samples = [{} for _ in range(len(self.text_df))]
        offset_mappings = []

        for preds, offset_mapping in preds_iter:
            offset_mappings.append(offset_mapping)

            pred = np.argmax(preds.numpy(), axis=1)
            pred_scr = np.max(preds.numpy(), axis=1)

            final_preds.append(pred.tolist())
            final_scores.append(pred_scr.tolist())

        for j in range(len(self.text_df)):
            valid_samples[j]["preds"] = final_preds[j][:]
            valid_samples[j]["pred_scores"] = final_scores[j][:]

        submission = []

        ids = self.text_df['id']
        texts = self.text_df['text']

        for ii, sample in enumerate(valid_samples): #tqdm(list(
            preds = sample["preds"]
            offset_mapping = offset_mappings[ii]
            sample_id = ids[ii]
            sample_text = texts[ii]
            sample_pred_scores = sample["pred_scores"]

            idx = 1
            phrase_preds = []
            while idx < len(offset_mapping) -1:
                start, _ = offset_mapping[idx]
                end = None
                label = preds[idx]
                label = label + (label % 2 == 1)

                phrase_scores = []
                phrase_scores.append(sample_pred_scores[idx])
                idx += 1
                while idx < len(offset_mapping) -1:
                    if preds[idx] == label:
                        _, end = offset_mapping[idx]
                        phrase_scores.append(sample_pred_scores[idx])
                        idx += 1
                    else:
                        break
                if end is not None:
                    phrase = sample_text[start:end]
                    phrase_preds.append((phrase, start, end, label, phrase_scores))

            temp_df = []
            for phrase, start, end, label, phrase_scores in phrase_preds:
                word_start = len(sample_text[:start].split())
                word_end = word_start + len(sample_text[start:end].split())
                word_end = min(word_end, len(sample_text.split()))
                ps = " ".join([str(x) for x in range(word_start, word_end)])
                if label != 0:
                    if sum(phrase_scores) / len(phrase_scores) >= self.proba_thresh[label]:
                        temp_df.append((sample_id, label, ps))

            temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

            submission.append(temp_df)

        submission = pd.concat(submission).reset_index(drop=True)
        submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

        def threshold(df):
            df = df.copy()
            for key, value in self.min_thresh.items():
                class_df = df.loc[df["class"] == key]
                class_df = class_df[class_df['len'] < value]
                index = class_df.index
                df.drop(index, inplace=True)
            return df

        submission = threshold(submission)

        # drop len
        submission = submission.drop(columns=["len"])

        submission['class'] = submission['class'].apply(lambda x: ids_to_labels[x][2:])
        scr = score_feedback_comp(submission, self.labels_df, return_class_scores=True)
        print(scr)

        return float(scr[0])
