import numpy as np
import pandas as pd

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

th_keys = {
    'Lead': 2,
    'Position': 4,
    'Claim': 6,
    'Counterclaim': 8,
    'Rebuttal': 10,
    'Evidence': 12,
    'Concluding Statement': 14,
}

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
    "Lead": 0.7,
    "Position": 0.6,
    "Evidence": 0.65,
    "Claim": 0.55,
    "Concluding Statement": 0.7,
    "Counterclaim": 0.6,
    "Rebuttal": 0.6,
}

min_thresh = {th_keys[k]:v for k,v in min_thresh.items()}
proba_thresh = {th_keys[k]:v for k,v in proba_thresh.items()}

def label_thresh(labels):
    new_labels = []

    for label in labels:
        if label[3] <= min_thresh[th_keys[label[0]]]:
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

    def __init__(self, datamodule, valid_df, is_cut=False):
        self.datamodule = datamodule
        self.valid_df = valid_df
        self.is_cut = is_cut

    def score(self, preds_iter):

        dataset = self.datamodule.valid_dataset

        final_preds = []
        final_scores = []

        if self.is_cut:
            dataset_length = len(set(dataset.texts))
        else:
            dataset_length = len(dataset)

        valid_samples = [{} for _ in range(dataset_length)]

        if self.is_cut:

            index = 0
            preds_tmp = []
            text_indexes = dataset.text_indexes

            overlap = dataset.stride // 2

            while index < len(preds_iter):

                text_index, _ = text_indexes[index]
                offset_mapping = dataset.offset_mappings[text_index]

                preds = np.zeros((len(offset_mapping), 15))

                while index < len(preds_iter):
                    curr_text_index, start = text_indexes[index]

                    if curr_text_index != text_index:
                        break

                    curr_preds = preds_iter[index].numpy()

                    if start == 0:
                        length = min(len(preds), len(curr_preds))
                        preds[:length] = curr_preds[:length]
                    elif start + len(curr_preds) > len(offset_mapping):
                        preds[-len(curr_preds)+overlap:] = curr_preds[overlap:]
                    else:
                        preds[start+overlap:start+len(curr_preds)] = curr_preds[overlap:]

                    index += 1

                preds_tmp.append(preds)

            preds_iter = preds_tmp

        for preds in preds_iter:

            if not self.is_cut:
                preds = preds.numpy()

            pred = np.argmax(preds, axis=1)
            pred_scr = np.max(preds, axis=1)

            final_preds.append(pred.tolist())
            final_scores.append(pred_scr.tolist())

        for j in range(dataset_length):
            valid_samples[j]["preds"] = final_preds[j][:]
            valid_samples[j]["pred_scores"] = final_scores[j][:]

        submission = []

        for ii, sample in enumerate(valid_samples):
            preds = sample["preds"]
            offset_mapping = dataset.offset_mappings[ii]
            sample_id = dataset.ids[ii]
            sample_text = dataset.texts[ii]
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
                    if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                        temp_df.append((sample_id, label, ps))

            temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

            submission.append(temp_df)

        submission = pd.concat(submission).reset_index(drop=True)
        submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

        def threshold(df):
            df = df.copy()
            for key, value in min_thresh.items():
                index = df.loc[df["class"] == key].query(f"len<{value}").index
                df.drop(index, inplace=True)
            return df

        submission = threshold(submission)

        # drop len
        submission = submission.drop(columns=["len"])

        submission['class'] = submission['class'].apply(lambda x: ids_to_labels[x][2:])
        scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
        #print(scr)

        return float(scr[0])
