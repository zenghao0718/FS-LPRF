import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

# EER/DET copied from pinned Layer-Wise train_util/eer.py; see docs/CODE_PROVENANCE.md.
def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(ground_truth, predictions):
    """
    Expecting ground_truth and predictions to be numpy arrays of the same length;
    Defining deepfakes (ground_truth == 1) as target scores and bonafide (ground_truth == 0) as nontarget scores.
    """
    assert ground_truth.shape == predictions.shape, f"ground_truth and predictions must have the same shape, got {ground_truth.shape} and {predictions.shape}"
    assert len(ground_truth.shape) == 1, "ground_truth and predictions must be 1D arrays"

    target_scores = predictions[ground_truth == 1]
    nontarget_scores = predictions[ground_truth == 0]

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


METRICS = ("eer", "auc", "ap", "acc")


def binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Metrics require finite scores and both Real=0 and Fake=1")
    eer, threshold = compute_eer(labels, scores)
    return dict(eer=float(eer), eer_threshold=float(threshold),
                auc=float(roc_auc_score(labels, scores)),
                ap=float(average_precision_score(labels, scores)),
                acc=float(accuracy_score(labels, scores >= 0)))


def macro_metrics(rows):
    return {name: float(np.mean([row[name] for row in rows])) for name in METRICS}


def mean_std(macros):
    # Population standard deviation across the five prescribed evaluation seeds.
    return {name: {"mean": float(np.mean([row[name] for row in macros])),
                   "std": float(np.std([row[name] for row in macros], ddof=0))}
            for name in METRICS}
