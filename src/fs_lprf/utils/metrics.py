import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from fs_lprf.upstream import enable_upstream

enable_upstream()
from tomato.train_util.eer import compute_eer

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
