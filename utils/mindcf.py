"""ASVspoof 5 Track 1 pooled minDCF.

Project scores are higher for Fake. The official cost uses the opposite
direction, so this module always evaluates ``-project_score`` with bona fide
as the target class. The resulting threshold is a metric, not a decision rule.
"""
import numpy as np

from utils.metrics import compute_det_curve

CMISS = 1.0
CFA = 10.0
PSPOOF = 0.05


def default_cost(cmiss=CMISS, cfa=CFA, pspoof=PSPOOF):
    return min(cmiss * (1 - pspoof), cfa * pspoof)


def beta(cmiss=CMISS, cfa=CFA, pspoof=PSPOOF):
    return (cmiss / cfa) * ((1 - pspoof) / pspoof)


def compute_mindcf(frr, far, thresholds, pspoof=PSPOOF, cmiss=CMISS, cfa=CFA):
    """Official Baseline-AASIST normalized minimum detection cost."""
    min_c_det = float("inf")
    min_c_det_threshold = thresholds
    for index in range(len(frr)):
        c_det = cmiss * frr[index] * (1 - pspoof) + cfa * far[index] * pspoof
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[index]
    normalizer = default_cost(cmiss, cfa, pspoof)
    return float(min_c_det / normalizer), float(min_c_det_threshold)


def pooled_mindcf(labels, project_scores, *, cmiss=CMISS, cfa=CFA, pspoof=PSPOOF):
    """One pooled minDCF over a complete evaluation set.

    ``labels`` use the project convention Real=0 and Fake=1. Scores stay in the
    project Fake-high direction and are negated internally.
    """
    labels = np.asarray(labels)
    scores = np.asarray(project_scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("minDCF requires one label and one project score per trial")
    if set(np.unique(labels).tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("minDCF requires finite scores and both Real=0 and Fake=1")
    official = -scores
    frr, far, thresholds = compute_det_curve(official[labels == 0], official[labels == 1])
    mindcf, threshold = compute_mindcf(frr, far, thresholds, pspoof, cmiss, cfa)
    return dict(mindcf=mindcf, threshold=float(threshold), cmiss=float(cmiss), cfa=float(cfa),
                pspoof=float(pspoof), beta=float(beta(cmiss, cfa, pspoof)),
                default_cost=float(default_cost(cmiss, cfa, pspoof)), score_direction="-project_score")
