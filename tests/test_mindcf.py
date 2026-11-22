import numpy as np

from utils.metrics import compute_det_curve
from utils.mindcf import compute_mindcf, pooled_mindcf


def official_compute_mindcf(frr, far, thresholds, pspoof, cmiss, cfa):
    """Reference copied from ASVspoof5 Baseline-AASIST calculate_modules.py."""
    min_c_det = float("inf")
    min_c_det_threshold = thresholds
    for i in range(0, len(frr)):
        c_det = cmiss * frr[i] * (1 - pspoof) + cfa * far[i] * pspoof
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(cmiss * (1 - pspoof), cfa * pspoof)
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def test_mindcf_matches_official_reference_on_synthetic_scores():
    rng = np.random.default_rng(17)
    target = rng.normal(1.0, 1.0, size=40)
    nontarget = rng.normal(-0.2, 1.0, size=80)
    frr, far, thresholds = compute_det_curve(target, nontarget)
    actual, threshold = compute_mindcf(frr, far, thresholds, 0.05, 1, 10)
    expected, expected_threshold = official_compute_mindcf(frr, far, thresholds, 0.05, 1, 10)
    assert actual == expected
    assert threshold == expected_threshold


def test_pooled_mindcf_uses_real_high_official_direction():
    # Separable after negation: project Real scores are low and Fake scores are high.
    labels = np.array([0, 0, 1, 1])
    project_scores = np.array([-2.0, -1.0, 0.0, 1.0])
    correct = pooled_mindcf(labels, project_scores)
    assert correct["score_direction"] == "-project_score"
    assert correct["mindcf"] == 0.0
    # The same numbers treated as already Real-high would be reversed and costly.
    reversed_scores = -project_scores
    reversed_cost = pooled_mindcf(labels, reversed_scores)
    assert reversed_cost["mindcf"] == 1.0
