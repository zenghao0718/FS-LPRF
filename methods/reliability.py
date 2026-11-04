import torch
import torch.nn.functional as F
from .prototypes import class_dispersion

EPSILON = 1e-8


def reliability_from_statistics(prototypes, dispersion):
    """Shared 3-way, 2-way and 33-class formula. Only Real-to-Fake distances."""
    prototypes, dispersion = prototypes.float(), dispersion.float()
    inter = (1 - F.cosine_similarity(prototypes[0:1], prototypes[1:], dim=-1)).sum(0)
    intra = dispersion.sum(0)
    # A cosine distance cannot be negative; eliminate float32 dot-product roundoff only.
    inter, intra = inter.clamp_min(0), intra.clamp_min(0)
    reliability = (inter + EPSILON) / (intra + EPSILON)
    root = reliability.sqrt()
    return (root / root.sum()).detach()


def support_reliability(support, labels, prototypes):
    dispersion = class_dispersion(support.float(), labels, prototypes.float())
    return reliability_from_statistics(prototypes, dispersion)
