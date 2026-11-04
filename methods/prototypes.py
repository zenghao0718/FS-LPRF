import torch
import torch.nn.functional as F


def build_prototypes(support, labels, num_classes):
    """[S,L,D] -> [C,L,D]. Average raw features, without extra normalization."""
    if support.ndim != 3 or labels.shape != (support.shape[0],):
        raise ValueError("Expected support [S,L,D] and labels [S]")
    if set(labels.tolist()) != set(range(num_classes)):
        raise ValueError("Support must contain every task class, with Real=0")
    return torch.stack([support[labels == c].mean(dim=0) for c in range(num_classes)])


def cosine_logits(query, prototypes):
    """Return [Q,L,C] cosine similarities; no temperature or learned classifier."""
    return F.cosine_similarity(query[:, None], prototypes[None], dim=-1).permute(0, 2, 1)


def class_dispersion(support, labels, prototypes):
    return torch.stack([
        (1 - F.cosine_similarity(support[labels == c], prototypes[c][None], dim=-1)).mean(0)
        for c in range(prototypes.shape[0])
    ])
