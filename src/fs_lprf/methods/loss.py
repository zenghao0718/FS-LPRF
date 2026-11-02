import torch
import torch.nn.functional as F
from .prototypes import build_prototypes, cosine_logits
from .reliability import support_reliability


def episode_loss(support, support_labels, query, query_labels):
    prototypes = build_prototypes(support, support_labels, 3)
    alpha = support_reliability(support, support_labels, prototypes)
    logits = cosine_logits(query, prototypes)
    q, layers, classes = logits.shape
    losses = F.cross_entropy(logits.reshape(q * layers, classes),
                             query_labels[:, None].expand(q, layers).reshape(-1),
                             reduction="none").view(q, layers).mean(0)
    return (alpha * losses).sum(), {"alpha": alpha, "prototypes": prototypes, "layer_losses": losses}


def step_loss(episodes):
    results = [episode_loss(*episode) for episode in episodes]
    return torch.stack([result[0] for result in results]).mean(), [result[1] for result in results]
