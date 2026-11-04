from dataclasses import dataclass
import torch
from .prototypes import build_prototypes, class_dispersion, cosine_logits
from .reliability import support_reliability, reliability_from_statistics


def kshot_task(support, labels):
    counts = torch.bincount(labels, minlength=2)
    if counts.numel() != 2 or counts[0] != counts[1] or counts[0] == 0:
        raise ValueError("K-shot requires equal, positive Real/Fake Support counts")
    prototypes = build_prototypes(support, labels, 2)
    return prototypes, support_reliability(support, labels, prototypes)


def kshot_scores(query, prototypes, alpha):
    logits = cosine_logits(query, prototypes)
    return ((logits[:, :, 1] - logits[:, :, 0]) * alpha).sum(-1)


def predict(scores):
    return (scores >= 0).long()


@dataclass
class PrototypeBank:
    classes: list[str]
    prototypes: torch.Tensor
    dispersion: torch.Tensor
    metadata: dict

    @property
    def alpha(self):
        return reliability_from_statistics(self.prototypes, self.dispersion)

    def save(self, path):
        torch.save(dict(classes=self.classes, prototypes=self.prototypes.cpu(),
                        dispersion=self.dispersion.cpu(), metadata=self.metadata), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=True))


def bank_from_features(features_by_class, metadata=None):
    """Used by synthetic tests; real construction selects exactly 1024/class upstream."""
    classes = ["real"] + sorted(set(features_by_class) - {"real"})
    prototypes, dispersions = [], []
    for name in classes:
        features = features_by_class[name].float()
        proto = features.mean(0, keepdim=True)
        prototypes.append(proto[0])
        labels = torch.zeros(len(features), dtype=torch.long, device=features.device)
        dispersions.append(class_dispersion(features, labels, proto)[0])
    return PrototypeBank(classes, torch.stack(prototypes), torch.stack(dispersions), metadata or {})


def zeroshot_scores(query, bank):
    prototypes = bank.prototypes.to(query.device)
    alpha = bank.alpha.to(query.device)
    logits = cosine_logits(query, prototypes)
    fake = logits[:, :, 1:].max(-1).values
    return ((fake - logits[:, :, 0]) * alpha).sum(-1)
