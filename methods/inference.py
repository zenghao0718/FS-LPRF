from dataclasses import dataclass
import torch
from .prototypes import build_prototypes, cosine_logits
from .layer_weights import METHOD_VERSION, layer_weights


def kshot_task(support, labels, alpha):
    counts = torch.bincount(labels, minlength=2)
    if counts.numel() != 2 or counts[0] != counts[1] or counts[0] == 0:
        raise ValueError("K-shot requires equal, positive Real/Fake Support counts")
    prototypes = build_prototypes(support, labels, 2)
    return prototypes, alpha


def kshot_scores(query, prototypes, alpha):
    logits = cosine_logits(query, prototypes)
    return ((logits[:, :, 1] - logits[:, :, 0]) * alpha).sum(-1)


def predict(scores):
    return (scores >= 0).long()


@dataclass
class PrototypeBank:
    classes: list[str]
    prototypes: torch.Tensor
    layer_logits: torch.Tensor
    metadata: dict

    def __post_init__(self):
        if (self.metadata.get("method_version") != METHOD_VERSION or
                self.layer_logits.shape != (25,) or not torch.isfinite(self.layer_logits).all()):
            raise ValueError("Incompatible FS-LPRF-LW Prototype Bank")

    @property
    def alpha(self):
        return layer_weights(self.layer_logits)

    def save(self, path):
        torch.save(dict(classes=self.classes, prototypes=self.prototypes.cpu(),
                        layer_logits=self.layer_logits.cpu(), metadata=self.metadata), path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "layer_logits" not in payload or "dispersion" in payload:
            raise ValueError("Incompatible FS-LPRF-LW Prototype Bank; rebuild from a LW checkpoint")
        return cls(**payload)


def bank_from_features(features_by_class, layer_logits, metadata=None):
    """Used by synthetic tests; real construction selects exactly 1024/class upstream."""
    classes = ["real"] + sorted(set(features_by_class) - {"real"})
    prototypes = []
    for name in classes:
        features = features_by_class[name].float()
        proto = features.mean(0, keepdim=True)
        prototypes.append(proto[0])
    metadata = dict(metadata or {}, method_version=METHOD_VERSION)
    return PrototypeBank(classes, torch.stack(prototypes), layer_logits.detach().cpu().clone(), metadata)


def zeroshot_scores(query, bank):
    prototypes = bank.prototypes.to(query.device)
    alpha = bank.alpha.to(query.device)
    logits = cosine_logits(query, prototypes)
    fake = logits[:, :, 1:].max(-1).values
    return ((fake - logits[:, :, 0]) * alpha).sum(-1)
