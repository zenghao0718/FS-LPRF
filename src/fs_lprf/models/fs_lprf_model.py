from argparse import Namespace
import torch
from torch import nn
from fs_lprf.upstream import enable_upstream
from .frontend_compat import install_frontend_compatibility

enable_upstream()
from tomato.models.frontend_only import XLSRTimeAttnOnly


class FSLPRF(XLSRTimeAttnOnly):
    """Official 25-layer time attention followed by one shared 1024 -> 256 Linear."""
    def __init__(self, cfg):
        device = torch.device(cfg["runtime"]["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The official Layer-Wise model requires an available CUDA device")
        install_frontend_compatibility()
        args = Namespace(**cfg["model"], cuda=device.index if device.index is not None else 0)
        super().__init__(args)
        self.bottleneck_proj = nn.Linear(self.frontend_dim, cfg["model"]["bottleneck_dim"])
        self.to(self.device)
        self.frontend_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frontend_model is not None:
            self.frontend_model.eval()
        return self

    def forward(self, source):
        pooled = super().forward(source)["feats"]
        if pooled.shape[1:] != (25, 1024):
            raise ValueError(f"Expected complete 25-layer official attention output, got {pooled.shape}")
        return {"feats": self.bottleneck_proj(pooled)}
