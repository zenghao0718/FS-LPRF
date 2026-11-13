"""Frozen XLS-R plus the original Layer-Wise time attention and shared projection.

Attention/extraction derived from XLSRAdapterBase and XLSRTimeAttnOnly by Yixuan,
Layer-Wise commit 88efe1376dfc5f5b9b62d016d33a7381c158d8af.
See docs/CODE_PROVENANCE.md. Parameter names and initialization order are retained.
"""
from argparse import Namespace
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .xlsr import XLSR
from methods.layer_weights import layer_weights


class FSLPRF(nn.Module):
    """25 independent 4-head time attentions, then a shared 1024 -> 256 Linear."""
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg["runtime"]["device"])
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The FS-LPRF XLS-R model requires an available CUDA device")
        options = cfg["model"]
        if (options["frontend"] != "XLSR" or not options["freeze_frontend"] or
                options["num_layers"] != 25 or options["frontend_dim"] != 1024 or
                options["layer_attn_head"] != 4 or options["reduce_dim"] or
                options["attn_cfg"] != "default" or options["have_padding_mask"] or
                options["bottleneck_dim"] != 256):
            raise ValueError("FS-LPRF requires the fixed frozen XLS-R/Attention/Bottleneck architecture")
        self.args = Namespace(**options, cuda=self.device.index if self.device.index is not None else 0)
        self.frontend = "XLSR"
        self.freeze_frontend = True
        self.have_padding_mask = False
        self.frontend_model = XLSR(self.device, self.args)
        for parameter in self.frontend_model.parameters():
            parameter.requires_grad = False
        self.frontend_model.eval()
        self.frontend_dim = 1024
        self.num_layers = 25
        self.layer_attn_head = 4
        # Preserve CPU initialization and device moves in their original order.
        self.time_attn_mlp1 = nn.Parameter(torch.Tensor(25, 1024, 4))
        nn.init.xavier_uniform_(self.time_attn_mlp1)
        self.to(self.device)
        self.bottleneck_proj = nn.Linear(self.frontend_dim, options["bottleneck_dim"])
        self.layer_logits = nn.Parameter(torch.zeros(25))
        self.to(self.device)

    def layer_weights(self):
        """Episode-independent global weights; keep the Softmax gradient intact."""
        return layer_weights(self.layer_logits)

    def train(self, mode=True):
        super().train(mode)
        self.frontend_model.eval()
        return self

    def extract_features(self, source):
        feats = source["feats"].to(self.device, non_blocking=True).squeeze(1)
        with torch.no_grad():
            output = self.frontend_model.model(feats, mask=False, features_only=True)
            post_features = rearrange(output["post_features"], "n t d -> t n d")
            return [post_features] + [layer[0] for layer in output["layer_results"]]

    def extract_frontend_features(self, waveform, *, to_cpu=False):
        """Complete pre-Attention states, [B,25,T,1024], with T from XLS-R."""
        hiddens = self.extract_features({"feats": waveform})
        if len(hiddens) != 25 or any(hidden.shape != hiddens[0].shape for hidden in hiddens):
            raise ValueError("XLS-R must return post_features followed by all 24 transformer states")
        if to_cpu:
            time, batch, dim = hiddens[0].shape
            result = torch.empty(batch, 25, time, dim, dtype=hiddens[0].dtype, device="cpu")
            for layer, hidden in enumerate(hiddens):
                result[:, layer].copy_(hidden.permute(1, 0, 2))
            return result
        return torch.stack([hidden.permute(1, 0, 2) for hidden in hiddens], dim=1)

    def forward_from_frontend_features(self, frontend_features):
        if frontend_features.ndim != 4 or frontend_features.shape[1] != 25 or frontend_features.shape[-1] != 1024:
            raise ValueError("Expected cached frontend features [B,25,T,1024]")
        states = frontend_features.to(self.device, dtype=torch.float32, non_blocking=True)
        if states.shape[2] <= 0 or not torch.isfinite(states).all():
            raise ValueError("Cached frontend features must have nonempty finite states")
        hiddens = [states[:, layer].permute(1, 0, 2) for layer in range(25)]
        return {"feats": self.bottleneck_proj(self.tnd_forward(hiddens))}

    def tnd_forward(self, hiddens, padding_mask=None):
        """Original XLSRTimeAttnOnly computation, without unused alternate heads."""
        T, N, D = hiddens[0].shape
        L = self.num_layers
        stacked_h = torch.stack(hiddens, dim=0)
        stacked_h = rearrange(stacked_h, "l t n d -> l (n t) d")
        time_attn = torch.bmm(stacked_h, self.time_attn_mlp1)
        time_score = torch.logsumexp(time_attn, dim=-1)
        time_score = rearrange(time_score, "l (n t) -> l n t", n=N, t=T)
        time_score = F.softmax(time_score, dim=-1)
        stacked_h = stacked_h.view(L, N, T, D)
        weighted_h = stacked_h * time_score.unsqueeze(-1)
        utt_h = torch.sum(weighted_h, dim=2)
        return rearrange(utt_h, "l n d -> n l d")

    def forward(self, source):
        pooled = self.tnd_forward(self.extract_features(source))
        if pooled.shape[1:] != (25, 1024):
            raise ValueError(f"Expected complete 25-layer attention output, got {pooled.shape}")
        return {"feats": self.bottleneck_proj(pooled)}

    def state_dict(self):
        """Preserve the existing backend-only training checkpoint schema."""
        state = super().state_dict()
        for key in self.frontend_model.state_dict():
            del state[f"frontend_model.{key}"]
        return state


def forward_batch(model, batch):
    """Shared Train/Eval dispatch; pooling and projection stay in the model."""
    if batch.get("input_kind") == "xlsr_feature":
        return model.forward_from_frontend_features(batch["feats"])["feats"]
    return model({"feats": batch["feats"]})["feats"]
