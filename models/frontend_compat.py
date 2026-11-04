"""Verified official Fairseq checkpoint I/O for the integrated XLS-R model.

The original file, tensors, network structure and hidden-state order are preserved.
"""
import hashlib
import logging
from pathlib import Path
import pickle
from argparse import Namespace
import torch
from utils.provenance import XLSR_URL as OFFICIAL_URL, XLSR_SHA256 as OFFICIAL_SHA256
LOAD_REPORT = {}

# The raw 2021 snapshot predates these fields. Fairseq's own loader merges the
# Wav2Vec2Config defaults; the upstream converter then writes a Namespace.
# Source: facebookresearch/fairseq v0.12.2, Wav2Vec2Config (not new model settings).
FAIRSEQ_COMPAT_DEFAULTS = dict(crop_seq_to_multiple=1, layer_type="transformer",
    mask_channel_before=False, quantizer_depth=1, quantizer_factor=3,
    required_seq_len_multiple=2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_official_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing official XLS-R checkpoint: {path}; run scripts/download_frontend.py")
    digest = sha256_file(path)
    if digest != OFFICIAL_SHA256:
        raise ValueError("Frontend loading (including any weights_only=False fallback) is restricted to the verified official XLS-R SHA-256")
    LOAD_REPORT.clear()
    LOAD_REPORT.update(path=str(path), map_location="cpu", unsafe_fallback=False,
                       sha256=digest, bytes=path.stat().st_size, source=OFFICIAL_URL)
    try:
        # Intentionally try the official interface/default first, as required by spec 1.3.
        payload = torch.load(path, map_location="cpu")
        LOAD_REPORT["default_load"] = "passed"
    except pickle.UnpicklingError as error:
        if "Weights only load failed" not in str(error):
            raise
        LOAD_REPORT["default_load"] = str(error)
        logging.getLogger("fs_lprf").info("Official Fairseq checkpoint requires weights_only=False on CPU (verified SHA-256)")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        LOAD_REPORT.update(unsafe_fallback=True, sha256=digest)
    # Original Fairseq training snapshots store cfg.model and model. The pinned
    # Integrated XLS-R loader expects the same objects as cfg and state_dict.
    if "state_dict" not in payload and "model" in payload:
        cfg = payload["cfg"]
        model_cfg = cfg["model"] if isinstance(cfg, dict) else cfg.model
        if not isinstance(model_cfg, dict):
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        missing = {key: value for key, value in FAIRSEQ_COMPAT_DEFAULTS.items() if key not in model_cfg}
        model_cfg = Namespace(**dict(FAIRSEQ_COMPAT_DEFAULTS, **model_cfg))
        payload = {"cfg": model_cfg, "state_dict": payload["model"]}
        LOAD_REPORT["schema_adapter"] = "cfg.model -> cfg; model -> state_dict (unchanged tensors)"
        LOAD_REPORT["fairseq_legacy_defaults"] = missing
    return payload
