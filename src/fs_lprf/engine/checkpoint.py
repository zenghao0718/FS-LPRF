from pathlib import Path
import torch
from fs_lprf.upstream import UPSTREAM_COMMIT
from fs_lprf.utils.seed import capture_rng, restore_rng


def save_checkpoint(path, model, optimizer, scheduler, config, step,
                    dev_metrics, best_eer, manifest_hash, train_fingerprint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1, "method_version": "2.4", "implementation_version": "1.3",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "config": config, "step": step,
        "train_seed": config["train"]["seed"], "dev_metrics": dev_metrics,
        "best_eer": best_eer, "frontend_path": config["model"]["frontend_path"],
        "upstream_commit": UPSTREAM_COMMIT, "rng": capture_rng(),
        "dev_manifest_sha256": manifest_hash, "train_fingerprint": train_fingerprint,
    }
    if any(key.startswith("frontend_model.") for key in payload["model"]):
        raise ValueError("Frozen frontend must be omitted by the official state_dict implementation")
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, restore_random=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("upstream_commit") != UPSTREAM_COMMIT or payload.get("method_version") != "2.4":
        raise ValueError("Incompatible FS-LPRF checkpoint")
    expected = set(model.state_dict())
    if set(payload["model"]) != expected:
        raise ValueError("Checkpoint must exactly match trainable Attention and shared Bottleneck keys")
    result = model.load_state_dict(payload["model"], strict=False)
    if result.unexpected_keys or any(not key.startswith("frontend_model.") for key in result.missing_keys):
        raise ValueError(f"Invalid checkpoint keys: {result}")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_random:
        restore_rng(payload["rng"])
    return payload
