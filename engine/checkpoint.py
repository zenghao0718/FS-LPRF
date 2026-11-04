import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import torch
from utils.provenance import UPSTREAM_COMMIT
from utils.seed import capture_rng, restore_rng


def save_checkpoint(path, model, optimizer, scheduler, config, step,
                    dev_metrics, best_eer, manifest_hash, train_fingerprint, *,
                    run_identity=None, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists; refusing to overwrite: {path}")
    payload = {
        "version": 1, "method_version": "2.4", "implementation_version": "1.3",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "config": config, "step": step,
        "train_seed": config["train"]["seed"], "dev_metrics": dev_metrics,
        "best_eer": best_eer, "frontend_path": config["model"]["frontend_path"],
        "upstream_commit": UPSTREAM_COMMIT, "rng": capture_rng(),
        "dev_manifest_sha256": manifest_hash, "train_fingerprint": train_fingerprint,
    }
    if run_identity is not None:
        payload["run_identity"] = run_identity
    if any(key.startswith("frontend_model.") for key in payload["model"]):
        raise ValueError("Frozen frontend must be omitted by the official state_dict implementation")
    # The temporary file belongs to this call, so failure cannot replace a node or
    # mistake a pre-existing temporary file for a finished checkpoint.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.stem}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp, path)
        else:
            # Atomic publication without an overwrite race. Both NTFS and ext4
            # support this same-directory hard link; it does not duplicate data.
            os.link(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def load_checkpoint(path, model, optimizer=None, scheduler=None, restore_random=False,
                    *, expected_run_identity=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("upstream_commit") != UPSTREAM_COMMIT or
            payload.get("method_version") != "2.4"):
        raise ValueError("Incompatible FS-LPRF checkpoint")
    if expected_run_identity is not None and payload.get("run_identity") != expected_run_identity:
        raise ValueError("Checkpoint run identity does not match this run")
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


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(reference, output_dir):
    """Resolve an explicit checkpoint path or a validated best-node index.

    The best index is relative to the configured run's checkpoints directory and
    can name only one real step_XXXX.pt node in that directory. It never falls
    back to an unrelated best.pt or trusts a path embedded in an arbitrary JSON.
    """
    if str(reference) != "best":
        path = Path(reference).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    run_dir = Path(output_dir).resolve()
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    if checkpoint_dir.parent != run_dir:
        raise ValueError("Checkpoint directory escapes the configured output directory")
    index_path = checkpoint_dir / "best_checkpoint.json"
    if index_path.resolve().parent != checkpoint_dir:
        raise ValueError("Best checkpoint index escapes the checkpoint directory")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    required = {"version", "checkpoint", "step", "mean_eer", "checkpoint_sha256",
                "run_identity", "dev_manifest_sha256", "train_fingerprint"}
    if not isinstance(index, dict) or not required.issubset(index):
        raise ValueError("Invalid best checkpoint index schema")
    step = index["step"]
    if (type(index["version"]) is not int or index["version"] != 1 or
            type(step) is not int or step <= 0):
        raise ValueError("Invalid best checkpoint index version or step")
    name = index["checkpoint"]
    if not isinstance(name, str) or re.fullmatch(r"step_[0-9]{4}\.pt", name) is None:
        raise ValueError("Best index must reference a step_XXXX.pt basename in checkpoints")
    if name != f"step_{step:04d}.pt":
        raise ValueError("Best checkpoint filename and index step differ")
    mean_eer = index["mean_eer"]
    if (type(mean_eer) not in (int, float) or not math.isfinite(mean_eer) or
            not 0 <= mean_eer <= 1):
        raise ValueError("Best checkpoint EER must be a finite ratio in [0,1]")
    for key in ("checkpoint_sha256", "run_identity", "dev_manifest_sha256", "train_fingerprint"):
        if not isinstance(index[key], str) or re.fullmatch(r"[0-9a-f]{64}", index[key]) is None:
            raise ValueError(f"Invalid best checkpoint {key}")
    path = (checkpoint_dir / name).resolve()
    if path.parent != checkpoint_dir or path.name != name:
        raise ValueError("Best checkpoint path escapes or aliases another node")
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint node does not exist: {path}")
    if checkpoint_sha256(path) != index["checkpoint_sha256"]:
        raise ValueError("Best checkpoint SHA-256 does not match its index")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("upstream_commit") != UPSTREAM_COMMIT or
            payload.get("method_version") != "2.4"):
        raise ValueError("Incompatible best FS-LPRF checkpoint")
    for key in ("step", "run_identity", "dev_manifest_sha256", "train_fingerprint"):
        if payload.get(key) != index[key]:
            raise ValueError(f"Best checkpoint {key} does not match its index")
    return path
