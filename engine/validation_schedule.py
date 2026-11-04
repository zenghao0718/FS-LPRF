"""Validation schedule and durable, checkpoint-bound full-Dev history."""
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


def validation_mode(step, train_config):
    if step % train_config["full_validation_interval_steps"] == 0:
        return "full"
    if step % train_config["light_validation_interval_steps"] == 0:
        return "light"
    return None


def milestone_steps(cfg):
    return list(range(cfg["train"]["full_validation_interval_steps"],
                      cfg["train"]["total_steps"] + 1,
                      cfg["train"]["full_validation_interval_steps"]))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temp.replace(path)


def make_run_identity(cfg, manifest_hash, light_manifest_hash, train_fingerprint):
    # Paths, worker count, microbatch and scientific settings are fixed for a resumed run.
    # CUDA ordinal may change when the same run is restored on a different host/GPU.
    config = json.loads(json.dumps(cfg))
    config["runtime"].pop("device", None)
    payload = dict(version=1, config=config, dev_manifest_sha256=manifest_hash,
                   light_manifest_sha256=light_manifest_hash, train_fingerprint=train_fingerprint)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_new_run(out):
    out = Path(out)
    names = ("run_state.json", "config_resolved.yaml", "train.log", "episodes.jsonl")
    if any((out / name).exists() for name in names):
        raise ValueError("Run already exists: resume a milestone or choose a new output_dir")
    for name in ("checkpoints", "tensorboard", "evaluation/full", "evaluation/light"):
        path = out / name
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Existing run artifacts in {path}; do not overwrite them")


class ValidationHistory:
    def __init__(self, out, cfg, run_identity, manifest_hash, train_fingerprint):
        self.out = Path(out).resolve()
        self.cfg, self.identity = cfg, run_identity
        self.manifest_hash, self.train_fingerprint = manifest_hash, train_fingerprint
        self.records = []

    def checkpoint(self, step):
        return self.out / "checkpoints" / f"step_{step:04d}.pt"

    def marker(self, step):
        return self.out / "evaluation" / "full" / f"dev_full_complete_step{step:04d}.json"

    def initialize(self, resume_step=None):
        state_path = self.out / "run_state.json"
        if resume_step is None:
            if state_path.exists():
                raise ValueError("Training run state already exists")
            atomic_json(state_path, dict(version=1, run_identity=self.identity,
                                         dev_manifest_sha256=self.manifest_hash,
                                         train_fingerprint=self.train_fingerprint))
            return
        if resume_step not in milestone_steps(self.cfg):
            raise ValueError("Resume requires a configured milestone step")
        if not state_path.is_file() or json.loads(state_path.read_text(encoding="utf-8")).get("run_identity") != self.identity:
            raise ValueError("Resume configuration/manifests/data identity does not match run_state.json")
        # Rewinding into a directory containing later results would overwrite its nodes.
        # Require the latest node instead, rather than silently mixing runs or discarding files.
        paths = list((self.out / "checkpoints").glob("*.pt"))
        for path in paths:
            match = re.fullmatch(r"step_(\d{4,})\.pt", path.name)
            if not match or int(match[1]) not in milestone_steps(self.cfg) or int(match[1]) > resume_step:
                raise ValueError("Existing future/legacy checkpoint: resume the latest milestone or use a separate output directory")
        for path in (self.out / "evaluation" / "full").glob("dev_full_complete_step*.json"):
            match = re.fullmatch(r"dev_full_complete_step(\d{4,})\.json", path.name)
            if not match or int(match[1]) > resume_step:
                raise ValueError("Future full-Dev history exists; refusing to mix it with an older resume node")
        for step in milestone_steps(self.cfg):
            if step > resume_step:
                break
            if not self.checkpoint(step).is_file():
                raise ValueError(f"Missing earlier milestone checkpoint: {step}")
            record = self.read_complete(step)
            if record is None:
                if step < resume_step:
                    raise ValueError(f"Earlier full-Dev history incomplete at {step}; retain all files and restore the matching complete reports/output directory")
            else:
                self.records.append(record)
        self.update_best()

    def _report_paths(self, step):
        names = ("per_class_seed", "per_class_mean_std", "macro_per_seed", "macro_mean_std", "predictions")
        return {f"evaluation/full/dev_full_{name}_step{step:04d}.csv" for name in names}

    def read_complete(self, step):
        path = self.marker(step)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None  # A partial marker is never considered successful validation.
        required = {"version", "run_identity", "step", "dev_manifest_sha256", "train_fingerprint",
                    "checkpoint_sha256", "reports", "summary"}
        if not isinstance(record, dict) or not required.issubset(record):
            return None
        if record["version"] != 1:
            raise ValueError("Unsupported full-Dev completion marker version")
        if (record.get("run_identity") != self.identity or record.get("step") != step or
                record.get("dev_manifest_sha256") != self.manifest_hash or
                record.get("train_fingerprint") != self.train_fingerprint):
            raise ValueError("Full-Dev completion marker identity mismatch")
        if record.get("checkpoint_sha256") != file_sha256(self.checkpoint(step)):
            raise ValueError("Full-Dev checkpoint checksum mismatch")
        reports = record.get("reports", {})
        if not isinstance(reports, dict) or set(reports) != self._report_paths(step):
            return None
        for relative, digest in reports.items():
            report_path = self.out / relative
            if (not report_path.resolve().is_relative_to(self.out) or not report_path.is_file() or
                    file_sha256(report_path) != digest):
                return None
        self._check_summary(record["summary"])
        return record

    def _check_summary(self, summary):
        values = summary["per_seed"]
        if [row["seed"] for row in values] != self.cfg["validation"]["seeds"]:
            raise ValueError("Full-Dev summary must contain all five seeds in order")
        for name in ("eer", "auc", "ap", "acc"):
            metric = summary["metrics"][name]
            if any(not math.isfinite(float(row[name])) or not 0 <= row[name] <= 1 for row in values):
                raise ValueError("Invalid full-Dev macro metric")
            mean = sum(row[name] for row in values) / len(values)
            std = (sum((row[name] - mean) ** 2 for row in values) / len(values)) ** .5
            if not math.isclose(mean, metric["mean"], abs_tol=1e-12) or not math.isclose(std, metric["std"], abs_tol=1e-12):
                raise ValueError("Full-Dev summary mean/std mismatch")
        if not math.isclose(summary["mean_eer"], summary["metrics"]["eer"]["mean"], abs_tol=1e-12):
            raise ValueError("Full-Dev EER summary mismatch")

    def complete(self, step, result):
        self._check_summary(result)
        reports = {}
        for name in result["reports"]:
            path = Path(name).resolve()
            if not path.is_relative_to(self.out):
                raise ValueError("Evaluation report is outside the run directory")
            relative = path.relative_to(self.out).as_posix()
            reports[relative] = file_sha256(path)
        if set(reports) != self._report_paths(step):
            raise ValueError("Full-Dev report set is incomplete")
        record = dict(version=1, step=step, run_identity=self.identity,
                      dev_manifest_sha256=self.manifest_hash, train_fingerprint=self.train_fingerprint,
                      checkpoint_sha256=file_sha256(self.checkpoint(step)), reports=reports,
                      summary=result)
        atomic_json(self.marker(step), record)  # Written only after all reports succeeded.
        self.records = [row for row in self.records if row["step"] != step] + [record]
        self.update_best()
        return record

    @property
    def best(self):
        return min(self.records, key=lambda row: (row["summary"]["mean_eer"], row["step"])) if self.records else None

    def update_best(self):
        record = self.best
        if record is None:
            # A stale index must not keep pointing at an unverified validation.
            # Preserve its original content while making `--checkpoint best` fail closed.
            path = self.out / "checkpoints" / "best_checkpoint.json"
            if path.exists():
                original = path.read_text(encoding="utf-8")
                try:
                    pending = json.loads(original)
                except ValueError:
                    pending = {}
                if not isinstance(pending, dict):
                    pending = {}
                if pending.get("status") != "pending_full_dev" or pending.get("run_identity") != self.identity:
                    atomic_json(path, dict(version=1, status="pending_full_dev", run_identity=self.identity,
                                           previous_index=original))
            return
        index = dict(version=1, checkpoint=self.checkpoint(record["step"]).name,
                     step=record["step"], mean_eer=record["summary"]["mean_eer"],
                     checkpoint_sha256=record["checkpoint_sha256"], run_identity=self.identity,
                     dev_manifest_sha256=self.manifest_hash, train_fingerprint=self.train_fingerprint,
                     completion_marker=self.marker(record["step"]).relative_to(self.out).as_posix())
        atomic_json(self.out / "checkpoints" / "best_checkpoint.json", index)
