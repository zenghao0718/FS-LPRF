"""Provenance for an augmented prototype bank. It does not alter waveforms."""
import hashlib
import json
from pathlib import Path
import subprocess

from data.augmentation import RIR_INDEX_FILES, get_default_args
from utils.provenance import XLSR_SHA256


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _sanitize(value.item())
        except (ValueError, AttributeError):
            pass
    return str(value)


def git_record(repo, destination):
    repo = Path(repo)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL)

    commit = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--short").decode()
    diff = git("diff", "--binary")
    record = dict(commit=commit, status=status, diff_sha256=hashlib.sha256(diff).hexdigest(), diff_path=None)
    if status.strip():
        path = Path(destination) / "worktree.diff"
        path.write_bytes(diff)
        record["diff_path"] = str(path.resolve())
        record["diff_bytes"] = path.stat().st_size
    return record


def rir_fingerprint(root, path_list=None):
    root = Path(root).resolve()
    indexes = {}
    digest = hashlib.sha256()
    for relative in RIR_INDEX_FILES:
        data = (root / relative).read_bytes()
        indexes[relative] = hashlib.sha256(data).hexdigest()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(data)
    paths = hashlib.sha256()
    if path_list is not None:
        for item in path_list:
            paths.update(str(item).encode() + b"\n")
    return dict(root=str(root), index_sha256=indexes, index_fingerprint=digest.hexdigest(),
                path_count=None if path_list is None else len(path_list),
                path_list_fingerprint=None if path_list is None else paths.hexdigest())


def cache_record(reader):
    identity = reader.identity
    summary = {}
    report = identity.get("codec_report")
    if isinstance(report, dict):
        summary = dict(available_families=report.get("available_families"), skipped=report.get("skipped"),
                       environment=report.get("environment"))
    kept = {key: identity.get(key) for key in (
        "mode", "split", "schema_version", "dataset_fingerprint", "builder_version", "codec_parameters",
        "source_cache", "implementation_sha256", "clipping_fallback_policy", "codec_storage_format",
        "source_commit", "rng_schema_version", "preprocessing")}
    return _sanitize(dict(path=str(reader.path), resolved=str(Path(reader.path).resolve()),
                          fingerprint=reader.metadata.get("fingerprint"),
                          content_fingerprint=reader.metadata.get("content_fingerprint"),
                          compatibility_audit_sha256=reader.metadata.get("compatibility_audit_sha256"),
                          samples=len(getattr(reader, "rows", {})), identity=kept, codec_report_summary=summary))


def rawboost_record(dataset, mode):
    source = dataset.augmentor.rawboost_args if mode == "full" else get_default_args()
    return dict(enabled=mode == "full", algorithm=source.algo, probability=float(1 / 5) if mode == "full" else 0.0,
                parameters={key: _sanitize(value) for key, value in vars(source).items()})


def write_prototype_info(path, *, bank_path, dataset, cfg, mode, repo):
    bank_path = Path(bank_path)
    info = dict(augmentation_mode=mode, bank_path=str(bank_path.resolve()), bank_sha256=_sha256(bank_path),
                bank_bytes=bank_path.stat().st_size, probabilities=dict(
                    codec=0.5 if mode != "clean" else 0.0, rir=1 / 3 if mode == "full" else 0.0,
                    rawboost=1 / 5 if mode == "full" else 0.0),
                order=["preprocess", "codec_or_original", "crop", "rir", "rawboost"],
                xlsr_sha256=XLSR_SHA256, xlsr_path=str(Path(cfg["model"]["frontend_path"]).resolve()),
                xlsr_file_sha256=_sha256(cfg["model"]["frontend_path"]),
                waveform_cache=cache_record(dataset.cache_reader),
                codec_cache=cache_record(dataset.codec_cache_reader),
                rir=rir_fingerprint(cfg["data"]["rir_root"], getattr(getattr(dataset, "augmentor", None), "rir", None)),
                rawboost=rawboost_record(dataset, mode), git=git_record(repo, bank_path.parent))
    if info["xlsr_file_sha256"] != XLSR_SHA256:
        raise ValueError("XLS-R file SHA256 does not match the formal frontend")
    destination = Path(path)
    destination.write_text(json.dumps(_sanitize(info), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info
