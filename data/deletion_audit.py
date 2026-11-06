"""Read-only source-removal audit for SpeechFake Train/Dev waveform caches.

This module never deletes, moves or rewrites source audio. Only individual
manifest-referenced WAV paths can become deletion suggestions.
"""
import os
from pathlib import Path
import shutil

from .cache import source_digest
from .speechfake_dataset import SpeechFakeDataset
from .waveform_cache import open_waveform_cache


def _key(path):
    return os.path.normcase(str(Path(path).resolve()))


def audit_source_deletion(cfg, datasets=None):
    if cfg["data"]["test_cache_mode"] != "none":
        raise ValueError("Deletion audit requires raw Test input")
    if datasets is None:
        datasets = {split: SpeechFakeDataset(cfg["data"], split, train_mode=False)
                    for split in ("train", "dev", "test")}
    if set(datasets) != {"train", "dev", "test"}:
        raise ValueError("Deletion audit requires all three manifests, including Test")
    errors, cache_reports, source_checks = [], {}, {}
    mappings = [getattr(datasets[split], "fake_classes", None) for split in ("train", "dev", "test")]
    if mappings[1:] != mappings[:-1]:
        errors.append("Train/Dev/Test class mappings differ")
    for split in ("train", "dev"):
        ds = datasets[split]
        if cfg["data"][f"{split}_cache_mode"] != "waveform":
            errors.append(f"{split}: Waveform Cache mode is required")
            continue
        try:
            reader = open_waveform_cache(ds, cfg, verify_checksums=True)
            cache_reports[split] = reader.verify()
            missing = sorted({path for path in ds.uttid2path.values() if not Path(path).is_file()})
            if missing:
                source_checks[split] = dict(status="missing_sources", paths=missing)
                errors.append(f"{split}: original source verification cannot finish; {len(missing)} missing paths")
            elif source_digest(ds, ds.uttids) != reader.identity["source_audio_sha256"]:
                source_checks[split] = dict(status="changed_since_cache_build")
                errors.append(f"{split}: current source SHA differs from cache build history")
            else:
                source_checks[split] = dict(status="matches_recorded_source_sha256")
        except (ValueError, OSError, RuntimeError) as error:
            errors.append(f"{split}: {error}")
    test_paths, test_files, test_ids, test_sizes = {}, {}, set(), {}
    for uid, raw in datasets["test"].uttid2path.items():
        path = Path(raw)
        key = _key(path)
        test_paths.setdefault(key, []).append(uid)
        test_files.setdefault(key, str(path.absolute()))
        try:
            stat = path.stat()
            if not path.is_file() or stat.st_size <= 0:
                raise ValueError("not a nonempty regular file")
            test_ids.add((stat.st_dev, stat.st_ino))
            test_sizes[key] = stat.st_size
        except (OSError, ValueError) as error:
            errors.append(f"test: missing/unusable original WAV {uid}: {path}: {error}")
    manifest_root = Path(cfg["data"]["data_path"]).resolve()
    essentials = [dict(path=str(manifest_root/f"{split}.{suffix}"), reason="manifest_and_labels")
                  for split in ("train", "dev", "test") for suffix in ("tsv", "txt")]
    weight = cfg["model"].get("frontend_path")
    if weight:
        essentials.append(dict(path=str(Path(weight).resolve()), reason="model_weights"))
    rir = cfg["data"].get("rir_root") or os.environ.get("FS_LPRF_RIR_ROOT")
    protected_dirs = [Path(cfg["cache"]["root"]).resolve()]
    essentials.append(dict(path=str(protected_dirs[0]), reason="waveform_cache"))
    if rir:
        protected_dirs.append(Path(rir).resolve())
        essentials.append(dict(path=str(protected_dirs[-1]), reason="RIR_resources"))
    protected_exact = {_key(row["path"]) for row in essentials}
    sources = {}
    for split in ("train", "dev"):
        ds = datasets[split]
        for uid, raw in ds.uttid2path.items():
            path = Path(raw)
            key = _key(path)
            row = sources.setdefault(key, dict(path=str(path.absolute()), uttids=[], splits=[],
                                               bytes=0, reasons=[]))
            row["uttids"].append(uid)
            if split not in row["splits"]:
                row["splits"].append(split)
            if any(field in ds.uttid2info[uid] for field in ("st", "dur")):
                row["reasons"].append("special_random_file_read_not_cacheable")
    eligible, retained = [], []
    for key, row in sources.items():
        path = Path(row["path"])
        resolved = path.resolve()
        if key in test_paths:
            row["reasons"].append("referenced_by_Test_manifest")
        if key in protected_exact or any(resolved.is_relative_to(directory) for directory in protected_dirs):
            row["reasons"].append("required_project_resource")
        if path.suffix.lower() != ".wav":
            row["reasons"].append("not_a_WAV_deletion_candidate")
        if path.is_symlink():
            row["reasons"].append("symlink_requires_manual_review")
        try:
            stat = path.stat()
            row["bytes"] = stat.st_size
            if not path.is_file():
                row["reasons"].append("not_a_regular_file")
            if (stat.st_dev, stat.st_ino) in test_ids:
                row["reasons"].append("same_physical_file_as_Test")
            if stat.st_nlink > 1:
                row["reasons"].append("hardlink_requires_manual_review")
        except OSError:
            row["reasons"].append("original_file_missing")
        if errors:
            row["reasons"].append("audit_prerequisites_not_satisfied")
        row["reasons"] = sorted(set(row["reasons"]))
        (retained if row["reasons"] else eligible).append(row)
    for key, raw in test_files.items():
        if key not in sources:
            retained.append(dict(path=raw, uttids=test_paths[key], splits=["test"],
                                 bytes=test_sizes.get(key), reasons=["Test_original_WAV_must_be_retained"]))
    probe = Path(cfg["cache"]["root"]).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return dict(status="passed" if not errors else "blocked", errors=errors,
                cache_verification=cache_reports, source_verification=source_checks,
                deletable_files=eligible, retained_files=retained, retained_essentials=essentials,
                deletable_file_count=len(eligible), estimated_reclaimable_logical_bytes=sum(row["bytes"] for row in eligible),
                cache_filesystem_free_bytes=shutil.disk_usage(probe).free,
                deletion_performed=False,
                note="Suggestions only. Never remove the dataset root. Test WAVs, manifests, labels, caches, RIRs "
                     "and weights must stay. Logical byte sums are estimates, not allocated disk bytes.")
