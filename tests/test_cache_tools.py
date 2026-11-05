import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from data.cache_pipeline import (cache_specs, estimate_cache, prepare_final_test_caches,
                                 evaluation_identities, require_cache)
from data.speechfake_dataset import SpeechFakeDataset
from utils.config import load_config
from test_cache_storage import cache_config
from test_cache_integration import _cli
from test_cached_features import synthetic_model


def test_waveform_builder_dry_run_build_and_integrity_cli(dataset, config, tmp_path, monkeypatch, capsys):
    import sys
    cli = _cli("build_cache", monkeypatch)
    cfg = cache_config(dataset, config, tmp_path)
    monkeypatch.setattr(cli, "load_config", Mock(return_value=cfg))
    base = ["build_cache.py", "--split", "train", "--mode", "waveform", "--limit", "2"]
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    cli.main()
    assert not Path(cfg["cache"]["root"]).exists()
    assert json.loads(capsys.readouterr().out)["dry_run"]
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    generated = capsys.readouterr().out
    assert '"status": "complete"' in generated
    monkeypatch.setattr(sys, "argv", base + ["--check"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    raw, identities = cache_specs(cfg, "train", "waveform", limit=2)
    assert identities[0]["smoke"]
    with pytest.raises(ValueError, match="Smoke/incomplete"):
        require_cache(cfg, identities[0])


def test_feature_estimate_measures_time_dimension_and_large_cache_confirmation(dataset, config, tmp_path, monkeypatch):
    from models import fs_lprf_model
    import sys
    cfg = cache_config(dataset, config, tmp_path)
    raw, identities = cache_specs(cfg, "train", "xlsr_feature", limit=2)
    estimate = estimate_cache(raw, cfg, identities[0], synthetic_model())
    assert estimate["frames"] == 3 and estimate["bytes"] == 2*25*3*1024*4
    cli = _cli("build_cache", monkeypatch)
    cfg["cache"]["warn_if_estimated_gb_above"] = 1e-9
    monkeypatch.setattr(cli, "load_config", Mock(return_value=cfg))
    monkeypatch.setattr(fs_lprf_model, "FSLPRF", lambda cfg: synthetic_model())
    monkeypatch.setattr(sys, "argv", ["build_cache.py", "--split", "train", "--mode", "xlsr_feature", "--limit", "2"])
    with pytest.raises(ValueError, match="confirm-large-cache"):
        cli.main()
    assert not Path(cfg["cache"]["root"]).exists()


def test_final_test_dry_run_has_no_cache_writes_or_backend_metrics(dataset, config, tmp_path, monkeypatch):
    from data.evaluation_manifest import load_or_create_manifest
    cfg = cache_config(dataset, config, tmp_path)
    ds = SpeechFakeDataset(cfg["data"], "test")
    path = tmp_path / "test_manifest.json"
    manifest = load_or_create_manifest(ds, path, 10, cfg["test"]["seeds"])
    model = synthetic_model()
    model.forward_from_frontend_features = Mock(side_effect=AssertionError("No backend or metrics during cache planning"))
    report = prepare_final_test_caches(ds, cfg, manifest, path, model, dry_run=True)
    assert report["dry_run"] and len(report["estimates"]) == 5
    assert not Path(cfg["cache"]["root"]).exists()
    model.forward_from_frontend_features.assert_not_called()
    cfg["cache"]["warn_if_estimated_gb_above"] = 1e-9
    with pytest.raises(ValueError, match="confirm-large-cache"):
        prepare_final_test_caches(ds, cfg, manifest, path, model)


@pytest.mark.parametrize("override", ["data.train_cache_mode=bad", "data.dev_cache_mode=waveform", "cache.feature_dtype=bfloat16",
                                     "cache.schema_version=2", "cache.feature_build_batch_size=0", "runtime.train_num_workers=-1",
                                     "runtime.prefetch_factor=0", "runtime.pin_memory=1", "cache.root=models/cached"])
def test_invalid_cache_and_worker_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config("configs/speechfake.yaml", [override])


def test_check_data_retains_audits_with_explicit_train_dev_only(dataset, config, rir_root, tmp_path, monkeypatch):
    import sys
    cli = _cli("check_data", monkeypatch)
    cfg = cache_config(dataset, config, tmp_path)
    cfg["data"]["rir_root"] = str(rir_root)
    original = cli.SpeechFakeDataset
    splits = []
    def create(options, split, **kwargs):
        assert split != "test"
        splits.append(split)
        result = original(options, split, **kwargs)
        # Synthetic fixtures share one WAV intentionally; use separate virtual paths
        # for the cross-split audit while keeping every class/capacity check live.
        if split == "dev":
            result.uttid2path = {uid: str(tmp_path / "dev_source.wav") for uid in result.uttids}
            (tmp_path / "dev_source.wav").write_bytes(b"metadata-only synthetic audit")
        return result
    monkeypatch.setattr(cli, "load_config", Mock(return_value=cfg))
    monkeypatch.setattr(cli, "SpeechFakeDataset", create)
    monkeypatch.setattr(sys, "argv", ["check_data.py", "--read-samples", "0", "--splits", "train", "dev"])
    cli.main()
    assert splits == ["train", "dev"]


def test_dev_benchmark_identity_binds_inputs_and_timing_settings(dataset, config, tmp_path):
    from engine.pipeline_benchmark import dev_measurement_identity
    cfg = cache_config(dataset, config, tmp_path)
    original = dev_measurement_identity(cfg)
    changed = copy.deepcopy(cfg)
    changed["cache"]["root"] = str(tmp_path / "another_cache_location")
    assert dev_measurement_identity(changed) == original
    changed["runtime"]["eval_num_workers"] = 8
    assert dev_measurement_identity(changed) != original
    Path(cfg["model"]["frontend_path"]).write_bytes(b"different synthetic frontend")
    assert dev_measurement_identity(cfg) != original


def test_source_checksums_are_not_reread_for_each_benchmark_dataset(dataset, config, tmp_path, monkeypatch):
    import builtins
    from data.cache import source_digest
    cfg = cache_config(dataset, config, tmp_path)
    first = SpeechFakeDataset(cfg["data"], "train")
    second = SpeechFakeDataset(cfg["data"], "train")
    path = Path(first.uttid2path[first.uttids[0]]).resolve()
    original_open = builtins.open
    reads = []
    def opened(filename, mode="r", *args, **kwargs):
        if Path(filename).resolve() == path and mode == "rb":
            reads.append(path)
        return original_open(filename, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", opened)
    initial = source_digest(first, first.uttids[:2])
    assert source_digest(second, second.uttids[:2]) == initial and len(reads) == 1
    path.write_bytes(path.read_bytes() + b"changed source")
    assert source_digest(second, second.uttids[:2]) != initial and len(reads) == 2
