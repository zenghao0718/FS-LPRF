import copy
from pathlib import Path

import pytest

from data.cache_pipeline import dev_manifests
from data.evaluation_manifest import load_or_create_manifest
from engine.evaluator import evaluate
from engine.validation_schedule import ValidationHistory, make_run_identity
from utils.config import load_config
from test_evaluation import FeatureDataset, FeatureModel
from test_validation_reports import CaptureLogger, csv_rows
from test_validation_schedule import harness, result


@pytest.mark.parametrize("seeds", ["[42]", "[17,42,71,100,123]"])
def test_only_two_dev_protocols_are_allowed(seeds):
    cfg = load_config("configs/speechfake.yaml", [f"validation.seeds={seeds}"])
    assert cfg["validation"]["light_seed"] == 42
    assert cfg["test"]["seeds"] == [17, 42, 71, 100, 123]


@pytest.mark.parametrize("seeds", ["[17]", "[42,17]", "[]", "[42.0]", "[17,42,71]"])
def test_other_dev_seed_lists_are_rejected(seeds):
    with pytest.raises(ValueError, match="validation.seeds"):
        load_config("configs/speechfake.yaml", [f"validation.seeds={seeds}"])


def test_single_seed_manifests_metrics_and_official_directory_isolation(dataset, config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["validation"]["seeds"] = [42]
    cfg["runtime"]["output_dir"] = str(tmp_path / "development")
    manifests = dev_manifests(dataset, cfg)
    full, light = manifests["full"][0], manifests["light"][0]
    assert full["seeds"] == light["seeds"] == [42]
    assert full["supports"] == light["supports"]
    for name in full["classes"]:
        assert set(light["queries"][name]) <= set(full["queries"][name])
        assert not set(full["queries"][name]) & set(full["supports"]["42"][name])
    official = copy.deepcopy(cfg)
    official["validation"]["seeds"] = [17, 42, 71, 100, 123]
    with pytest.raises(ValueError, match="separate run directory"):
        dev_manifests(dataset, official)
    official["runtime"]["output_dir"] = str(tmp_path / "official")
    assert dev_manifests(dataset, official)["full"][0]["seeds"] == official["validation"]["seeds"]
    ds = FeatureDataset(dataset)
    summary = evaluate(FeatureModel(), ds, full, cfg, tmp_path / "reports", "dev", 800,
                       mode="full", logger=CaptureLogger())
    assert summary["seed_count"] == 1 and summary["aggregation"] == "single_seed"
    assert [row["seed"] for row in summary["per_seed"]] == [42]
    assert all(value["std"] == 0 for value in summary["metrics"].values())
    macro = csv_rows(tmp_path / "reports/full/dev_full_macro_mean_std_step0800.csv")
    assert all(row["seed_count"] == "1" and row["aggregation"] == "single_seed" for row in macro)
    assert make_run_identity(cfg, "full", "light", "train") != make_run_identity(official, "full", "light", "train")


def test_single_seed_best_resume_and_cross_protocol_rejection(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["validation"]["seeds"] = [42]
    cfg, history, _, _, _ = harness(tmp_path, cfg)
    history.initialize()
    history.checkpoint(800).parent.mkdir(parents=True)
    history.checkpoint(800).write_bytes(b"synthetic checkpoint for history hash only")
    summary = result(tmp_path / "evaluation", 800, "full")
    summary["per_seed"] = [row for row in summary["per_seed"] if row["seed"] == 42]
    with pytest.raises(ValueError, match="seed_count"):
        history.complete(800, summary)
    summary.update(seed_count=1, aggregation="single_seed")
    history.complete(800, summary)
    resumed = ValidationHistory(tmp_path, cfg, history.identity, "a" * 64, "c" * 64)
    resumed.initialize(800)
    assert resumed.best["step"] == 800
    official = copy.deepcopy(cfg)
    official["validation"]["seeds"] = [17, 42, 71, 100, 123]
    other = ValidationHistory(tmp_path, official, make_run_identity(official, "a" * 64, "b" * 64, "c" * 64),
                               "a" * 64, "c" * 64)
    with pytest.raises(ValueError, match="identity"):
        other.initialize(800)
