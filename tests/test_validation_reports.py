import copy
import csv
from pathlib import Path
import numpy as np
import pytest
import torch
from data.evaluation_manifest import load_or_create_manifest
from engine.evaluator import evaluate
from utils.metrics import METRICS
from utils.seed import capture_rng, seed_all
from test_evaluation import FeatureDataset, FeatureModel


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifests(dataset, tmp_path):
    full = load_or_create_manifest(dataset, tmp_path / "full_manifest.json", 10, [17, 42, 71, 100, 123])
    light = copy.deepcopy(full)
    light["seeds"] = [42]
    light["supports"] = {"42": full["supports"]["42"]}
    light["queries"] = {name: ids[:5] for name, ids in full["queries"].items()}
    light["queries"]["real"] = full["queries"]["real"][:4]
    light["queries"][dataset.fake_classes[0]] = full["queries"][dataset.fake_classes[0]][:2]
    return full, light


def assert_rng_equal(first, second):
    assert first["python"] == second["python"]
    assert torch.equal(first["numpy"][1], second["numpy"][1])
    assert first["numpy"][2:] == second["numpy"][2:]
    assert torch.equal(first["torch"], second["torch"])
    assert len(first["cuda"]) == len(second["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(first["cuda"], second["cuda"]))


def test_full_dev_reports_all_classes_population_std_logs_and_file_isolation(dataset, config, tmp_path):
    ds = FeatureDataset(dataset)
    seed_all(42)
    model = FeatureModel().train()
    logger = CaptureLogger()
    full, light = manifests(ds, tmp_path)
    before = capture_rng()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    light_result = evaluate(model, ds, light, config, tmp_path / "evaluation", "dev", 200, mode="light", logger=logger)
    first = evaluate(model, ds, full, config, tmp_path / "evaluation", "dev", 800, mode="full", logger=logger)
    assert_rng_equal(before, capture_rng())
    assert model.training and all(value.grad is None for value in model.parameters())
    assert all(torch.equal(value, parameters[name]) for name, value in model.named_parameters())
    assert len(first["reports"]) == 5 and len(first["per_class_summary"]) == 32
    paths = {Path(path).name: Path(path) for path in first["reports"]}
    expected_names = {f"dev_full_{kind}_step0800.csv" for kind in
                      ("per_class_seed", "per_class_mean_std", "macro_per_seed", "macro_mean_std", "predictions")}
    assert set(paths) == expected_names
    raw = csv_rows(paths["dev_full_per_class_seed_step0800.csv"])
    classes = csv_rows(paths["dev_full_per_class_mean_std_step0800.csv"])
    macros = csv_rows(paths["dev_full_macro_per_seed_step0800.csv"])
    overall = csv_rows(paths["dev_full_macro_mean_std_step0800.csv"])
    assert (len(raw), len(classes), len(macros), len(overall)) == (160, 32, 5, 4)
    assert {row["fake_class"] for row in classes} == set(ds.fake_classes)
    assert set(row["seed"] for row in raw) == {"17", "42", "71", "100", "123"}
    assert all("eer_threshold" in row for row in raw)
    for row in classes:
        seed_rows = [item for item in raw if item["fake_class"] == row["fake_class"]]
        assert len(seed_rows) == 5
        assert int(row["real_query_count_min"]) == len(full["queries"]["real"])
        assert int(row["fake_query_count_min"]) == len(full["queries"][row["fake_class"]])
        for metric in METRICS:
            values = [float(item[metric]) for item in seed_rows]
            assert float(row[f"{metric}_mean"]) == pytest.approx(np.mean(values))
            assert float(row[f"{metric}_std"]) == pytest.approx(np.std(values, ddof=0))
            assert 0 <= float(row[f"{metric}_mean"]) <= 1
    for row in macros:
        seed_rows = [item for item in raw if item["seed"] == row["seed"]]
        for metric in METRICS:
            assert float(row[metric]) == pytest.approx(np.mean([float(item[metric]) for item in seed_rows]))
    for row in overall:
        values = [float(item[row["metric"]]) for item in macros]
        assert float(row["mean"]) == pytest.approx(np.mean(values))
        assert float(row["std"]) == pytest.approx(np.std(values, ddof=0))
    prediction_rows = csv_rows(paths["dev_full_predictions_step0800.csv"])
    expected_predictions = 5 * (32 * len(full["queries"]["real"]) +
                                sum(len(full["queries"][name]) for name in ds.fake_classes))
    assert len(prediction_rows) == expected_predictions
    assert all(int(row["prediction"]) == int(float(row["score"]) >= 0) for row in prediction_rows)
    assert any("[FULL DEV step=0800] Macro" in message for message in logger.messages)
    assert any("START" in message and "seeds=[17, 42, 71, 100, 123]" in message for message in logger.messages)
    assert any("END elapsed_seconds=" in message and "reports=" in message for message in logger.messages)
    for name in ds.fake_classes:
        assert any(f"[FULL DEV step=0800] {name} " in message for message in logger.messages)
    saved_bytes = {path: path.read_bytes() for path in map(Path, first["reports"] + light_result["reports"])}
    second = evaluate(model, ds, full, config, tmp_path / "evaluation", "dev", 1600, mode="full", logger=logger)
    assert first["metrics"] == second["metrics"]
    assert all(path.read_bytes() == content for path, content in saved_bytes.items())
    assert all("step1600" in path for path in second["reports"])
    assert not list((tmp_path / "evaluation").rglob(".dev_*"))


def test_light_dev_unequal_queries_use_all_classes_and_equal_class_macro(dataset, config, tmp_path, monkeypatch):
    from engine import evaluator
    ds = FeatureDataset(dataset)
    model = FeatureModel().train()
    _, light = manifests(ds, tmp_path)
    counts = []
    def metrics(labels, scores):
        fake = int((labels == 1).sum())
        real = int((labels == 0).sum())
        counts.append((real, fake, len(scores)))
        value = 0.0 if fake == 2 else 1.0
        return dict(eer=value, eer_threshold=0.25, auc=value, ap=value, acc=value)
    monkeypatch.setattr(evaluator, "binary_metrics", metrics)
    result = evaluate(model, ds, light, config, tmp_path, "dev", 200, mode="light", logger=CaptureLogger())
    assert len(counts) == 32 and counts[0] == (4, 2, 6)
    assert all(item == (4, 5, 9) for item in counts[1:])
    assert result["mean_eer"] == 31 / 32
    weighted = (31 * 5) / (2 + 31 * 5)
    assert result["mean_eer"] != weighted
    assert all(value["mean"] == 31 / 32 and value["std"] == 0 for value in result["metrics"].values())
    raw = csv_rows(tmp_path / "light/dev_light_per_class_seed_step0200.csv")
    assert len(raw) == 32 and {row["seed"] for row in raw} == {"42"}
    assert raw[0]["real_query_count"] == "4" and raw[0]["fake_query_count"] == "2"
    assert result["query_counts"][ds.fake_classes[0]] == dict(real_query_count=4, fake_query_count=2)
    assert not (tmp_path / "full").exists()


def test_failed_full_dev_restores_rng_and_mode_and_does_not_commit_partial_csv(dataset, config, tmp_path, monkeypatch):
    from engine import evaluator
    ds = FeatureDataset(dataset)
    model = FeatureModel().train()
    full, _ = manifests(ds, tmp_path)
    before = capture_rng()
    logger = CaptureLogger()
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic metric failure")
    monkeypatch.setattr(evaluator, "binary_metrics", fail)
    with pytest.raises(RuntimeError, match="synthetic metric failure"):
        evaluate(model, ds, full, config, tmp_path / "evaluation", "dev", 800, mode="full", logger=logger)
    assert model.training
    assert_rng_equal(before, capture_rng())
    assert not list((tmp_path / "evaluation").rglob("*.csv"))
    assert any("FAILED elapsed_seconds=" in message for message in logger.messages)


def test_test_mode_keeps_original_outputs_and_rejects_training_dev_modes(dataset, config, tmp_path):
    ds = FeatureDataset(dataset)
    model = FeatureModel().train()
    full, _ = manifests(ds, tmp_path)
    ds.split = "test"
    full["split"] = "test"
    result = evaluate(model, ds, full, config, tmp_path / "evaluation", "test")
    assert set(result) == {"mean_eer", "std_eer", "metrics", "per_seed"}
    assert (tmp_path / "evaluation/test_predictions.csv").is_file()
    assert (tmp_path / "evaluation/test_macro_mean_std.csv").is_file()
    assert all(len(csv_rows(tmp_path / f"evaluation/test_per_class_seed{seed}.csv")) == 32 for seed in full["seeds"])
    with pytest.raises(ValueError, match="restricted to Dev"):
        evaluate(model, ds, full, config, tmp_path / "evaluation", "test", 800, mode="full")


def test_explicit_full_mode_requires_fixed_seed_shot_and_all_classes(dataset, config, tmp_path):
    ds = FeatureDataset(dataset)
    model = FeatureModel()
    full, light = manifests(ds, tmp_path)
    with pytest.raises(ValueError, match="fixed seed"):
        evaluate(model, ds, light, config, tmp_path, "dev", 800, mode="full")
    full["classes"] = full["classes"][:-1]
    with pytest.raises(ValueError, match="32 Fake classes"):
        evaluate(model, ds, full, config, tmp_path, "dev", 800, mode="full")
