"""Pure scheduling / crash-recovery simulations: never execute real training."""
import copy
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from engine import trainer
from engine.checkpoint import load_checkpoint, resolve_checkpoint
from engine.validation_schedule import (ValidationHistory, check_new_run,
    make_run_identity, milestone_steps, validation_mode)
from utils.seed import seed_all, capture_rng


class CounterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = torch.nn.Parameter(torch.zeros(()))


class CounterScheduler:
    def __init__(self):
        self.count = 0
    def step(self):
        self.count += 1
    def state_dict(self):
        return {"count": self.count}
    def load_state_dict(self, state):
        self.count = state["count"]


class Writer:
    def __init__(self):
        self.points = []
    def add_scalar(self, tag, value, step):
        self.points.append((tag, value, step))
    def flush(self):
        pass


def result(out, step, mode, eer=.3):
    seeds = [17, 42, 71, 100, 123] if mode == "full" else [42]
    names = ("per_class_seed", "per_class_mean_std", "macro_per_seed", "macro_mean_std", "predictions")
    reports = []
    for name in names:
        path = Path(out) / mode / f"dev_{mode}_{name}_step{step:04d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic,state\ncomplete,1\n", encoding="utf-8")
        reports.append(str(path.resolve()))
    metrics = {name: {"mean": eer if name == "eer" else .7, "std": 0.} for name in ("eer", "auc", "ap", "acc")}
    return dict(mean_eer=eer, std_eer=0., metrics=metrics, reports=reports,
                per_seed=[dict(seed=seed, **{name: val["mean"] for name, val in metrics.items()}) for seed in seeds])


def harness(tmp_path, config):
    cfg = copy.deepcopy(config)
    cfg["runtime"]["output_dir"] = str(tmp_path)
    identity = make_run_identity(cfg, "a" * 64, "b" * 64, "c" * 64)
    history = ValidationHistory(tmp_path, cfg, identity, "a" * 64, "c" * 64)
    model = CounterModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return cfg, history, model, optimizer, CounterScheduler()


def mock_update(model, ds, episodes, cfg, optimizer, step):
    # Only an increment and RNG draws; no forward/backward, audio or training data.
    with torch.no_grad():
        model.counter.add_(1)
    random.random()
    np.random.rand()
    torch.rand(1)
    return .25, .5, []


def run(cfg, history, model, optimizer, scheduler, start, writer, logger):
    trainer.run_steps(model, SimpleNamespace(uttids=[]), None,
        SimpleNamespace(sample_step=lambda: []), cfg, optimizer, scheduler,
        {}, {}, history, "a" * 64, "c" * 64, start, writer, logger, io.StringIO())


def test_4000_step_schedule_is_exact_pure_logic(config):
    light = [s for s in range(1, 4001) if validation_mode(s, config["train"]) == "light"]
    full = [s for s in range(1, 4001) if validation_mode(s, config["train"]) == "full"]
    assert len(light) == 15 and len(full) == 5
    assert full == milestone_steps(config) == [800, 1600, 2400, 3200, 4000]
    assert not set(light) & set(full)


def test_mocked_loop_only_saves_five_nodes_full_only_best_and_tags(tmp_path, config, monkeypatch):
    cfg, history, model, optimizer, scheduler = harness(tmp_path, config)
    history.initialize()
    writer, logger, events = Writer(), Mock(), []
    monkeypatch.setattr(trainer, "optimize_step", mock_update)
    def evaluate(model, ds, manifest, cfg, out, split, step, *, mode, logger):
        assert split == "dev"
        events.append((step, mode))
        if mode == "full":
            assert history.checkpoint(step).is_file()  # Save happens before full evaluation.
        return result(out, step, mode, .01 if mode == "light" else .3)
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    run(cfg, history, model, optimizer, scheduler, 0, writer, logger)
    assert scheduler.count == 4000 and model.counter.item() == 4000
    assert sum(mode == "light" for _, mode in events) == 15
    assert sum(mode == "full" for _, mode in events) == 5
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("*.pt")) == [f"step_{s:04d}.pt" for s in milestone_steps(cfg)]
    assert history.best["step"] == 800  # Light's lower EER is ignored; full ties keep earliest.
    assert resolve_checkpoint("best", tmp_path) == history.checkpoint(800)
    tags = {}
    for tag, value, step in writer.points:
        tags.setdefault(tag, []).append(step)
    for name in ("loss", "learning_rate", "gradient_norm"):
        assert tags[f"train/{name}"] == list(range(1, 4001))
    assert len(tags["dev/light_macro_eer"]) == 15
    assert tags["dev/full_macro_eer_mean"] == [800, 1600, 2400, 3200, 4000]
    assert "dev/macro_eer_mean" not in tags
    train_logs = [call.args for call in logger.info.call_args_list if call.args[0].startswith("step=%d")]
    assert [args[1] for args in train_logs] == list(range(1, 4001))
    assert any("best_step" in str(call) for call in logger.info.call_args_list)


def test_resume_incomplete_full_dev_precedes_next_update_and_restores_state(tmp_path, config, monkeypatch):
    cfg, history, model, optimizer, scheduler = harness(tmp_path, config)
    cfg["train"]["total_steps"] = 1600  # Mock orchestration only, not a valid production config.
    history.initialize()
    monkeypatch.setattr(trainer, "optimize_step", mock_update)
    def interrupted(*args, **kwargs):
        if kwargs["mode"] == "light":
            return result(args[4], args[6], "light")
        random.random()
        raise RuntimeError("simulated interruption during full Dev")
    monkeypatch.setattr(trainer, "evaluate", interrupted)
    seed_all(42)
    with pytest.raises(RuntimeError, match="interruption"):
        run(cfg, history, model, optimizer, scheduler, 0, Writer(), Mock())
    assert history.checkpoint(800).exists() and not history.marker(800).exists()
    saved_rng = capture_rng()
    restored = CounterModel()
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    new_scheduler = CounterScheduler()
    load_checkpoint(history.checkpoint(800), restored, new_optimizer, new_scheduler,
                    restore_random=True, expected_run_identity=history.identity)
    assert restored.counter.item() == 800 and new_scheduler.count == 800
    assert capture_rng()["python"] == saved_rng["python"]
    resumed = ValidationHistory(tmp_path, cfg, history.identity, "a" * 64, "c" * 64)
    resumed.initialize(800)
    calls = []
    def evaluate(model, ds, manifest, cfg, out, split, step, *, mode, logger):
        calls.append((step, mode, model.counter.item()))
        # Validate wrapper RNG preservation even when an evaluator consumes random values.
        random.random(); np.random.rand(); torch.rand(1)
        return result(out, step, mode)
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    run(cfg, resumed, restored, new_optimizer, new_scheduler, 800, Writer(), Mock())
    assert calls[0] == (800, "full", 800)
    assert restored.counter.item() == 1600 and new_scheduler.count == 1600
    assert resumed.marker(800).is_file() and resumed.marker(1600).is_file()
    assert resumed.best["step"] == 800


def test_completed_node_rebuilds_best_and_does_not_revalidate(tmp_path, config, monkeypatch):
    cfg, history, model, optimizer, scheduler = harness(tmp_path, config)
    cfg["train"]["total_steps"] = 800
    history.initialize()
    monkeypatch.setattr(trainer, "optimize_step", mock_update)
    monkeypatch.setattr(trainer, "evaluate", lambda m, d, manifest, c, out, split, step, **kw: result(out, step, kw["mode"]))
    run(cfg, history, model, optimizer, scheduler, 0, Writer(), Mock())
    index = tmp_path / "checkpoints" / "best_checkpoint.json"
    index.write_text("{}", encoding="utf-8")  # Simulate missing/stale Best after marker committed.
    restored = ValidationHistory(tmp_path, cfg, history.identity, "a" * 64, "c" * 64)
    restored.initialize(800)
    assert json.loads(index.read_text())["step"] == 800
    forbidden = Mock(side_effect=AssertionError("must not repeat completed validation"))
    monkeypatch.setattr(trainer, "evaluate", forbidden)
    replay_writer = Writer()
    run(cfg, restored, model, optimizer, scheduler, 800, replay_writer, Mock())
    forbidden.assert_not_called()
    assert ("dev/full_macro_eer_mean", .3, 800) in replay_writer.points
    # Missing/corrupted report invalidates completion even if marker exists.
    one_report = tmp_path / next(iter(restored.records[0]["reports"]))
    one_report.write_text("partial", encoding="utf-8")
    assert restored.read_complete(800) is None
    pending = ValidationHistory(tmp_path, cfg, history.identity, "a" * 64, "c" * 64)
    pending.initialize(800)
    assert json.loads(index.read_text())["status"] == "pending_full_dev"
    with pytest.raises(ValueError):
        resolve_checkpoint("best", tmp_path)


def test_resume_rejects_future_nodes_identity_changes_and_new_run_overwrite(tmp_path, config, monkeypatch):
    cfg, history, model, optimizer, scheduler = harness(tmp_path, config)
    history.initialize()
    with pytest.raises(ValueError, match="Run already exists"):
        check_new_run(tmp_path)
    history.checkpoint(800).parent.mkdir(parents=True, exist_ok=True)
    history.checkpoint(800).write_bytes(b"node")
    history.checkpoint(1600).write_bytes(b"future")
    with pytest.raises(ValueError, match="future/legacy"):
        ValidationHistory(tmp_path, cfg, history.identity, "a" * 64, "c" * 64).initialize(800)
    with pytest.raises(ValueError, match="identity"):
        ValidationHistory(tmp_path, cfg, "f" * 64, "a" * 64, "c" * 64).initialize(800)
    changed = copy.deepcopy(cfg)
    changed["validation"]["light_max_query_per_class"] = 80
    assert make_run_identity(changed, "a" * 64, "b" * 64, "c" * 64) != history.identity


def test_preflight_manifests_alone_do_not_prevent_new_training(tmp_path):
    path = tmp_path / "evaluation" / "dev_episode_manifest.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    check_new_run(tmp_path)


def test_real_report_result_can_commit_and_rebuild_history(dataset, tmp_path, config):
    from test_evaluation import FeatureDataset, FeatureModel
    from data.evaluation_manifest import load_or_create_manifest
    from engine.evaluator import evaluate
    from engine.checkpoint import save_checkpoint
    cfg, history, _, _, _ = harness(tmp_path / "run", config)
    history.initialize()
    ds, model = FeatureDataset(dataset), FeatureModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CounterScheduler()
    save_checkpoint(history.checkpoint(800), model, optimizer, scheduler, cfg, 800,
                    None, float("inf"), "a" * 64, "c" * 64, run_identity=history.identity)
    manifest = load_or_create_manifest(ds, tmp_path / "source.json", 10, cfg["validation"]["seeds"])
    evaluated = evaluate(model, ds, manifest, cfg, history.out / "evaluation", "dev", 800,
                         mode="full", logger=Mock())
    history.complete(800, evaluated)
    assert history.read_complete(800)["summary"]["mean_eer"] == evaluated["mean_eer"]
    restored = ValidationHistory(history.out, cfg, history.identity, "a" * 64, "c" * 64)
    restored.initialize(800)
    assert resolve_checkpoint("best", history.out) == history.checkpoint(800)
    history.marker(800).write_text("[]", encoding="utf-8")
    assert restored.read_complete(800) is None


def test_logging_writes_each_step_to_terminal_and_file(tmp_path, capsys):
    from utils.logging import configure_logging
    logger = configure_logging(tmp_path)
    try:
        logger.info("step=%d loss=%.6f lr=%.8g gradient_norm=%.6f", 1, .2, 1e-6, .5)
        assert "step=1 loss=0.200000" in (tmp_path / "train.log").read_text(encoding="utf-8")
        assert "step=1 loss=0.200000" in capsys.readouterr().err
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
