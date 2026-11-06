import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch

from data.speechfake_dataset import SpeechFakeDataset
from data.evaluation_manifest import load_or_create_manifest
from data.cache_pipeline import evaluation_ids
from engine.evaluator import evaluate
from engine.trainer import optimize_step, make_optimizer
from engine.pipeline_benchmark import benchmark_train, total_experiment_times
from data.episode_sampler import EpisodeSampler
from utils.seed import capture_rng
from test_engine import SyntheticDataset, SyntheticBackend
from test_evaluation import FeatureDataset, FeatureModel
from test_cache_storage import cache_config


class MemoryFeatures:
    """Synthetic reader for full evaluator protocol tests, not a real cache."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.identity = dict(split="dev")
        self.reads = {}

    def get(self, uid):
        self.reads[uid] = self.reads.get(uid, 0) + 1
        tensor = torch.zeros(25, 1, 1024)
        tensor[0, 0, :9] = self.dataset.synthetic_features[self.dataset.uttid2idx[uid], 0]
        return tensor


class CacheFeatureDataset(FeatureDataset):
    use_cache = SpeechFakeDataset.use_cache

    def __getitem__(self, index):
        if self.cache_mode == "xlsr_feature":
            return SpeechFakeDataset.__getitem__(self, index)
        return super().__getitem__(index)

    @staticmethod
    def collate_fn(batch):
        if batch[0].get("input_kind") == "xlsr_feature":
            return SpeechFakeDataset.collate_fn(batch)
        return FeatureDataset.collate_fn(batch)


class CacheFeatureModel(FeatureModel):
    def __init__(self):
        super().__init__()
        self.online_calls = self.cached_calls = 0

    def forward(self, source):
        self.online_calls += 1
        return super().forward(source)

    def forward_from_frontend_features(self, features):
        self.cached_calls += 1
        return dict(feats=self.proj(features[:, 0, 0, :9]).view(-1, 25, 7))


def test_cached_full_evaluator_matches_scores_and_reads_shared_real_once(dataset, config, tmp_path):
    ds, model = CacheFeatureDataset(dataset), CacheFeatureModel().train()
    manifest = load_or_create_manifest(ds, tmp_path / "manifest.json", 10, config["validation"]["seeds"])
    before = capture_rng()
    baseline = evaluate(model, ds, manifest, config, tmp_path / "baseline", "dev", 800, mode="full")
    cfg = copy.deepcopy(config)
    cfg["data"]["dev_cache_mode"] = "xlsr_feature"
    readers = {seed: MemoryFeatures(ds) for seed in manifest["seeds"]}
    ds.feature_caches = {("full", seed): reader for seed, reader in readers.items()}
    online_calls = model.online_calls
    actual = evaluate(model, ds, manifest, cfg, tmp_path / "cached", "dev", 800, mode="full")
    assert actual["metrics"] == baseline["metrics"] and model.training
    assert model.online_calls == online_calls and model.cached_calls > 0
    assert (tmp_path / "baseline/full/dev_full_predictions_step0800.csv").read_bytes() == (
            tmp_path / "cached/full/dev_full_predictions_step0800.csv").read_bytes()
    for seed, reader in readers.items():
        assert set(reader.reads) == set(evaluation_ids(manifest, seed))
        assert all(count == 1 for count in reader.reads.values())
    after = capture_rng()
    assert before["python"] == after["python"] and torch.equal(before["torch"], after["torch"])
    assert torch.equal(before["numpy"][1], after["numpy"][1])


def test_cached_training_keeps_one_update_and_all_eight_episodes(config):
    class CachedTrain(SyntheticDataset):
        def __getitem__(self, index):
            feats = torch.zeros(25, 1, 1024)
            feats[0, 0, :9] = self.features[index, 0]
            return dict(feats=feats)

        @staticmethod
        def collate_fn(batch):
            return dict(feats=torch.stack([row["feats"] for row in batch]), input_kind="xlsr_feature")

    class CachedBackend(SyntheticBackend):
        def forward(self, source):
            raise AssertionError("XLS-R must not run for cached Train features")

        def forward_from_frontend_features(self, features):
            self.seen_batch_sizes.append(len(features))
            return dict(feats=self.projection(features[:, 0, 0, :9]).reshape(-1, 25, 7))

    dataset, model = CachedTrain(), CachedBackend()
    episodes = EpisodeSampler(dataset.pools).sample_step()
    optimizer, _ = make_optimizer(model, config)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, norm, diagnostics = optimize_step(model, dataset, episodes, config, optimizer, 1)
    assert optimizer.step.call_count == 1 and len(diagnostics) == 8
    assert sum(model.seen_batch_sizes) == 240 and torch.isfinite(torch.tensor([loss, norm])).all()
    assert model.projection.weight.grad.abs().sum() > 0


def _cli(name, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(f"cache_test_{name}", root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_test_happens_after_best_resolution_without_cache_access(config, tmp_path, monkeypatch):
    cli = _cli("evaluate", monkeypatch)
    cfg = copy.deepcopy(config)
    cfg["runtime"]["output_dir"] = str(tmp_path)
    cfg["data"]["test_cache_mode"] = "none"
    (tmp_path / "evaluation").mkdir()
    events = []
    node = tmp_path / "step_0800.pt"
    def resolved(reference, output):
        events.append("best")
        return node
    monkeypatch.setattr(cli, "load_config", Mock(return_value=cfg))
    monkeypatch.setattr(cli, "resolve_checkpoint", resolved)
    monkeypatch.setattr(cli, "SpeechFakeDataset", lambda *args: events.append("test_dataset") or object())
    monkeypatch.setattr(cli, "load_or_create_manifest", lambda *args: events.append("manifest") or {})
    monkeypatch.setattr(cli, "FSLPRF", lambda *args: object())
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args: events.append("load_checkpoint"))
    forbidden_cache = Mock(side_effect=AssertionError("Test must never build/read caches"))
    monkeypatch.setattr(cli, "prepare_final_test_caches", forbidden_cache)
    monkeypatch.setattr(cli, "evaluate", lambda *args: events.append("test_metrics") or {"mean_eer": .2})
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "best"])
    cli.main()
    assert events.index("best") < events.index("test_dataset") < events.index("load_checkpoint")
    assert events.index("load_checkpoint") < events.index("test_metrics")
    forbidden_cache.assert_not_called()
    cfg["data"]["test_cache_mode"] = "waveform"
    with pytest.raises(ValueError, match="original WAVs"):
        cli.main()
    cfg["data"]["test_cache_mode"] = "none"
    forbidden = Mock(side_effect=AssertionError("Test must stay untouched without Best"))
    monkeypatch.setattr(cli, "SpeechFakeDataset", forbidden)
    monkeypatch.setattr(cli, "resolve_checkpoint", Mock(side_effect=ValueError("Best not selected")))
    with pytest.raises(ValueError, match="Best not selected"):
        cli.main()
    forbidden.assert_not_called()


def test_train_preflight_never_initializes_test_or_builds_dev(dataset, config, tmp_path, monkeypatch):
    from engine import trainer
    from data import cache_pipeline
    from models import fs_lprf_model
    cfg = cache_config(dataset, config, tmp_path)
    cfg["data"]["dev_cache_mode"] = "xlsr_feature"
    cfg["runtime"]["output_dir"] = str(tmp_path / "new_run")
    calls = []
    original = SpeechFakeDataset
    def create(options, split, **kwargs):
        assert split != "test"
        calls.append(split)
        return original(options, split, **kwargs)
    monkeypatch.setattr(trainer, "SpeechFakeDataset", create)
    monkeypatch.setattr(cache_pipeline, "SpeechFakeDataset", create)
    frontend = Mock(side_effect=AssertionError("No model initialization before cache preflight"))
    monkeypatch.setattr(fs_lprf_model, "FSLPRF", frontend)
    with pytest.raises(ValueError, match="Build explicitly"):
        trainer.train(cfg)
    assert calls == ["train", "dev"]
    frontend.assert_not_called()
    assert not list(Path(cfg["runtime"]["output_dir"]).rglob("*test*"))


def test_benchmark_uses_disposable_weights_and_reports_unmeasured_dev(config):
    dataset, model = SyntheticDataset(), SyntheticBackend()
    dataset.class_indices = dataset.pools
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rows = benchmark_train(model, config, modes=("none",), workers=(0,), warmup=1, steps=2,
                            dataset_factory=lambda cfg: dataset)
    assert len(rows) == 1 and rows[0]["samples_per_step"] == 240 and len(rows[0]["raw_steps"]) == 2
    assert rows[0]["xlsr_forward_seconds"] == 0 and rows[0]["backward_optimizer_seconds"] > 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    projection = total_experiment_times(rows, {}, config)[0]
    assert projection["total_seconds"] is None
    dev = dict(none=dict(cache_build_seconds=0, light_seconds=2, full_seconds=10))
    total = total_experiment_times(rows, dev, config)[0]
    assert total["light_validations"] == 15 and total["full_validations"] == 5
    assert total["total_seconds"] == pytest.approx(4000 * rows[0]["mean_step_seconds"] + 15*2 + 5*10)
