import copy
import os
import random

import numpy as np
import pytest
import torch

from data.audio import FADTrim
from data.loaders import ReusableAudioLoader, audio_loader, worker_options, _worker_import_environment
from data.randomness import sample_seed, sample_rng
from engine.validation_schedule import make_run_identity
from utils.config import load_config
from utils.seed import capture_rng, restore_rng


class RandomAudioDataset(torch.utils.data.Dataset):
    sample_segment = FADTrim.sample_segment

    def __init__(self):
        self.uttids = ["a", "b"]
        self.max_len = 64000
        self.wave = torch.linspace(-.2, .2, 128001)[None]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        wave = self.sample_segment(self.wave)
        # Mock stochastic augmentation using all three RNGs.
        factor = random.random() + np.random.rand() + torch.rand(1).item()
        return dict(feats=wave * factor, pid=os.getpid(), import_threads=os.environ.get("OMP_NUM_THREADS"))

    @staticmethod
    def collate_fn(batch):
        return dict(feats=torch.stack([item["feats"] for item in batch]), pids=[item["pid"] for item in batch],
                    import_threads=[item["import_threads"] for item in batch])


def rng_equal(a, b):
    assert a["python"] == b["python"] and a["numpy"][2:] == b["numpy"][2:]
    assert torch.equal(a["numpy"][1], b["numpy"][1]) and torch.equal(a["torch"], b["torch"])


@pytest.mark.parametrize("workers", [0, 2, 4, 8])
def test_workers_match_sample_inputs_and_survive_across_steps(config, workers):
    ds = RandomAudioDataset()
    cfg = copy.deepcopy(config)
    cfg["train"]["audio_microbatch_size"] = 2
    cfg["runtime"].update(train_num_workers=workers, persistent_train_workers=True, pin_memory=False)
    expected_cfg = copy.deepcopy(cfg)
    expected_cfg["runtime"]["train_num_workers"] = 0
    indices = [0, 1] * 8  # Eight microbatches exercise every configured worker.
    with ReusableAudioLoader(ds, expected_cfg) as reference:
        expected = torch.cat([batch["feats"] for batch in reference.for_step(indices, 7)])
    before = capture_rng()
    parent_threads = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    worker_processes = []
    with ReusableAudioLoader(ds, cfg) as reusable:
        loader_id = id(reusable.loader)
        first_batches = list(reusable.for_step(indices, 7))
        first = torch.cat([batch["feats"] for batch in first_batches])
        again_batches = list(reusable.for_step(indices, 7))
        again = torch.cat([batch["feats"] for batch in again_batches])
        assert torch.equal(first, expected) and torch.equal(first, again)
        assert first_batches[0]["pids"] == again_batches[0]["pids"]
        assert id(reusable.loader) == loader_id
        if workers:
            worker_processes = list(reusable.loader._iterator._workers)
            assert reusable.loader.persistent_workers
            assert all(pid != os.getpid() for batch in first_batches for pid in batch["pids"])
            assert len({pid for batch in first_batches for pid in batch["pids"]}) == workers
            assert all(value == "1" for batch in first_batches for value in batch["import_threads"])
        else:
            assert reusable.loader.prefetch_factor is None and not reusable.loader.persistent_workers
        changed = torch.cat([batch["feats"] for batch in reusable.for_step(indices, 8)])
        assert not torch.equal(changed, first)
    rng_equal(before, capture_rng())
    assert all(not process.is_alive() for process in worker_processes)
    assert parent_threads == {key: os.environ.get(key) for key in parent_threads}
    assert not torch.equal(first[0], first[2])  # Same uttid, different episode position.


def test_rng_restore_replays_data_without_worker_rng_checkpoint(config):
    ds = RandomAudioDataset()
    state = capture_rng()
    indices = random.choices(range(2), k=4)
    with ReusableAudioLoader(ds, config) as first:
        expected = torch.cat([row["feats"] for row in first.for_step(indices, 800)])
    restore_rng(state)
    repeated = random.choices(range(2), k=4)
    assert indices == repeated
    with ReusableAudioLoader(ds, config) as restarted:
        actual = torch.cat([row["feats"] for row in restarted.for_step(repeated, 800)])
    assert torch.equal(expected, actual)


def test_loader_refuses_to_overwrite_unconsumed_step(config):
    with ReusableAudioLoader(RandomAudioDataset(), config) as loader:
        stream = loader.for_step([0, 1, 0, 1], 1)
        next(stream)
        with pytest.raises(RuntimeError, match="Fully consume"):
            next(loader.for_step([0], 2))
        stream.close()
        with pytest.raises(RuntimeError, match="Fully consume"):
            next(loader.for_step([0], 2))


def test_rng_context_restores_after_failure_and_uses_stable_digest():
    before = capture_rng()
    with pytest.raises(RuntimeError):
        with sample_rng(sample_seed("train", 42, 17, 1, "a")):
            random.random(); np.random.rand(); torch.rand(1)
            raise RuntimeError()
    rng_equal(before, capture_rng())
    assert sample_seed("train", 42, 17, 1, "a") == sample_seed("train", 42, 17, 1, "a")
    assert len({sample_seed("train", 42, step, position, "a") for step in (1, 2) for position in (1, 2)}) == 4


def test_pipeline_defaults_and_legacy_worker_override(config):
    defaults = load_config("configs/speechfake.yaml")
    assert defaults["data"]["train_cache_mode"] == "waveform"
    assert defaults["data"]["dev_cache_mode"] == "waveform"
    assert defaults["data"]["test_cache_mode"] == "none"
    assert defaults["cache"]["feature_dtype"] == "float32"
    assert defaults["runtime"]["train_num_workers"] == 4 and defaults["runtime"]["eval_num_workers"] == 2
    old = load_config("configs/speechfake.yaml", ["runtime.num_workers=0"])
    with pytest.warns(FutureWarning, match="deprecated"):
        assert worker_options(old, True)["num_workers"] == 0
    with pytest.warns(FutureWarning, match="deprecated"):
        assert worker_options(old)["num_workers"] == 0


def test_run_identity_ignores_worker_performance_but_binds_cache_and_rng(config):
    bindings = dict(rng_schema_version=1, caches=[dict(fingerprint="one", content_fingerprint="contents")])
    original = make_run_identity(config, "full", "light", "train", bindings)
    changed = copy.deepcopy(config)
    changed["runtime"].update(train_num_workers=8, eval_num_workers=4, prefetch_factor=1,
                               pin_memory=False, worker_torch_num_threads=2, persistent_train_workers=False, persistent_eval_workers=False,
                               device="cuda:7")
    assert make_run_identity(changed, "full", "light", "train", bindings) == original
    for altered in (dict(bindings, rng_schema_version=2), dict(bindings, caches=[dict(fingerprint="different")])):
        assert make_run_identity(config, "full", "light", "train", altered) != original
    changed["data"]["train_cache_mode"] = "xlsr_feature"
    assert make_run_identity(changed, "full", "light", "train", bindings) != original


def test_worker_import_limits_restore_environment_even_after_error(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    before = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    with pytest.raises(RuntimeError, match="startup failure"):
        with _worker_import_environment(1):
            assert all(os.environ[key] == "1" for key in before)
            raise RuntimeError("startup failure")
    assert before == {key: os.environ.get(key) for key in before}
