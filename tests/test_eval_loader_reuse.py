import copy
import os

import numpy as np
import pytest
import torch

from data.audio import FADTrim
from data.audio_benchmark import RealAudioBenchmarkDataset, scan_real_audio
from data.cache import CacheWriter, cache_path, make_cache_identity
from data.cache_pipeline import build_dataset_cache, view_tokens
from data.evaluation_manifest import load_or_create_manifest
from data.loaders import ReusableAudioLoader, PreprocessedWaveformDataset, worker_options
from data.randomness import sample_rng
from data.speechfake_dataset import SpeechFakeDataset
from engine.evaluator import evaluate, extract_features, extract_support_features
from test_cache_storage import cache_config
from test_evaluation import FeatureDataset, FeatureModel


class IndexedAudioDataset(torch.utils.data.Dataset):
    sample_segment = FADTrim.sample_segment

    def __init__(self):
        self.uttids = [f"audio-{index}" for index in range(19)]
        self.uttid2idx = {uid: index for index, uid in enumerate(self.uttids)}
        self.split, self.max_len = "benchmark", 64000
        self.wave = torch.linspace(-.2, .2, 100003)[None]
        self.fail = None

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, index):
        if index == self.fail:
            raise ValueError("intentional decode failure")
        return dict(uttid=self.uttids[index], feats=self.sample_segment(self.wave),
                    label=index % 2, pid=os.getpid())

    @staticmethod
    def collate_fn(batch):
        return dict(uttids=[row["uttid"] for row in batch],
                    feats=torch.stack([row["feats"] for row in batch]),
                    labels=torch.tensor([row["label"] for row in batch]),
                    pids=[row["pid"] for row in batch])


@pytest.mark.parametrize("workers,persistent,prefetch", [(0, True, 2), (2, True, 2), (2, False, 4)])
def test_evaluation_reuse_preserves_view_tokens_order_labels_and_workers(config, workers, persistent, prefetch):
    ds = IndexedAudioDataset()
    cfg = copy.deepcopy(config)
    cfg["train"]["audio_microbatch_size"] = 4
    cfg["runtime"].update(eval_num_workers=workers, persistent_eval_workers=persistent,
                          prefetch_factor=prefetch, pin_memory=False)
    processes, previous_pids = [], None
    with ReusableAudioLoader(ds, cfg, training=False, seed=42) as loader:
        for indices in ([7, 0, 18, 1, 4, 3, 6, 12, 11], [11, 7, 0], [0, 1, 2, 3, 4]):
            tokens = view_tokens(ds, [ds.uttids[i] for i in indices], 42, "full")
            expected = []
            for index, seed in tokens:
                with sample_rng(seed):
                    expected.append(ds[index]["feats"])
            batches = list(loader.for_indices(tokens))
            assert [uid for batch in batches for uid in batch["uttids"]] == [ds.uttids[i] for i in indices]
            assert torch.equal(torch.cat([batch["labels"] for batch in batches]), torch.tensor([i % 2 for i in indices]))
            assert torch.equal(torch.cat([batch["feats"] for batch in batches]), torch.stack(expected))
            if workers and persistent:
                current_pids = {worker.pid for worker in loader.loader._iterator._workers}
                processes = list(loader.loader._iterator._workers)
                assert previous_pids is None or current_pids == previous_pids
                previous_pids = current_pids
        assert list(loader.for_indices([])) == []
    assert all(not process.is_alive() for process in processes)


@pytest.mark.parametrize("persistent", [False, True])
def test_abandoned_or_failed_stream_releases_eval_workers(config, persistent):
    cfg = copy.deepcopy(config)
    cfg["runtime"].update(eval_num_workers=2, persistent_eval_workers=persistent, pin_memory=False)
    cfg["train"]["audio_microbatch_size"] = 2
    ds = IndexedAudioDataset()
    with ReusableAudioLoader(ds, cfg, training=False) as loader:
        stream = loader.for_indices(range(19))
        next(stream)
        processes = list(loader._active_iterator._workers)
        with pytest.raises(RuntimeError, match="Fully consume"):
            next(loader.for_indices([0]))
        stream.close()
        assert loader.closed and loader.incomplete
        assert all(not process.is_alive() for process in processes)
    ds.fail = 0
    with ReusableAudioLoader(ds, cfg, training=False) as loader:
        with pytest.raises(ValueError, match="intentional decode failure"):
            list(loader.for_indices([0, 1]))
        assert loader.closed and loader._active_iterator is None


def test_merged_support_restores_33_class_ids_and_matches_individual_encoding(dataset, config):
    ds = FeatureDataset(dataset)
    model = FeatureModel().eval()
    classes = ["real"] + ds.fake_classes
    supports = {name: [ds.uttids[i] for i in reversed(ds.class_indices[name][:10])] for name in classes}
    with torch.no_grad(), ReusableAudioLoader(ds, config, training=False, seed=42) as loader:
        merged = extract_support_features(model, ds, supports, config, 42, loader=loader)
        for name in classes:
            expected = extract_features(model, ds, supports[name], config, 42, loader=loader)
            torch.testing.assert_close(merged[name], expected, atol=2e-6, rtol=2e-6)
    supports[ds.fake_classes[0]][0] = supports["real"][0]
    with ReusableAudioLoader(ds, config, training=False, seed=42) as loader:
        with pytest.raises(ValueError, match="unique uttids"):
            extract_support_features(model, ds, supports, config, 42, loader=loader)


def test_evaluator_builds_one_loader_per_seed_and_reads_real_query_once(dataset, config, tmp_path, monkeypatch):
    import data.loaders as loaders
    ds = FeatureDataset(dataset)
    manifest = load_or_create_manifest(ds, tmp_path/"manifest.json", 10, [17, 42, 71, 100, 123])
    created, requests = [], []
    original_factory, original_indices = loaders.audio_loader, loaders.ReusableAudioLoader.for_indices
    def factory(*args, **kwargs):
        result = original_factory(*args, **kwargs)
        created.append(result)
        return result
    def indices(owner, tokens):
        tokens = list(tokens)
        requests.append((owner.seed, [owner.dataset.uttids[token[0]] for token in tokens]))
        return original_indices(owner, tokens)
    monkeypatch.setattr(loaders, "audio_loader", factory)
    monkeypatch.setattr(loaders.ReusableAudioLoader, "for_indices", indices)
    evaluate(FeatureModel(), ds, manifest, config, tmp_path, "dev")
    assert len(created) == 5
    for seed in manifest["seeds"]:
        streams = [ids for current, ids in requests if current == seed]
        assert len(streams) == 34  # All Support, shared Real Query, 32 Fake Query.
        assert streams[0] == [uid for name in manifest["classes"] for uid in manifest["supports"][str(seed)][name]]
        assert len(streams[0]) == 330
        assert streams[1] == manifest["queries"]["real"]
        assert streams[2:] == [manifest["queries"][name] for name in ds.fake_classes]


def test_parallel_waveform_build_resumes_verified_prefix_without_crop(dataset, config, tmp_path):
    cfg = cache_config(dataset, config, tmp_path)
    cfg["runtime"].update(eval_num_workers=2, persistent_eval_workers=True, pin_memory=False)
    raw = SpeechFakeDataset(cfg["data"], "train")
    ids = raw.uttids[:5]
    identity = make_cache_identity(raw, cfg, "waveform", uttids=ids, smoke=True)
    target = cache_path(cfg, identity)
    # Simulate interruption after a committed shard using the existing writer.
    with CacheWriter(target, identity, 2) as writer:
        for uid in ids[:2]:
            writer.add(uid, raw.load_preprocessed_waveform(uid))
    reader = build_dataset_cache(raw, cfg, identity, resume=True)
    assert [row["uttid"] for row in reader.index] == ids
    assert len(reader.metadata["shards"]) == 3
    for uid in ids:
        assert torch.equal(reader.get(uid), raw.load_preprocessed_waveform(uid))
        assert reader.get(uid).shape[1] == 128000  # Not the 64000-sample crop.
    reader.verify()


def test_real_benchmark_reuses_official_processing_and_stays_outside_protocol(wav, tmp_path):
    records, report = scan_real_audio(tmp_path, 512)
    assert report["selected_files"] == report["total_audio_files"] == 1
    assert report["sampled_source_counts"] == {"(root)": 1}
    ds = RealAudioBenchmarkDataset(tmp_path, records)
    assert ds.split == "benchmark" and not hasattr(ds, "fake_classes")
    with sample_rng(42):
        original = FADTrim.get_audio(ds, ds.uttids[0])
    with sample_rng(42):
        actual = ds[0]
    assert torch.equal(original, actual["feats"])
    assert actual["label"] == 0 and actual["attacker"] == "real"
    wrapper = PreprocessedWaveformDataset(ds)
    uid, whole = wrapper[0]
    assert uid == ds.uttids[0] and whole.shape[1] == 128000


def test_eval_persistence_configuration_defaults_and_validation(config):
    from utils.config import load_config
    cfg = load_config("configs/speechfake.yaml")
    assert worker_options(cfg)["persistent_workers"] is True
    assert load_config("configs/speechfake.yaml", ["runtime.persistent_eval_workers=false"])["runtime"]["persistent_eval_workers"] is False
    with pytest.raises(ValueError, match="booleans"):
        load_config("configs/speechfake.yaml", ["runtime.persistent_eval_workers=1"])
