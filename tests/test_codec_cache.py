"""Small PCM fixtures only: never downloads weights, builds a real corpus or trains."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from data.cache import CacheReader, CacheWriter, cache_path, make_cache_identity, file_sha
from data.cache_pipeline import cache_bindings, open_train_dataset
from data.codec_augmentation import CodecAugmentor
from data.codec_cache import (build_codec_cache, choose_spec, estimate_space, make_codec_identity,
                              open_codec_cache, select_smoke_ids, validate_location)
from data.loaders import ReusableAudioLoader
from data.randomness import sample_rng
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import publish_waveform_reference
from engine.validation_schedule import make_run_identity
from utils.seed import capture_rng
from test_pipeline_workers import rng_equal


def fake_backend(spec, audio):
    return audio * np.float32(.5), {"payload": b"unit-test-codec"}


@pytest.fixture
def source_cache(dataset, config, tmp_path):
    cfg = copy.deepcopy(config)
    root = tmp_path / "processed"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    # 12/class permits valid unchanged 3-way, 5-support + 5-query Episodes.
    for suffix in ("tsv", "txt"):
        rows = (tmp_path / f"train.{suffix}").read_text(encoding="utf-8").splitlines()
        (manifests / f"train.{suffix}").write_text("\n".join(row for i, row in enumerate(rows) if i % 65 < 12) + "\n", encoding="utf-8")
    cfg["data"].update(data_path=str(manifests), do_augment=False, train_cache_mode="waveform")
    cfg["data"]["codec_augment"].update(enabled=True, mode="offline", codecs=["mp3", "opus"],
                                       offline_cache_root=str(tmp_path / "codec"))
    cfg["cache"].update(root=str(root), shard_size=17)
    cfg["runtime"].update(output_dir=str(tmp_path / "run"), pin_memory=False)
    ds = SpeechFakeDataset(cfg["data"], "train", train_mode=False)
    identity = make_cache_identity(ds, cfg, "waveform", recorded_source_sha256="a" * 64)
    with CacheWriter(cache_path(cfg, identity), identity, 17) as writer:
        for i, uid in enumerate(ds.uttids):
            n = [1, 63, 320, 1001, 4000][i % 5]
            writer.add(uid, torch.full((1, n), .08 + (i % 10) * .005))
        source = publish_waveform_reference(writer.finish(), cfg)
    # All fixture manifests point here. No operation below may read raw Train audio.
    Path(ds.uttid2path[ds.uttids[0]]).unlink()
    augmentor = CodecAugmentor(probability=1, codecs=["mp3", "opus"], backend=fake_backend)
    return cfg, ds, source, augmentor


def build(source_cache, *, smoke=False, limit=66, target=None, **kwargs):
    cfg, ds, source, augmentor = source_cache
    ids = select_smoke_ids(ds, source, limit) if smoke else source.identity["uttids"]
    identity = make_codec_identity(source, cfg, augmentor, uttids=ids, smoke=smoke)
    reader = build_codec_cache(source, cfg, target or cfg["data"]["codec_augment"]["offline_cache_root"],
                               identity, augmentor, **kwargs)
    return reader, identity


def test_full_build_preserves_source_order_lengths_labels_and_rng(source_cache):
    cfg, ds, source, _ = source_cache
    before = capture_rng()
    hashes = [file_sha(source.path / row["file"]) for row in source.metadata["shards"]]
    reader, identity = build(source_cache)
    rng_equal(before, capture_rng())
    assert reader.identity["mode"] == "codec_waveform"
    assert reader.identity["source_cache"]["content_fingerprint"] == source.metadata["content_fingerprint"]
    assert list(reader.rows) == ds.uttids and len(reader.rows) == 396
    assert hashes == [file_sha(source.path / row["file"]) for row in source.metadata["shards"]]
    for uid in ds.uttids:
        assert torch.equal(reader.get(uid), source.get(uid) * .5)
        assert reader.rows[uid]["shape"] == source.rows[uid]["shape"]
        assert reader.rows[uid]["codec"]["status"] == "ok"
    assert {choose_spec(identity, uid) for uid in reversed(ds.uttids)} == {("mp3", "mp3", 0), ("opus", "opus", 0)}
    assert reader.verify()["samples"] == len(ds.uttids)
    training = open_train_dataset(cfg)
    assert training.codec_augmentor is None
    for name, expected in (("real", 0), (training.fake_classes[0], 1)):
        item = training[training.class_indices[name][0]]
        assert item["label"] == expected and item["feats"].shape == (1, 64000)
    dev = Mock(cache_reader=None, feature_caches={}, cache_mode="waveform")
    bindings, _ = cache_bindings(training, dev)
    assert [row["mode"] for row in bindings["caches"]] == ["waveform", "codec_waveform"]
    changed = copy.deepcopy(bindings)
    changed["caches"][-1]["content_fingerprint"] = "changed"
    assert make_run_identity(cfg, "a", "b", "c", bindings) != make_run_identity(cfg, "a", "b", "c", changed)


def test_smoke_is_explicit_stratified_and_never_a_formal_training_input(source_cache):
    cfg, ds, source, _ = source_cache
    reader, _ = build(source_cache, smoke=True, limit=330)
    assert reader.identity["smoke"] and len(reader.rows) == 330
    assert all(sum(ds.uttids[i] in reader.rows for i in pool) == 10 for pool in ds.class_indices.values())
    with pytest.raises(ValueError, match="Smoke/incomplete"):
        open_train_dataset(cfg)
    with open_codec_cache(source, cfg, allow_smoke=True) as allowed:
        assert allowed.stats()["smoke"]
    for i in (0, 1, 12, 63):
        uid = ds.uttids[i]
        assert choose_spec(reader.identity, uid) == choose_spec(reader.identity, uid)


def test_interruption_resumes_committed_shards_and_refuses_changed_identity(source_cache):
    cfg, ds, source, augmentor = source_cache
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    identity = make_codec_identity(source, cfg, augmentor)
    calls = 0
    original = augmentor.apply_array
    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 21:
            raise RuntimeError("simulated interruption")
        return original(*args)
    augmentor.apply_array = interrupt
    with pytest.raises(RuntimeError, match="interruption"):
        build_codec_cache(source, cfg, target, identity, augmentor)
    assert not target.exists()
    assert json.loads((target.with_name("codec.building") / "failure.json").read_text())["status"] == "error"
    with pytest.raises(ValueError, match="Invalid cache"):
        open_codec_cache(source, cfg)
    altered = copy.deepcopy(identity)
    altered["codec_parameters"]["build_seed"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        build_codec_cache(source, cfg, target, altered, augmentor, resume=True)
    with pytest.raises(FileExistsError, match="--resume"):
        build_codec_cache(source, cfg, target, identity, augmentor)
    augmentor.apply_array = original
    before = estimate_space(source, ds.uttids, target, reserve_bytes=0, resume=True)
    assert before["committed_pcm_bytes"] > 0
    resumed = build_codec_cache(source, cfg, target, identity, augmentor, resume=True)
    fresh, _ = build(source_cache, target=target.with_name("fresh"))
    assert {key: value for key, value in resumed.index[0]["codec"].items() if key != "seconds"} == {
        key: value for key, value in fresh.index[0]["codec"].items() if key != "seconds"}
    for uid in ds.uttids:
        assert torch.equal(resumed.get(uid), fresh.get(uid))
        assert choose_spec(resumed.identity, uid) == choose_spec(fresh.identity, uid)
    with pytest.raises(FileExistsError, match="already exists"):
        build_codec_cache(source, cfg, target, identity, augmentor, resume=True)


@pytest.mark.parametrize("change", ["seed", "family", "implementation", "source", "manifest", "preset"])
def test_identity_changes_fail_closed(source_cache, monkeypatch, change):
    import data.codec_cache as module
    cfg, _, source, _ = source_cache
    build(source_cache)
    altered = copy.deepcopy(cfg)
    if change == "seed": altered["data"]["codec_augment"]["build_seed"] += 1
    if change == "family": altered["data"]["codec_augment"]["codecs"] = ["mp3"]
    if change == "implementation": monkeypatch.setattr(module, "implementation_hashes", lambda: {"changed": "hash"})
    if change == "source": source.metadata = dict(source.metadata, content_fingerprint="changed")
    if change == "manifest": source.identity = dict(source.identity, manifest_sha256={"changed": "hash"})
    if change == "preset":
        # This is a code change to the codec preset definitions, bound by implementation hashes.
        original = module.implementation_hashes()
        monkeypatch.setattr(module, "implementation_hashes", lambda: dict(original, **{"codec_augmentation.py": "new-presets"}))
    with pytest.raises(ValueError, match="mismatch"):
        open_codec_cache(source, altered)


@pytest.mark.parametrize("damage", ["shard", "index", "metadata", "missing"])
def test_corruption_rejected_even_with_baseline_checksums_disabled(source_cache, damage):
    cfg, _, source, _ = source_cache
    reader, _ = build(source_cache)
    path = reader.path
    reader.close()
    cfg["cache"]["verify_checksums"] = False
    if damage == "shard":
        shard = path / "shard_00000.bin"
        content = shard.read_bytes()
        shard.write_bytes(b"bad!" + content[4:])
    if damage == "index": (path / "index.json").write_text("[]")
    if damage == "metadata": (path / "metadata.json").write_text("{}")
    if damage == "missing": (path / "shard_00000.bin").unlink()
    with pytest.raises(ValueError, match="Invalid cache"):
        open_codec_cache(source, cfg)


def test_locations_and_bad_source_inputs_are_rejected(source_cache):
    cfg, _, source, augmentor = source_cache
    for path in (source.path, source.path / "child", Path(cfg["cache"]["root"]), Path(cfg["runtime"]["output_dir"]) / "codec"):
        with pytest.raises(ValueError, match="independent"):
            validate_location(path, source, cfg)
    for key, value in (("split", "dev"), ("split", "test"), ("mode", "xlsr_feature"), ("smoke", True)):
        invalid = copy.copy(source)
        invalid.identity = dict(source.identity, **{key: value})
        with pytest.raises(ValueError, match="complete Train"):
            make_codec_identity(invalid, cfg, augmentor)


def test_source_shard_corruption_is_rejected_with_checksums_disabled(source_cache):
    cfg, _, source, _ = source_cache
    build(source_cache)
    source.close()
    cfg["cache"]["verify_checksums"] = False
    shard = source.path / source.metadata["shards"][0]["file"]
    data = shard.read_bytes()
    shard.write_bytes(b"bad!" + data[4:])
    with pytest.raises(ValueError, match="checksum"):
        open_train_dataset(cfg)


@pytest.mark.parametrize("workers", [0, 4, 8])
def test_offline_branches_and_crops_match_across_workers(source_cache, workers):
    cfg, ds, source, _ = source_cache
    build(source_cache)
    training = open_train_dataset(cfg)
    indices = list(range(16))
    with ReusableAudioLoader(training, cfg, batch_size=2) as loader:
        expected = torch.cat([row["feats"] for row in loader.for_step(indices, 7)])
        next_step = torch.cat([row["feats"] for row in loader.for_step(indices, 8)])
    assert not torch.equal(expected, next_step)
    # Fixture tensors are constant, so branch identity is directly observable.
    ratios = [float(expected[i, 0, 0] / source.get(ds.uttids[index])[0, 0]) for i, index in enumerate(indices)]
    assert set(ratios) == {.5, 1.0}
    cfg["runtime"]["train_num_workers"] = workers
    before = capture_rng()
    with ReusableAudioLoader(training, cfg, batch_size=2) as loader:
        actual = torch.cat([row["feats"] for row in loader.for_step(indices, 7)])
        repeated = torch.cat([row["feats"] for row in loader.for_step(indices, 7)])
    assert torch.equal(expected, actual) and torch.equal(actual, repeated)
    rng_equal(before, capture_rng())


def test_offline_choice_precedes_same_crop_and_rawboost_path(source_cache):
    cfg, _, _, _ = source_cache
    build(source_cache)
    training = open_train_dataset(cfg)
    events = []
    training.codec_probability = 1
    original = training.codec_cache_reader.get
    training.codec_cache_reader.get = lambda uid: (events.append("codec_read"), original(uid))[1]
    crop = training.sample_segment
    training.sample_segment = lambda audio: (events.append("crop"), crop(audio))[1]
    training.do_augment = True
    training.augmentor = lambda audio: (events.extend(["RIR", "RawBoost"]), audio)[1]
    training[0]
    assert events == ["codec_read", "crop", "RIR", "RawBoost"]


def test_builder_cli_dry_run_build_verify_inspection_and_no_backend_on_read(source_cache, monkeypatch, capsys):
    import sys
    from test_cache_integration import _cli
    cfg, _, source, augmentor = source_cache
    cli = _cli("build_codec_cache", monkeypatch)
    monkeypatch.setattr(cli, "load_config", lambda *args: cfg)
    monkeypatch.setattr(cli, "CodecAugmentor", lambda **kw: augmentor)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    base = ["build_codec_cache.py", "--output-dir", str(target), "--limit", "66", "--reserve-gib", "0"]
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    cli.main()
    assert not target.exists()
    assert '"preflight"' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    assert (target / "build_report.json").is_file()
    capsys.readouterr()
    monkeypatch.setattr(cli, "CodecAugmentor", Mock(side_effect=AssertionError("verify must not load backends")))
    monkeypatch.setattr(sys, "argv", ["build_codec_cache.py", "--output-dir", str(target), "--verify"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["samples"] == 66
    inspect = _cli("inspect_codec_cache", monkeypatch)
    monkeypatch.setattr(inspect, "load_config", lambda *args: cfg)
    destination = target.with_name("inspection")
    monkeypatch.setattr(sys, "argv", ["inspect_codec_cache.py", "--cache-dir", str(target),
                                     "--samples", "3", "--output-dir", str(destination)])
    inspect.main()
    report = json.loads((destination / "inspection.json").read_text())
    assert len(report["records"]) == 3 and len(list(destination.glob("*.wav"))) == 6


def test_smoke_benchmark_uses_identical_valid_episodes_and_records_build_separately(source_cache, monkeypatch):
    from engine import pipeline_benchmark as module
    cfg, _, _, _ = source_cache
    reader, _ = build(source_cache, smoke=True, limit=330)
    original = CodecAugmentor
    monkeypatch.setattr("data.speechfake_dataset.CodecAugmentor",
                        lambda **kw: original(**kw, backend=fake_backend))
    monkeypatch.setattr(module, "measure_codec_latency", lambda *args, **kw: [])
    seen = []
    def step(model, dataset, episodes, local, optimizer, number, *, loader, timings):
        ids = [index for episode in episodes for index in episode.indices]
        assert all(dataset.uttids[index] in reader.rows for index in ids)
        assert all(len(set(episode.indices)) == 30 for episode in episodes)
        seen.append(ids)
        # Consume the real dual-cache loader; no model update or training occurs.
        assert sum(len(batch["uttids"]) for batch in loader.for_step(ids, number)) == 240
    monkeypatch.setattr(module, "optimize_step", step)
    model = torch.nn.Linear(1, 1)
    model.device = torch.device("cpu")
    modes = ("waveform", "waveform_codec", "waveform_codec_offline")
    rows = module.benchmark_train(model, cfg, modes=modes, workers=(0,), warmup=0, steps=1,
                                 smoke_codec_cache=reader.path)
    assert seen[0] == seen[1] == seen[2]
    assert len({row["episode_sha256"] for row in rows}) == 1
    assert all(row["restricted_smoke"] and row["coverage_samples"] == 330 for row in rows)
    assert rows[-1]["codec_cache"]["build_seconds"] > 0
    totals = module.total_experiment_times(rows, {}, cfg)
    assert totals[-1]["one_time_codec_build_seconds"] == rows[-1]["codec_cache"]["build_seconds"]
    assert totals[-1]["total_seconds"] is None
