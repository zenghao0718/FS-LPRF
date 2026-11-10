import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from data.cache import (CacheReader, CacheWriter, HIDDEN_ORDER, PREPROCESSING,
                        atomic_json, canonical_sha, file_sha, make_cache_identity, cache_path)
from data.cache_pipeline import build_dataset_cache, open_train_dataset, evaluation_identities, dev_manifests
from data.speechfake_dataset import SpeechFakeDataset
from data.audio import FADTrim


def storage_identity(mode="waveform", ids=("a", "b", "c"), dtype="float32"):
    identity = dict(schema_version=1, split="train", mode=mode, validation_mode=None, seed=42,
                    uttids=list(ids), smoke=False, dtype=dtype, rng_schema_version=1,
                    dataset_fingerprint="synthetic", manifest_sha256={"tsv": "synthetic", "txt": "synthetic"},
                    source_audio_sha256="synthetic", preprocessing=PREPROCESSING)
    if mode == "xlsr_feature":
        identity.update(hidden_order=HIDDEN_ORDER, frontend_sha256="synthetic", feature_views=1)
    return identity


def small_cache(path, identity=None):
    identity = identity or storage_identity()
    with CacheWriter(path, identity, 2, verify_checksums=True) as writer:
        for number, uid in enumerate(identity["uttids"]):
            writer.add(uid, torch.arange(7 + number, dtype=torch.float32)[None])
        return writer.finish()


def cache_config(dataset, config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["data"].update(data_path=dataset.args.data_path, do_augment=False)
    cfg["cache"].update(root=str(tmp_path / "cache"), shard_size=2)
    cfg["runtime"]["output_dir"] = str(tmp_path / "run")
    weights = tmp_path / "synthetic_frontend.bin"
    weights.write_bytes(b"synthetic frontend identity, not model weights")
    cfg["model"]["frontend_path"] = str(weights)
    return cfg


@pytest.mark.parametrize("length", [32000, 128000])
def test_waveform_cache_exact_before_and_after_original_crop(dataset, config, tmp_path, monkeypatch, length):
    import soundfile as sf
    cfg = cache_config(dataset, config, tmp_path)
    path = Path(dataset.uttid2path[dataset.uttids[0]])
    time = np.arange(length, dtype=np.float32) / 16000
    sf.write(path, (.05 + .01*time) * np.sin(2*np.pi*(130*time+13*time*time)), 16000, subtype="FLOAT")
    raw = SpeechFakeDataset(cfg["data"], "train")
    uid = raw.uttids[0]
    identity = make_cache_identity(raw, cfg, "waveform", uttids=[uid], smoke=True)
    crop, augment = Mock(side_effect=AssertionError("must not crop during waveform build")), Mock()
    monkeypatch.setattr(raw, "sample_segment", crop)
    raw.augmentor = augment
    reader = build_dataset_cache(raw, cfg, identity)
    crop.assert_not_called()
    augment.assert_not_called()
    expected = raw.load_preprocessed_waveform(uid)
    assert expected.shape == (1, length) and expected.dtype == torch.float32
    assert torch.equal(reader.get(uid), expected)
    cached = SpeechFakeDataset(cfg["data"], "train").use_cache(reader, "waveform")
    raw.sample_segment = FADTrim.sample_segment.__get__(raw)
    np.random.seed(71)
    original = raw[0]["feats"]
    np.random.seed(71)
    actual = cached[0]["feats"]
    assert torch.equal(original, actual) and actual.shape == (1, 64000)
    assert reader.index[0]["shape"] == [1, length]
    assert not list(reader.path.glob("*.pt"))


def test_cached_training_still_crops_before_dynamic_augmentation(dataset, config, tmp_path, monkeypatch):
    cfg = cache_config(dataset, config, tmp_path)
    raw = SpeechFakeDataset(cfg["data"], "train")
    identity = make_cache_identity(raw, cfg, "waveform", uttids=raw.uttids[:1], smoke=True)
    reader = build_dataset_cache(raw, cfg, identity)
    cached = SpeechFakeDataset(cfg["data"], "train").use_cache(reader, "waveform")
    cached.do_augment = cached.train_mode = True
    seen = []
    def crop(wave):
        seen.append("crop")
        return FADTrim.sample_segment(cached, wave)
    def augment(wave):
        assert wave.shape == (1, 64000)
        seen.append("augment")
        return wave * 2
    monkeypatch.setattr(cached, "sample_segment", crop)
    cached.augmentor = augment
    first, second = cached[0]["feats"], cached[0]["feats"]
    assert seen == ["crop", "augment", "crop", "augment"]
    assert not torch.equal(first, second)
    assert torch.equal(reader.get(raw.uttids[0]), raw.load_preprocessed_waveform(raw.uttids[0]))


def test_shards_are_memmapped_lazily_and_pickle_without_maps(tmp_path):
    reader = small_cache(tmp_path / "cache")
    assert not reader._maps and len(reader.metadata["shards"]) == 2
    assert torch.equal(reader.get("b"), torch.arange(8, dtype=torch.float32)[None])
    assert reader._maps and not reader.__getstate__()["_maps"]
    with pytest.raises(ValueError, match="Missing cache record"):
        reader.get("missing")


def test_interrupted_build_resumes_only_committed_shards(tmp_path):
    target, identity = tmp_path / "cache", storage_identity()
    with pytest.raises(RuntimeError, match="interruption"):
        with CacheWriter(target, identity, 1) as writer:
            writer.add("a", torch.ones(1, 7))
            raise RuntimeError("interruption")
    assert not target.exists()
    stage = target.with_name("cache.building")
    # Simulate a crash during the next index publication; resume.json remains durable.
    (stage / "index.json").write_text("partial", encoding="utf-8")
    with CacheWriter(target, identity, 1, resume=True) as writer:
        assert [row["uttid"] for row in writer.index] == ["a"]
        writer.add("b", torch.ones(1, 8))
        writer.add("c", torch.ones(1, 9))
        reader = writer.finish()
    assert reader.metadata["complete"] and len(reader.verify()["shapes"]) == 3


def test_resume_refuses_changed_identity_and_force_preserves_original(tmp_path):
    target, identity = tmp_path / "cache", storage_identity()
    old = small_cache(target, identity)
    old_sha = file_sha(target / old.metadata["shards"][0]["file"])
    with pytest.raises(FileExistsError):
        CacheWriter(target, identity, 2)
    with CacheWriter(target, identity, 2, force=True) as writer:
        for uid in identity["uttids"]:
            writer.add(uid, torch.full((1, 10), 5.))
        writer.finish()
    preserved = list(tmp_path.glob("cache.preserved-*"))
    assert len(preserved) == 1
    assert file_sha(preserved[0] / old.metadata["shards"][0]["file"]) == old_sha
    changed = dict(identity, seed=17)
    with CacheWriter(tmp_path / "other", identity, 2):
        with pytest.raises(RuntimeError, match="Another cache builder"):
            CacheWriter(tmp_path / "other", identity, 2, resume=True)
    with pytest.raises(ValueError, match="identity mismatch"):
        CacheWriter(tmp_path / "other", changed, 2, resume=True)


def _reseal(path):
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["index_sha256"] = file_sha(path / "index.json")
    metadata["content_fingerprint"] = canonical_sha([metadata["fingerprint"], metadata["index_sha256"], metadata["shards"]])
    atomic_json(path / "metadata.json", metadata)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "offset", "shape", "dtype", "length", "shard_path", "checksum", "truncated", "index_hash"])
def test_cache_integrity_fails_closed(tmp_path, damage):
    path = tmp_path / "cache"
    small_cache(path)
    index = json.loads((path / "index.json").read_text())
    if damage == "missing": index.pop()
    if damage == "duplicate": index[1]["uttid"] = index[0]["uttid"]
    if damage == "offset": index[1]["offset"] = 10**9
    if damage == "shape": index[1]["shape"] = [25, 7, 1024]
    if damage == "dtype": index[1]["dtype"] = "float16"
    if damage == "length": index[1]["length"] = 1000
    if damage == "shard_path": index[1]["shard"] = "../outside.bin"
    atomic_json(path / "index.json", index)
    if damage != "index_hash": _reseal(path)
    else: (path / "index.json").write_text("[]")
    if damage in ("checksum", "truncated"):
        shard = path / "shard_00000.bin"
        content = shard.read_bytes()
        shard.write_bytes(content[:-4] if damage == "truncated" else b"1234" + content[4:])
    with pytest.raises(ValueError, match="Invalid cache"):
        CacheReader(path, verify_checksums=True)


def test_nan_and_smoke_caches_are_not_formal_inputs(tmp_path):
    identity = storage_identity(ids=("a",))
    with CacheWriter(tmp_path / "nan", identity, 1) as writer:
        with pytest.raises(ValueError, match="NaN/Inf"):
            writer.add("a", torch.full((1, 7), float("nan")))
    smoke = dict(identity, smoke=True)
    with CacheWriter(tmp_path / "smoke", smoke, 1) as writer:
        writer.add("a", torch.ones(1, 7))
        writer.finish()
    with pytest.raises(ValueError, match="Smoke/incomplete"):
        CacheReader(tmp_path / "smoke")
    reader = CacheReader(tmp_path / "smoke", allow_smoke=True)
    shard = reader.path / "shard_00000.bin"
    data = np.ones(7, dtype=np.float32)
    data[0] = np.nan
    shard.write_bytes(data.tobytes())
    unchecked = CacheReader(reader.path, allow_smoke=True, verify_checksums=False)
    with pytest.raises(ValueError, match="NaN/Inf"):
        unchecked.get("a")


def test_identity_binds_data_preprocessing_weights_seed_and_mode(dataset, config, tmp_path):
    cfg = cache_config(dataset, config, tmp_path)
    raw = SpeechFakeDataset(cfg["data"], "train")
    ids = raw.uttids[:1]
    wave = make_cache_identity(raw, cfg, "waveform", uttids=ids)
    feature = make_cache_identity(raw, cfg, "xlsr_feature", seed=42, uttids=ids)
    Path(cfg["model"]["frontend_path"]).write_bytes(b"changed synthetic weights")
    assert make_cache_identity(raw, cfg, "waveform", uttids=ids) == wave
    assert make_cache_identity(raw, cfg, "xlsr_feature", seed=42, uttids=ids) != feature
    assert make_cache_identity(raw, cfg, "xlsr_feature", seed=17, uttids=ids) != feature
    path = Path(raw.uttid2path[ids[0]])
    path.write_bytes(path.read_bytes() + b"new source bytes")
    assert make_cache_identity(raw, cfg, "waveform", uttids=ids) != wave
    raw.uttid2info[ids[0]].update(st=0., dur=8.)
    with pytest.raises(ValueError, match="st/dur"):
        make_cache_identity(raw, cfg, "waveform", uttids=ids)


def test_light_full_seed_identities_and_missing_train_cache(dataset, config, tmp_path):
    cfg = cache_config(dataset, config, tmp_path)
    manifests = dev_manifests(dataset, cfg)
    identities = []
    for mode, (manifest, path) in manifests.items():
        identities.extend(evaluation_identities(dataset, cfg, manifest, path, mode, limit=2))
    assert len(identities) == 6 and len({canonical_sha(row) for row in identities}) == 6
    assert all(row["split"] == "dev" and row["smoke"] for row in identities)
    cfg["data"]["train_cache_mode"] = "waveform"
    with pytest.raises(ValueError, match="Build explicitly"):
        open_train_dataset(cfg)
