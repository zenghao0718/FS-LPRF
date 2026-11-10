"""Synthetic, bounded build checks. No real codec, CUDA run, or speed claims."""
import copy
from concurrent.futures import Future
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from data.cache import CacheReader, atomic_json, canonical_sha
from data.codec_augmentation import CodecAugmentor
from data.codec_cache import (build_codec_cache, choose_spec, make_codec_identity, open_codec_cache,
                              LEGACY_CPU_IMPLEMENTATION)
from data.codec_storage import CodecCacheWriter
from data.codec_workers import CodecWorkerPool
from test_codec_cache import source_cache, build


def slow_backend(spec, audio):
    # Different task costs exercise out-of-order process completion.
    time.sleep(.006 if spec[0] == "mp3" else .001)
    return audio * np.float32(.5), {"payload": b"synthetic codec"}


def synthetic_worker_factory(**kwargs):
    import data.cache as storage
    original = storage.file_sha
    def guarded(path):
        if Path(path).suffix == ".bin" or Path(path).name == "index.json":
            raise AssertionError("Worker attempted a PCM/index hash")
        return original(path)
    storage.file_sha = guarded
    return CodecAugmentor(**kwargs, backend=slow_backend)


def test_normal_source_codec_read_build_resume_never_hash_pcm_or_index(source_cache, monkeypatch):
    import data.cache as storage
    cfg, ds, source, augmentor = source_cache
    cfg["cache"]["verify_checksums"] = False
    original = storage.file_sha
    def guarded(path):
        if Path(path).suffix == ".bin" or Path(path).name == "index.json":
            pytest.fail(f"Large-file hash: {path}")
        return original(path)
    monkeypatch.setattr(storage, "file_sha", guarded)
    with CacheReader(source.path) as legacy:
        assert legacy.metadata["fingerprint"] == source.metadata["fingerprint"]
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:35], smoke=True)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    with CodecCacheWriter(target, identity, 5) as writer:
        for uid in identity["uttids"][:5]:
            wave, info = augmentor.apply_array(source.get(uid).numpy().reshape(-1), choose_spec(identity, uid))
            record = dict({k: v for k, v in info.items() if k != "payload"}, uttid=uid, seconds=0)
            writer.add(uid, torch.from_numpy(wave)[None], codec_record=record)
    # Resume must use the journal prefix, not scan any completed PCM.
    reader = build_codec_cache(source, cfg, target, identity, augmentor, resume=True)
    with open_codec_cache(source, cfg, path=target, allow_smoke=True) as opened:
        assert opened.verify()["samples"] == 35
    assert all(row["sha256"] is None for row in reader.metadata["shards"])


@pytest.mark.parametrize("damage", ["nan", "inf", "length"])
def test_v2_rejects_bad_codec_output_before_publication(source_cache, monkeypatch, damage):
    cfg, ds, source, augmentor = source_cache
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:3], smoke=True)
    original = augmentor.apply_array
    def broken(*args):
        audio, info = original(*args)
        if damage == "length":
            audio = np.pad(audio, (0, 1))
        else:
            audio[0] = np.nan if damage == "nan" else np.inf
        return audio, info
    monkeypatch.setattr(augmentor, "apply_array", broken)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    with pytest.raises(ValueError, match="length/finite"):
        build_codec_cache(source, cfg, target, identity, augmentor)
    assert not target.exists()


@pytest.mark.parametrize("damage", ["id", "offset", "dtype", "range", "missing_pcm"])
def test_v2_resume_rejects_broken_incremental_index_without_scanning_pcm(source_cache, damage):
    cfg, ds, source, augmentor = source_cache
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:10], smoke=True)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    with CodecCacheWriter(target, identity, 5) as writer:
        for uid in identity["uttids"][:5]:
            audio, info = augmentor.apply_array(source.get(uid).numpy().reshape(-1), choose_spec(identity, uid))
            writer.add(uid, torch.from_numpy(audio)[None], codec_record=dict(
                {k: v for k, v in info.items() if k != "payload"}, uttid=uid, seconds=0))
    stage = target.with_name(target.name + ".building")
    commit_file = stage / "commit_00000.json"
    commit = json.loads(commit_file.read_text())
    if damage == "id": commit["index"][1]["uttid"] = commit["index"][0]["uttid"]
    if damage == "offset": commit["index"][1]["offset"] = 999999
    if damage == "dtype": commit["index"][1]["dtype"] = "float16"
    if damage == "range": commit["index"][1]["shape"] = [1, 10**9]
    if damage == "missing_pcm": (stage / "shard_00000.bin").unlink()
    atomic_json(commit_file, commit)
    with pytest.raises(ValueError, match="Invalid cache"):
        CodecCacheWriter(target, identity, 5, resume=True)


def test_incremental_flush_writes_only_new_records_and_small_progress(source_cache, monkeypatch):
    import data.codec_storage as storage
    cfg, ds, source, augmentor = source_cache
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:15], smoke=True)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    writes = []
    original = storage.atomic_json
    def saved(path, payload):
        writes.append((Path(path).name, len(json.dumps(payload))))
        original(path, payload)
    monkeypatch.setattr(storage, "atomic_json", saved)
    with CodecCacheWriter(target, identity, 5) as writer:
        for uid in identity["uttids"]:
            spec = choose_spec(identity, uid)
            wave, info = augmentor.apply_array(source.get(uid).numpy().reshape(-1), spec)
            record = dict({k: v for k, v in info.items() if k != "payload"}, uttid=uid, seconds=0)
            writer.add(uid, torch.from_numpy(wave)[None], codec_record=record)
        assert not (writer.stage / "index.json").exists()
        assert not (writer.stage / "metadata.json").exists()
        for commit in sorted(writer.stage.glob("commit_*.json")):
            assert len(json.loads(commit.read_text())["index"]) == 5
        writer.finish().close()
    names = [name for name, size in writes]
    assert names.count("identity.json") == names.count("index.json") == names.count("metadata.json") == 1
    assert len([name for name in names if name.startswith("commit_")]) == 3
    assert max(size for name, size in writes if name == "progress.json") < 600
    metadata = json.loads((target / "metadata.json").read_text())
    assert "identity" not in metadata and metadata["identity_file"] == "identity.json"


@pytest.mark.parametrize("crash_point", ["identity", "first_progress", "commit", "progress", "final_index", "final_metadata"])
def test_crash_publication_resumes_only_authoritative_prefix(source_cache, monkeypatch, crash_point):
    import data.codec_storage as storage
    cfg, ds, source, augmentor = source_cache
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:17], smoke=True)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    original = storage.atomic_json
    fired = False
    def interrupt(path, payload):
        nonlocal fired
        name = Path(path).name
        selected = ((crash_point == "identity" and name == "identity.json") or
                    (crash_point == "first_progress" and name == "progress.json" and payload["committed_shards"] == 0) or
                    (crash_point == "commit" and name == "commit_00001.json") or
                    (crash_point == "progress" and name == "progress.json" and payload["committed_shards"] == 2) or
                    (crash_point == "final_index" and name == "index.json") or
                    (crash_point == "final_metadata" and name == "metadata.json"))
        if selected and not fired:
            fired = True
            raise RuntimeError("simulated atomic publication crash")
        original(path, payload)
    monkeypatch.setattr(storage, "atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="publication crash"):
        with CodecCacheWriter(target, identity, 5) as writer:
            for uid in identity["uttids"]:
                spec = choose_spec(identity, uid)
                wave, info = augmentor.apply_array(source.get(uid).numpy().reshape(-1), spec)
                writer.add(uid, torch.from_numpy(wave)[None], codec_record=dict(
                    {k: v for k, v in info.items() if k != "payload"}, uttid=uid, seconds=0))
            writer.finish()
    monkeypatch.setattr(storage, "atomic_json", original)
    stage = target.with_name(target.name + ".building")
    checkpoint = json.loads((stage / "progress.json").read_text()) if (stage / "progress.json").exists() else {"records": 0}
    with CodecCacheWriter(target, identity, 7, resume=True) as writer:
        assert len(writer.index) == checkpoint["records"]
    resumed = build_codec_cache(source, cfg, target, identity, augmentor, resume=True)
    assert list(resumed.rows) == identity["uttids"] and len(set(resumed.rows)) == 17
    for uid in resumed.rows:
        assert torch.equal(resumed.get(uid), source.get(uid) * .5)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_spawn_workers_fixed_assignments_order_labels_lengths_and_bounded_pcm(source_cache, workers):
    cfg, ds, source, _ = source_cache
    cfg["cache"]["verify_checksums"] = False
    ids = ds.uttids[:66]
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    with CodecWorkerPool(source, ["mp3", "opus"], workers=workers, max_inflight=3,
                         factory=synthetic_worker_factory) as pool:
        identity = make_codec_identity(source, cfg, pool, uttids=ids, smoke=True)
        reader = build_codec_cache(source, cfg, target, identity, pool=pool)
        assert pool.peak_inflight <= 3
    assert list(reader.rows) == ids
    for uid in ids:
        assert torch.equal(reader.get(uid), source.get(uid) * .5)
        assert reader.rows[uid]["shape"] == source.rows[uid]["shape"]
        row = reader.rows[uid]["codec"]
        assert (row["family"], row["variant"], row["param"]) == choose_spec(identity, uid)
        assert ds.uttid2info[uid]["label"] == (0 if ds.uttid2info[uid]["attacker"] == "real" else 1)


def test_old_whole_wave_cpu_codec_format_still_reads_without_hashes(source_cache, monkeypatch):
    from data.cache import CacheWriter
    cfg, _, source, _ = source_cache
    current, identity = build(source_cache, smoke=True, limit=33)
    legacy = copy.deepcopy(identity)
    legacy.pop("codec_storage_format")
    legacy.pop("encodec_device")
    legacy["implementation_sha256"] = LEGACY_CPU_IMPLEMENTATION
    target = current.path.with_name("legacy_cpu_v1")
    with CacheWriter(target, legacy, 10, verify_checksums=True) as writer:
        for uid in legacy["uttids"]:
            writer.add(uid, current.get(uid), codec_record=current.rows[uid]["codec"])
        writer.finish().close()
    cfg["cache"]["verify_checksums"] = False
    import data.cache as storage
    monkeypatch.setattr(storage, "file_sha", lambda path: pytest.fail("Legacy PCM/index must not be rehashed"))
    with open_codec_cache(source, cfg, path=target, allow_smoke=True) as opened:
        assert opened.metadata["fingerprint"] == canonical_sha(legacy)


def test_parallel_resume_can_change_worker_count_but_rejects_device_change(source_cache):
    cfg, ds, source, _ = source_cache
    ids = ds.uttids[:33]
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    with pytest.raises(RuntimeError, match="interrupt after committed shard"):
        with CodecWorkerPool(source, ["mp3", "opus"], workers=1, max_inflight=3,
                             factory=synthetic_worker_factory) as pool:
            identity = make_codec_identity(source, cfg, pool, uttids=ids, smoke=True)
            with CodecCacheWriter(target, identity, 5) as writer:
                tasks = ((uid, choose_spec(identity, uid), 42) for uid in ids)
                for number, (uid, audio, record) in enumerate(pool.map(tasks), 1):
                    writer.add(uid, torch.from_numpy(audio)[None], codec_record=record)
                    if number == 7:
                        raise RuntimeError("interrupt after committed shard")
    changed = dict(identity, encodec_device="cuda:0")
    with pytest.raises(ValueError, match="identity mismatch"):
        CodecCacheWriter(target, changed, 5, resume=True)
    with CodecWorkerPool(source, ["mp3", "opus"], workers=2, max_inflight=4,
                         factory=synthetic_worker_factory) as pool:
        assert make_codec_identity(source, cfg, pool, uttids=ids, smoke=True) == identity
        reader = build_codec_cache(source, cfg, target, identity, pool=pool, resume=True)
    assert list(reader.rows) == ids
    assert all(torch.equal(reader.get(uid), source.get(uid) * .5) for uid in ids)


class RoutingExecutor:
    """Fake executors for CUDA routing only; never invokes a CUDA backend."""
    instances = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.families, self.device = kwargs["initargs"][1:3]
        self.jobs = []
        self.instances.append(self)
    def submit(self, fn, *args):
        result = Future()
        if not args:
            augmentor = CodecAugmentor(codecs=self.families, backend=slow_backend)
            result.set_result(augmentor.report())
        else:
            self.jobs.append(args)
            uid, spec, seed = args
            result.set_result((np.ones(7, np.float32), dict(uttid=uid, spec=spec)))
        return result
    def shutdown(self, **kwargs): pass


def test_cuda_routing_owns_one_spawn_process_and_cpu_workers_exclude_encodec(source_cache):
    cfg, ds, source, augmentor = source_cache
    RoutingExecutor.instances = []
    with CodecWorkerPool(source, ["mp3", "opus", "amr", "encodec"], workers=4,
                         encodec_device="cuda:0", max_inflight=2, executor_factory=RoutingExecutor) as pool:
        assert pool.cpu_pool.kwargs["max_workers"] == 4 and "encodec" not in pool.cpu_pool.families
        assert pool.gpu_pool.kwargs["max_workers"] == 1 and pool.gpu_pool.families == ["encodec"]
        assert all(p.kwargs["mp_context"].get_start_method() == "spawn" for p in RoutingExecutor.instances)
        tasks = [(f"id_{i}", (name, name, 0), 42) for i, name in enumerate(["opus", "encodec", "mp3", "encodec", "amr"])]
        assert [uid for uid, audio, record in pool.map(tasks)] == [row[0] for row in tasks]
        assert all(row[1][0] == "encodec" for row in pool.gpu_pool.jobs)
        assert all(row[1][0] != "encodec" for row in pool.cpu_pool.jobs)
        assert pool.peak_inflight == 2
    cpu = make_codec_identity(source, cfg, augmentor, encodec_device="cpu")
    gpu = make_codec_identity(source, cfg, augmentor, encodec_device="cuda:0")
    assert cpu != gpu and cpu["builder_version"] == gpu["builder_version"] == 1
