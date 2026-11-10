"""Synthetic codec failures and pinned v2 recovery; no AutoDL or real PCM migration."""
import copy
import json
from pathlib import Path
import subprocess
import types
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from data.cache import CacheReader, atomic_json
from data.codec_augmentation import CodecAugmentor, CodecClippingError
from data.codec_cache import build_codec_cache, choose_spec, make_codec_identity, open_codec_cache
from data.codec_compat import (AUDIT_NAME, RELEASE_PATH, authorize_clipping_recovery,
                               recovery_requested_identity, validate_compatibility)
from data.codec_records import encode_cached_sample, summarize_codec_results
from data.codec_storage import CodecCacheWriter
from data.codec_workers import CodecWorkerPool
from test_codec_cache import source_cache

BASE_COMMIT = "5f538056ec070b2e1ad6938bcf9a21af5de4ffa0"


def clipping_backend(spec, audio):
    if spec[0] == "opus":
        raise CodecClippingError(2.782, .000399, spec)
    return audio * np.float32(.5), {"payload": b"synthetic-codec"}


def clipping_worker_factory(**kwargs):
    return CodecAugmentor(**kwargs, backend=clipping_backend)


def test_typed_clipping_preserves_exact_original_rule_and_details(monkeypatch):
    import data.codec_augmentation as module
    def backend(spec, audio):
        decoded = audio.copy()
        decoded[0] = 2.782
        return decoded, {"payload": b"test"}
    augmentor = CodecAugmentor(codecs=["opus"], backend=backend)
    monkeypatch.setattr(module, "_fit_length", Mock(side_effect=AssertionError("Must not normalize clipped output")))
    spec = ("opus", "opus", 32)
    with pytest.raises(CodecClippingError) as caught:
        augmentor.apply_array(np.full(2506, .1, np.float32), spec)
    assert caught.value.peak == pytest.approx(2.782)
    assert caught.value.clipping_fraction == pytest.approx(1/2506)
    assert caught.value.spec == spec and "Severe Codec clipping" in str(caught.value)


def test_successful_pcm_and_info_are_identical_to_pinned_previous_code():
    payload = subprocess.check_output(["git", "show", BASE_COMMIT+":data/codec_augmentation.py"])
    legacy = types.ModuleType("data._pre_fallback_reference")
    legacy.__file__ = str(Path("data/codec_augmentation.py").resolve())
    legacy.__package__ = "data"
    exec(compile(payload, legacy.__file__, "exec"), legacy.__dict__)
    def backend(spec, audio):
        return audio * np.float32(.8), {"payload": b"same-synthetic-codec"}
    old = legacy.CodecAugmentor(codecs=["mp3", "opus", "amr", "encodec"], backend=backend)
    new = CodecAugmentor(codecs=["mp3", "opus", "amr", "encodec"], backend=backend)
    for length in (3, 2506, 64003):
        signal = np.linspace(-.25, .25, length, dtype=np.float32)
        for spec in (("mp3", "mp3", 32), ("opus", "opus", 32), ("amr", "amr_nb", 7), ("encodec", "encodec", 3.0)):
            before, old_info = old.apply_array(signal, spec)
            after, new_info = new.apply_array(signal, spec)
            assert np.array_equal(before, after) and new_info == old_info


def test_only_current_source_read_is_used_and_fallback_is_bit_identical(source_cache, caplog):
    _, ds, source, _ = source_cache
    uid = ds.uttids[4]
    original = source.get(uid).numpy().reshape(-1)
    reader = Mock()
    reader.get.return_value = torch.from_numpy(original)[None]
    augmentor = CodecAugmentor(codecs=["opus"], backend=clipping_backend)
    audio, record = encode_cached_sample(reader, augmentor, uid, ("opus", "opus", 32), 42)
    reader.get.assert_called_once_with(uid)
    assert audio.dtype == np.float32 and np.array_equal(audio, original)
    assert record["status"] == "fallback_original" and record["param"] == 32
    assert record["decoded_peak"] == 2.782 and record["clipping_fraction"] == .000399
    assert uid in caplog.text and "fallback_original" in caplog.text


@pytest.mark.parametrize("failure", ["source_nan", "source_inf", "decoded_nan", "decoded_inf", "length", "runtime", "oom", "string_only"])
def test_other_errors_are_never_fallbacks(failure):
    original = np.full(4000, .1, np.float32)
    if failure.startswith("source_"):
        original[0] = np.nan if failure == "source_nan" else np.inf
    reader = Mock()
    reader.get.return_value = torch.from_numpy(original)[None]
    def backend(spec, audio):
        if failure == "runtime": raise RuntimeError("decoder failed")
        if failure == "oom": raise torch.cuda.OutOfMemoryError("synthetic OOM; no GPU operation")
        if failure == "string_only": raise RuntimeError("Severe Codec clipping: ordinary untyped error")
        output = audio.copy()
        if failure == "decoded_nan": output[0] = np.nan
        if failure == "decoded_inf": output[0] = np.inf
        if failure == "length": output = output[:1]
        return output, {"payload": b"test"}
    with pytest.raises(RuntimeError) as caught:
        encode_cached_sample(reader, CodecAugmentor(codecs=["opus"], backend=backend), "test_uid", ("opus", "opus", 32), 42)
    assert not isinstance(caught.value, CodecClippingError)


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_serial_parallel_fallbacks_cover_every_id_and_report_honestly(source_cache, workers):
    cfg, ds, source, _ = source_cache
    ids = ds.uttids[:66]
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    if workers:
        with CodecWorkerPool(source, ["mp3", "opus"], workers=workers, max_inflight=4,
                             factory=clipping_worker_factory) as pool:
            identity = make_codec_identity(source, cfg, pool, uttids=ids, smoke=True)
            reader = build_codec_cache(source, cfg, target, identity, pool=pool)
    else:
        augmentor = clipping_worker_factory(codecs=["mp3", "opus"])
        identity = make_codec_identity(source, cfg, augmentor, uttids=ids, smoke=True)
        reader = build_codec_cache(source, cfg, target, identity, augmentor)
    assert list(reader.rows) == ids and len(reader.rows) == 66
    expected_fallback = 0
    for uid in ids:
        spec = choose_spec(identity, uid)
        fallback = spec[0] == "opus"
        expected_fallback += fallback
        assert reader.rows[uid]["codec"]["status"] == ("fallback_original" if fallback else "ok")
        assert torch.equal(reader.get(uid), source.get(uid) * (1 if fallback else .5))
    summary = summarize_codec_results(reader.index)
    assert summary["total_samples"] == 66 and summary["fallback_original_count"] == expected_fallback
    assert summary["codec_success_count"] == 66-expected_fallback
    assert summary["fallback_fraction"] == expected_fallback/66
    assert summary["fallback_by_family"]["opus"] == expected_fallback
    assert len(summary["fallback_details"]) == expected_fallback and 0 < expected_fallback < 66


@pytest.mark.parametrize("damage", ["status", "reason", "message", "evidence", "assignment"])
def test_reader_rejects_invalid_fallback_records(source_cache, damage):
    cfg, ds, source, _ = source_cache
    augmentor = clipping_worker_factory(codecs=["mp3", "opus"])
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:33], smoke=True)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    reader = build_codec_cache(source, cfg, target, identity, augmentor)
    rows = copy.deepcopy(reader.index)
    record = next(row["codec"] for row in rows if row["codec"]["status"] == "fallback_original")
    if damage == "status": record["status"] = "skipped"
    if damage == "reason": record.pop("fallback_reason")
    if damage == "message": record["exception_message"] = ""
    if damage == "evidence": record.update(decoded_peak=1.9, clipping_fraction=.009)
    if damage == "assignment": record["param"] = 999
    atomic_json(target/"index.json", rows)
    with pytest.raises(ValueError):
        open_codec_cache(source, cfg, path=target, allow_smoke=True)


@pytest.fixture
def interrupted_5f_cache(source_cache):
    cfg, ds, source, augmentor = source_cache
    ids = ds.uttids[:33]
    old = make_codec_identity(source, cfg, augmentor, uttids=ids, smoke=True, encodec_device="cuda:0")
    release = json.loads(RELEASE_PATH.read_text())
    old.pop("clipping_fallback_policy")
    old["implementation_sha256"] = release["source_implementation_sha256"]
    old["source_commit"] = BASE_COMMIT
    old["codec_report"]["execution"] = dict(encodec_device="cuda:0", worker_threads=1)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    with CodecCacheWriter(target, old, 5) as writer:
        for uid in ids[:10]:
            audio, record = encode_cached_sample(source, augmentor, uid, choose_spec(old, uid), 42)
            writer.add(uid, torch.from_numpy(audio)[None], codec_record=record)
    stage = target.with_name(target.name+".building")
    before = {p.name: p.read_bytes() for p in stage.iterdir() if p.is_file()}
    return cfg, source, augmentor, old, target, stage, before


def recover(fixture, **kwargs):
    cfg, source, _, _, target, _, _ = fixture
    return authorize_clipping_recovery(target, source, cfg, expected_shards=2, expected_records=10,
                                       expected_total=33, **kwargs)


def test_controlled_resume_preserves_original_files_and_completes_with_fallback(interrupted_5f_cache):
    cfg, source, _, old, target, stage, before = interrupted_5f_cache
    requested = recovery_requested_identity(source, cfg, old, "cuda:0", 1)
    with pytest.raises(ValueError, match="no controlled"):
        CodecCacheWriter(target, requested, 5, resume=True)
    assert recover(interrupted_5f_cache, dry_run=True)["status"] == "dry_run"
    assert not (stage/AUDIT_NAME).exists()
    audit = recover(interrupted_5f_cache)["audit"]
    assert audit["boundary"]["records"] == 10
    assert {name: (stage/name).read_bytes() for name in before} == before
    audit_bytes = (stage/AUDIT_NAME).read_bytes()
    assert recover(interrupted_5f_cache)["status"] == "already_authorized"
    assert (stage/AUDIT_NAME).read_bytes() == audit_bytes
    reader = build_codec_cache(source, cfg, target, requested,
                               clipping_worker_factory(codecs=["mp3", "opus"]), resume=True)
    assert reader.identity == old and len(reader.rows) == 33
    assert reader.metadata["compatibility_audit_sha256"] == audit["audit_sha256"]
    assert reader.metadata["compatibility_boundary"]["records"] == 10
    for name, payload in before.items():
        if name == "identity.json" or name.startswith(("shard_", "commit_")):
            assert (target/name).read_bytes() == payload
    assert summarize_codec_results(reader.index)["fallback_original_count"] > 0
    assert len(open_codec_cache(source, cfg, path=target, allow_smoke=True).rows) == 33


def test_recovery_does_not_read_or_hash_pcm(interrupted_5f_cache, monkeypatch):
    import data.cache as storage
    original = storage.file_sha
    def guarded(path):
        if Path(path).suffix == ".bin" or Path(path).name == "index.json": pytest.fail("PCM/index hash during recovery")
        return original(path)
    monkeypatch.setattr(storage, "file_sha", guarded)
    monkeypatch.setattr(CacheReader, "get", Mock(side_effect=AssertionError("Recovery must not read PCM")))
    assert recover(interrupted_5f_cache)["status"] == "authorized"


def test_recovery_remains_idempotent_after_new_fallback_commits(interrupted_5f_cache):
    cfg, source, _, old, target, stage, before = interrupted_5f_cache
    audit = recover(interrupted_5f_cache)["audit"]
    requested = recovery_requested_identity(source, cfg, old, "cuda:0", 1)
    augmentor = clipping_worker_factory(codecs=["mp3", "opus"])
    with CodecCacheWriter(target, requested, 5, resume=True) as writer:
        for uid in old["uttids"][10:25]:
            audio, record = encode_cached_sample(source, augmentor, uid, choose_spec(old, uid), 42)
            writer.add(uid, torch.from_numpy(audio)[None], codec_record=record)
    assert recover(interrupted_5f_cache)["audit"] == audit
    assert json.loads((stage/"progress.json").read_text())["records"] == 25
    reader = build_codec_cache(source, cfg, target, requested, augmentor, resume=True)
    assert len(reader.rows) == 33 and reader.metadata["compatibility_boundary"] == audit["boundary"]
    for name, payload in before.items():
        if name.startswith(("shard_", "commit_")):
            assert (target/name).read_bytes() == payload


def test_fatal_error_after_committed_fallback_keeps_resume_point(source_cache):
    cfg, ds, source, _ = source_cache
    cfg["cache"]["codec_shard_size"] = 5
    augmentor = clipping_worker_factory(codecs=["mp3", "opus"])
    identity = make_codec_identity(source, cfg, augmentor, uttids=ds.uttids[:33], smoke=True)
    target = Path(cfg["data"]["codec_augment"]["offline_cache_root"])
    apply = augmentor.apply_array
    calls = 0
    def failing(audio, spec):
        nonlocal calls
        calls += 1
        if calls == 13:
            raise RuntimeError("real decoder failure")
        return apply(audio, spec)
    augmentor.apply_array = failing
    with pytest.raises(RuntimeError, match="real decoder failure"):
        build_codec_cache(source, cfg, target, identity, augmentor)
    stage = target.with_name(target.name+".building")
    assert json.loads((stage/"progress.json").read_text())["records"] == 10
    augmentor.apply_array = apply
    reader = build_codec_cache(source, cfg, target, identity, augmentor, resume=True)
    assert list(reader.rows) == ds.uttids[:33]
    assert summarize_codec_results(reader.index)["fallback_original_count"] > 0


def test_gpu_and_cpu_worker_entry_share_typed_fallback(source_cache, monkeypatch):
    import data.codec_workers as workers
    _, ds, source, _ = source_cache
    def backend(spec, audio):
        raise CodecClippingError(2.782, .000399, spec)
    monkeypatch.setattr(workers, "_SOURCE", source)
    monkeypatch.setattr(workers, "_AUGMENTOR", CodecAugmentor(codecs=["opus", "encodec"], backend=backend))
    for spec in (("opus", "opus", 32), ("encodec", "encodec", 3.0)):
        audio, record = workers._encode_task(ds.uttids[0], spec, 42)
        assert record["status"] == "fallback_original" and record["family"] == spec[0]
        assert np.array_equal(audio, source.get(ds.uttids[0]).numpy().reshape(-1))


def test_unapproved_target_release_cannot_authorize_recovery(interrupted_5f_cache, monkeypatch):
    import data.codec_cache as module
    original = module.implementation_hashes()
    monkeypatch.setattr(module, "implementation_hashes", lambda: dict(original, **{"codec_cache.py": "changed"}))
    with pytest.raises(ValueError, match="not the explicitly approved"):
        recover(interrupted_5f_cache)


def test_recovery_cli_runs_without_backends_and_supports_dry_run(interrupted_5f_cache, monkeypatch, capsys):
    import sys
    from test_cache_integration import _cli
    cfg, _, _, _, target, stage, _ = interrupted_5f_cache
    cli = _cli("recover_codec_cache", monkeypatch)
    monkeypatch.setattr(cli, "load_config", lambda *args: cfg)
    args = ["recover_codec_cache.py", "--cache-dir", str(target), "--expected-shards", "2",
            "--expected-records", "10", "--expected-total", "33", "--encodec-device", "cuda:0"]
    monkeypatch.setattr(sys, "argv", args+["--dry-run"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    assert not (stage/AUDIT_NAME).exists()
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "authorized"


@pytest.mark.parametrize("point", ["before", "after"])
def test_atomic_recovery_interruption_is_retryable(interrupted_5f_cache, monkeypatch, point):
    import data.codec_compat as module
    _, _, _, _, _, stage, before = interrupted_5f_cache
    original = module.atomic_json
    def interrupted(path, payload):
        if point == "after": original(path, payload)
        raise RuntimeError("simulated recovery interruption")
    monkeypatch.setattr(module, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="recovery interruption"):
        recover(interrupted_5f_cache)
    assert {name: (stage/name).read_bytes() for name in before} == before
    monkeypatch.setattr(module, "atomic_json", original)
    assert recover(interrupted_5f_cache)["status"] == ("already_authorized" if point == "after" else "authorized")


@pytest.mark.parametrize("change", ["counts", "source", "seed", "device", "threads", "version", "chain"])
def test_recovery_rejects_incompatible_or_damaged_prefix(interrupted_5f_cache, change):
    cfg, source, _, _, target, stage, _ = interrupted_5f_cache
    if change == "counts":
        with pytest.raises(ValueError, match="counts differ"):
            authorize_clipping_recovery(target, source, cfg, expected_shards=3, expected_records=15, expected_total=33)
        return
    if change == "source": source.metadata = dict(source.metadata, content_fingerprint="changed")
    if change == "seed": cfg["data"]["codec_augment"]["build_seed"] += 1
    if change == "version":
        identity = json.loads((stage/"identity.json").read_text())
        identity["source_commit"] = "unapproved"
        atomic_json(stage/"identity.json", identity)
    if change == "chain":
        commit = json.loads((stage/"commit_00000.json").read_text())
        commit["index"][0]["codec"]["seconds"] = 999
        atomic_json(stage/"commit_00000.json", commit)
    options = {"device": "cpu"} if change == "device" else {"threads": 2} if change == "threads" else {}
    with pytest.raises(ValueError): recover(interrupted_5f_cache, **options)
    assert not (stage/AUDIT_NAME).exists()
