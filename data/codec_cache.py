"""Train-only, fixed-per-utterance codec PCM; no source WAV or backend at runtime."""
import copy
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .cache import CacheReader, CacheWriter, atomic_json, file_sha
from .codec_augmentation import alignment_diagnostics, codec_settings
from .randomness import sample_rng, sample_seed

BUILDER_VERSION = 1


def implementation_hashes():
    return {name: file_sha(Path(__file__).with_name(name)) for name in
            ("codec_cache.py", "codec_augmentation.py", "cache.py", "randomness.py")}


def build_parameters(cfg):
    settings = codec_settings(cfg["data"])
    return {key: settings[key] for key in ("build_seed", "codecs", "sample_rate")}


def choose_spec(identity, uid):
    # Local RNG only: independent of traversal, workers, and interrupted shards.
    rng = np.random.default_rng(sample_seed("codec_cache", BUILDER_VERSION,
                                           identity["codec_parameters"]["build_seed"], uid))
    families = identity["codec_parameters"]["codecs"]
    family = families[int(rng.integers(len(families)))]
    choices = identity["codec_report"]["choices"][family]
    choice = choices[int(rng.integers(len(choices)))]
    return family, choice["variant"], choice["param"]


def select_smoke_ids(dataset, source, limit):
    """Cover every class and its length range; retain original manifest order."""
    if type(limit) is not int or not 0 < limit <= len(source.index):
        raise ValueError("--limit must be positive and at most the source sample count")
    classes = sorted(dataset.class_indices)
    selected = set()
    remaining = limit
    for position, name in enumerate(classes):
        pool = [dataset.uttids[i] for i in dataset.class_indices[name]]
        pool.sort(key=lambda uid: (source.rows[uid]["length"], uid))
        count = min(len(pool), remaining // (len(classes) - position))
        selected.update(pool[int(i)] for i in np.linspace(0, len(pool) - 1, count, dtype=int))
        remaining -= count
    if remaining:
        selected.update(uid for uid in source.identity["uttids"] if uid not in selected and
                        len(selected) < limit)
    return [uid for uid in source.identity["uttids"] if uid in selected]


def make_codec_identity(source, cfg, augmentor, *, uttids=None, smoke=False):
    if source.identity["mode"] != "waveform" or source.identity["split"] != "train" or source.identity["smoke"]:
        raise ValueError("Codec builds require the complete Train Waveform Cache")
    ids = list(source.identity["uttids"] if uttids is None else uttids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(source.rows):
        raise ValueError("Invalid codec cache ID subset")
    if not smoke and ids != source.identity["uttids"]:
        raise ValueError("Full codec cache must cover every source ID in order")
    identity = copy.deepcopy(source.identity)
    identity.update(mode="codec_waveform", dtype="float32", uttids=ids, smoke=bool(smoke),
                    builder_version=BUILDER_VERSION, codec_parameters=build_parameters(cfg),
                    source_cache=dict(fingerprint=source.metadata["fingerprint"],
                                      content_fingerprint=source.metadata["content_fingerprint"]),
                    implementation_sha256=implementation_hashes(), codec_report=augmentor.report())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    identity["source_commit"] = commit
    return identity


def estimate_space(source, ids, target, *, reserve_bytes=5 * 1024**3, resume=False):
    target = Path(target).resolve()
    existing = target
    while not existing.exists():
        existing = existing.parent
    pcm = sum(source.rows[uid]["length"] * 4 for uid in ids)
    # Index/records, staging shard, filesystem overhead; avoid trusting a historic 160 GiB figure.
    committed = 0
    stage = target.with_name(target.name + ".building")
    if resume and stage.exists():
        with CacheReader(stage, allow_smoke=True, allow_incomplete=True) as partial:
            if partial.identity["uttids"] != list(ids):
                raise ValueError("Resume IDs/limit changed")
            committed = sum(shard["bytes"] for shard in partial.metadata["shards"])
    overhead = max(16 * 1024**2, int(pcm * .03) + len(ids) * 4096)
    free = shutil.disk_usage(existing).free
    return dict(samples=len(ids), pcm_bytes=pcm, committed_pcm_bytes=committed, estimated_bytes=pcm - committed + overhead,
                reserve_bytes=reserve_bytes, free_bytes=free,
                enough_space=free >= pcm - committed + overhead + reserve_bytes, path=str(target))


def validate_location(target, source, cfg):
    target = Path(target).resolve()
    protected = [source.path, Path(cfg["cache"]["root"]).resolve(),
                 Path(cfg["data"]["data_path"]).resolve().parent,
                 Path(cfg["runtime"]["output_dir"]).resolve()]
    if any(target.is_relative_to(path) or path.is_relative_to(target) for path in protected):
        raise ValueError("Codec output must be independent of source caches, processed data and experiment outputs")
    return target


def validate_codec_reader(reader, source, cfg):
    identity = reader.identity
    if identity["mode"] != "codec_waveform" or identity["split"] != "train":
        raise ValueError("Expected Train codec_waveform cache")
    if identity.get("builder_version") != BUILDER_VERSION or identity.get("implementation_sha256") != implementation_hashes():
        raise ValueError("Codec builder/implementation identity mismatch")
    if identity.get("codec_parameters") != build_parameters(cfg):
        raise ValueError("Codec parameters/build_seed mismatch")
    binding = dict(fingerprint=source.metadata["fingerprint"], content_fingerprint=source.metadata["content_fingerprint"])
    if identity.get("source_cache") != binding:
        raise ValueError("Codec source cache content identity mismatch")
    for key in ("manifest_sha256", "dataset_fingerprint", "source_audio_sha256", "preprocessing", "rng_schema_version"):
        if identity[key] != source.identity[key]:
            raise ValueError(f"Codec source {key} mismatch")
    expected = [uid for uid in source.identity["uttids"] if uid in reader.rows]
    if identity["uttids"] != expected or (not identity["smoke"] and expected != source.identity["uttids"]):
        raise ValueError("Codec IDs must match source manifest order/coverage")
    report = identity["codec_report"]
    if report["available_families"] != identity["codec_parameters"]["codecs"] or report["skipped"]:
        raise ValueError("Incomplete codec preflight inventory")
    for uid, row in reader.rows.items():
        record = row.get("codec", {})
        spec = choose_spec(identity, uid)
        if (row["shape"] != source.rows[uid]["shape"] or record.get("status") != "ok" or
                record.get("uttid") != uid or record.get("input_samples") != row["length"] or
                record.get("output_samples") != row["length"] or
                (record.get("family"), record.get("variant"), record.get("param")) != spec):
            raise ValueError(f"Codec record/length/assignment mismatch: {uid}")
    return reader


def open_codec_cache(source, cfg, *, allow_smoke=False, path=None):
    # Always verify checksums, even when baseline cache verification was disabled.
    target = path or codec_settings(cfg["data"])["offline_cache_root"]
    reader = CacheReader(target, verify_checksums=True, allow_smoke=allow_smoke)
    try:
        return validate_codec_reader(reader, source, cfg)
    except (KeyError, TypeError, ValueError) as error:
        reader.close()
        raise ValueError(f"Invalid offline Codec Cache: {error}; no online/WAV fallback") from error


def preflight_waveforms(source, augmentor):
    """Exercise each preset on actual cached short/median/long utterances."""
    ordered = sorted(source.index, key=lambda row: row["length"])
    selected = list(dict.fromkeys(ordered[i]["uttid"] for i in (0, len(ordered)//2, len(ordered)-1)))
    rows = []
    for specs in augmentor.choices.values():
        for spec in specs:
            for uid in selected:
                started = time.perf_counter()
                source_audio = source.get(uid).numpy().reshape(-1)
                audio, info = augmentor.apply_array(source_audio, spec)
                rows.append(dict(uttid=uid, seconds=time.perf_counter()-started,
                                 alignment=alignment_diagnostics(source_audio, audio),
                                 **{key: value for key, value in info.items() if key != "payload"}))
    return rows


def build_codec_cache(source, cfg, target, identity, augmentor, *, resume=False, progress=None):
    target = validate_location(target, source, cfg)
    # No implicit reuse/overwrite: an existing completed cache must be explicitly verified.
    with CacheWriter(target, identity, cfg["cache"]["shard_size"], resume=resume, progress=progress) as writer:
        if progress:
            writer.progress = lambda done, total, elapsed: progress(done, total, elapsed + writer.previous_seconds)
        for uid in identity["uttids"][len(writer.index):]:
            spec = choose_spec(identity, uid)
            started = time.perf_counter()
            try:
                with sample_rng(sample_seed("codec_encode", identity["codec_parameters"]["build_seed"], uid)):
                    audio, info = augmentor.apply_array(source.get(uid).numpy().reshape(-1), spec)
            except Exception as error:
                atomic_json(writer.stage / "failure.json", dict(uttid=uid, family=spec[0], variant=spec[1],
                            param=spec[2], input_samples=source.rows[uid]["length"], status="error", error=str(error)))
                raise
            record = {key: value for key, value in info.items() if key != "payload"}
            record.update(uttid=uid, seconds=time.perf_counter()-started)
            if len(audio) != source.rows[uid]["length"] or not np.isfinite(audio).all():
                raise ValueError(f"Codec output length/finite check failed: {uid}")
            writer.add(uid, torch.from_numpy(audio).unsqueeze(0), codec_record=record)
        reader = writer.finish()
    return validate_codec_reader(reader, source, cfg)
