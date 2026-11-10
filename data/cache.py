"""Validated, sharded caches; tensors are read on demand, never at Dataset init."""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

import numpy as np
import torch

from .evaluation_manifest import dataset_fingerprint
from .randomness import RNG_SCHEMA_VERSION

PREPROCESSING = dict(sample_rate=16000, mono="first_channel", trim_top_db=30,
                     trim_frame_length=1024, trim_hop_length=256, target_rms_db=-20,
                     clip=[-1, 1], max_samples=64000, fix_len=True)
HIDDEN_ORDER = ["post_features"] + [f"layer_results[{i}][0]" for i in range(24)]


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=None)
def _stat_sha(path, size, modified, changed):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _stat_sha(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_digest(dataset, uttids):
    digest = hashlib.sha256()
    memo = getattr(dataset, "_cache_source_hashes", {})
    dataset._cache_source_hashes = memo
    for uid in uttids:
        path = Path(dataset.uttid2path[uid]).resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        saved = memo.get(str(path))
        if saved is None or saved[0] != signature:
            saved = memo[str(path)] = (signature, file_sha(path))
        row = [uid, str(path), saved[1], dataset.uttid2info[uid]]
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
    return digest.hexdigest()


def make_cache_identity(dataset, cfg, mode, *, validation_mode=None, seed=None,
                        manifest=None, manifest_path=None, uttids=None, smoke=False, recorded_source_sha256=None):
    if dataset.split == "test":
        raise ValueError("Test always uses original WAVs; Test caches are forbidden")
    if mode not in ("waveform", "xlsr_feature") or dataset.split not in ("train", "dev"):
        raise ValueError("Caches are restricted to SpeechFake Train/Dev")
    ids = list(dataset.uttids if uttids is None else uttids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(dataset.uttids):
        raise ValueError("Cache uttids must be a nonempty, unique dataset subset")
    if mode == "waveform" and any("st" in dataset.uttid2info[uid] or "dur" in dataset.uttid2info[uid] for uid in ids):
        raise ValueError("Full Waveform Cache cannot freeze the optional random st/dur file-read branch; use none")
    if recorded_source_sha256 is not None and (
            mode != "waveform" or not isinstance(recorded_source_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", recorded_source_sha256)):
        raise ValueError("Runtime waveform identity requires a recorded source SHA-256")
    root = Path(cfg["data"]["data_path"])
    preprocessing = dict(PREPROCESSING, trim_silence=cfg["data"]["trim_silence"],
                         implementation_sha256=file_sha(Path(__file__).with_name("audio.py")))
    identity = dict(schema_version=cfg["cache"]["schema_version"], split=dataset.split, mode=mode,
                    validation_mode=validation_mode, seed=seed, uttids=ids, smoke=bool(smoke),
                    dataset_fingerprint=dataset_fingerprint(dataset),
                    manifest_sha256={suffix: file_sha(root / f"{dataset.split}.{suffix}") for suffix in ("tsv", "txt")},
                    source_audio_sha256=(source_digest(dataset, ids) if recorded_source_sha256 is None
                                         else recorded_source_sha256), preprocessing=preprocessing,
                    rng_schema_version=RNG_SCHEMA_VERSION,
                    dtype="float32" if mode == "waveform" else cfg["cache"]["feature_dtype"])
    if mode == "xlsr_feature":
        if seed is None:
            raise ValueError("Feature caches require a fixed view seed")
        identity.update(frontend_sha256=file_sha(cfg["model"]["frontend_path"]),
                        hidden_order=HIDDEN_ORDER, feature_views=1,
                        feature_build_batch_size=cfg["cache"]["feature_build_batch_size"],
                        train_feature_view_seed=seed if dataset.split == "train" else None)
        if dataset.split != "train":
            if validation_mode not in ("light", "full") or manifest is None or manifest_path is None:
                raise ValueError("Dev/Test cache requires its validated mode, seed and evaluation manifest")
            if json.loads(Path(manifest_path).read_text(encoding="utf-8")) != manifest:
                raise ValueError("Evaluation manifest differs from the file on disk")
            identity["evaluation_manifest_sha256"] = file_sha(manifest_path)
        else:
            augment = dict(enabled=cfg["data"]["do_augment"], rir_probability=1/3,
                           rawboost_probability=1/5, rawboost_algorithm=5,
                           order=["RIR", "RawBoost"],
                           implementation_sha256=file_sha(Path(__file__).with_name("augmentation.py")))
            if augment["enabled"]:
                from .speechfake_dataset import ConfiguredRawBoost
                rir_root = cfg["data"].get("rir_root") or os.environ.get("FS_LPRF_RIR_ROOT")
                if not rir_root:
                    raise ValueError("Train feature cache requires data.rir_root")
                augmentor = ConfiguredRawBoost(rir_root)
                candidates = augmentor.rir
                augment["rawboost_parameters"] = vars(augmentor.rawboost_args)
                augment["rir_sha256"] = canonical_sha([(str(path), file_sha(path)) for path in candidates])
            identity["augmentation"] = augment
    return identity


def cache_path(cfg, identity):
    path = Path(cfg["cache"]["root"]).resolve() / identity["split"] / identity["mode"]
    if identity["validation_mode"] is not None:
        path = path / identity["validation_mode"] / f"seed_{identity['seed']:03d}"
    return path / canonical_sha(identity)


def _check_shape(shape, mode):
    if not isinstance(shape, list) or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Cache shape must contain positive integer dimensions")
    if mode in ("waveform", "codec_waveform"):
        valid = len(shape) == 2 and shape[0] == 1
    else:
        valid = len(shape) == 3 and shape[0] == 25 and shape[-1] == 1024
    if not valid:
        raise ValueError(f"Invalid {mode} cache shape: {shape}")


class CacheReader:
    def __init__(self, path, expected=None, *, verify_checksums=False, allow_smoke=False, allow_incomplete=False):
        self.path, self.verify_checksums = Path(path).resolve(), verify_checksums
        try:
            if allow_incomplete and (self.path / "progress.json").exists():
                from .codec_storage import load_codec_snapshot
                self.metadata, self.index = load_codec_snapshot(self.path)
            elif allow_incomplete:
                progress = json.loads((self.path / "resume.json").read_text(encoding="utf-8"))
                self.metadata, self.index = progress["metadata"], progress["index"]
            else:
                self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
                self.index = json.loads((self.path / "index.json").read_text(encoding="utf-8"))
                if self.metadata.get("format_version") == 2:
                    if self.metadata.get("identity_file") != "identity.json":
                        raise ValueError("Invalid codec identity reference")
                    self.metadata["identity"] = json.loads((self.path / "identity.json").read_text(encoding="utf-8"))
            self.identity = self.metadata["identity"]
            required = {"schema_version", "split", "mode", "validation_mode", "seed", "uttids", "smoke", "dtype",
                        "rng_schema_version", "dataset_fingerprint", "manifest_sha256", "source_audio_sha256", "preprocessing"}
            if not required.issubset(self.identity) or self.identity["split"] not in ("train", "dev", "test"):
                raise ValueError("Invalid cache identity schema")
            if self.identity["mode"] not in ("waveform", "xlsr_feature", "codec_waveform") or (
                    self.identity["mode"] == "waveform" and self.identity["split"] not in ("train", "dev")):
                raise ValueError("Invalid cache split/mode")
            if self.identity["mode"] == "codec_waveform" and self.identity["split"] != "train":
                raise ValueError("Codec caches are Train-only")
            if self.identity["mode"] == "codec_waveform" and not {
                    "builder_version", "codec_parameters", "source_cache", "implementation_sha256",
                    "codec_report", "source_commit"}.issubset(self.identity):
                raise ValueError("Incomplete codec cache identity")
            if not isinstance(self.identity["uttids"], list) or not self.identity["uttids"]:
                raise ValueError("Cache identity requires nonempty ordered uttids")
            if expected is not None and self.identity != expected:
                raise ValueError("Cache identity mismatch")
            if (type(self.identity["schema_version"]) is not int or self.identity["schema_version"] != 1 or
                    self.metadata["fingerprint"] != canonical_sha(self.identity)):
                raise ValueError("Cache schema/fingerprint mismatch")
            if type(self.identity["smoke"]) is not bool or type(self.identity["rng_schema_version"]) is not int:
                raise ValueError("Invalid cache smoke/RNG schema metadata")
            if self.identity["mode"] == "xlsr_feature" and (type(self.identity["seed"]) is not int or self.identity["seed"] < 0):
                raise ValueError("Invalid feature view seed")
            if self.metadata["complete"] is not True and not allow_incomplete:
                raise ValueError("Cache is incomplete")
            if self.identity["smoke"] and not allow_smoke:
                raise ValueError("Smoke/incomplete cache is forbidden for formal training/evaluation")
            if self.verify_checksums and self.metadata.get("index_sha256") is not None:
                index_sha = hashlib.sha256(json.dumps(self.index, ensure_ascii=False, sort_keys=True,
                                                     allow_nan=False).encode()).hexdigest() if allow_incomplete else file_sha(self.path / "index.json")
                if self.metadata["index_sha256"] != index_sha:
                    raise ValueError("Cache index checksum mismatch")
            if not isinstance(self.index, list):
                raise ValueError("Invalid cache index")
            ids = [row["uttid"] for row in self.index]
            expected_ids = self.identity["uttids"][:len(ids)] if allow_incomplete else self.identity["uttids"]
            if ids != expected_ids or len(ids) != len(set(ids)):
                raise ValueError("Missing, duplicate or reordered cache uttids")
            self.rows = {row["uttid"]: row for row in self.index}
            self._validate_shards()
            if self.metadata["output_shapes"] != [list(shape) for shape in sorted({tuple(row["shape"]) for row in self.index})]:
                raise ValueError("Cache output-shape metadata mismatch")
            content = canonical_sha([self.metadata["fingerprint"],
                                     self.metadata.get("index_fingerprint", self.metadata["index_sha256"]), self.metadata["shards"]])
            if content != self.metadata["content_fingerprint"]:
                raise ValueError("Cache content fingerprint mismatch")
        except (KeyError, TypeError, OSError, ValueError) as error:
            raise ValueError(f"Invalid cache at {self.path}: {error}; rebuild explicitly, no WAV fallback") from error
        self._maps = OrderedDict()

    def _validate_shards(self):
        dtype = np.dtype(self.identity["dtype"])
        if dtype.name not in ("float32", "float16") or (self.identity["mode"] in ("waveform", "codec_waveform") and dtype.name != "float32"):
            raise ValueError("Invalid cache dtype")
        if self.identity["mode"] == "xlsr_feature" and self.identity["hidden_order"] != HIDDEN_ORDER:
            raise ValueError("Incorrect 25-layer hidden state order")
        shards = self.metadata["shards"]
        names = [shard["file"] for shard in shards]
        if len(names) != len(set(names)) or any(not re.fullmatch(r"shard_\d{5,}\.bin", name) for name in names):
            raise ValueError("Invalid or duplicate shard filename")
        grouped = {name: [] for name in names}
        for row in self.index:
            _check_shape(row["shape"], self.identity["mode"])
            length = row["shape"][-1] if self.identity["mode"] in ("waveform", "codec_waveform") else row["shape"][1]
            if (row["dtype"] != dtype.name or row["shard"] not in grouped or
                    type(row["offset"]) is not int or row["offset"] < 0 or
                    row["length"] != length):
                raise ValueError("Invalid cache dtype, length or offset")
            if self.identity["mode"] == "codec_waveform":
                from .codec_records import validate_codec_record
                validate_codec_record(row.get("codec", {}), row["uttid"], length)
            grouped[row["shard"]].append(row)
        for shard in shards:
            offset = 0
            rows = grouped[shard["file"]]
            for row in rows:
                if row["offset"] != offset:
                    raise ValueError("Out-of-bounds or overlapping cache offset")
                offset += math.prod(row["shape"]) * dtype.itemsize
            path = self.path / shard["file"]
            if not rows or len(rows) != shard["records"] or offset != shard["bytes"] or path.stat().st_size != offset:
                raise ValueError("Cache shard size/record count mismatch")
            if self.verify_checksums and shard.get("sha256") is not None and file_sha(path) != shard["sha256"]:
                raise ValueError("Cache shard checksum mismatch")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = OrderedDict()
        return state

    def _array(self, uid):
        if uid not in self.rows:
            raise ValueError(f"Missing cache record: {uid}")
        row = self.rows[uid]
        name = row["shard"]
        if name not in self._maps:
            self._maps[name] = np.memmap(self.path / name, dtype="uint8", mode="r")
            if len(self._maps) > 4:
                self._maps.popitem(last=False)
        self._maps.move_to_end(name)
        return np.ndarray(row["shape"], dtype=row["dtype"], buffer=self._maps[name], offset=row["offset"])

    def get(self, uid):
        array = self._array(uid)
        if not np.isfinite(array).all():
            raise ValueError(f"Cache contains NaN/Inf: {uid}")
        return torch.from_numpy(array.copy())

    def verify(self, *, scan_waveforms=False):
        """Structural validation by default; full PCM scanning is explicitly opt-in."""
        self._validate_shards()
        if scan_waveforms:
            for uid in self.rows:
                if not np.isfinite(self._array(uid)).all():
                    raise ValueError(f"Cache contains NaN/Inf: {uid}")
        self._maps.clear()
        return self.stats()

    def close(self):
        """Release current file mappings; later get() can reopen them lazily."""
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def stats(self):
        return dict(path=str(self.path), samples=len(self.rows), dtype=self.identity["dtype"],
                    format_version=self.metadata.get("format_version", 1),
                    checksum_policy=self.metadata.get("checksum_policy", "historical"),
                    fingerprint=self.metadata["fingerprint"], content_fingerprint=self.metadata["content_fingerprint"],
                    bytes=sum(shard["bytes"] for shard in self.metadata["shards"]),
                    build_seconds=self.metadata.get("build_seconds", 0),
                    shapes=sorted({tuple(row["shape"]) for row in self.index}), smoke=self.identity["smoke"])


class _WriterLock:
    """OS releases the lock on process exit, so a crashed build can resume."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        if path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError(f"Another cache builder holds {path}") from error

    def close(self):
        if not self.handle.closed:
            self.handle.close()


class CacheWriter:
    """Commit complete shards to a resumable staging directory, then publish."""
    def __init__(self, target, identity, shard_size, *, resume=False, force=False, progress=None, verify_checksums=False):
        target = Path(target)
        self.verify_checksums = verify_checksums
        self.lock = _WriterLock(target.with_name(target.name + ".lock"))
        try:
            self._initialize(target, identity, shard_size, resume=resume, force=force, progress=progress)
        except BaseException:
            self.lock.close()
            raise

    def _initialize(self, target, identity, shard_size, *, resume, force, progress):
        self.target, self.identity, self.shard_size = Path(target), identity, shard_size
        self.stage = self.target.with_name(self.target.name + ".building")
        self.progress, self.started, self.force = progress, time.perf_counter(), force
        self.index, self.shards, self.pending, self.handle = [], [], [], None
        self.previous_seconds = 0
        if self.target.exists() and not force:
            raise FileExistsError(f"Cache already exists: {self.target}; check/reuse it or request --force")
        if self.stage.exists():
            if force:
                self.stage.rename(self.stage.with_name(self.stage.name + ".preserved-" + uuid.uuid4().hex[:8]))
            elif not resume:
                raise FileExistsError(f"Interrupted cache exists: {self.stage}; use --resume")
            else:
                saved = CacheReader(self.stage, identity, allow_smoke=True, allow_incomplete=True,
                                    verify_checksums=self.verify_checksums)
                self.index, self.shards = saved.index, saved.metadata["shards"]
                self.previous_seconds = saved.metadata.get("build_seconds", 0)
        self.stage.mkdir(parents=True, exist_ok=True)
        self._save(complete=False)

    def _save(self, complete):
        atomic_json(self.stage / "index.json", self.index)
        fingerprint = canonical_sha(self.identity)
        index_sha = file_sha(self.stage / "index.json") if self.verify_checksums else None
        metadata = dict(identity=self.identity, fingerprint=fingerprint, index_sha256=index_sha,
                        checksum_policy="sha256" if self.verify_checksums else "none",
                        shards=self.shards, complete=complete,
                        output_shapes=sorted({tuple(row["shape"]) for row in self.index}),
                        build_seconds=self.previous_seconds + time.perf_counter() - self.started,
                        content_fingerprint=canonical_sha([fingerprint, index_sha, self.shards]))
        atomic_json(self.stage / "metadata.json", metadata)
        # One atomic snapshot is authoritative for resume even if index/metadata
        # publication was interrupted between their two file replacements.
        atomic_json(self.stage / "resume.json", dict(metadata=metadata, index=self.index))

    def add(self, uid, tensor, *, codec_record=None):
        if (self.identity["mode"] == "codec_waveform") != (codec_record is not None):
            raise ValueError("codec_waveform writes require a codec_record; other modes forbid it")
        position = len(self.index) + len(self.pending)
        if position >= len(self.identity["uttids"]) or uid != self.identity["uttids"][position]:
            raise ValueError("Cache writes must follow the complete unique uttid order")
        array = tensor.detach().cpu().numpy().astype(self.identity["dtype"], copy=False)
        _check_shape(list(array.shape), self.identity["mode"])
        if not np.isfinite(array).all():
            raise ValueError(f"Cannot cache NaN/Inf: {uid}")
        if self.handle is None:
            name = f"shard_{len(self.shards):05d}.bin"
            self.temporary = self.stage / (name + "." + uuid.uuid4().hex[:8] + ".tmp")
            self.handle = self.temporary.open("xb")
        offset = self.handle.tell()
        self.handle.write(array.tobytes(order="C"))
        length = array.shape[-1] if self.identity["mode"] in ("waveform", "codec_waveform") else array.shape[1]
        self.pending.append(dict(uttid=uid, shard=f"shard_{len(self.shards):05d}.bin", offset=offset,
                                 length=length, dtype=array.dtype.name, shape=list(array.shape)))
        if codec_record is not None:
            self.pending[-1]["codec"] = codec_record
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        final = self.stage / self.pending[0]["shard"]
        self.temporary.replace(final)
        self.shards.append(dict(file=final.name, bytes=final.stat().st_size,
                                records=len(self.pending), sha256=file_sha(final) if self.verify_checksums else None))
        self.index.extend(self.pending)
        self.pending = []
        self._save(complete=False)
        if self.progress:
            elapsed = time.perf_counter() - self.started
            self.progress(len(self.index), len(self.identity["uttids"]), elapsed)

    def finish(self):
        self.flush()
        if len(self.index) != len(self.identity["uttids"]):
            raise ValueError("Cache remains incomplete")
        self._save(complete=True)
        reader = CacheReader(self.stage, self.identity, allow_smoke=True, verify_checksums=self.verify_checksums)
        reader.metadata["build_seconds"] = self.previous_seconds + time.perf_counter() - self.started
        atomic_json(self.stage / "metadata.json", reader.metadata)
        atomic_json(self.stage / "resume.json", dict(metadata=reader.metadata, index=self.index))
        if self.target.exists():
            if not self.force:
                raise FileExistsError(f"Cache published by another builder: {self.target}")
            self.target.rename(self.target.with_name(self.target.name + ".preserved-" + uuid.uuid4().hex[:8]))
        self.stage.rename(self.target)
        reader.path = self.target.resolve()
        return reader

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.lock.close()
