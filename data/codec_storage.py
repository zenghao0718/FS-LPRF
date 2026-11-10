"""Codec-only format v2: immutable identity, per-shard commits, O(1) progress.

No PCM or full-index hashes. The progress file is the commit point; files beyond
its shard count are uncommitted and are safely replaced when that work is retried.
"""
import json
import os
from pathlib import Path
import time

from .cache import CacheReader, CacheWriter, atomic_json, canonical_sha

FORMAT_VERSION = 2


def _metadata(identity, fingerprint, index, shards, seconds, token, *, complete):
    return dict(format_version=FORMAT_VERSION, identity_file="identity.json", identity=identity,
                fingerprint=fingerprint, complete=complete, checksum_policy="none",
                index_sha256=None, index_fingerprint=token, shards=shards,
                output_shapes=[list(shape) for shape in sorted({tuple(row["shape"]) for row in index})], build_seconds=seconds,
                content_fingerprint=canonical_sha([fingerprint, token, shards]))


def load_codec_snapshot(path):
    """Recover exactly the prefix named by the atomic progress commit point."""
    path = Path(path)
    progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    identity = json.loads((path / "identity.json").read_text(encoding="utf-8"))
    count = progress["committed_shards"]
    if progress["format_version"] != FORMAT_VERSION or type(count) is not int or count < 0:
        raise ValueError("Invalid codec recovery progress")
    index, shards = [], []
    for number in range(count):
        commit = json.loads((path / f"commit_{number:05d}.json").read_text(encoding="utf-8"))
        if (commit["format_version"] != FORMAT_VERSION or
                commit["shard"]["file"] != f"shard_{number:05d}.bin" or
                any(row["shard"] != commit["shard"]["file"] for row in commit["index"])):
            raise ValueError("Invalid codec shard commit/order")
        index.extend(commit["index"])
        shards.append(commit["shard"])
    if type(progress["records"]) is not int or len(index) != progress["records"]:
        raise ValueError("Codec progress record count mismatch")
    if sum(shard["bytes"] for shard in shards) != progress["committed_bytes"]:
        raise ValueError("Codec progress byte count mismatch")
    return _metadata(identity, progress["fingerprint"], index, shards, progress["build_seconds"],
                     progress["index_fingerprint"], complete=False), index


class CodecCacheWriter(CacheWriter):
    """Reuse validated tensor writes and OS locks, without legacy index rewrites."""
    def _initialize(self, target, identity, shard_size, *, resume, force, progress):
        if (force or identity["mode"] != "codec_waveform" or identity.get("codec_storage_format") != FORMAT_VERSION
                or type(shard_size) is not int or shard_size <= 0):
            raise ValueError("Codec v2 requires codec_waveform, positive shard size and no overwrite")
        self.target, self.identity, self.shard_size = Path(target), identity, shard_size
        self.stage = self.target.with_name(self.target.name + ".building")
        self.progress, self.started, self.force = progress, time.perf_counter(), False
        self.index, self.shards, self.pending, self.handle = [], [], [], None
        self.previous_seconds = 0
        self.committed_bytes = 0
        self.fingerprint = canonical_sha(identity)
        self.token = self.fingerprint
        if self.target.exists():
            raise FileExistsError(f"Cache already exists: {self.target}; use a new output directory")
        if self.stage.exists():
            if not resume:
                raise FileExistsError(f"Interrupted cache exists: {self.stage}; use --resume")
            if not (self.stage / "progress.json").is_file():
                # A crash before the first progress publication has no committed data.
                identity_file = self.stage / "identity.json"
                if not identity_file.is_file():
                    if any(not (p.name.startswith("identity.json.") and p.name.endswith(".tmp"))
                           for p in self.stage.iterdir()):
                        raise ValueError("Legacy/incompatible codec staging format; preserve it and use a new directory")
                    atomic_json(identity_file, identity)
                if json.loads(identity_file.read_text(encoding="utf-8")) != identity:
                    raise ValueError("Codec cache identity mismatch")
                self._save_progress()
            else:
                with CacheReader(self.stage, allow_smoke=True, allow_incomplete=True) as saved:
                    if saved.identity != identity:
                        from .codec_compat import validate_compatibility
                        validate_compatibility(self.stage, saved.identity, identity)
                        # Keep the original identity and chain anchor. The atomic
                        # audit binds this exact target implementation separately.
                        self.identity = saved.identity
                        self.fingerprint = saved.metadata["fingerprint"]
                    self.index, self.shards = saved.index, saved.metadata["shards"]
                    self.previous_seconds = saved.metadata["build_seconds"]
                    self.token = saved.metadata["index_fingerprint"]
                    self.committed_bytes = sum(shard["bytes"] for shard in self.shards)
        else:
            self.stage.mkdir(parents=True)
            atomic_json(self.stage / "identity.json", identity)  # The full ID list is written once.
            self._save_progress()

    def _save_progress(self):
        atomic_json(self.stage / "progress.json", dict(format_version=FORMAT_VERSION,
                    fingerprint=self.fingerprint, committed_shards=len(self.shards), records=len(self.index),
                    committed_bytes=self.committed_bytes,
                    index_fingerprint=self.token,
                    build_seconds=self.previous_seconds + time.perf_counter() - self.started))

    def flush(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        number = len(self.shards)
        final = self.stage / f"shard_{number:05d}.bin"
        self.temporary.replace(final)
        shard = dict(file=final.name, bytes=final.stat().st_size, records=len(self.pending), sha256=None)
        commit = dict(format_version=FORMAT_VERSION, shard=shard, index=self.pending)
        atomic_json(self.stage / f"commit_{number:05d}.json", commit)
        # A small per-shard in-memory digest, never a scan of PCM or the full index.
        self.token = canonical_sha([self.token, commit])
        self.shards.append(shard)
        self.committed_bytes += shard["bytes"]
        self.index.extend(self.pending)
        self.pending = []
        self._save_progress()  # Publication of this scalar snapshot commits the shard.
        if self.progress:
            self.progress(len(self.index), len(self.identity["uttids"]), time.perf_counter()-self.started)

    def finish(self):
        self.flush()
        if len(self.index) != len(self.identity["uttids"]):
            raise ValueError("Cache remains incomplete")
        atomic_json(self.stage / "index.json", self.index)  # One final materialization.
        metadata = _metadata(self.identity, self.fingerprint, self.index, self.shards,
                             self.previous_seconds + time.perf_counter()-self.started, self.token, complete=True)
        metadata.pop("identity")  # Immutable identity remains in its single authoritative file.
        atomic_json(self.stage / "metadata.json", metadata)
        reader = CacheReader(self.stage, self.identity, allow_smoke=True)  # Structural checks only.
        if self.target.exists():
            raise FileExistsError(f"Cache published by another builder: {self.target}")
        self.stage.rename(self.target)
        reader.path = self.target.resolve()
        return reader
