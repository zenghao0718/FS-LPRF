"""One pinned transition: 5f53805 successful v2 prefix -> clipping fallback v1.

The original identity, progress, commit records and PCM stay untouched. A single
atomic authorization binds the exact source/target code and immutable prefix.
"""
import copy
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path

from .cache import CacheReader, _WriterLock, atomic_json, canonical_sha
from .codec_records import FALLBACK_POLICY

AUDIT_NAME = "clipping_compatibility.json"
RELEASE_PATH = Path(__file__).resolve().parents[1] / "configs" / "codec_clipping_recovery_v1.json"
TRANSITION = "5f53805_to_severe_clipping_original_v1"
_CHANGED_KEYS = {"implementation_sha256", "source_commit", "clipping_fallback_policy"}


def _reject(message):
    return ValueError(f"Codec builder/implementation identity mismatch: {message}")


def _release():
    from .codec_cache import implementation_hashes
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if release["transition"] != TRANSITION or release["target_implementation_sha256"] != implementation_hashes():
        raise _reject("this checkout is not the explicitly approved clipping recovery release")
    return release


def _common(identity):
    return {key: value for key, value in identity.items() if key not in _CHANGED_KEYS}


def _source_identity(identity, release):
    if (identity.get("source_commit") != release["source_commit"] or
            identity.get("implementation_sha256") != release["source_implementation_sha256"] or
            identity.get("codec_storage_format") != 2 or identity.get("builder_version") != 1 or
            identity.get("mode") != "codec_waveform" or identity.get("split") != "train" or
            "clipping_fallback_policy" in identity):
        raise _reject("only the exact 5f53805 v2 implementation is eligible")


def _prefix(path, identity, shard_count):
    """Verify only small commit JSON and file sizes; never open PCM/index.json."""
    token, records, size = canonical_sha(identity), 0, 0
    for number in range(shard_count):
        commit = json.loads((Path(path) / f"commit_{number:05d}.json").read_text(encoding="utf-8"))
        shard, rows = commit["shard"], commit["index"]
        if (commit["format_version"] != 2 or shard["file"] != f"shard_{number:05d}.bin" or
                len(rows) != shard["records"] or (Path(path)/shard["file"]).stat().st_size != shard["bytes"]):
            raise ValueError("Invalid compatibility commit prefix")
        records += len(rows)
        size += shard["bytes"]
        token = canonical_sha([token, commit])
    return dict(committed_shards=shard_count, records=records, committed_bytes=size, index_fingerprint=token)


def validate_compatibility(path, identity, requested=None):
    """Validate the sole audited exception; no general identity bypass exists."""
    audit_file = Path(path) / AUDIT_NAME
    if not audit_file.is_file():
        raise _reject("no controlled clipping recovery authorization; use scripts/recover_codec_cache.py")
    release = _release()
    _source_identity(identity, release)
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if (audit.get("audit_sha256") != canonical_sha(payload) or audit.get("transition") != TRANSITION or
            audit.get("release_sha256") != canonical_sha(release) or
            audit.get("original_identity_fingerprint") != canonical_sha(identity) or
            audit.get("original_implementation_sha256") != release["source_implementation_sha256"] or
            audit.get("target_implementation_sha256") != release["target_implementation_sha256"] or
            audit.get("original_source_commit") != release["source_commit"] or
            audit.get("fallback_policy") != FALLBACK_POLICY or
            audit.get("common_identity_sha256") != canonical_sha(_common(identity))):
        raise _reject("invalid or changed clipping compatibility authorization")
    if requested is not None:
        if (requested.get("implementation_sha256") != release["target_implementation_sha256"] or
                requested.get("clipping_fallback_policy") != FALLBACK_POLICY or
                _common(requested) != _common(identity) or
                audit.get("requested_identity_fingerprint") != canonical_sha(requested)):
            raise _reject("resume parameters, environment, source, order, device or target revision changed")
    boundary = audit["boundary"]
    if (type(boundary.get("committed_shards")) is not int or boundary["committed_shards"] < 0 or
            _prefix(path, identity, boundary["committed_shards"]) != boundary):
        raise _reject("original committed prefix no longer matches the authorized boundary")
    progress = json.loads((Path(path)/"progress.json").read_text(encoding="utf-8"))
    count = progress["committed_shards"]
    if type(count) is not int or count < boundary["committed_shards"]:
        raise _reject("progress precedes the authorized prefix")
    full_prefix = _prefix(path, identity, count)
    if any(progress.get(key) != value for key, value in full_prefix.items()):
        raise _reject("current incremental chain no longer matches progress")
    return audit


def recovery_requested_identity(source, cfg, old_identity, device, threads):
    """Use recorded backend provenance; recovery never initializes/executes codecs."""
    from .codec_cache import make_codec_identity
    execution = old_identity["codec_report"].get("execution", {})
    if old_identity.get("encodec_device") != device or execution != dict(encodec_device=device, worker_threads=threads):
        raise _reject("requested recovery device/threads differ from the interrupted build")
    class RecordedReport:
        def report(self):
            return copy.deepcopy(old_identity["codec_report"])
    return make_codec_identity(source, cfg, RecordedReport(), uttids=old_identity["uttids"],
                               smoke=old_identity["smoke"], encodec_device=device)


def authorize_clipping_recovery(target, source, cfg, *, expected_shards, expected_records, expected_total,
                                device="cuda:0", threads=1, dry_run=False):
    """Create only an atomic audit under the same lock used by the v2 writer."""
    from .codec_cache import _validate_codec_contents, validate_location
    target = validate_location(target, source, cfg)
    stage = target.with_name(target.name + ".building")
    if target.exists() or not stage.is_dir():
        raise ValueError("Recovery requires an interrupted .building directory and no completed target")
    if any(type(value) is not int or value < 0 for value in (expected_shards, expected_records, expected_total)):
        raise ValueError("Expected recovery counts must be nonnegative integers")
    with closing(_WriterLock(target.with_name(target.name + ".lock"))):
        with CacheReader(stage, allow_smoke=True, allow_incomplete=True, verify_checksums=False) as reader:
            release = _release()
            _source_identity(reader.identity, release)
            requested = recovery_requested_identity(source, cfg, reader.identity, device, threads)
            if _common(reader.identity) != _common(requested):
                raise _reject("source cache, full IDs, seed, codecs or other invariant fields changed")
            selected = set(reader.identity["uttids"])
            if reader.identity["uttids"] != [uid for uid in source.identity["uttids"] if uid in selected]:
                raise _reject("full ordered ID list differs from the source Manifest")
            existing = validate_compatibility(stage, reader.identity, requested) if (stage/AUDIT_NAME).is_file() else None
            # A completed-cache check requires all IDs. Here require the precise
            # original Manifest prefix, then validate that subset against source.
            partial = copy.copy(reader)
            partial.identity = dict(reader.identity, uttids=[row["uttid"] for row in reader.index], smoke=True)
            partial.metadata = dict(reader.metadata)
            if existing:
                partial.metadata["compatibility_audit_sha256"] = existing["audit_sha256"]
            if partial.index:
                _validate_codec_contents(partial, source, cfg)
            if len(reader.identity["uttids"]) != expected_total:
                raise _reject("expected full sample count differs")
            progress = json.loads((stage / "progress.json").read_text(encoding="utf-8"))
            full_prefix = _prefix(stage, reader.identity, progress["committed_shards"])
            if any(progress.get(key) != value for key, value in full_prefix.items()):
                raise _reject("incremental index chain or progress counts are damaged")
            if (stage/AUDIT_NAME).is_file():
                audit = existing
                boundary = audit["boundary"]
                if boundary["committed_shards"] != expected_shards or boundary["records"] != expected_records:
                    raise _reject("existing authorization has a different migration boundary")
                if progress["committed_shards"] < expected_shards or progress["records"] < expected_records:
                    raise _reject("build progress precedes the authorized prefix")
                return dict(status="already_authorized", audit=audit)
            if progress["committed_shards"] != expected_shards or len(reader.index) != expected_records:
                raise _reject("committed shard/record counts differ from the operator's expected boundary")
            if any(row["codec"]["status"] != "ok" for row in reader.index):
                raise _reject("the pre-fix committed prefix must contain only successful codec records")
            payload = dict(schema_version=1, transition=TRANSITION, release_sha256=canonical_sha(release),
                           created_at_utc=datetime.now(timezone.utc).isoformat(),
                           original_source_commit=reader.identity["source_commit"],
                           target_source_commit=requested["source_commit"],
                           original_implementation_sha256=reader.identity["implementation_sha256"],
                           target_implementation_sha256=requested["implementation_sha256"],
                           original_identity_fingerprint=canonical_sha(reader.identity),
                           requested_identity_fingerprint=canonical_sha(requested),
                           common_identity_sha256=canonical_sha(_common(reader.identity)),
                           fallback_policy=FALLBACK_POLICY, boundary=full_prefix)
            audit = dict(payload, audit_sha256=canonical_sha(payload))
            if not dry_run:
                atomic_json(stage/AUDIT_NAME, audit)  # No original file is changed.
            return dict(status="dry_run" if dry_run else "authorized", audit=audit)
