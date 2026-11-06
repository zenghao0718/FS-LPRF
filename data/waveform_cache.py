"""SpeechFake Train/Dev waveform lookup using metadata, never source WAV I/O."""
import json
from pathlib import Path
import re

from .cache import CacheReader, atomic_json, canonical_sha, make_cache_identity

_SHA = re.compile(r"[0-9a-f]{64}")


def waveform_binding(dataset, cfg, *, uttids=None, smoke=False):
    identity = make_cache_identity(dataset, cfg, "waveform", uttids=uttids, smoke=smoke,
                                   recorded_source_sha256="0" * 64)
    identity.pop("source_audio_sha256")
    return identity


def _reference_path(cfg, binding):
    return (Path(cfg["cache"]["root"]).resolve() / binding["split"] / "waveform" /
            "references" / (canonical_sha(binding) + ".json"))


def publish_waveform_reference(reader, cfg):
    """Publish only a completed, verified cache. Failed builds leave old refs intact."""
    identity = reader.identity
    if identity["mode"] != "waveform":
        return reader
    if identity["split"] not in ("train", "dev") or reader.metadata["complete"] is not True:
        raise ValueError("Only complete Train/Dev waveform caches can be referenced")
    binding = dict(identity)
    source = binding.pop("source_audio_sha256")
    if not isinstance(source, str) or not _SHA.fullmatch(source):
        raise ValueError("Waveform cache lacks a valid historical source SHA-256")
    fingerprint = reader.metadata["fingerprint"]
    if reader.path.name != fingerprint:
        raise ValueError("Cache directory must match its metadata fingerprint")
    atomic_json(_reference_path(cfg, binding),
                dict(schema_version=1, binding_sha256=canonical_sha(binding),
                     cache_fingerprint=fingerprint))
    return reader


def open_waveform_cache(dataset, cfg, *, uttids=None, allow_smoke=False, verify_checksums=None):
    """Bind manifests, labels, preprocessing and shards after the sources are gone.

    The source digest was established at build time. Runtime checks its recorded
    value and the cache fingerprint instead of attempting to hash removed WAVs.
    Old Train caches without a reference remain usable only if unambiguous.
    """
    binding = waveform_binding(dataset, cfg, uttids=uttids, smoke=allow_smoke)
    reference = _reference_path(cfg, binding)
    base = reference.parent.parent
    try:
        if reference.is_file():
            saved = json.loads(reference.read_text(encoding="utf-8"))
            fingerprint = saved.get("cache_fingerprint")
            if (saved.get("schema_version") != 1 or saved.get("binding_sha256") != canonical_sha(binding)
                    or not isinstance(fingerprint, str) or not _SHA.fullmatch(fingerprint)):
                raise ValueError("Invalid waveform cache reference")
            selected = base / fingerprint
        else:
            matches = []
            mismatches = set()
            for path in base.iterdir() if base.is_dir() else ():
                if not path.is_dir() or not _SHA.fullmatch(path.name):
                    continue
                try:
                    identity = json.loads((path / "metadata.json").read_text(encoding="utf-8"))["identity"]
                    candidate = {key: value for key, value in identity.items() if key != "source_audio_sha256"}
                    if candidate == binding:
                        matches.append(path)
                    else:
                        mismatches.update(key for key, value in binding.items() if candidate.get(key) != value)
                except (OSError, ValueError, KeyError, TypeError):
                    mismatches.add("unreadable_metadata")
            if len(matches) != 1:
                detail = "ambiguous source histories" if len(matches) > 1 else (
                    "missing cache or incompatible " + ", ".join(sorted(mismatches)))
                raise ValueError(detail)
            selected = matches[0]
        metadata = json.loads((selected / "metadata.json").read_text(encoding="utf-8"))
        source = metadata["identity"]["source_audio_sha256"]
        if not isinstance(source, str) or not _SHA.fullmatch(source):
            raise ValueError("Invalid recorded source SHA-256")
        expected = dict(binding, source_audio_sha256=source)
        reader = CacheReader(selected, expected,
                             verify_checksums=cfg["cache"]["verify_checksums"] if verify_checksums is None else verify_checksums,
                             allow_smoke=allow_smoke)
        if reader.path.name != reader.metadata["fingerprint"]:
            raise ValueError("Cache directory/fingerprint mismatch")
        return reader
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Required {dataset.split} Waveform Cache unavailable: {error}. "
            f"Build explicitly: python scripts/build_cache.py --split {dataset.split} --mode waveform. "
            "No WAV fallback.") from error
