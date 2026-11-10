"""Shared offline build policy and record schema. No fallback for other errors."""
import logging
import math
import time

import numpy as np

from .codec_augmentation import CodecClippingError, KNOWN_CODECS, preset_parameters
from .randomness import sample_rng, sample_seed

FALLBACK_POLICY = "severe_clipping_original_v1"
FALLBACK_REASON = "severe_codec_clipping"


def encode_cached_sample(reader, augmentor, uid, spec, seed):
    started = time.perf_counter()
    with sample_rng(sample_seed("codec_encode", seed, uid)):
        original = reader.get(uid).numpy().reshape(-1)
        if original.dtype != np.float32 or not original.size or not np.isfinite(original).all():
            raise RuntimeError(f"Invalid source Waveform Cache PCM: {uid}")
        try:
            audio, info = augmentor.apply_array(original.copy(), spec)
        except CodecClippingError as error:
            if error.spec != tuple(spec):
                raise RuntimeError("Codec clipping error spec differs from requested assignment") from error
            # Retain exact source samples, without fitting, gain changes or another read.
            audio = original.copy()
            info = dict(uttid=uid, family=spec[0], variant=spec[1], param=spec[2],
                        parameters=preset_parameters(spec), status="fallback_original",
                        input_samples=int(original.size), output_samples=int(original.size),
                        fallback_reason=FALLBACK_REASON, exception_type="CodecClippingError",
                        exception_message=str(error), decoded_peak=error.peak,
                        clipping_fraction=error.clipping_fraction)
            logging.getLogger(__name__).warning(
                "Codec fallback_original uttid=%s spec=%s peak=%.6g fraction=%.6g",
                uid, spec, error.peak, error.clipping_fraction)
    record = {key: value for key, value in info.items() if key != "payload"}
    record.update(uttid=uid, seconds=time.perf_counter()-started)
    return audio, record


def validate_codec_record(record, uid, length):
    if (record.get("status") not in ("ok", "fallback_original") or record.get("uttid") != uid or
            record.get("input_samples") != length or record.get("output_samples") != length):
        raise ValueError("Invalid codec record status/length/ID")
    if record["status"] == "fallback_original":
        peak, fraction = record.get("decoded_peak"), record.get("clipping_fraction")
        if (record.get("fallback_reason") != FALLBACK_REASON or record.get("exception_type") != "CodecClippingError" or
                not isinstance(record.get("exception_message"), str) or not record["exception_message"] or
                record.get("family") not in KNOWN_CODECS or not isinstance(record.get("variant"), str) or not record["variant"] or
                type(record.get("param")) not in (int, float) or not math.isfinite(record["param"]) or
                type(peak) not in (int, float) or type(fraction) not in (int, float) or
                not math.isfinite(peak) or not math.isfinite(fraction) or peak < 0 or not 0 <= fraction <= 1 or
                not (peak > 2 or fraction > .01)):
            raise ValueError("Invalid codec fallback reason/assignment/clipping evidence")


def summarize_codec_results(index):
    fallbacks = []
    by_family = {name: 0 for name in KNOWN_CODECS}
    successes = 0
    for row in index:
        record = row["codec"]
        validate_codec_record(record, row["uttid"], row["length"])
        if record["status"] == "ok":
            successes += 1
        else:
            by_family[record["family"]] += 1
            fallbacks.append({key: record[key] for key in (
                "uttid", "family", "variant", "param", "fallback_reason", "exception_message",
                "decoded_peak", "clipping_fraction")})
    count = len(index)
    return dict(total_samples=count, codec_success_count=successes, fallback_original_count=len(fallbacks),
                fallback_fraction=len(fallbacks)/count if count else 0., fallback_by_family=by_family,
                fallback_details=fallbacks)
