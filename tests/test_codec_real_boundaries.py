"""Real bitstreams only. Missing backends/speech fixture are explicit skips."""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.codec_augmentation import CodecAugmentor, alignment_diagnostics, probe_codecs, _resample


@pytest.mark.parametrize("delay", [0, 73, 321, 1105])
def test_speech_band_calibration_recovers_delay_without_impulse_threshold(delay):
    from data.codec_augmentation import _estimate_lead, _lead_signal
    source = _lead_signal(16000)
    decoded = np.pad(source * .6, (delay, 1600))
    assert _estimate_lead(decoded, source) == delay


def available_augmentor(family):
    inventory = probe_codecs()
    if not inventory[family]["available"]:
        pytest.skip(inventory[family]["reason"])
    # An installed-but-broken backend/weight/preset is a failure, never a skip.
    return CodecAugmentor(probability=1, codecs=[family])


@pytest.mark.parametrize("family", ["mp3", "opus", "amr", "encodec"])
@pytest.mark.parametrize("kind,length", [("voiced", 3), ("voiced", 160), ("voiced", 2049),
                                         ("voiced", 16003), ("voiced", 48001), ("silence", 16003)])
def test_real_all_presets_short_silent_variable_length_bitstreams(family, kind, length):
    augmentor = available_augmentor(family)
    t = np.arange(length, dtype=np.float32) / 16000
    audio = (0.15 * np.sin(2*np.pi*(130*t+23*t*t)) + .06*np.sin(2*np.pi*680*t)).astype(np.float32)
    if kind == "silence": audio[:] = 0
    for spec in augmentor.choices[family]:
        restored, info = augmentor.apply_array(audio, spec)
        assert info["encoded_bytes"] > 0 and info["payload"]
        assert restored.shape == audio.shape and restored.dtype == np.float32
        assert np.isfinite(restored).all() and np.max(np.abs(restored)) <= 1
        assert info["pad_ratio"] <= .01 and info["clipping_fraction"] <= .01
        assert info["decoded_samples"] >= length + info["lead_samples"]
        assert info["start_rms"] >= 0 and info["end_rms"] >= 0
        if kind == "voiced" and length >= 16000:
            diagnostics = alignment_diagnostics(audio, restored)
            assert any(row["conclusive"] for row in diagnostics), (spec, diagnostics)
            assert all(abs(row.get("residual_lag_samples", 0)) <= 480 for row in diagnostics if row["conclusive"])
            # Ending content must survive, rather than a repeated final-sample pad.
            assert np.std(restored[-320:]) > 1e-4


@pytest.mark.parametrize("family", ["mp3", "opus", "amr", "encodec"])
def test_real_speech_fixture_at_every_bitrate(family):
    location = os.environ.get("FS_LPRF_CODEC_SPEECH_FIXTURE")
    if not location or not Path(location).is_file():
        pytest.skip("Set FS_LPRF_CODEC_SPEECH_FIXTURE to a local real speech WAV; synthetic speech is not a real-speech pass")
    augmentor = available_augmentor(family)
    audio, sr = sf.read(location, dtype="float32", always_2d=True)
    audio = _resample(audio[:, 0], sr, 16000)
    if len(audio) < 16000 or np.max(np.abs(audio)) < 1e-3:
        pytest.fail("Real speech fixture must have at least one second of audible speech")
    for length in (160, 2049, min(len(audio), 48001)):
        for spec in augmentor.choices[family]:
            restored, info = augmentor.apply_array(audio[:length], spec)
            assert info["pad_samples"] == 0
            alignment_diagnostics(audio[:length], restored)


def test_residual_delay_check_rejects_a_bad_alignment():
    rng = np.random.default_rng(42)
    audio = rng.normal(0, .05, 32000).astype(np.float32)
    delayed = np.pad(audio, (1600, 0))[:len(audio)]
    with pytest.raises(RuntimeError, match="residual codec delay"):
        alignment_diagnostics(audio, delayed)
