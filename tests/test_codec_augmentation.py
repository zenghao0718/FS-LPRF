"""Codec augmentation: real round-trips, train-only gating, and baseline preservation."""
import copy
import shutil
import subprocess

import numpy as np
import pytest
import torch

from data.audio import GeneralFAD
from data.codec_augmentation import (CodecAugmentor, _amr_nb_roundtrip, _fit_length, _mp3_magic,
                                     _opus_magic, _resample, codec_settings, probe_codecs)
from data.randomness import sample_rng
from data.speechfake_dataset import SpeechFakeDataset
from engine.pipeline_benchmark import configure_benchmark_mode
from engine.validation_schedule import make_run_identity
from utils.config import load_config


def _tone(samples=16000, sample_rate=16000):
    time = np.arange(samples, dtype=np.float32) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 220 * time) + 0.1 * np.sin(2 * np.pi * 660 * time)).astype(np.float32)


def test_codec_augmentation_is_off_by_default_and_preserves_baseline_identity(config):
    assert config["data"]["codec_augment"]["enabled"] is False
    assert config["data"]["codec_augment"]["probability"] == 0.5
    bare = copy.deepcopy(config)
    bare["data"].pop("codec_augment")
    bindings = dict(rng_schema_version=1, caches=[])
    assert make_run_identity(config, "full", "light", "train", bindings) == make_run_identity(bare, "full", "light", "train", bindings)
    enabled = copy.deepcopy(config)
    enabled["data"]["codec_augment"]["enabled"] = True
    assert make_run_identity(enabled, "full", "light", "train", bindings) != make_run_identity(bare, "full", "light", "train", bindings)


def test_invalid_codec_settings_are_rejected():
    with pytest.raises(ValueError, match="probability"):
        load_config("configs/speechfake.yaml", ["data.codec_augment.probability=1.5"])
    with pytest.raises(ValueError, match="codecs"):
        load_config("configs/speechfake.yaml", ["data.codec_augment.codecs=[aac]"])
    assert load_config("configs/speechfake.yaml", ["data.codec_augment.enabled=true"])["data"]["codec_augment"]["enabled"] is True


def test_dev_and_test_ignore_an_enabled_codec_block(dataset, config, tmp_path):
    options = dict(config["data"], data_path=str(tmp_path),
                   codec_augment=dict(enabled=True, probability=1.0, codecs=["mp3"], sample_rate=16000))
    dev = SpeechFakeDataset(options, "dev")
    test = SpeechFakeDataset(options, "test")
    assert dev.do_codec_augment is False and dev.codec_augmentor is None
    assert test.do_codec_augment is False and not test.do_augment
    with sample_rng(5):
        original = GeneralFAD.get_audio(dev, dev.uttids[0])
    with sample_rng(5):
        actual = dev.get_audio(dev.uttids[0])
    assert torch.equal(original, actual)


def test_disabled_train_get_audio_matches_the_original_method(dataset, config, tmp_path, rir_root):
    options = dict(config["data"], data_path=str(tmp_path), rir_root=str(rir_root))
    training = SpeechFakeDataset(options, "train", train_mode=True)
    assert training.do_codec_augment is False
    with sample_rng(11):
        original = GeneralFAD.get_audio(training, training.uttids[0])
    with sample_rng(11):
        actual = training.get_audio(training.uttids[0])
    assert torch.equal(original, actual)
    real = training[0]
    fake_index = training.class_indices[training.fake_classes[0]][0]
    fake = training[fake_index]
    assert real["label"] == 0 and fake["label"] == 1
    assert real["feats"].shape == fake["feats"].shape == (1, 64000)


def test_codec_is_applied_before_crop_and_rawboost_without_changing_labels(dataset, config, tmp_path, rir_root):
    options = dict(config["data"], data_path=str(tmp_path), rir_root=str(rir_root))
    training = SpeechFakeDataset(options, "train", train_mode=True)
    events = []

    def codec(audio):
        events.append("codec")
        return audio + 0.01

    def crop(audio):
        events.append("crop")
        assert torch.allclose(audio[:, :1], training.load_preprocessed_waveform(training.uttids[0])[:, :1] + 0.01)
        return GeneralFAD.sample_segment(training, audio)

    def augment(audio):
        events.append("augment")
        return audio

    training.codec_augmentor = codec
    training.do_codec_augment = True
    training.sample_segment = crop
    training.augmentor = augment
    real = training[0]
    fake = training[training.class_indices[training.fake_classes[0]][0]]
    assert events == ["codec", "crop", "augment", "codec", "crop", "augment"]
    assert real["label"] == 0 and fake["label"] == 1
    assert real["feats"].shape == (1, 64000)


def test_probability_keeps_unencoded_samples_and_is_reproducible():
    seen = []

    def backend(spec, audio):
        seen.append(spec[0])
        return np.ascontiguousarray(audio, dtype=np.float32) * np.float32(0.5), {"payload": b"codec-bytes"}

    augmentor = CodecAugmentor(probability=0.5, codecs=["mp3", "opus"], backend=backend)
    wave = torch.from_numpy(_tone())[None]
    encoded = 0
    for seed in range(40):
        with sample_rng(seed):
            output = augmentor(wave.clone())
        with sample_rng(seed):
            again = augmentor(wave.clone())
        assert torch.equal(output, again)
        encoded += int(not torch.equal(output, wave))
    assert seen
    assert 0 < encoded < 40
    assert torch.equal(wave, torch.from_numpy(_tone())[None])


def test_cached_waveform_tensor_is_not_modified(dataset, config, tmp_path, rir_root):
    options = dict(config["data"], data_path=str(tmp_path), rir_root=str(rir_root))
    training = SpeechFakeDataset(options, "train", train_mode=True)
    stored = training.load_preprocessed_waveform(training.uttids[0]).clone()

    class Reader:
        def __init__(self):
            self.identity = {"split": "train"}
            self.wave = stored.clone()

        def get(self, uttid):
            return self.wave.clone()

    reader = Reader()
    training.use_cache(reader, "waveform")
    training.codec_augmentor = lambda audio: audio * 0.2
    training.do_codec_augment = True
    result = training.get_audio(training.uttids[0])
    assert result.shape == (1, 64000)
    assert torch.equal(reader.wave, stored)
    assert not torch.equal(result, stored[:, :64000])


def test_feature_cache_cannot_be_combined_with_codec_augmentation(dataset):
    dataset.do_codec_augment = True
    reader = type("Reader", (), {"identity": {"split": dataset.split}})()
    with pytest.raises(ValueError, match="Codec Augmentation"):
        dataset.use_cache(reader, "xlsr_feature")


def test_length_repair_handles_delay_padding_and_gain():
    source = _tone(1000)
    longer = np.concatenate([np.zeros(12, np.float32), source, np.zeros(20, np.float32)]) * 2
    fitted = _fit_length(longer, source.size, 12, float(np.max(np.abs(source))))
    assert fitted.shape == source.shape
    assert fitted.dtype == np.float32 and np.isfinite(fitted).all() and np.max(np.abs(fitted)) <= 1
    shorter = source[:700]
    padded = _fit_length(shorter, source.size, 0, 0.3)
    assert padded.shape == source.shape
    assert np.all(padded[700:] == padded[699])


def test_benchmark_modes_separate_baseline_from_codec(config):
    baseline = configure_benchmark_mode(copy.deepcopy(config), "waveform")
    codec = configure_benchmark_mode(copy.deepcopy(config), "waveform_codec")
    assert baseline["data"]["train_cache_mode"] == "waveform" and baseline["data"]["codec_augment"]["enabled"] is False
    assert codec["data"]["train_cache_mode"] == "waveform"
    assert codec["data"]["codec_augment"]["enabled"] is True
    assert codec["data"]["codec_augment"]["probability"] == 0.5


def test_missing_codec_settings_stay_disabled():
    assert codec_settings({})["enabled"] is False
    assert codec_settings({"codec_augment": {"enabled": False}})["codecs"] == ["mp3", "opus", "amr", "encodec"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not probe_codecs()["mp3"]["available"], reason="ffmpeg libmp3lame is unavailable")
def test_mp3_roundtrip_is_a_real_bitstream_and_restores_pcm():
    augmentor = CodecAugmentor(probability=1, codecs=["mp3"])
    source = _tone(32000)
    restored, info = augmentor.apply_array(source, ("mp3", "mp3", 64))
    assert _mp3_magic(info["payload"])
    assert restored.shape == source.shape and np.isfinite(restored).all()
    assert np.max(np.abs(restored - source)) > 1e-4
    assert np.corrcoef(source, restored)[0, 1] > 0.9


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not probe_codecs()["opus"]["available"], reason="ffmpeg libopus is unavailable")
def test_opus_roundtrip_is_ogg_opus():
    augmentor = CodecAugmentor(probability=1, codecs=["opus"])
    source = _tone(32000)
    restored, info = augmentor.apply_array(source, ("opus", "opus", 24))
    assert _opus_magic(info["payload"])
    assert restored.shape == source.shape and np.isfinite(restored).all()
    assert np.max(np.abs(restored - source)) > 1e-4


@pytest.mark.skipif(not probe_codecs()["amr"]["available"], reason="OpenCORE/VisualOn AMR libraries are unavailable")
def test_amr_roundtrip_is_not_a_resample_and_ffmpeg_can_decode_it():
    source = _tone(16000)
    decoded, payload = _amr_nb_roundtrip(source, 7, 16000)
    assert payload.startswith(b"#!AMR\n") and len(payload) > len(b"#!AMR\n") + 16
    resampled = _resample(_resample(source, 16000, 8000), 8000, 16000)
    limit = min(decoded.size, resampled.size, source.size)
    assert np.max(np.abs(decoded[:limit] - resampled[:limit])) > 1e-3
    if shutil.which("ffmpeg"):
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
        assert len(completed.stdout) > 1000
    augmentor = CodecAugmentor(probability=1, codecs=["amr"])
    for variant, param in (("amr_nb", 7), ("amr_wb", 2)):
        restored, info = augmentor.apply_array(source, ("amr", variant, param))
        assert restored.shape == source.shape and np.isfinite(restored).all()
        assert info["payload"].startswith(b"#!AMR")


@pytest.mark.skipif(not probe_codecs()["encodec"]["available"], reason="encodec package is unavailable")
def test_encodec_roundtrip_uses_discrete_codes():
    augmentor = CodecAugmentor(probability=1, codecs=["encodec"])
    source = _tone(16000)
    restored, info = augmentor.apply_array(source, ("encodec", "encodec", 6.0))
    assert info["payload"].startswith(b"EC")
    assert int.from_bytes(info["payload"][2:10], "little") > 0
    assert restored.shape == source.shape and np.isfinite(restored).all()
    assert np.max(np.abs(restored - source)) > 1e-3
    with sample_rng(19):
        first = augmentor(torch.from_numpy(source)[None])
    with sample_rng(19):
        second = augmentor(torch.from_numpy(source)[None])
    assert torch.equal(first, second)
    assert first.shape == (1, source.size)


def test_train_smoke_keeps_shape_label_and_finite_values(dataset, config, tmp_path, rir_root):
    if not probe_codecs()["mp3"]["available"]:
        pytest.skip("ffmpeg libmp3lame is unavailable")
    options = dict(config["data"], data_path=str(tmp_path), rir_root=str(rir_root),
                   codec_augment=dict(enabled=True, probability=1.0, codecs=["mp3"], sample_rate=16000))
    training = SpeechFakeDataset(options, "train", train_mode=True)
    assert training.do_codec_augment and "mp3" in training.codec_augmentor.families
    real_index = training.class_indices["real"][0]
    fake_index = training.class_indices[training.fake_classes[0]][0]
    for index, label in ((real_index, 0), (fake_index, 1)):
        item = training[index]
        assert item["label"] == label
        assert item["feats"].shape == (1, 64000)
        assert item["feats"].dtype == torch.float32
        assert torch.isfinite(item["feats"]).all()
    assert training.codec_augmentor is training.codec_augmentor
