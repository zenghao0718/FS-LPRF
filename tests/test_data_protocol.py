import json
import random
from unittest.mock import Mock
import numpy as np
import pytest
import soundfile as sf
import torch
from fs_lprf.data.speechfake_dataset import SpeechFakeDataset, ConfiguredRawBoost, validate_manifest
from fs_lprf.data.episode_sampler import EpisodeSampler
from fs_lprf.data.evaluation_manifest import load_or_create_manifest
from fs_lprf.data.loaders import audio_loader
from fs_lprf.utils.seed import seed_all, capture_rng, preserved_rng
from fs_lprf.methods.prototype_bank import select_bank_samples, REAL_QUOTAS


def test_episode_sampling_is_independent_balanced_and_disjoint(dataset):
    seed_all(42)
    sampler = EpisodeSampler(dataset.class_indices)
    episodes = sampler.sample_step()
    assert len(episodes) == 8
    assert len({episode.classes for episode in episodes}) > 1
    for episode in episodes:
        assert episode.classes[0] == "real" and len(set(episode.classes)) == 3
        assert len(episode.indices) == 30 and len(set(episode.indices)) == 30
        assert not set(episode.support_indices) & set(episode.query_indices)
        assert episode.support_labels == tuple([0] * 5 + [1] * 5 + [2] * 5)
        assert episode.query_labels == episode.support_labels


def test_query_count_configuration(dataset):
    with pytest.raises(ValueError, match="balanced"):
        EpisodeSampler(dataset.class_indices, total_query_per_episode=14)
    episode = EpisodeSampler(dataset.class_indices, total_query_per_episode=6).sample_episode()
    assert len(episode.query_indices) == 6 and len(episode.support_indices) == 15


def test_five_seed_manifest_created_once_shared_query_and_support(dataset, tmp_path):
    path = tmp_path / "dev_episode_manifest.json"
    seeds = [17, 42, 71, 100, 123]
    manifest = load_or_create_manifest(dataset, path, 10, seeds)
    original = path.read_bytes()
    for name in manifest["classes"]:
        support_union = {uid for seed in seeds for uid in manifest["supports"][str(seed)][name]}
        query = set(manifest["queries"][name])
        pool = {dataset.uttids[i] for i in dataset.class_indices[name]}
        assert not query & support_union and query | support_union == pool
    seed_all(999)
    assert load_or_create_manifest(dataset, path, 10, seeds) == manifest
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_manifest(dataset, path, 5, seeds)


def test_tampered_or_changed_manifest_rejected(dataset, tmp_path):
    path = tmp_path / "dev_episode_manifest.json"
    manifest = load_or_create_manifest(dataset, path, 10, [17, 42, 71, 100, 123])
    manifest["queries"]["real"].append(manifest["supports"]["17"]["real"][0])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_or_create_manifest(dataset, path, 10, [17, 42, 71, 100, 123])


@pytest.mark.parametrize("workers", [0, 2])
def test_official_random_crop_repeats_per_seed_across_checkpoints(dataset, workers):
    def read(seed):
        seed_all(seed)
        return next(iter(audio_loader(dataset, [0, 1], 2, seed, workers)))["feats"]
    first = read(17)
    # Simulate arbitrary intervening training and a subsequent checkpoint validation.
    seed_all(999)
    torch.rand(40)
    np.random.rand(40)
    assert torch.equal(first, read(17))
    assert not torch.equal(first, read(42))
    assert first.shape == (2, 1, 64000)


def test_evaluation_rng_context_restores_python_numpy_cpu_and_cuda():
    seed_all(42)
    expected = capture_rng()
    with preserved_rng():
        seed_all(17)
        random.random()
        np.random.rand()
        torch.rand(1)
        if torch.cuda.is_available():
            torch.rand(1, device="cuda")
    actual = capture_rng()
    assert actual["python"] == expected["python"]
    assert torch.equal(actual["numpy"][1], expected["numpy"][1])
    assert torch.equal(actual["torch"], expected["torch"])
    assert all(torch.equal(a, b) for a, b in zip(actual["cuda"], expected["cuda"]))


@pytest.mark.parametrize("rate,length,channels", [(16000, 16000, 1), (8000, 8000, 1), (16000, 80000, 2)])
def test_official_audio_resample_mono_repeat_and_length(dataset, tmp_path, rate, length, channels):
    t = np.arange(length, dtype=np.float32) / rate
    first = 0.1 * np.sin(2 * np.pi * 123 * t)
    wave = first if channels == 1 else np.stack([first, np.ones_like(first) * 0.8], axis=1)
    path = tmp_path / "variant.wav"
    sf.write(path, wave, rate, subtype="FLOAT")
    dataset.uttid2path[dataset.uttids[0]] = str(path)
    result = dataset[0]
    assert result["feats"].shape == (1, 64000)
    assert torch.isfinite(result["feats"]).all()
    assert result["feats"].abs().max() <= 1
    assert not (result["feats"][:, 32000:] == 0).all()


def test_bad_audio_never_replaced_by_other_class(dataset, tmp_path):
    uid = dataset.uttids[0]
    dataset.uttid2path[uid] = str(tmp_path / "does_not_exist.wav")
    with pytest.raises(RuntimeError, match=uid):
        dataset[0]


def test_manifest_rejects_header_duplicates_unknown_label(dataset, tmp_path):
    path = tmp_path / "dev.txt"
    initial = path.read_text(encoding="utf-8")
    path.write_text("uttid\torigin_ds\tspeaker\tattacker\tlabel\n" + initial, encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        validate_manifest(tmp_path, "dev")
    path.write_text(initial + initial.splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(tmp_path, "dev")
    path.write_text(initial.replace("bonafide", "unknown", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        validate_manifest(tmp_path, "dev")


def test_rir_and_rawboost_official_probabilities_order_and_arguments(rir_root, monkeypatch):
    augmentor = ConfiguredRawBoost(rir_root)
    assert augmentor.rir == [rir_root / "simulated_rirs/rir_list/synthetic_rir.wav"]
    assert augmentor.rir_prob == 1 / 3 and augmentor.noise_prob == 1 / 5
    assert augmentor.rawboost_args.algo == 5
    events = []
    def rir(audio):
        events.append("rir")
        return audio + 1, None
    def raw(audio, sample_rate, args, algo):
        events.append("rawboost")
        assert audio.mean() == 1 and sample_rate == 16000 and algo == 5
        return audio + 1
    monkeypatch.setattr(augmentor, "convolve_rir", rir)
    monkeypatch.setattr(augmentor, "process_Rawboost_feature", raw)
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    assert torch.equal(augmentor(torch.zeros(1, 64000)), torch.full((1, 64000), 2.0))
    assert events == ["rir", "rawboost"]


def test_real_rir_convolution_is_reused(dataset, rir_root):
    augmentor = ConfiguredRawBoost(rir_root)
    audio = dataset[0]["feats"]
    augmentor.audio_power = float((audio**2).mean())
    result, _ = augmentor.convolve_rir(audio)
    assert result.shape == audio.shape and torch.isfinite(result).all()
    assert float((result**2).mean()) == pytest.approx(augmentor.audio_power, rel=1e-4)


def test_official_rawboost_algorithm5_runs_with_pinned_numpy(dataset, rir_root, monkeypatch):
    augmentor = ConfiguredRawBoost(rir_root)
    original = np.random.rand
    first_draws = iter([1.0, 0.0])  # Only force the two gate draws; keep the algorithm unchanged.
    def gates(*shape):
        if not shape and gates.count < 2:
            gates.count += 1
            return next(first_draws)
        return original(*shape)
    gates.count = 0
    monkeypatch.setattr(np.random, "rand", gates)
    result = augmentor(dataset[0]["feats"])
    assert result.shape == (1, 64000) and result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_train_adapter_uses_configured_official_augmentation_only_on_train(dataset, config, tmp_path, rir_root, monkeypatch):
    options = dict(config["data"], data_path=str(tmp_path), rir_root=str(rir_root))
    training = SpeechFakeDataset(options, "train", train_mode=True)
    assert training.do_augment and isinstance(training.augmentor, ConfiguredRawBoost)
    monkeypatch.setattr(np.random, "rand", lambda: 1.0)
    assert training[0]["feats"].shape == (1, 64000)
    assert not SpeechFakeDataset(options, "test").do_augment
    with pytest.raises(ValueError, match="restricted"):
        SpeechFakeDataset(options, "dev", train_mode=True)


def test_bank_source_quotas_and_capacity_reporting(dataset):
    dataset.split = "train"
    with pytest.raises(ValueError, match="capacity mismatch"):
        select_bank_samples(dataset, 17)
    assert sum(REAL_QUOTAS.values()) == 1024
