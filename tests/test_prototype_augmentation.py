import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from data.randomness import sample_seed
from data.speechfake_dataset import SpeechFakeDataset
from methods.prototype_bank import (dataset_config_for_mode, select_bank_samples,
                                   validate_augmented_reference, build_prototype_bank)
from utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_augmentation_mode_is_clean(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("scripts.build_prototype_bank")
    args = module.build_parser().parse_args(["--checkpoint", "best", "--output", "bank.pt"])
    assert args.augmentation_mode == "clean"
    assert args.reference_bank is None


def _snapshot(config):
    cfg = load_config(ROOT / "configs" / "speechfake.yaml")
    cfg["data"].update(config["data"])
    cfg["data"]["codec_augment"] = dict(enabled=True, mode="offline", probability=0.5,
                                        codecs=["mp3", "opus", "amr", "encodec"], sample_rate=16000,
                                        build_seed=42, offline_cache_root="/tmp/codec")
    cfg["data"]["do_augment"] = True
    cfg["data"]["rir_root"] = "/tmp/rirs"
    cfg["data"]["train_cache_mode"] = "waveform"
    return cfg


def test_mode_config_keeps_codec_probability_and_toggles_only_rir_rawboost(config):
    cfg = _snapshot(config)
    clean = dataset_config_for_mode(cfg, "clean")
    codec50 = dataset_config_for_mode(cfg, "codec50")
    full = dataset_config_for_mode(cfg, "full")
    assert cfg["data"]["do_augment"] is True
    assert clean["data"]["do_augment"] is True and clean["data"]["codec_augment"]["enabled"] is True
    assert codec50["data"]["codec_augment"]["enabled"] is True
    assert codec50["data"]["codec_augment"]["mode"] == "offline"
    assert codec50["data"]["codec_augment"]["probability"] == 0.5
    assert codec50["data"]["do_augment"] is False
    assert full["data"]["do_augment"] is True
    assert full["data"]["codec_augment"]["probability"] == 0.5
    with pytest.raises(ValueError, match="probability 0.5"):
        broken = dataset_config_for_mode(cfg, "codec50")
        broken["data"]["codec_augment"]["enabled"] = False
        dataset_config_for_mode(broken, "full")


def _write_manifests(root, wav, count):
    root.mkdir(parents=True, exist_ok=True)
    names = ["real"] + [f"TTS__generator{i:02d}" for i in range(32)]
    paths, infos = [], []
    for name in names:
        for index in range(count):
            uid = f"train_{name}_{index:03d}"
            paths.append(f"{uid}\t{wav}\n")
            infos.append(f"{uid}\tAISHELL-1\tspk{index}\t{name}\t{'bonafide' if name == 'real' else 'spoof'}\n")
    (root / "train.tsv").write_text("".join(paths), encoding="utf-8")
    (root / "train.txt").write_text("".join(infos), encoding="utf-8")
    for split in ("dev", "test"):
        (root / f"{split}.tsv").write_text(paths[0].replace("train_", f"{split}_"), encoding="utf-8")
        (root / f"{split}.txt").write_text(infos[0].replace("train_", f"{split}_"), encoding="utf-8")


class _Reader:
    def __init__(self, waveform, rows):
        self.waveform = waveform
        self.rows = rows
        self.calls = []

    def get(self, uid):
        self.calls.append(uid)
        return self.waveform.clone()


def _augmented_dataset(tmp_path, wav, rir_root, config, mode):
    _write_manifests(tmp_path, wav, 1)
    cfg = _snapshot(config)
    cfg["data"]["data_path"] = str(tmp_path)
    cfg["data"]["rir_root"] = str(rir_root)
    cfg["data"]["do_augment"] = mode == "full"
    dataset = SpeechFakeDataset(cfg["data"], "train", train_mode=True)
    waveform = (0.1 * torch.sin(torch.linspace(0, 80, 64000))).view(1, -1)
    rows = {uid: {"codec": dict(family="mp3", variant="cbr", param=64, status="ok")} for uid in dataset.uttids}
    dataset.cache_mode = "waveform"
    dataset.cache_reader = _Reader(waveform, rows)
    dataset.codec_cache_reader = _Reader(waveform + 0.25, rows)
    dataset.record_augmentation = True
    return dataset


def _choices(dataset):
    choices = {}
    for uid in dataset.uttids:
        seed = sample_seed("view", dataset.split, "full", 17, uid)
        item = dataset[(dataset.uttid2idx[uid], seed)]
        choices[uid] = item["augmentation_trace"]["codec_applied"]
    return choices


def test_codec50_and_full_share_codec_choices_and_training_order(tmp_path, wav, rir_root, config):
    codec50 = _augmented_dataset(tmp_path / "b", wav, rir_root, config, "codec50")
    full = _augmented_dataset(tmp_path / "c", wav, rir_root, config, "full")
    assert codec50.do_codec_augment and codec50.codec_mode == "offline" and not codec50.do_augment
    assert full.do_augment and full.augmentor.rir_prob == 1 / 3
    assert full.augmentor.noise_prob == 1 / 5 and full.augmentor.rawboost_args.algo == 5
    assert _choices(codec50) == _choices(full)
    assert set(_choices(full).values()) == {False, True}
    order = []
    full.codec_cache_reader.get = lambda uid, original=full.codec_cache_reader.get: order.append("codec") or original(uid)
    full.load_preprocessed_waveform = lambda uid, original=full.load_preprocessed_waveform: order.append("original") or original(uid)
    full.sample_segment = lambda audio, original=full.sample_segment: order.append("crop") or original(audio)

    class _OrderAug:
        def __init__(self, inner):
            object.__setattr__(self, "inner", inner)

        def __setattr__(self, name, value):
            setattr(self.inner, name, value)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __call__(self, audio):
            order.append("rir_rawboost")
            return self.inner(audio)

    full.augmentor = _OrderAug(full.augmentor)
    uid = full.uttids[0]
    full[(full.uttid2idx[uid], sample_seed("view", "train", "full", 17, uid))]
    assert order[0] in {"codec", "original"} and order[1:] == ["crop", "rir_rawboost"]


def test_codec_cache_mismatch_has_no_online_fallback(monkeypatch, tmp_path):
    from data.codec_cache import open_codec_cache

    class Reader:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr("data.codec_cache.CacheReader", lambda *args, **kwargs: reader)

    def fail(*args, **kwargs):
        raise ValueError("Codec parameters/build_seed mismatch")

    monkeypatch.setattr("data.codec_cache.validate_codec_reader", fail)
    with pytest.raises(ValueError, match="no online/WAV fallback"):
        open_codec_cache(object(), {"cache": {"verify_checksums": False}}, path=str(tmp_path))
    assert reader.closed


def test_full_mode_missing_rir_fails(tmp_path, wav, config):
    _write_manifests(tmp_path, wav, 1)
    cfg = _snapshot(config)
    cfg["data"]["data_path"] = str(tmp_path)
    cfg["data"]["rir_root"] = str(tmp_path / "missing_rir")
    cfg["data"]["do_augment"] = True
    with pytest.raises(FileNotFoundError, match="RIR resource root"):
        SpeechFakeDataset(cfg["data"], "train", train_mode=True)


def _reference_dataset():
    names = [f"NV__g{i:02d}" for i in range(32)]
    dataset = type("Dataset", (), {})()
    dataset.split = "train"
    dataset.fake_classes = names
    dataset.do_augment = False
    dataset.do_codec_augment = True
    dataset.codec_mode = "offline"
    dataset.codec_probability = 0.5
    dataset.cache_mode = "waveform"
    dataset.codec_cache_reader = object()
    dataset.uttids = []
    dataset.uttid2info = {}
    dataset.class_indices = {"real": []}
    for source in ("Aishell", "Aishell3", "LibriTTS", "VCTK-Corpus"):
        for number in range(260):
            uid = f"{source}_{number}"
            dataset.class_indices["real"].append(len(dataset.uttids))
            dataset.uttids.append(uid)
            dataset.uttid2info[uid] = {"origin_ds": source, "attacker": "real"}
    for name in names:
        dataset.class_indices[name] = []
        for number in range(1030):
            uid = f"{name}_{number}"
            dataset.class_indices[name].append(len(dataset.uttids))
            dataset.uttids.append(uid)
            dataset.uttid2info[uid] = {"origin_ds": "LibriTTS", "attacker": name}
    dataset.uttid2idx = {uid: index for index, uid in enumerate(dataset.uttids)}
    return dataset


def test_augmented_reference_reuses_clean_ids_and_keeps_dimensions(tmp_path, monkeypatch):
    dataset = _reference_dataset()
    dataset.do_codec_augment = False
    selected = select_bank_samples(dataset, 17)
    dataset.do_codec_augment = True
    assert validate_augmented_reference(dataset, selected, 17, "codec50") == selected
    dataset.record_augmentation = True

    def extract(model, source, ids, cfg, seed):
        source.augmentation_trace_sink.extend(dict(
            uttid=uid, codec_applied=index % 2 == 0, codec_family="opus", codec_variant="audio",
            codec_param=16, codec_status="fallback_original", rir_applied=False, rir_path="",
            rawboost_applied=False) for index, uid in enumerate(ids))
        return torch.ones(len(ids), 25, 256)

    def misaligned(model, source, ids, cfg, seed):
        source.augmentation_trace_sink.extend(dict(
            uttid="missing", codec_applied=False, codec_family="", codec_variant="", codec_param="",
            codec_status="", rir_applied=False, rir_path="", rawboost_applied=False) for _ in ids)
        return torch.ones(len(ids), 25, 256)

    monkeypatch.setattr("methods.prototype_bank.extract_features", extract)
    monkeypatch.setattr("methods.prototype_bank.dataset_fingerprint", lambda source: "synthetic")
    checkpoint = tmp_path / "step_4000.pt"
    checkpoint.write_bytes(b"checkpoint")
    rows = []
    model = torch.nn.Linear(1, 1)
    bank = build_prototype_bank(model, dataset, {"train": {"audio_microbatch_size": 32}}, 17, checkpoint,
                                selected_samples=selected, augmentation_mode="codec50", trace_rows=rows)
    assert bank.prototypes.shape == (33, 25, 256) and bank.dispersion.shape == (33, 25)
    assert torch.isfinite(bank.prototypes).all() and torch.isfinite(bank.dispersion).all()
    from engine.checkpoint import checkpoint_sha256
    assert bank.metadata["checkpoint_sha256"] == checkpoint_sha256(checkpoint)
    assert bank.metadata["sample_ids"] == selected
    assert [row["uttid"] for row in rows] == [uid for group in selected.values() for uid in group]
    assert all(len(group) == 1024 for group in bank.metadata["sample_ids"].values())
    assert bank.metadata["augmentation_mode"] == "codec50"
    assert bank.metadata["rir_probability"] == 0.0 and bank.metadata["codec_probability"] == 0.5
    monkeypatch.setattr("methods.prototype_bank.extract_features", misaligned)
    with pytest.raises(ValueError, match="lost or reordered"):
        build_prototype_bank(model, dataset, {}, 17, checkpoint, selected_samples=selected,
                             augmentation_mode="codec50", trace_rows=[])
