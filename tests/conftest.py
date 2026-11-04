from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from utils.config import load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "speechfake.yaml")


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "signal.wav"
    t = np.arange(128000, dtype=np.float32) / 16000
    signal = (0.04 + 0.02 * t) * np.sin(2 * np.pi * (130 * t + 23 * t**2))
    sf.write(path, signal, 16000, subtype="FLOAT")
    return path


@pytest.fixture
def rir_root(tmp_path):
    root = tmp_path / "RIRS_NOISES"
    impulse = np.zeros(1600, dtype=np.float32)
    impulse[0], impulse[320] = 1.0, 0.2
    for directory, filename in (
        ("simulated_rirs/smallroom", "Room001/a.wav"),
        ("simulated_rirs/mediumroom", "Room001/c.wav"),
        ("simulated_rirs/largeroom", "Room001/d.wav"),
        ("real_rirs_isotropic_noises", "real.wav"),
    ):
        path = root / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, impulse, 16000, subtype="FLOAT")
        (root / directory / "rir_list").write_text(
            f"--rir-id={directory.replace('/', '_')} --room-id=Room001 RIRS_NOISES/{directory}/{filename}\n",
            encoding="utf-8")
    return root


@pytest.fixture
def dataset(tmp_path, wav, config):
    from data.speechfake_dataset import SpeechFakeDataset
    names = ["real"] + [f"TTS__generator{i:02d}" for i in range(32)]
    for split in ("train", "dev", "test"):
        paths, infos = [], []
        for name in names:
            for i in range(65):
                uid = f"{split}_{name}_{i:03d}"
                paths.append(f"{uid}\t{wav}\n")
                infos.append(f"{uid}\tAISHELL-1\tspk{i}\t{name}\t{'bonafide' if name == 'real' else 'spoof'}\n")
        (tmp_path / f"{split}.tsv").write_text("".join(paths), encoding="utf-8")
        (tmp_path / f"{split}.txt").write_text("".join(infos), encoding="utf-8")
    cfg = dict(config["data"], data_path=str(tmp_path))
    return SpeechFakeDataset(cfg, "dev")


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
