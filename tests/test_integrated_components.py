"""Behavior regressions for the integrated components, without external source trees."""
from argparse import Namespace
import importlib.abc
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from fs_lprf.data.audio import get_audio
from fs_lprf.models.fs_lprf_model import FSLPRF
from fs_lprf.models.xlsr import Wav2Vec2Model
from fs_lprf.models import frontend_compat
from fs_lprf.utils.metrics import compute_eer


def test_all_modules_import_without_external_checkout(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    program = """
import importlib.abc
import pkgutil
from pathlib import Path
import sys

class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tomato', 'fairseq'}:
            raise AssertionError('Unexpected external import: ' + fullname)

sys.meta_path.insert(0, BlockExternal())
sys.path.insert(0, sys.argv[1])
import fs_lprf
for module in pkgutil.walk_packages(fs_lprf.__path__, fs_lprf.__name__ + '.'):
    __import__(module.name)
assert not any('third_party' in path for path in sys.path)
print('All FS-LPRF modules imported independently')
"""
    result = subprocess.run([sys.executable, "-I", "-c", program, str(source)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "imported independently" in result.stdout


def test_transformer_matches_pinned_original_reference():
    """The fixture contains old-code synthetic outputs, never pretrained weights."""
    reference = json.loads((Path(__file__).parent / "fixtures/xlsr_small_reference.json").read_text())
    torch.manual_seed(42)
    np.random.seed(42)
    model = Wav2Vec2Model(Namespace(**reference["config"])).eval()
    assert {key: list(value.shape) for key, value in model.state_dict().items()} == reference["parameter_shapes"]
    source = torch.randn(1, 80, generator=torch.Generator().manual_seed(71))
    with torch.no_grad():
        output = model(source, mask=False, features_only=True)
    hidden = [output["post_features"]] + [item[0] for item in output["layer_results"]]
    assert len(hidden) == len(reference["hidden"]) == 3
    for actual, expected in zip(hidden, reference["hidden"]):
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(output["x"], torch.tensor(reference["final"]), rtol=1e-6, atol=1e-6)
    # Odd frame count exercises the original sequence-padding/undo-padding path.
    assert output["x"].shape == (1, 19, 8)


def _synthetic_backend():
    """Exercise the real pooling/checkpoint methods without loading a 300M model."""
    model = FSLPRF.__new__(FSLPRF)
    torch.nn.Module.__init__(model)
    model.frontend_model = torch.nn.Linear(4, 4).requires_grad_(False)
    model.frontend_dim = 1024
    model.num_layers = 25
    model.time_attn_mlp1 = torch.nn.Parameter(torch.randn(25, 1024, 4) * 0.02)
    model.bottleneck_proj = torch.nn.Linear(1024, 256)
    return model


def test_pooling_matches_independent_per_layer_per_utterance_calculation_and_gradients():
    torch.manual_seed(42)
    model = _synthetic_backend().train()
    hidden = torch.randn(25, 7, 2, 1024)
    actual = model.tnd_forward(list(hidden.unbind()))
    expected = []
    for batch in range(2):
        layers = []
        for layer in range(25):
            features = hidden[layer, :, batch]
            scores = (features @ model.time_attn_mlp1[layer]).logsumexp(dim=-1).softmax(dim=0)
            layers.append((features * scores[:, None]).sum(dim=0))
        expected.append(torch.stack(layers))
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)
    assert actual.shape == (2, 25, 1024)
    projected = model.bottleneck_proj(actual)
    assert projected.shape == (2, 25, 256)
    projected.square().mean().backward()
    assert not model.frontend_model.training
    assert all(parameter.grad is None for parameter in model.frontend_model.parameters())
    assert all(model.time_attn_mlp1.grad[layer].isfinite().all() and
               model.time_attn_mlp1.grad[layer].abs().sum() > 0 for layer in range(25))
    for parameter in model.bottleneck_proj.parameters():
        assert parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0


def test_backend_keeps_legacy_parameter_names_and_count():
    model = _synthetic_backend()
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == {
        "time_attn_mlp1": (25, 1024, 4),
        "bottleneck_proj.weight": (256, 1024),
        "bottleneck_proj.bias": (256,),
    }
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 364800


def test_audio_keeps_first_channel_and_original_rms_normalization(tmp_path):
    first = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
    stereo = np.stack([first, np.full_like(first, 0.8)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 16000, subtype="FLOAT")
    raw, rate = get_audio(path, norm=False, trim_sil=False)
    assert rate == 16000 and torch.equal(raw, torch.from_numpy(first[None]))
    normalized, _ = get_audio(path, trim_sil=False)
    expected = np.clip(first * (0.1 / np.sqrt(np.square(first).mean())), -1.0, 1.0)
    assert torch.equal(normalized, torch.from_numpy(expected[None]))


@pytest.mark.parametrize("scores,eer,threshold", [
    ([-2., -1., 1., 2.], 0., -1.),
    ([2., 1., -1., -2.], 1., -1.),
    ([0., 0., 0., 0.], 1., 0.),
])
def test_eer_preserves_original_threshold_and_stable_tie_order(scores, eer, threshold):
    actual = compute_eer(np.array([0, 0, 1, 1]), np.array(scores))
    assert actual == (eer, threshold)


def test_official_loader_retains_verified_fallback_and_unmodified_tensors(tmp_path, monkeypatch):
    path = tmp_path / "verified.pt"
    path.write_bytes(b"synthetic checkpoint I/O test")
    monkeypatch.setattr(frontend_compat, "sha256_file", lambda _: frontend_compat.OFFICIAL_SHA256)
    tensor = torch.arange(3)
    calls = []
    def load(file, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise pickle.UnpicklingError("Weights only load failed")
        return {"cfg": {"model": {"_name": "wav2vec2"}}, "model": {"synthetic": tensor}}
    monkeypatch.setattr(torch, "load", load)
    payload = frontend_compat.load_official_checkpoint(path)
    assert calls == [{"map_location": "cpu"}, {"map_location": "cpu", "weights_only": False}]
    assert payload["state_dict"]["synthetic"] is tensor
    assert payload["cfg"].layer_type == "transformer"
    assert frontend_compat.LOAD_REPORT["unsafe_fallback"]
