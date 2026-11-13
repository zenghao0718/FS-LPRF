"""CPU-only checks of production initialization, logging and evaluation wiring."""
import copy
import math
from unittest.mock import Mock

import pytest
import torch

from engine.evaluator import build_support_tasks
from engine.trainer import record_layer_weights
from models.fs_lprf_model import FSLPRF
from test_engine import SyntheticBackend


def test_production_constructor_initializes_only_25_additional_parameters(config, monkeypatch):
    # Run the actual constructor with a tiny frozen frontend and CPU device moves.
    import models.fs_lprf_model as module
    cfg = copy.deepcopy(config)
    cfg["runtime"]["device"] = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "XLSR", lambda *_: torch.nn.Linear(4, 4))
    monkeypatch.setattr(FSLPRF, "to", lambda self, *_: self)
    model = FSLPRF(cfg)
    assert model.layer_logits.requires_grad
    assert torch.equal(model.layer_logits, torch.zeros(25))
    torch.testing.assert_close(model.layer_weights(), torch.full((25,), .04))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 364825
    assert all(not p.requires_grad for p in model.frontend_model.parameters())
    assert set(model.state_dict()) == {"time_attn_mlp1", "bottleneck_proj.weight",
                                       "bottleneck_proj.bias", "layer_logits"}


@pytest.mark.parametrize("concentrated", [False, True])
def test_tensorboard_records_weights_statistics_without_mutation(concentrated):
    model = SyntheticBackend()
    if concentrated:
        with torch.no_grad():
            model.layer_logits[0] = 1000
    before = model.layer_logits.detach().clone()
    writer = Mock()
    record_layer_weights(writer, model, 200)
    points = {call.args[0]: call.args[1:] for call in writer.add_scalar.call_args_list}
    assert len(points) == 28
    alpha = model.layer_weights()
    for layer in range(25):
        assert points[f"layer_weight/layer_{layer:02d}"] == (alpha[layer].item(), 200)
    assert points["layer_weight/max"] == (alpha.max().item(), 200)
    assert points["layer_weight/min"] == (alpha.min().item(), 200)
    assert points["layer_weight/entropy"][0] == pytest.approx(0 if concentrated else math.log(25))
    assert torch.equal(model.layer_logits, before) and model.layer_logits.grad is None


def test_all_support_tasks_use_the_same_checkpoint_weights():
    model = SyntheticBackend()
    with torch.no_grad():
        model.layer_logits.copy_(torch.linspace(-2, 2, 25))
    support = {name: torch.randn(10, 25, 7) for name in ("real", "fake_a", "fake_b")}
    first = build_support_tasks(model, support, ["fake_a", "fake_b"])
    assert first["fake_a"][1] is first["fake_b"][1]
    support["fake_a"] += 5
    second = build_support_tasks(model, support, ["fake_a", "fake_b"])
    assert torch.equal(first["fake_a"][1], second["fake_a"][1])
    assert not torch.equal(first["fake_a"][0], second["fake_a"][0])
    torch.testing.assert_close(second["fake_b"][1], model.layer_weights())
