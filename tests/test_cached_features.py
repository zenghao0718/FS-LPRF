import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from data.cache import CacheReader, CacheWriter
from data.cache_pipeline import build_dataset_cache, cache_specs, evaluation_view, bind_evaluation_caches
from data.speechfake_dataset import SpeechFakeDataset
from methods.loss import step_loss
from methods.inference import kshot_task, kshot_scores
from models.fs_lprf_model import FSLPRF, forward_batch
from utils.numerics import tensor_errors
from test_cache_storage import storage_identity, cache_config


class FakeFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.13), requires_grad=False)
        self.calls = 0

    def forward(self, wave, **kwargs):
        self.calls += 1
        dimension = torch.linspace(-.3, .3, 1024, device=wave.device)
        post = wave[:, :3, None] * (1 + dimension) + dimension + self.bias
        layers = [(torch.sin(post * (1 + layer / 24) + layer * .07).permute(1, 0, 2), None)
                  for layer in range(24)]
        return dict(post_features=post, layer_results=layers)


def synthetic_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = FSLPRF.__new__(FSLPRF)
        torch.nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.frontend_model = torch.nn.Module()
        model.frontend_model.model = FakeFrontend()
        model.num_layers, model.frontend_dim = 25, 1024
        model.time_attn_mlp1 = torch.nn.Parameter(torch.randn(25, 1024, 4) * .02)
        model.bottleneck_proj = torch.nn.Linear(1024, 256)
        return model.train()


def episode_loss_scores(features):
    labels = torch.arange(15) // 5
    loss, _ = step_loss([(features[:15], labels, features[15:], labels)])
    binary_labels = torch.tensor([0] * 5 + [1] * 5)
    task = kshot_task(features[:10], binary_labels)
    scores = kshot_scores(features[15:], *task)
    return loss, scores


def test_feature_order_shape_fp32_equivalence_and_gradients():
    model = synthetic_model()
    waves = torch.randn(2, 1, 64000, generator=torch.Generator().manual_seed(42))
    original_hiddens = model.extract_features({"feats": waves})
    cached = model.extract_frontend_features(waves, to_cpu=True)
    assert cached.shape == (2, 25, 3, 1024) and not cached.requires_grad
    for layer in range(25):
        assert torch.equal(cached[:, layer], original_hiddens[layer].permute(1, 0, 2))
    online = model({"feats": waves})["feats"]
    online.square().mean().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    restored = model.forward_from_frontend_features(cached)["feats"]
    torch.testing.assert_close(online, restored, atol=1e-6, rtol=1e-6)
    restored.square().mean().backward()
    assert not model.frontend_model.training
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
    assert all(model.time_attn_mlp1.grad[layer].abs().sum() > 0 for layer in range(25))
    assert set(model.state_dict()) == {"time_attn_mlp1", "bottleneck_proj.weight", "bottleneck_proj.bias"}
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 364800


def test_fp16_disk_roundtrip_converts_to_fp32_and_reports_loss_scores(tmp_path):
    model = synthetic_model()
    waves = torch.randn(30, 1, 64000, generator=torch.Generator().manual_seed(17))
    states = model.extract_frontend_features(waves, to_cpu=True)
    identity = storage_identity("xlsr_feature", ids=tuple(f"u{i}" for i in range(30)), dtype="float16")
    with CacheWriter(tmp_path / "features", identity, 8) as writer:
        for uid, state in zip(identity["uttids"], states):
            writer.add(uid, state)
        reader = writer.finish()
    saved = torch.stack([reader.get(uid) for uid in identity["uttids"]])
    assert all(row["length"] == states.shape[2] for row in reader.index)
    assert saved.dtype == torch.float16
    full = model.forward_from_frontend_features(states)["feats"]
    reduced = model.forward_from_frontend_features(saved)["feats"]
    assert reduced.dtype == torch.float32
    loss32, scores32 = episode_loss_scores(full)
    loss16, scores16 = episode_loss_scores(reduced)
    report = dict(scope="synthetic 30-audio episode and synthetic frozen frontend",
                  hidden_states=tensor_errors(states, saved.float()), backend=tensor_errors(full, reduced),
                  loss_fp32=float(loss32.detach()), loss_fp16=float(loss16.detach()),
                  loss_absolute_error=float((loss32-loss16).abs().detach()), scores=tensor_errors(scores32, scores16))
    (tmp_path / "fp16_report.json").write_text(json.dumps(report, indent=2))
    assert report["hidden_states"]["cosine_similarity"] > .99999
    assert report["backend"]["relative_l2_error"] < .002
    assert report["loss_absolute_error"] < .01
    assert report["scores"]["max_absolute_error"] < .01
    reduced.square().mean().backward()
    assert model.time_attn_mlp1.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 24, 3, 1024), (1, 25, 3, 256), (25, 3, 1024), (1, 25, 0, 1024)])
def test_invalid_frontend_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="Expected|nonempty"):
        synthetic_model().forward_from_frontend_features(torch.zeros(shape))


def test_train_feature_cache_builds_one_view_and_never_repeats_frontend(dataset, config, rir_root, tmp_path):
    cfg = cache_config(dataset, config, tmp_path)
    cfg["data"].update(do_augment=True, rir_root=str(rir_root))
    cfg["data"]["train_cache_mode"] = "xlsr_feature"
    raw, identities = cache_specs(cfg, "train", "xlsr_feature", limit=3)
    augmentation = Mock(side_effect=lambda waveform: waveform * .9)
    raw.augmentor = augmentation
    model = synthetic_model()
    reader = build_dataset_cache(raw, cfg, identities[0], model=model)
    assert augmentation.call_count == 3
    assert all(call.args[0].shape == (1, 64000) for call in augmentation.call_args_list)
    calls = model.frontend_model.model.calls
    assert reader.identity["feature_views"] == 1 and reader.identity["train_feature_view_seed"] == 42
    cached = SpeechFakeDataset(cfg["data"], "train").use_cache(reader, "xlsr_feature")
    cached.sample_segment = Mock(side_effect=AssertionError("must not crop cached features"))
    cached.augmentor = Mock(side_effect=AssertionError("must not augment cached features"))
    batch = cached.collate_fn([cached[0], cached[0]])
    assert torch.equal(batch["feats"][0], batch["feats"][1])
    result = forward_batch(model, batch)
    result.square().mean().backward()
    assert model.frontend_model.model.calls == calls
    assert not model.frontend_model.training and model.time_attn_mlp1.grad.abs().sum() > 0


def test_dev_cache_must_be_bound_and_never_builds_during_training(dataset, config, tmp_path):
    cfg = cache_config(dataset, config, tmp_path)
    cfg["data"]["dev_cache_mode"] = "xlsr_feature"
    with pytest.raises(ValueError, match="not bound"):
        evaluation_view(dataset, cfg, "full", 42)
    with pytest.raises(ValueError, match="never silently generate"):
        bind_evaluation_caches(dataset, cfg, {}, build_missing=True, model=Mock())
    with pytest.raises(ValueError, match="Test must read original WAVs"):
        cache_specs(cfg, "test", "xlsr_feature")


def test_prototype_bank_refuses_augmented_train_feature_views():
    from methods.prototype_bank import select_bank_samples
    dataset = SimpleNamespace(split="train", do_augment=False, cache_mode="xlsr_feature")
    with pytest.raises(ValueError, match="augmented Train feature-cache"):
        select_bank_samples(dataset, 17)
