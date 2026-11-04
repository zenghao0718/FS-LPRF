from argparse import Namespace
import copy
import math
from pathlib import Path
from unittest.mock import Mock
import random
import numpy as np
import pytest
import torch
from utils.config import load_config
from data.episode_sampler import EpisodeSampler
from engine.trainer import make_optimizer, optimize_step
from engine.checkpoint import save_checkpoint, load_checkpoint
from models.frontend_compat import load_official_checkpoint
from utils.seed import seed_all
from utils.provenance import UPSTREAM_COMMIT


class SyntheticDataset(torch.utils.data.Dataset):
    """Only tests the engine's batching and update contract; never used by the real model."""
    def __init__(self):
        self.pools = {name: list(range(c * 12, c * 12 + 12)) for c, name in enumerate(
            ["real"] + [f"VC__g{i:02d}" for i in range(32)])}
        self.features = torch.randn(33 * 12, 1, 9, generator=torch.Generator().manual_seed(71))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"feats": self.features[index]}

    @staticmethod
    def collate_fn(batch):
        return {"feats": torch.stack([item["feats"] for item in batch])}


class SyntheticBackend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.projection = torch.nn.Linear(9, 25 * 7)
        self.seen_batch_sizes = []

    def forward(self, source):
        self.seen_batch_sizes.append(len(source["feats"]))
        return {"feats": self.projection(source["feats"].squeeze(1)).reshape(-1, 25, 7)}


def _microbatch_sizes(total_audio, microbatch):
    """Match DataLoader slicing, including a shorter final batch."""
    full, remainder = divmod(total_audio, microbatch)
    sizes = [microbatch] * full
    if remainder:
        sizes.append(remainder)
    if len(sizes) != math.ceil(total_audio / microbatch) or sum(sizes) != total_audio:
        raise AssertionError("Microbatch split does not cover every audio record")
    return sizes


@pytest.mark.parametrize("microbatch", [8, 16, 32, 48, 64])
def test_engine_one_backward_one_update_microbatches_and_independent_episodes(config, monkeypatch, microbatch):
    seed_all(42)
    dataset, model = SyntheticDataset(), SyntheticBackend()
    episodes = EpisodeSampler(dataset.pools).sample_step()
    cfg = copy.deepcopy(config)
    cfg["train"]["audio_microbatch_size"] = microbatch
    optimizer, scheduler = make_optimizer(model, cfg)
    step_spy = Mock(wraps=optimizer.step)
    monkeypatch.setattr(optimizer, "step", step_spy)
    original_backward = torch.Tensor.backward
    calls = []
    def backward(tensor, *args, **kwargs):
        calls.append(tensor)
        return original_backward(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    before = model.projection.weight.detach().clone()
    with torch.no_grad():
        expected_prototypes = [model.projection(dataset.features[list(episode.support_indices)].squeeze(1))
                               .view(3, 5, 25, 7).mean(1) for episode in episodes]
    loss, norm, diagnostics = optimize_step(model, dataset, episodes, cfg, optimizer, 1)
    total_audio = sum(len(episode.indices) for episode in episodes)
    expected_sizes = _microbatch_sizes(total_audio, cfg["train"]["audio_microbatch_size"])
    assert len(calls) == 1 and step_spy.call_count == 1
    assert model.seen_batch_sizes == expected_sizes
    assert len(diagnostics) == 8 and np.isfinite(loss) and np.isfinite(norm)
    assert not torch.equal(before, model.projection.weight)
    assert not torch.equal(diagnostics[0]["alpha"], diagnostics[1]["alpha"])
    # Different GEMM batch sizes differ by <= 2.4e-7 in float32 on the test host.
    assert all(torch.allclose(details["prototypes"], expected, atol=1e-6, rtol=1e-5) for details, expected in
               zip(diagnostics, expected_prototypes))


def test_checkpoint_roundtrip_optimizer_scheduler_and_rng(config, tmp_path):
    seed_all(42)
    model = SyntheticBackend()
    optimizer, scheduler = make_optimizer(model, config)
    model({"feats": torch.randn(2, 1, 9)})["feats"].square().mean().backward()
    optimizer.step()
    scheduler.step()
    expected = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(optimizer.state_dict())
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scheduler, config, 1, dict(mean_eer=0.25),
                    0.25, "manifest-hash", "train-fingerprint")
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    with torch.no_grad():
        model.projection.weight.add_(1)
    payload = load_checkpoint(path, model, optimizer, scheduler, restore_random=True)
    assert payload["step"] == 1 and payload["best_eer"] == 0.25
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in expected.items())
    assert scheduler.state_dict()["last_epoch"] == 1
    for key, state in expected_opt["state"].items():
        assert torch.equal(optimizer.state_dict()["state"][key]["exp_avg"], state["exp_avg"])
    actual_random = (random.random(), np.random.rand(), torch.rand(2))
    assert actual_random[:2] == expected_random[:2]
    assert torch.equal(actual_random[2], expected_random[2])
    # Checkpoint schema is intentionally compatible with weights_only=True.
    assert torch.load(path, weights_only=True)["dev_manifest_sha256"] == "manifest-hash"


def test_scheduler_warmup_cosine_endpoints_and_resume(config):
    model = SyntheticBackend()
    optimizer, scheduler = make_optimizer(model, config)
    factor = scheduler.lr_lambdas[0]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert factor(99) * 1e-4 == pytest.approx(1e-4)
    assert factor(3999) * 1e-4 == pytest.approx(1e-6)
    assert factor(1099) > factor(2099) > factor(3099)


def test_untrusted_pickle_cannot_trigger_unsafe_frontend_load(tmp_path):
    path = tmp_path / "untrusted.pt"
    torch.save({"cfg": Namespace(_name="wav2vec2"), "state_dict": {}}, path)
    with pytest.raises(ValueError, match="restricted"):
        load_official_checkpoint(path)


def test_protocol_configuration_rejects_unrequested_architecture_changes(config, tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / "speechfake.yaml"
    with pytest.raises(ValueError, match="Phase 1 protocol"):
        load_config(path, ["model.layer_attn_head=8"])
    with pytest.raises(ValueError, match="divisible"):
        load_config(path, ["train.total_query_per_episode=14"])
    assert load_config(path, ["train.audio_microbatch_size=1"])["train"]["audio_microbatch_size"] == 1


def test_integrated_imports_and_historical_source_revision():
    from data.audio import GeneralFAD
    from data.speechfake_dataset import SpeechFakeDataset
    from utils.metrics import compute_eer
    from models.fs_lprf_model import FSLPRF
    assert issubclass(FSLPRF, torch.nn.Module)
    assert issubclass(SpeechFakeDataset, GeneralFAD)
    assert compute_eer.__module__ == "utils.metrics"
    assert UPSTREAM_COMMIT == "88efe1376dfc5f5b9b62d016d33a7381c158d8af"
