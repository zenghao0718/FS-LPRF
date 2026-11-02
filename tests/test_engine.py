from argparse import Namespace
import copy
from pathlib import Path
from unittest.mock import Mock
import random
import numpy as np
import pytest
import torch
from fs_lprf.config import load_config
from fs_lprf.data.episode_sampler import EpisodeSampler
from fs_lprf.engine.trainer import make_optimizer, optimize_step
from fs_lprf.engine.checkpoint import save_checkpoint, load_checkpoint
from fs_lprf.models.frontend_compat import load_official_checkpoint
from fs_lprf.utils.seed import seed_all
from fs_lprf.upstream import enable_upstream, UPSTREAM_COMMIT, UPSTREAM


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


def test_engine_one_backward_one_update_microbatches_and_independent_episodes(config, monkeypatch):
    seed_all(42)
    dataset, model = SyntheticDataset(), SyntheticBackend()
    episodes = EpisodeSampler(dataset.pools).sample_step()
    optimizer, scheduler = make_optimizer(model, config)
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
    loss, norm, diagnostics = optimize_step(model, dataset, episodes, config, optimizer, 1)
    assert len(calls) == 1 and step_spy.call_count == 1
    assert len(model.seen_batch_sizes) == 30 and set(model.seen_batch_sizes) == {8}
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


def test_upstream_imports_and_exact_submodule_revision():
    import subprocess
    enable_upstream()
    from tomato.models.frontend_only import XLSRTimeAttnOnly
    from tomato.data.fad_trim import GeneralFAD
    from tomato.train_util.eer import compute_eer
    from fs_lprf.models.fs_lprf_model import FSLPRF
    assert issubclass(FSLPRF, XLSRTimeAttnOnly)
    assert issubclass(__import__("fs_lprf.data.speechfake_dataset", fromlist=["SpeechFakeDataset"]).SpeechFakeDataset, GeneralFAD)
    revision = subprocess.check_output(["git", "-c", f"safe.directory={UPSTREAM.as_posix()}",
                                        "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert revision == UPSTREAM_COMMIT
    assert not subprocess.check_output(["git", "-c", f"safe.directory={UPSTREAM.as_posix()}",
                                        "-C", str(UPSTREAM), "diff", "--name-only"], text=True).strip()
