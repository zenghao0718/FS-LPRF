import numpy as np
import pytest
import torch
import torch.nn.functional as F
from methods.prototypes import build_prototypes, cosine_logits, class_dispersion
from methods.reliability import support_reliability, reliability_from_statistics
from methods.loss import episode_loss, step_loss
from methods.inference import kshot_task, kshot_scores, predict, bank_from_features, zeroshot_scores, PrototypeBank
from utils.metrics import binary_metrics, macro_metrics, mean_std


def task(seed, dim=7):
    generator = torch.Generator().manual_seed(seed)
    support = torch.randn(15, 25, dim, generator=generator, requires_grad=True)
    query = torch.randn(15, 25, dim, generator=generator, requires_grad=True)
    labels = torch.arange(3).repeat_interleave(5)
    return support, labels, query, labels


def test_prototypes_logits_and_reliability_match_scalar_definition():
    support, labels, query, qlabels = task(17)
    prototypes = build_prototypes(support, labels, 3)
    assert prototypes.shape == (3, 25, 7)
    assert torch.equal(prototypes[1], support[5:10].mean(0))
    alpha = support_reliability(support, labels, prototypes)
    inter = sum(1 - F.cosine_similarity(prototypes[0], prototypes[c], dim=-1) for c in (1, 2))
    intra = sum((1 - F.cosine_similarity(support[labels == c], prototypes[c], dim=-1)).mean(0) for c in range(3))
    reliability = (inter + 1e-8) / (intra + 1e-8)
    expected = reliability.sqrt() / reliability.sqrt().sum()
    assert torch.allclose(alpha, expected)
    assert alpha.shape == (25,) and alpha.dtype == torch.float32
    assert bool((alpha >= 0).all()) and torch.allclose(alpha.sum(), torch.tensor(1.0))
    assert not alpha.requires_grad and alpha.grad_fn is None
    assert cosine_logits(query, prototypes).shape == (15, 25, 3)


def test_query_layer_episode_aggregation_and_support_gradient():
    tasks = [task(17), task(42)]
    loss, diagnostics = step_loss(tasks)
    scalar_losses = []
    for args, details in zip(tasks, diagnostics):
        support, labels, query, qlabels = args
        logits = cosine_logits(query, details["prototypes"])
        layers = torch.stack([F.cross_entropy(logits[:, l], qlabels) for l in range(25)])
        scalar_losses.append((details["alpha"] * layers).sum())
        assert torch.allclose(layers, details["layer_losses"])
    assert torch.allclose(loss, torch.stack(scalar_losses).mean())
    assert not torch.equal(diagnostics[0]["alpha"], diagnostics[1]["alpha"])
    assert not torch.equal(diagnostics[0]["prototypes"], diagnostics[1]["prototypes"])
    loss.backward()
    for support, labels, query, qlabels in tasks:
        assert support.grad is not None and support.grad.abs().sum() > 0
        assert query.grad is not None and query.grad.abs().sum() > 0


def test_support_gradient_reaches_backend_parameters():
    torch.manual_seed(42)
    projection = torch.nn.Linear(9, 256)
    support_input = torch.randn(15, 25, 9, requires_grad=True)
    support = projection(support_input)
    # Query is constant here: all projection gradients must pass through Support prototypes.
    query = torch.randn(15, 25, 256)
    labels = torch.arange(3).repeat_interleave(5)
    loss, details = episode_loss(support, labels, query, labels)
    loss.backward()
    assert support_input.grad.abs().sum() > 0
    assert projection.weight.grad.abs().sum() > 0
    assert not details["alpha"].requires_grad


def test_float32_reliability_and_degenerate_support_are_finite():
    support = torch.ones(15, 25, 8, dtype=torch.float16)
    labels = torch.arange(3).repeat_interleave(5)
    alpha = support_reliability(support, labels, build_prototypes(support, labels, 3))
    assert alpha.dtype == torch.float32 and torch.isfinite(alpha).all()
    assert torch.allclose(alpha, torch.full((25,), 1 / 25))


@pytest.mark.parametrize("shot", [1, 3, 10])
def test_kshot_sign_zero_threshold_and_parameterized_shot(shot):
    real = torch.tensor([1., 0.]).expand(shot, 25, 2)
    fake = torch.tensor([-1., 0.]).expand(shot, 25, 2)
    prototypes, alpha = kshot_task(torch.cat([real, fake]), torch.tensor([0] * shot + [1] * shot))
    queries = torch.stack([real[0], fake[0], torch.tensor([0., 1.]).expand(25, 2)])
    scores = kshot_scores(queries, prototypes, alpha)
    assert torch.allclose(scores, torch.tensor([-2., 2., 0.]))
    assert predict(scores).tolist() == [0, 1, 1]


def test_zeroshot_33_class_bank_and_per_layer_maximum(tmp_path):
    generator = torch.Generator().manual_seed(17)
    by_class = {"real": torch.randn(5, 25, 7, generator=generator)}
    by_class.update({f"VC__g{i}": torch.randn(5, 25, 7, generator=generator) for i in range(32)})
    bank = bank_from_features(by_class)
    assert bank.prototypes.shape == (33, 25, 7) and bank.dispersion.shape == (33, 25)
    query = torch.randn(4, 25, 7, generator=generator)
    logits = cosine_logits(query, bank.prototypes)
    expected = ((logits[:, :, 1:].max(-1).values - logits[:, :, 0]) * bank.alpha).sum(-1)
    assert torch.allclose(zeroshot_scores(query, bank), expected)
    bank.save(tmp_path / "bank.pt")
    restored = PrototypeBank.load(tmp_path / "bank.pt")
    assert torch.equal(restored.prototypes, bank.prototypes)
    assert torch.equal(restored.dispersion, bank.dispersion)
    assert torch.equal(zeroshot_scores(query, restored), expected)


def test_metrics_use_fake_positive_and_macro_not_pooled():
    values = binary_metrics([0, 0, 1, 1], [-1, -0.1, 0, 1])
    assert values["eer"] == 0 and values["acc"] == 1 and values["auc"] == 1 and values["ap"] == 1
    first = dict(eer=0.1, auc=0.9, ap=0.8, acc=0.7)
    second = dict(eer=0.3, auc=0.7, ap=0.6, acc=0.5)
    assert macro_metrics([first, second])["eer"] == pytest.approx(0.2)
    assert mean_std([first, second])["eer"] == pytest.approx(dict(mean=0.2, std=0.1))
