import numpy as np
import pytest
import torch
import torch.nn.functional as F
from methods.prototypes import build_prototypes, cosine_logits
from methods.layer_weights import layer_weights
from methods.loss import episode_loss, step_loss
from methods.inference import kshot_task, kshot_scores, predict, bank_from_features, zeroshot_scores, PrototypeBank
from utils.metrics import binary_metrics, macro_metrics, mean_std


def task(seed, dim=7):
    generator = torch.Generator().manual_seed(seed)
    support = torch.randn(15, 25, dim, generator=generator, requires_grad=True)
    query = torch.randn(15, 25, dim, generator=generator, requires_grad=True)
    labels = torch.arange(3).repeat_interleave(5)
    return support, labels, query, labels


def test_prototypes_logits_and_global_softmax_match_scalar_definition():
    support, labels, query, qlabels = task(17)
    prototypes = build_prototypes(support, labels, 3)
    assert prototypes.shape == (3, 25, 7)
    assert torch.equal(prototypes[1], support[5:10].mean(0))
    theta = torch.linspace(-2, 2, 25, requires_grad=True)
    alpha = layer_weights(theta)
    expected = theta.exp() / theta.exp().sum()
    assert torch.allclose(alpha, expected)
    assert alpha.shape == (25,) and alpha.dtype == torch.float32
    assert bool((alpha >= 0).all()) and torch.allclose(alpha.sum(), torch.tensor(1.0))
    assert alpha.requires_grad and alpha.grad_fn is not None
    assert cosine_logits(query, prototypes).shape == (15, 25, 3)


def test_query_layer_episode_aggregation_and_support_gradient():
    tasks = [task(seed) for seed in range(8)]
    theta = torch.nn.Parameter(torch.linspace(-1, 1, 25))
    loss, diagnostics = step_loss(tasks, layer_weights(theta))
    scalar_losses = []
    for args, details in zip(tasks, diagnostics):
        support, labels, query, qlabels = args
        logits = cosine_logits(query, details["prototypes"])
        layers = torch.stack([F.cross_entropy(logits[:, l], qlabels) for l in range(25)])
        scalar_losses.append((details["alpha"] * layers).sum())
        assert torch.allclose(layers, details["layer_losses"])
    assert torch.allclose(loss, torch.stack(scalar_losses).mean())
    assert all(details["alpha"] is diagnostics[0]["alpha"] for details in diagnostics)
    assert not torch.equal(diagnostics[0]["prototypes"], diagnostics[1]["prototypes"])
    loss.backward()
    mean_layer_losses = torch.stack([details["layer_losses"].detach() for details in diagnostics]).mean(0)
    alpha = layer_weights(theta.detach())
    expected_gradient = alpha * (mean_layer_losses - (alpha * mean_layer_losses).sum())
    torch.testing.assert_close(theta.grad, expected_gradient, atol=1e-8, rtol=1e-5)
    assert theta.grad.abs().sum() > 0
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
    theta = torch.nn.Parameter(torch.zeros(25))
    loss, details = episode_loss(support, labels, query, labels, layer_weights(theta))
    loss.backward()
    assert support_input.grad.abs().sum() > 0
    assert projection.weight.grad.abs().sum() > 0
    assert details["alpha"].requires_grad and theta.grad.abs().sum() > 0


def test_zero_initialization_and_unconstrained_softmax():
    theta = torch.nn.Parameter(torch.zeros(25))
    assert theta.shape == (25,) and torch.count_nonzero(theta) == 0
    alpha = layer_weights(theta)
    assert alpha.dtype == torch.float32 and torch.isfinite(alpha).all()
    assert torch.allclose(alpha, torch.full((25,), 1 / 25))
    with torch.no_grad():
        theta[0] = 1000
    assert layer_weights(theta)[0] == 1  # No clipping, bounding or temperature.


def test_support_changes_prototypes_without_changing_alpha():
    support, labels, query, qlabels = task(42)
    theta = torch.nn.Parameter(torch.linspace(-2, 2, 25))
    alpha = layer_weights(theta)
    _, first = episode_loss(support, labels, query, qlabels, alpha)
    changed = support.detach().clone()
    changed[:5] += 3
    _, second = episode_loss(changed, labels, query, qlabels, layer_weights(theta))
    assert torch.equal(first["alpha"], second["alpha"])
    assert not torch.equal(first["prototypes"], second["prototypes"])


@pytest.mark.parametrize("shot", [1, 3, 10])
def test_kshot_sign_zero_threshold_and_parameterized_shot(shot):
    real = torch.tensor([1., 0.]).expand(shot, 25, 2)
    fake = torch.tensor([-1., 0.]).expand(shot, 25, 2)
    prototypes, alpha = kshot_task(torch.cat([real, fake]), torch.tensor([0] * shot + [1] * shot),
                                  layer_weights(torch.zeros(25)))
    queries = torch.stack([real[0], fake[0], torch.tensor([0., 1.]).expand(25, 2)])
    scores = kshot_scores(queries, prototypes, alpha)
    assert torch.allclose(scores, torch.tensor([-2., 2., 0.]))
    assert predict(scores).tolist() == [0, 1, 1]


def test_zeroshot_33_class_bank_and_per_layer_maximum(tmp_path):
    generator = torch.Generator().manual_seed(17)
    by_class = {"real": torch.randn(5, 25, 7, generator=generator)}
    by_class.update({f"VC__g{i}": torch.randn(5, 25, 7, generator=generator) for i in range(32)})
    theta = torch.linspace(-2, 2, 25)
    bank = bank_from_features(by_class, theta)
    assert bank.prototypes.shape == (33, 25, 7) and torch.equal(bank.layer_logits, theta)
    query = torch.randn(4, 25, 7, generator=generator)
    logits = cosine_logits(query, bank.prototypes)
    expected = ((logits[:, :, 1:].max(-1).values - logits[:, :, 0]) * bank.alpha).sum(-1)
    assert torch.allclose(zeroshot_scores(query, bank), expected)
    bank.save(tmp_path / "bank.pt")
    restored = PrototypeBank.load(tmp_path / "bank.pt")
    assert torch.equal(restored.prototypes, bank.prototypes)
    assert torch.equal(restored.layer_logits, bank.layer_logits)
    assert torch.equal(zeroshot_scores(query, restored), expected)


def test_zeroshot_rejects_legacy_bank(tmp_path):
    path = tmp_path / "legacy_bank.pt"
    torch.save(dict(classes=["real", "fake"], prototypes=torch.randn(2, 25, 7),
                    dispersion=torch.ones(2, 25), metadata=dict(method_version="2.4")), path)
    with pytest.raises(ValueError, match="Incompatible"):
        PrototypeBank.load(path)


def test_kshot_nonuniform_weighted_score_matches_formula():
    support, labels, query, _ = task(71)
    theta = torch.linspace(-3, 1, 25)
    prototypes, alpha = kshot_task(support[:10], labels[:10], layer_weights(theta))
    expected = torch.stack([
        sum(alpha[l] * (F.cosine_similarity(q[l], prototypes[1, l], dim=0) -
                        F.cosine_similarity(q[l], prototypes[0, l], dim=0)) for l in range(25))
        for q in query])
    torch.testing.assert_close(kshot_scores(query, prototypes, alpha), expected)


def test_metrics_use_fake_positive_and_macro_not_pooled():
    values = binary_metrics([0, 0, 1, 1], [-1, -0.1, 0, 1])
    assert values["eer"] == 0 and values["acc"] == 1 and values["auc"] == 1 and values["ap"] == 1
    first = dict(eer=0.1, auc=0.9, ap=0.8, acc=0.7)
    second = dict(eer=0.3, auc=0.7, ap=0.6, acc=0.5)
    assert macro_metrics([first, second])["eer"] == pytest.approx(0.2)
    assert mean_std([first, second])["eer"] == pytest.approx(dict(mean=0.2, std=0.1))
