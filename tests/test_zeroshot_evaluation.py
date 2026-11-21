"""Local 0-shot tests with synthetic waveforms and backend features."""
import csv
import importlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from engine.checkpoint import checkpoint_sha256
from methods import prototype_bank
from methods.inference import PrototypeBank, predict, zeroshot_scores
from test_external_evaluation import fixture_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("scripts.evaluate_zeroshot")


class MockModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, source):
        assert not self.training and not torch.is_grad_enabled()
        audio = source["feats"]
        self.batch_sizes.append(len(audio))
        return {"feats": audio[:, :, :256].expand(-1, 25, -1)}


def test_production_bank_means_and_dispersions_round_trip(tmp_path, monkeypatch):
    classes = ["real"] + [f"VC__{i:02d}" for i in range(32)]
    selected = {name: [f"{name}-{i}" for i in range(2)] for name in classes}
    features = torch.randn((33, 2, 25, 256), generator=torch.Generator().manual_seed(17))
    by_ids = {tuple(ids): values for ids, values in zip(selected.values(), features)}
    monkeypatch.setattr(prototype_bank, "select_bank_samples", lambda dataset, seed: selected)
    monkeypatch.setattr(prototype_bank, "dataset_fingerprint", lambda dataset: "synthetic")
    dataset = type("Dataset", (), {"uttid2info": {uid: {"origin_ds": "Aishell"} for uid in selected["real"]}})()
    model = MockModel()

    def extract(model, dataset, ids, cfg, seed):
        assert not model.training and not torch.is_grad_enabled()
        return by_ids[tuple(ids)]

    monkeypatch.setattr(prototype_bank, "extract_features", extract)
    checkpoint = tmp_path / "step_0800.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    bank = prototype_bank.build_prototype_bank(model, dataset, {}, 17, checkpoint)
    assert model.training
    bank.save(tmp_path / "bank.pt")
    restored = PrototypeBank.load(tmp_path / "bank.pt")
    expected_prototypes = features.mean(1)
    expected_dispersion = (1 - F.cosine_similarity(features, expected_prototypes[:, None], dim=-1)).mean(1)
    assert restored.classes == classes
    assert torch.equal(restored.prototypes, expected_prototypes)
    assert torch.allclose(restored.dispersion, expected_dispersion)
    assert torch.equal(restored.alpha, bank.alpha)
    assert restored.metadata["sample_ids"] == selected
    assert restored.metadata["seed"] == 17
    assert restored.metadata["checkpoint_sha256"] == checkpoint_sha256(checkpoint)


def test_weighted_per_layer_max_over_all_fake_classes_with_zero_boundary():
    prototypes = torch.zeros((33, 25, 256))
    prototypes[0, :, 0] = 1
    prototypes[1:, :, 0] = -1
    prototypes[-1, :12, :2] = torch.tensor([0., 1.])
    prototypes[-2, 12:, :2] = torch.tensor([0., 1.])
    dispersion = torch.arange(1., 26.).expand(33, -1)
    bank = PrototypeBank(["real"] + [f"g{i:02d}" for i in range(32)], prototypes, dispersion, {})
    queries = torch.zeros((4, 25, 256))
    queries[0, :, 0] = 1
    queries[1, :, 1] = 1
    queries[2, :, 2] = 1
    queries[3, :12, 1] = 1
    queries[3, 12:, 0] = 1
    root = ((63 + 1e-8) / (33 * torch.arange(1., 26.) + 1e-8)).sqrt()
    alpha = root / root.sum()
    expected = torch.stack((torch.tensor(-1.), torch.tensor(1.), torch.tensor(0.),
                            alpha[:12].sum() - alpha[12:].sum()))
    actual = zeroshot_scores(queries, bank)
    assert torch.allclose(bank.alpha, alpha)
    assert torch.isfinite(actual).all() and torch.allclose(actual, expected)
    assert predict(actual[:3]).tolist() == [0, 1, 1]


@pytest.fixture
def evaluation(tmp_path, config, monkeypatch, cli):
    # A01 is perfect. A02 has two ties and an erroneous Fake score below every Real.
    trials = [("u1", "bonafide", "bonafide", (1, 0)),
              ("u2", "spoof", "A01", (-1, 0)),
              ("u3", "bonafide", "-", (1, 1)),
              ("u4", "spoof", "A02", (0, 1)),
              ("u5", "spoof", "A02", (1, 0)),
              ("u6", "spoof", "A02", (0, 1))]
    audio = tmp_path / "flac_E_eval"
    audio.mkdir()
    manifest = tmp_path / "eval.track_1.tsv"
    rows = []
    for uid, label, attack, _ in trials:
        (audio / f"{uid}.flac").touch()
        rows.append(f"speaker {uid} F C05 2 source tag {attack} {label} -")
    manifest.write_text("\n".join(rows), encoding="utf-8")

    reads = []
    vectors = {uid: vector for uid, _, _, vector in trials}

    def get_audio(dataset, uid):
        assert dataset.uttid2info[uid]["label"] is None
        assert "attack_id" not in dataset.by_id[uid]
        reads.append(uid)
        wav = torch.zeros((1, 64000))
        wav[0, :2] = torch.tensor(vectors[uid])
        return wav

    model = MockModel()
    monkeypatch.setattr(cli.ExternalDataset, "get_audio", get_audio)
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    checkpoint = fixture_checkpoint(tmp_path / "training/checkpoints/step_0800.pt", model, config)
    classes = ["real"] + [f"VC__{i:02d}" for i in range(32)]
    prototypes = torch.zeros((33, 25, 256))
    prototypes[0, :, 0] = 1
    prototypes[1:, :, 0] = -1
    metadata = dict(checkpoint_sha256=checkpoint_sha256(checkpoint), samples_per_class=1024,
                    real_quotas=prototype_bank.REAL_QUOTAS, seed=17,
                    sample_ids={name: [f"{name}-{i}" for i in range(1024)] for name in classes})
    bank = PrototypeBank(classes, prototypes, torch.ones((33, 25)), metadata)
    bank_path = tmp_path / "bank.pt"
    bank.save(bank_path)
    out = tmp_path / "result"
    args = cli.build_parser().parse_args([
        "--manifest", str(manifest), "--mapping", str(ROOT / "configs/external/asvspoof5.yaml"),
        "--checkpoint", str(checkpoint), "--prototype-bank", str(bank_path), "--output-dir", str(out),
        "--set", "runtime.eval_num_workers=0", "--set", "runtime.pin_memory=false",
        "--set", "train.audio_microbatch_size=2",
    ])
    return args, bank, manifest, reads, model


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_external_scores_all_queries_once_and_macros_by_attack(evaluation, cli):
    args, bank, manifest, reads, model = evaluation
    out = Path(args.output_dir)
    report = cli.run(args)
    assert reads == ["u1", "u2", "u3", "u4", "u5", "u6"]
    assert model.batch_sizes == [2, 2, 2]
    assert set(path.name for path in out.iterdir()) == {
        "zeroshot_run.json", "prototype_bank_info.json", "zeroshot_predictions.csv",
        "zeroshot_per_attack.csv", "zeroshot_summary.json", "failures.jsonl"}
    assert (out / "failures.jsonl").read_text() == ""
    assert report["status"] == "complete" and report["threshold"] == 0
    assert report["checkpoint"]["sha256"] == bank.metadata["checkpoint_sha256"]
    assert report["prototype_bank"]["sha256"] == checkpoint_sha256(args.prototype_bank)
    assert report["manifest"]["sha256"] == checkpoint_sha256(manifest)
    assert report["input_mapping"]["label_mapping"] == {"bonafide": 0, "spoof": 1}
    predictions = read_csv(out / "zeroshot_predictions.csv")
    assert [row["attack_id"] for row in predictions] == ["bonafide", "A01", "-", "A02", "A02", "A02"]
    assert [float(row["score"]) for row in predictions] == pytest.approx([-2, 2, -(2 ** 0.5), 0, -2, 0])
    assert [row["prediction"] for row in predictions] == ["0", "1", "0", "1", "0", "1"]
    systems = read_csv(out / "zeroshot_per_attack.csv")
    assert [(row["attack_id"], row["real_count"], row["fake_count"]) for row in systems] == [
        ("A01", "2", "1"), ("A02", "2", "3")]
    assert float(systems[0]["eer"]) == 0
    assert float(systems[1]["eer"]) > 0
    summary = json.loads((out / "zeroshot_summary.json").read_text(encoding="utf-8"))
    assert summary["macro"]["eer"] == pytest.approx(float(systems[1]["eer"]) / 2)
    assert summary["macro"]["acc"] == pytest.approx((1 + 4 / 5) / 2)

    with pytest.raises(FileExistsError):
        cli.run(args)
    assert len(reads) == 6


def test_scores_independent_of_changed_labels_and_systems(evaluation, cli):
    args, _, manifest, reads, _ = evaluation
    cli.run(args)
    first = read_csv(Path(args.output_dir) / "zeroshot_predictions.csv")
    protocol = manifest.read_text().replace("bonafide bonafide", "B99 spoof").replace("A01 spoof", "- bonafide")
    manifest.write_text(protocol, encoding="utf-8")
    args.output_dir += "_relabeled"
    cli.run(args)
    second = read_csv(Path(args.output_dir) / "zeroshot_predictions.csv")
    assert len(reads) == 12
    assert [row["label"] for row in first] != [row["label"] for row in second]
    assert [(row["uttid"], row["score"]) for row in first] == [(row["uttid"], row["score"]) for row in second]


def test_no_real_reports_invalid_attacks_and_failed_run(evaluation, cli):
    args, _, manifest, reads, _ = evaluation
    manifest.write_text("\n".join(line for line in manifest.read_text().splitlines() if "spoof" in line))
    with pytest.raises(ValueError, match="No Attack/System has both"):
        cli.run(args)
    out = Path(args.output_dir)
    assert len(reads) == 4
    assert [row["status"] for row in read_csv(out / "zeroshot_per_attack.csv")] == ["invalid", "invalid"]
    assert json.loads((out / "zeroshot_run.json").read_text())["status"] == "failed"
    assert not (out / "zeroshot_summary.json").exists()
    assert len((out / "failures.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("problem,expected", [
    ("checkpoint", "Checkpoint SHA256"), ("old_quota", "current BD four-source Real quotas"),
    ("class_order", "classes in order"), ("shape", "shape"), ("nan", "finite"),
    ("duplicates", "duplicate sample IDs"),
])
def test_invalid_bank_fails_before_audio(evaluation, cli, problem, expected):
    args, bank, _, reads, _ = evaluation
    if problem == "checkpoint":
        bank.metadata["checkpoint_sha256"] = "wrong"
    elif problem == "old_quota":
        bank.metadata["real_quotas"] = {
            "AISHELL-1": 205, "AISHELL-3": 205, "CommonVoice": 205, "LibriTTS": 205, "VCTK": 204,
        }
    elif problem == "class_order":
        bank.classes[-2:] = bank.classes[-2:][::-1]
    elif problem == "shape":
        bank.dispersion = bank.dispersion[:, :1]
    elif problem == "nan":
        bank.prototypes[0, 0, 0] = float("nan")
    elif problem == "duplicates":
        bank.metadata["sample_ids"][bank.classes[1]][0] = bank.metadata["sample_ids"]["real"][0]
    bank.save(args.prototype_bank)
    with pytest.raises(ValueError, match=expected):
        cli.run(args)
    assert reads == []
    out = Path(args.output_dir)
    assert json.loads((out / "zeroshot_run.json").read_text())["status"] == "failed"
    assert json.loads((out / "failures.jsonl").read_text())["error_type"] == "ValueError"
