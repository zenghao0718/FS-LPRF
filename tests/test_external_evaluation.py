"""Temporary inputs and synthetic backends only; no formal external experiment."""
import copy
import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from data.audio import FADTrim, get_audio
from data.evaluation_manifest import DEV_SEEDS, load_or_create_manifest
from data.external_dataset import ExternalDataset
from data.external_manifest import load_label_mapping, read_manifest, validate_label_mapping
from data.loaders import ReusableAudioLoader
from data.randomness import sample_rng
from engine.evaluator import build_support_tasks, evaluate, extract_support_features, feature_batches
from engine.external_protocol import labeled_protocol, parse_external_seeds, unlabeled_protocol
from methods.inference import kshot_scores, kshot_task, predict
from utils.metrics import binary_metrics
from test_evaluation import FeatureDataset, FeatureModel


def write_manifest(path, rows, fields=("uttid", "audio_path", "label", "attack_id")):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writeheader()
        writer.writerows(rows)
    return path


def metadata(tmp_path, classes=("real", "A01", "A02"), count=65):
    rows = []
    for name in classes:
        for i in range(count):
            path = tmp_path / f"{name}-{i}.wav"
            path.touch()
            rows.append(dict(uttid=f"{name}-{i}", audio_path=str(path), label=int(name != "real"),
                             raw_label="bonafide" if name == "real" else "spoof",
                             attack_id="" if name == "real" else name))
    return rows


@pytest.mark.parametrize("mapping,raw,expected", [
    ({"bonafide": 0, "spoof": 1}, "spoof", 1),
    ({"real": 0, "fake": 1}, "real", 0),
    ({"genuine": 0, "synthetic": 1}, "synthetic", 1),
    ({"0": 0, "1": 1}, "1", 1), ({"1": 0, "0": 1}, "1", 0)])
@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_explicit_mapping_and_relative_paths(tmp_path, wav, mapping, raw, expected, suffix):
    path = write_manifest(tmp_path / f"input{suffix}", [dict(uttid="a", audio_path=wav.name, label=raw, attack_id="A01")])
    records = read_manifest(path, mapping)
    assert records[0]["label"] == expected and records[0]["raw_label"] == raw
    assert records[0]["audio_path"] == str(wav.resolve())


@pytest.mark.parametrize("mapping", [{}, {"r": 0}, {0: 0, 1: 1}, {"r": False, "f": True},
                                      {"r": 0, "f": 2}, {" r": 0, "f": 1}, {"r": 0.0, "f": 1}])
def test_invalid_maps_rejected(mapping):
    with pytest.raises(ValueError):
        validate_label_mapping(mapping)


@pytest.mark.parametrize("text", ["label_mapping:\n  r: 0\n  f: 1\n  r: 1\n",
                                       '{"r": 0, "f": 1, "r": 1}',
                                       'label_mapping:\n  0: 0\n  1: 1\n'])
def test_duplicate_or_unquoted_numeric_mapping_rejected(tmp_path, text):
    path = tmp_path / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_label_mapping(str(path))


def test_yaml_mapping_and_presets(tmp_path):
    path = tmp_path / "labels.yaml"
    path.write_text('label_mapping:\n  "1": 0\n  "0": 1\n', encoding="utf-8")
    assert load_label_mapping(str(path)) == {"1": 0, "0": 1}
    assert load_label_mapping("bonafide-spoof") == {"bonafide": 0, "spoof": 1}


@pytest.mark.parametrize("raw", ["unknown", "", "0", "1", " real"])
def test_unknown_missing_numeric_labels_not_guessed(tmp_path, wav, raw):
    path = write_manifest(tmp_path / "input.csv", [dict(uttid="a", audio_path=str(wav), label=raw, attack_id="")])
    with pytest.raises(ValueError):
        read_manifest(path, {"real": 0, "fake": 1})


@pytest.mark.parametrize("case", ["missing_label_column", "duplicate_id", "duplicate_path", "missing_file", "extra_cell", "duplicate_header"])
def test_manifest_validation(tmp_path, wav, case):
    path = tmp_path / "bad.csv"
    path.write_text(f"uttid,audio_path,label\na,{wav},real\n", encoding="utf-8")
    if case == "missing_label_column":
        path.write_text(f"uttid,audio_path\na,{wav}\n", encoding="utf-8")
    elif case == "duplicate_header":
        path.write_text(f"uttid,audio_path,label,label\na,{wav},real,real\n", encoding="utf-8")
    elif case == "extra_cell":
        path.write_text(f"uttid,audio_path,label\na,{wav},real,unexpected\n", encoding="utf-8")
    elif case == "missing_file":
        path.write_text("uttid,audio_path,label\na,missing.wav,real\n", encoding="utf-8")
    else:
        other = tmp_path / "other.wav"
        other.touch()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{'a' if case == 'duplicate_id' else 'b'},{other if case == 'duplicate_id' else wav},fake\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        read_manifest(path, {"real": 0, "fake": 1})


@pytest.mark.parametrize("group_by,classes", [("none", ("real", "A01")), ("none", ("real", "A01", "A02")),
                                              ("attack_id", ("real", "A01")), ("attack_id", ("real", "A01", "A02"))])
def test_labeled_protocol_uses_existing_sampler(tmp_path, group_by, classes):
    ds = ExternalDataset(metadata(tmp_path, classes), group_by=group_by)
    first = labeled_protocol(ds, tmp_path / "episodes.json")
    legacy = load_or_create_manifest(ds, tmp_path / "legacy.json", 10, DEV_SEEDS)
    assert first == legacy == labeled_protocol(ds, tmp_path / "episodes.json")
    assert first == labeled_protocol(ds, tmp_path / "repeat.json")
    assert first["seeds"] == [17, 42, 71, 100, 123]
    assert len(ds.fake_classes) == (len(classes) - 1 if group_by == "attack_id" else 1)
    for name in first["classes"]:
        union = set()
        for seed in first["seeds"]:
            ids = first["supports"][str(seed)][name]
            assert len(ids) == len(set(ids)) == 10
            union.update(ids)
        assert not union & set(first["queries"][name])
        assert union | set(first["queries"][name]) == {ds.uttids[i] for i in ds.class_indices[name]}


@pytest.mark.parametrize("count", [9, 10])
def test_labeled_shortages(tmp_path, count):
    ds = ExternalDataset(metadata(tmp_path, count=count))
    with pytest.raises(ValueError, match="Support candidates|no Query remains"):
        labeled_protocol(ds, tmp_path / "episodes.json")


def test_attack_id_required_only_for_grouped_fake(tmp_path):
    rows = metadata(tmp_path, ("real", "A01"))
    for row in rows:
        row["attack_id"] = ""
    ExternalDataset(rows)
    with pytest.raises(ValueError, match="attack_id"):
        ExternalDataset(rows, group_by="attack_id")


def unlabeled_data(tmp_path, real_count=10, fake_count=10):
    rows = metadata(tmp_path, ("real",), real_count) + metadata(tmp_path, ("fake",), fake_count)
    query = tmp_path / "query.wav"
    query.touch()
    rows.append(dict(uttid="query", audio_path=str(query), label=None, raw_label="", attack_id=""))
    ds = ExternalDataset(rows)
    supports = {name: [row["uttid"] for row in rows if row["label"] == label]
                for name, label in (("real", 0), ("fake", 1))}
    return ds, supports


@pytest.mark.parametrize("real_count,fake_count", [(0, 10), (9, 10), (10, 11)])
def test_unlabeled_support_count_rejected(tmp_path, real_count, fake_count):
    ds, supports = unlabeled_data(tmp_path, real_count, fake_count)
    with pytest.raises(ValueError, match="10-shot"):
        unlabeled_protocol(ds, supports, ["query"], tmp_path / "ep.json")


def test_unlabeled_no_fabricated_labels_or_metrics(tmp_path, config):
    ds, supports = unlabeled_data(tmp_path)
    manifest = unlabeled_protocol(ds, supports, ["query"], tmp_path / "ep.json")
    assert ds.by_id["query"]["label"] is None
    model = FeatureModel().train()
    before = copy.deepcopy(model.state_dict())
    result = evaluate(model, FeatureDataset(ds), manifest, config, tmp_path, "external")
    assert result == {"prediction_count": 1} and model.training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    with (tmp_path / "external_predictions.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert not {"label", "eer", "auc", "ap", "acc"} & set(reader.fieldnames)
        row = next(reader)
    assert row["prediction"] == ("fake" if float(row["score"]) >= 0 else "real")
    assert not list(tmp_path.glob("*macro*")) and not list(tmp_path.glob("*per_class*"))


def test_unlabeled_overlap_and_hidden_labels_rejected(tmp_path):
    ds, supports = unlabeled_data(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        unlabeled_protocol(ds, supports, [supports["real"][0]], tmp_path / "ep.json")
    ds.by_id["query"]["label"] = 1
    with pytest.raises(ValueError, match="must not have labels"):
        unlabeled_protocol(ds, supports, ["query"], tmp_path / "ep.json")


@pytest.mark.parametrize("group_by", ["none", "attack_id"])
def test_scores_prototypes_alpha_metrics_and_loader_reuse(tmp_path, config, monkeypatch, group_by):
    ds = FeatureDataset(ExternalDataset(metadata(tmp_path), group_by=group_by))
    manifest = labeled_protocol(ds, tmp_path / "ep.json")
    model = FeatureModel().eval()
    requests = []
    original = ReusableAudioLoader.for_indices
    def track(owner, indices):
        tokens = list(indices)
        requests.append((id(owner), owner.seed, [owner.dataset.uttids[token[0]] for token in tokens]))
        return original(owner, tokens)
    monkeypatch.setattr(ReusableAudioLoader, "for_indices", track)
    result = evaluate(model, ds, manifest, config, tmp_path, "external")
    for seed in DEV_SEEDS:
        streams = [(owner, ids) for owner, value, ids in requests if value == seed]
        assert len({owner for owner, ids in streams}) == 1
        assert len(streams) == 2 + len(ds.fake_classes)
        assert streams[1][1] == manifest["queries"]["real"]
    saved = list(csv.DictReader((tmp_path / "external_predictions.csv").open()))
    for seed in DEV_SEEDS:
        with torch.no_grad(), ReusableAudioLoader(ds, config, training=False, seed=seed) as loader:
            features = extract_support_features(model, ds, manifest["supports"][str(seed)], config, seed, loader=loader)
            tasks = build_support_tasks(model, features, ds.fake_classes)
            for name in ds.fake_classes:
                support = torch.cat([features["real"], features[name]])
                expected = kshot_task(support, torch.tensor([0] * 10 + [1] * 10))
                for old, new in zip(expected, tasks[name]):
                    assert torch.equal(old, new)
                rows = [row for row in saved if row["seed"] == str(seed) and row["fake_class"] == name]
                ids = [row["uttid"] for row in rows]
                # Preserve production class boundaries/microbatches for bitwise scoring.
                expected_scores = []
                for class_name in ("real", name):
                    for _, query in feature_batches(model, ds, manifest["queries"][class_name], config, seed, loader=loader):
                        expected_scores.extend(kshot_scores(query, *expected).tolist())
                assert ids == manifest["queries"]["real"] + manifest["queries"][name]
                assert np.array_equal(np.array(expected_scores), np.array([float(row["score"]) for row in rows]))
                labels = [int(row["label"]) for row in rows]
                metrics = binary_metrics(labels, expected_scores)
                actual = next(row for row in result["per_class"] if row["seed"] == seed and row["fake_class"] == name)
                assert all(actual[key] == value for key, value in metrics.items())
    assert len(result["per_class_summary"]) == len(ds.fake_classes)


@pytest.mark.parametrize("rate,channels,fmt", [(16000, 1, "WAV"), (48000, 1, "WAV"),
                                               (16000, 2, "WAV"), (48000, 2, "WAV"), (16000, 1, "FLAC")])
def test_audio_pipeline_matches_original_first_channel(tmp_path, rate, channels, fmt):
    t = np.arange(rate * 5, dtype=np.float32) / rate
    wave = .1 * np.sin(2 * np.pi * (130 * t + 17 * t**2))
    path = tmp_path / ("external.flac" if fmt == "FLAC" else "external.wav")
    mono = tmp_path / "mono.wav"
    subtype = "PCM_16" if fmt == "FLAC" else "FLOAT"
    sf.write(path, wave if channels == 1 else np.stack([wave, wave * .5], axis=1), rate, subtype=subtype)
    decoded, _ = sf.read(path, dtype="float32", always_2d=True)
    # Compare decoded samples: FLAC and PCM WAV encoders round differently.
    sf.write(mono, decoded[:, 0], rate, subtype="FLOAT")
    ds = ExternalDataset([dict(uttid="a", audio_path=str(path), label=None, raw_label="")])
    with sample_rng(71):
        expected = FADTrim.sample_segment(ds, get_audio(str(mono))[0])
    with sample_rng(71):
        actual = ds[0]["feats"]
    assert torch.equal(actual, expected)
    assert actual.shape == (1, 64000)


def test_bad_audio_identifies_input(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_text("not audio", encoding="utf-8")
    ds = ExternalDataset([dict(uttid="bad", audio_path=str(path), label=None)])
    with pytest.raises(RuntimeError, match="uttid=bad, path="):
        ds[0]


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_asvspoof5_conversion_official_released_layout(tmp_path, suffix, cli):
    from data.external_manifest import load_input_mapping
    for uid in ("E_0000000001", "E_0000000002"):
        (tmp_path / f"{uid}.flac").touch()
    protocol = tmp_path / "ASVspoof5.eval.track_1.tsv"
    protocol.write_text("E_0001 E_0000000001 F C05 2 E_1 AC1 bonafide bonafide -\n"
                        "E_0002 E_0000000002 M - - - AC2 A26 spoof -\n", encoding="utf-8")
    config = load_input_mapping(Path(__file__).resolve().parents[1] / "configs/external/asvspoof5.yaml")
    config["paths"]["root"] = str(tmp_path)
    rows = read_manifest(protocol, input_mapping=config)
    assert len(rows) == 2
    assert [row["label"] for row in rows] == [0, 1]
    assert [row["attack_id"] for row in rows] == ["bonafide", "A26"]
    assert rows[0]["metadata"]["codec"] == "C05" and rows[0]["metadata"]["speaker"] == "E_0001"
    # Normalized input export remains exclusive; the source protocol is untouched.
    out = tmp_path / f"asv{suffix}"
    cli.write_inputs(out, rows)
    exported = read_manifest(out, {"0": 0, "1": 1})
    assert [row["label"] for row in exported] == [0, 1]
    with pytest.raises(FileExistsError):
        cli.write_inputs(out, rows)
    protocol.write_text("E_0001 E_0000000001 F A26 spoof\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 10 columns"):
        read_manifest(protocol, input_mapping=config)


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("scripts.evaluate_external")


def fixture_checkpoint(path, model, config):
    from utils.provenance import UPSTREAM_COMMIT
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), method_version="2.4", upstream_commit=UPSTREAM_COMMIT,
                    config=config, step=800), path)
    return path


@pytest.mark.parametrize("mode", ["labeled", "unlabeled"])
def test_cli_old_checkpoint_reports_and_no_update(tmp_path, config, monkeypatch, cli, mode):
    model = FeatureModel()
    checkpoint = fixture_checkpoint(tmp_path / "old_training" / "checkpoints" / "step_0800.pt", model, config)
    before = checkpoint.read_bytes()
    out = tmp_path / "external"
    rows = metadata(tmp_path, ("real", "A01"))
    input_path = write_manifest(tmp_path / "input.csv", [dict(uttid=row["uttid"], audio_path=row["audio_path"],
                                     label=row["raw_label"], attack_id=row["attack_id"]) for row in rows])
    options = ["--mode", mode, "--checkpoint", str(checkpoint), "--output-dir", str(out), "--label-map", "bonafide-spoof"]
    if mode == "labeled":
        options += ["--manifest", str(input_path)]
    else:
        support = write_manifest(tmp_path / "support.csv", list(csv.DictReader(input_path.open(encoding="utf-8-sig")))[:10] +
                                 list(csv.DictReader(input_path.open(encoding="utf-8-sig")))[65:75])
        query = write_manifest(tmp_path / "query.csv", [dict(uttid="new", audio_path=rows[11]["audio_path"])], ("uttid", "audio_path"))
        options += ["--support-manifest", str(support), "--query-manifest", str(query)]
    args = cli.build_parser().parse_args(options)
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    actual = cli.ExternalDataset
    monkeypatch.setattr(cli, "ExternalDataset", lambda *a, **kw: FeatureDataset(actual(*a, **kw)))
    report = cli.run(args)
    assert report["status"] == "complete" and checkpoint.read_bytes() == before
    assert report["checkpoint_step"] == 800 and len(report["checkpoint_sha256"]) == 64
    assert report["label_mapping"] == {"bonafide": 0, "spoof": 1}
    assert report["manifest_sha256"] and report["code"]["source_sha256"]
    assert report["elapsed_seconds"] > 0 and not (out / "failures.jsonl").read_text()
    assert (out / "inputs.csv").exists() and (out / "episode_manifest.json").exists()
    if mode == "unlabeled":
        assert report["summary"] == {"prediction_count": 1}
    with pytest.raises(FileExistsError):
        cli.run(args)


def test_cli_records_audio_failure_and_protects_training(tmp_path, config, cli, monkeypatch):
    class AudioBackend(FeatureModel):
        def forward(self, source):
            raise AssertionError("Corrupt audio should fail before model forward")
    model = AudioBackend()
    checkpoint = fixture_checkpoint(tmp_path / "old" / "checkpoints" / "step_0800.pt", model, config)
    ds, supports = unlabeled_data(tmp_path)
    input_path = write_manifest(tmp_path / "support.csv", [dict(uttid=row["uttid"], audio_path=row["audio_path"],
                         label=row["raw_label"]) for row in ds.records[:-1]], ("uttid", "audio_path", "label"))
    query = write_manifest(tmp_path / "query.csv", [dict(uttid="query", audio_path=ds.records[-1]["audio_path"])], ("uttid", "audio_path"))
    args = cli.build_parser().parse_args(["--mode", "unlabeled", "--checkpoint", str(checkpoint),
                    "--output-dir", str(tmp_path / "failed"), "--label-map", "bonafide-spoof",
                    "--support-manifest", str(input_path), "--query-manifest", str(query)])
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    with pytest.raises(RuntimeError, match="Audio read failed"):
        cli.run(args)
    report = json.loads((tmp_path / "failed" / "external_run.json").read_text())
    assert report["status"] == "failed" and "uttid=real-" in report["failure"]["traceback"]
    assert (tmp_path / "failed" / "failures.jsonl").stat().st_size
    args.output_dir = str(tmp_path / "old" / "evaluation")
    with pytest.raises(ValueError, match="isolated"):
        cli.run(args)


def test_cli_missing_support_is_explicit(cli):
    args = cli.build_parser().parse_args(["--mode", "unlabeled", "--checkpoint", "x.pt", "--output-dir", "out", "--query-dir", "query"])
    with pytest.raises(ValueError, match="10-shot requires"):
        cli.prepare_inputs(args)


@pytest.mark.parametrize("workers", [0, 2])
def test_directory_audio_cli_smoke(tmp_path, config, monkeypatch, cli, workers):
    class WaveBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(8, 25 * 7)
            self.device = torch.device("cpu")

        def forward(self, source):
            return {"feats": self.projection(source["feats"][:, 0, ::8000]).reshape(-1, 25, 7)}

    model = WaveBackend()
    ckpt = fixture_checkpoint(tmp_path / "training" / "checkpoints" / "step_0800.pt", model, config)
    t = np.arange(32000, dtype=np.float32) / 16000
    dirs = [tmp_path / name for name in ("real", "fake", "query")]
    for directory, count in zip(dirs, (10, 10, 2)):
        directory.mkdir()
        for i in range(count):
            sf.write(directory / f"{i}.wav", .1 * np.sin(2 * np.pi * (130 + i) * t), 16000)
    args = cli.build_parser().parse_args(["--mode", "unlabeled", "--checkpoint", str(ckpt),
                  "--output-dir", str(tmp_path / "out"), "--real-support-dir", str(dirs[0]),
                  "--fake-support-dir", str(dirs[1]), "--query-dir", str(dirs[2]),
                  "--set", f"runtime.eval_num_workers={workers}"])
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    report = cli.run(args)
    assert report["summary"] == {"prediction_count": 2}
    assert report["label_mapping"] == {"real": 0, "fake": 1}
    inputs = list(csv.DictReader((tmp_path / "out" / "inputs.csv").open(encoding="utf-8")))
    assert sum(row["label"] == "0" for row in inputs) == 10
    assert sum(row["label"] == "1" for row in inputs) == 10
    assert sum(row["label"] == "" for row in inputs) == 2
    assert not list((tmp_path / "out").glob("*macro*"))


def test_directory_support_query_alias_rejected(tmp_path, cli):
    from data.external_manifest import directory_records
    directory = tmp_path / "same"
    directory.mkdir()
    (directory / "one.wav").touch()
    rows = directory_records(directory, "support/real", 0) + directory_records(directory, "query")
    with pytest.raises(ValueError, match="disjoint resolved audio paths"):
        ExternalDataset(rows)


def test_unlabeled_nonfinite_scores_raise(tmp_path, config):
    class BadModel(FeatureModel):
        def forward(self, source):
            result = super().forward(source)
            result["feats"].fill_(float("nan"))
            return result
    ds, supports = unlabeled_data(tmp_path)
    manifest = unlabeled_protocol(ds, supports, ["query"], tmp_path / "ep.json")
    with pytest.raises(ValueError, match="Non-finite Query scores"):
        evaluate(BadModel(), FeatureDataset(ds), manifest, config, tmp_path, "external")


def test_zero_threshold_unchanged():
    assert predict(torch.tensor([-1e-8, 0., 1e-8])).tolist() == [0, 1, 1]


@pytest.mark.parametrize("text", ["", "42,", "42,42", "0", "-1", "x", "042", "42.0"])
def test_external_seed_text_rejected(text):
    with pytest.raises(ValueError):
        parse_external_seeds(text)


def test_external_seed_text_accepts_development_and_paper_protocols():
    assert parse_external_seeds(None) == [42]
    assert parse_external_seeds("42") == [42]
    assert parse_external_seeds(" 17, 42,71,100,123 ") == DEV_SEEDS


def test_single_seed_manifest_is_stable_and_keeps_five_seed_support(tmp_path):
    ds = ExternalDataset(metadata(tmp_path))
    first = labeled_protocol(ds, tmp_path / "one.json", [42])
    second = labeled_protocol(ds, tmp_path / "other.json", [42])
    five = labeled_protocol(ds, tmp_path / "five.json")
    assert first["seeds"] == [42] == second["seeds"]
    assert first["supports"] == second["supports"] and first["queries"] == second["queries"]
    assert first["supports"]["42"] == five["supports"]["42"]
    assert first["queries"] != five["queries"]
    for name in first["classes"]:
        support = set(first["supports"]["42"][name])
        query = set(first["queries"][name])
        pool = {ds.uttids[i] for i in ds.class_indices[name]}
        assert len(support) == 10 and not support & query and support | query == pool


def test_single_seed_external_reports_point_metrics(tmp_path, config):
    ds = FeatureDataset(ExternalDataset(metadata(tmp_path)))
    manifest = labeled_protocol(ds, tmp_path / "ep.json", [42])
    result = evaluate(FeatureModel().eval(), ds, manifest, config, tmp_path / "single", "external")
    assert result["aggregation"] == "single_seed" and result["seed"] == 42
    assert set(("eer", "auc", "ap", "acc")).issubset(result)
    assert "mean_eer" not in result and "std_eer" not in result
    saved = list(csv.DictReader((tmp_path / "single" / "external_metrics.csv").open()))
    assert len(saved) == 1
    assert all(float(saved[0][name]) == result[name] for name in ("eer", "auc", "ap", "acc"))
    assert not (tmp_path / "single" / "external_macro_mean_std.csv").exists()
    assert not (tmp_path / "single" / "external_per_class_mean_std.csv").exists()
    five = labeled_protocol(ds, tmp_path / "five.json")
    five_result = evaluate(FeatureModel().eval(), ds, five, config, tmp_path / "five", "external")
    assert "mean_eer" in five_result and "std_eer" in five_result
    assert (tmp_path / "five" / "external_macro_mean_std.csv").is_file()


def test_cli_default_seed_reused_across_output_directories(tmp_path, config, monkeypatch, cli):
    model = FeatureModel()
    checkpoint = fixture_checkpoint(tmp_path / "training" / "checkpoints" / "step_0800.pt", model, config)
    rows = metadata(tmp_path, ("real", "A01"))
    source = write_manifest(tmp_path / "input.csv", [dict(uttid=row["uttid"], audio_path=row["audio_path"],
                             label=row["raw_label"], attack_id=row["attack_id"]) for row in rows])
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    actual = cli.ExternalDataset
    monkeypatch.setattr(cli, "ExternalDataset", lambda *a, **kw: FeatureDataset(actual(*a, **kw)))
    reports = []
    for name in ("ckpt_a", "ckpt_b"):
        args = cli.build_parser().parse_args(["--mode", "labeled", "--manifest", str(source),
                        "--label-map", "bonafide-spoof", "--checkpoint", str(checkpoint),
                        "--output-dir", str(tmp_path / name), "--set", "runtime.eval_num_workers=0"])
        assert args.seeds == "42"
        reports.append(cli.run(args))
    assert all(report["seeds"] == [42] and report["aggregation"] == "single_seed" and not report["official_five_seed"]
               for report in reports)
    manifests = [json.loads((tmp_path / name / "episode_manifest.json").read_text()) for name in ("ckpt_a", "ckpt_b")]
    assert manifests[0]["supports"] == manifests[1]["supports"]
    assert manifests[0]["queries"] == manifests[1]["queries"]
    scores = []
    for name in ("ckpt_a", "ckpt_b"):
        rows = list(csv.DictReader((tmp_path / name / "external_predictions.csv").open()))
        scores.append([(row["uttid"], row["score"]) for row in rows])
    assert scores[0] == scores[1]
    with pytest.raises(ValueError, match="labeled sampling"):
        cli.run(cli.build_parser().parse_args(["--mode", "unlabeled", "--checkpoint", str(checkpoint),
                "--output-dir", str(tmp_path / "bad"), "--seeds", "17,42", "--query-dir", "q"]))


def test_speechfake_matches_frozen_pre_refactor_evaluator(tmp_path, config):
    """Golden baseline came from Git a56b04e, not the current evaluator."""
    baseline = json.loads((Path(__file__).parent / "fixtures/evaluator_pre_external.json").read_text())
    names = ["real"] + [f"TTS__g{i:02d}" for i in range(32)]
    ids = [f"test/{name}/{i}" for name in names for i in range(65)]
    infos = {uid: dict(origin_ds="fixture", speaker="fixture", attacker=uid.split('/')[1],
                       label=int(uid.split('/')[1] != "real")) for uid in ids}
    meta = SimpleNamespace(uttids=ids, uttid2idx={uid: i for i, uid in enumerate(ids)},
              uttid2info=infos, uttid2path={uid: f"fixture/{uid}.wav" for uid in ids},
              class_indices={name: list(range(i*65, (i+1)*65)) for i, name in enumerate(names)},
              fake_classes=names[1:], split="test", do_augment=False, cache_mode="none")
    ds = FeatureDataset(meta)
    torch.manual_seed(777)
    model = FeatureModel()
    manifest = load_or_create_manifest(ds, tmp_path / "ep.json", 10, DEV_SEEDS)
    result = evaluate(model, ds, manifest, config, tmp_path, "test")
    assert result == baseline["summary"]
    with (tmp_path / "test_predictions.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == baseline["prediction_fields"]
        rows = {(row["seed"], row["fake_class"], row["uttid"]): row for row in reader}
    for expected in baseline["predictions"]:
        actual = rows[(expected["seed"], expected["fake_class"], expected["uttid"])]
        assert float(actual.pop("score")) == pytest.approx(float(expected["score"]), abs=2e-6)
        assert actual == {key: value for key, value in expected.items() if key != "score"}
