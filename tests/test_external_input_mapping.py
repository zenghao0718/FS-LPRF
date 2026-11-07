"""New source layouts require configuration only, never new Dataset/Evaluator code."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from data.external_dataset import ExternalDataset
from data.external_manifest import load_input_mapping, read_manifest, validate_input_mapping
from data.evaluation_manifest import DEV_SEEDS
from engine.external_protocol import labeled_protocol
from test_evaluation import FeatureDataset, FeatureModel
from test_external_evaluation import cli, fixture_checkpoint


@pytest.fixture
def layouts(tmp_path):
    """The same accessible audio identities in three unrelated table schemas."""
    audio = tmp_path / "audio"
    audio.mkdir()
    rows = []
    wave = np.sin(np.arange(1024, dtype=np.float32) * .137) * .1
    for name, label in (("natural", 0), ("synth_red", 1), ("synth_blue", 1)):
        for i in range(65):
            uid = f"{name}_{i:03d}"
            path = audio / f"{uid}.flac"
            sf.write(path, wave, 16000)
            rows.append(dict(uttid=uid, audio_path=str(path.resolve()), label=label,
                             attack_id="bonafide" if label == 0 else name))
    a, b, c = tmp_path / "vendor.csv", tmp_path / "registry.tsv", tmp_path / "trials.protocol"
    with a.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "source_path", "truth", "generator", "speaker_name"])
        writer.writerows([row["uttid"], f"audio/{row['uttid']}.flac", "natural" if row["label"] == 0 else "generated",
                          row["attack_id"], "speaker"] for row in rows)
    with b.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["target", "location", "audio_id"])
        writer.writerows(["1" if row["label"] == 0 else "0", row["audio_path"], row["uttid"]] for row in rows)
    with c.open("w", encoding="utf-8") as handle:
        for row in rows:
            fields = ["speaker", row["uttid"], "F", "C05", "2", "source", "condition",
                      row["attack_id"], "bonafide" if row["label"] == 0 else "spoof", "-"]
            handle.write(" \t  ".join(fields) + "\n")
    schemas = [
        dict(format=dict(delimiter=",", header=True),
             columns=dict(uttid="clip", audio_path="source_path", label="truth", attack_id="generator"),
             labels=dict(real=["natural"], fake=["generated"]), metadata=dict(speaker="speaker_name")),
        dict(format=dict(delimiter="\t", header=True),
             columns=dict(uttid=2, audio_path="location", label=0), labels=dict(real=["1"], fake=["0"])),
        load_input_mapping(Path(__file__).resolve().parents[1] / "configs/external/asvspoof5.yaml")]
    schemas[2]["paths"]["root"] = "audio"
    result = []
    for kind, source, schema, group_by in zip(("csv_names", "tsv_indices", "whitespace"), (a, b, c), schemas,
                                             ("attack_id", "none", "attack_id")):
        config = tmp_path / f"{kind}.yaml"
        config.write_text(yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8")
        result.append(dict(kind=kind, source=source, mapping=config, group_by=group_by))
    return rows, result


def test_three_formats_read_the_same_audio_labels_and_fixed_protocol(layouts, tmp_path):
    expected, formats = layouts
    reference = [(row["uttid"], row["audio_path"], row["label"]) for row in expected]
    protocols = []
    for item in formats:
        options = load_input_mapping(item["mapping"])
        records = read_manifest(item["source"], input_mapping=options)
        assert [(row["uttid"], row["audio_path"], row["label"]) for row in records] == reference
        assert all(set(row) == {"uttid", "audio_path", "label", "raw_label", "attack_id", "metadata"} for row in records)
        ds = ExternalDataset(records, group_by=item["group_by"])
        protocol = labeled_protocol(ds, tmp_path / f"{item['kind']}.json")
        assert len(ds.fake_classes) == (1 if item["group_by"] == "none" else 2)
        assert protocol["seeds"] == DEV_SEEDS
        for name in protocol["classes"]:
            union = set()
            for seed in DEV_SEEDS:
                support = protocol["supports"][str(seed)][name]
                assert len(support) == len(set(support)) == 10
                union.update(support)
            assert not union & set(protocol["queries"][name])
        protocols.append(protocol)
    # Input names, delimiter and label strings do not alter task sampling.
    assert protocols[0]["supports"] == protocols[2]["supports"]
    assert protocols[0]["queries"] == protocols[2]["queries"]


@pytest.mark.parametrize("index", [0, 1, 2], ids=["csv_names", "tsv_indices", "mixed_whitespace"])
def test_three_formats_complete_same_cli_without_dataset_selection(layouts, tmp_path, config, monkeypatch, cli, index):
    _, formats = layouts
    item = formats[index]
    torch.manual_seed(777)
    model = FeatureModel()
    checkpoint = fixture_checkpoint(tmp_path / "training/checkpoints/step_0800.pt", model, config)
    checkpoint_bytes = checkpoint.read_bytes()
    args = cli.build_parser().parse_args(["--mode", "labeled", "--manifest", str(item["source"]),
                  "--mapping", str(item["mapping"]), "--group-by", item["group_by"],
                  "--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / f"result_{index}"),
                  "--set", "runtime.eval_num_workers=0"])
    actual = cli.ExternalDataset
    monkeypatch.setattr(cli, "ExternalDataset", lambda *a, **kw: FeatureDataset(actual(*a, **kw)))
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    report = cli.run(args)
    assert report["status"] == "complete" and report["seeds"] == DEV_SEEDS
    assert len(report["summary"]["per_class"]) == 5 * (1 if item["group_by"] == "none" else 2)
    assert checkpoint.read_bytes() == checkpoint_bytes
    details = report["input_mappings"]["labeled"]
    assert details["mapping_sha256"] == hashlib.sha256(item["mapping"].read_bytes()).hexdigest()
    assert details["options"]["label_mapping"] == report["label_mapping"]
    assert details["options"]["paths"]["root"]
    with (Path(args.output_dir) / "inputs.csv").open(encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    assert isinstance(json.loads(first["metadata"]), dict)
    if index == 1:
        assert report["label_mapping"] == {"1": 0, "0": 1}


def test_alternate_formats_have_identical_scores_with_identical_tasks(layouts, tmp_path, config):
    from engine.evaluator import evaluate
    _, formats = layouts
    torch.manual_seed(71)
    model = FeatureModel()
    predictions = []
    summaries = []
    for item in (formats[0], formats[2]):
        rows = read_manifest(item["source"], input_mapping=load_input_mapping(item["mapping"]))
        ds = FeatureDataset(ExternalDataset(rows, group_by="attack_id"))
        manifest = labeled_protocol(ds, tmp_path / f"{item['kind']}_episodes.json")
        out = tmp_path / item["kind"]
        summaries.append(evaluate(model, ds, manifest, config, out, "external"))
        with (out / "external_predictions.csv").open(encoding="utf-8") as handle:
            predictions.append([{key: row[key] for key in ("uttid", "fake_class", "seed", "label", "score", "prediction")}
                                for row in csv.DictReader(handle)])
    assert summaries[0] == summaries[1] and predictions[0] == predictions[1]


def test_unknown_dataset_path_template_and_metadata(tmp_path):
    directory = tmp_path / "recordings" / "speaker with spaces"
    directory.mkdir(parents=True)
    audio = directory / "clip-23.wav"
    audio.touch()
    source = tmp_path / "never_seen.table"
    source.write_text('sid;audio;tag;person\nrecord23;23;H;"speaker with spaces"\n', encoding="utf-8")
    options = dict(format=dict(delimiter=";", header=True),
                   columns=dict(uttid="sid", audio_path="audio", label="tag"),
                   paths=dict(root="recordings", template="{speaker}/clip-{value}.wav"),
                   metadata=dict(speaker="person"), labels=dict(real=["H"], fake=["AI"]))
    row = read_manifest(source, input_mapping=options)[0]
    assert row["audio_path"] == str(audio.resolve()) and row["label"] == 0
    assert row["metadata"]["speaker"] == "speaker with spaces"
    ds = ExternalDataset([row])
    assert ds.uttid2info["record23"]["speaker"] == "speaker with spaces"


@pytest.mark.parametrize("payload", [
    [], {1: "bad", "format": {}}, {"dataset": "special"}, {"format": []}, {"format": {"header": "false"}},
    {"format": {"delimiter": "::"}}, {"format": {"delimiter": "\n"}},
    {"format": {"column_count": 0}}, {"format": {"encoding": "not-an-encoding"}},
    {"columns": {"path": 1}}, {"columns": {"uttid": -1}}, {"columns": {"uttid": True}},
    {"format": {"header": False}, "columns": {"uttid": "name"}},
    {"paths": {"root": ""}}, {"paths": {"template": "{value.__class__}"}},
    {"paths": {"template": "{value[0]}"}}, {"paths": {"template": "{value!r}"}},
    {"labels": {"real": ["r"]}}, {"labels": {"real": ["same"], "fake": ["same"]}},
    {"labels": {"real": [1], "fake": [0]}},
    {"labels": {"real": ["r"], "fake": ["f"]}, "label_mapping": {"r": 0, "f": 1}},
    {"metadata": {"uttid": 0}}, {"metadata": {"speaker": -1}}
])
def test_wrong_input_configuration_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_input_mapping(payload)


@pytest.mark.parametrize("text", ["format:\n  header: true\n  header: false\n",
                                      "columns:\n  uttid: 0\n  uttid: 1\n"])
def test_duplicate_configuration_fields_rejected(tmp_path, text):
    path = tmp_path / "mapping.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_input_mapping(path)


@pytest.mark.parametrize("raw,labels,expected", [
    ("bonafide", {"real": ["bonafide"], "fake": ["spoof"]}, 0),
    ("fake", {"real": ["real"], "fake": ["fake"]}, 1),
    ("genuine", {"real": ["genuine"], "fake": ["synthetic"]}, 0),
    ("1", {"real": ["0"], "fake": ["1"]}, 1),
    ("1", {"real": ["1"], "fake": ["0"]}, 0)])
def test_configured_label_semantics_are_explicit(tmp_path, wav, raw, labels, expected):
    source = tmp_path / "data.csv"
    source.write_text(f"id,path,target\nx,{wav},{raw}\n", encoding="utf-8")
    options = dict(columns=dict(uttid="id", audio_path="path", label="target"), labels=labels)
    row = read_manifest(source, input_mapping=options)[0]
    assert row["label"] == expected and row["raw_label"] == raw


@pytest.mark.parametrize("case", ["unknown_label", "missing_label", "missing_column", "index_outside",
                                   "row_width", "no_header_mapping", "ambiguous_separator", "template_unknown", "duplicate_path"])
def test_mapping_does_not_guess_bad_input(tmp_path, wav, case):
    source = tmp_path / "input.csv"
    source.write_text(f"id,file,target\na,{wav},R\n", encoding="utf-8")
    options = dict(columns=dict(uttid="id", audio_path="file", label="target"), labels=dict(real=["R"], fake=["F"]))
    if case == "unknown_label":
        options["labels"]["real"] = ["different"]
    elif case == "missing_label":
        source.write_text(f"id,file,target\na,{wav},\n", encoding="utf-8")
    elif case == "missing_column":
        options["columns"]["label"] = "missing"
    elif case == "index_outside":
        options["columns"]["label"] = 5
    elif case == "row_width":
        with source.open("a", encoding="utf-8") as handle:
            handle.write("one,two\n")
    elif case == "no_header_mapping":
        options = dict(format=dict(header=False), labels=options["labels"])
    elif case == "ambiguous_separator":
        source = source.rename(tmp_path / "input.txt")
    elif case == "template_unknown":
        options["paths"] = dict(template="{missing}.wav")
    elif case == "duplicate_path":
        with source.open("a", encoding="utf-8") as handle:
            handle.write(f"b,{wav},F\n")
    with pytest.raises(ValueError):
        read_manifest(source, input_mapping=options)


def test_no_header_csv_and_whitespace_header(tmp_path, wav):
    csv_path = tmp_path / "plain.csv"
    csv_path.write_text(f"R,{wav.name},id\n", encoding="utf-8")
    row = read_manifest(csv_path, input_mapping=dict(format=dict(header=False),
                       columns=dict(uttid=2, audio_path=1, label=0), labels=dict(real=["R"], fake=["F"])))[0]
    assert row["uttid"] == "id" and row["label"] == 0
    ws = tmp_path / "header.txt"
    ws.write_text(f"key\t file  truth\nx\t {wav.name} R\n", encoding="utf-8")
    row = read_manifest(ws, input_mapping=dict(format=dict(delimiter="whitespace", header=True),
                       columns=dict(uttid="key", audio_path="file", label="truth"),
                       labels=dict(real=["R"], fake=["F"])))[0]
    assert row["uttid"] == "x"


def test_unlabeled_support_and_query_can_use_different_schemas(tmp_path, config, monkeypatch, cli):
    rows = []
    for label in ("H", "AI"):
        for i in range(10):
            path = tmp_path / f"{label}_{i}.wav"
            path.touch()
            rows.append([path.name, label, f"{label}{i}"])
    support = tmp_path / "support.table"
    support.write_text("\n".join(";".join(row) for row in rows) + "\n", encoding="utf-8")
    support_map = tmp_path / "support.yaml"
    support_map.write_text(yaml.safe_dump(dict(format=dict(delimiter=";", header=False),
                           columns=dict(uttid=2, audio_path=0, label=1), labels=dict(real=["H"], fake=["AI"]))), encoding="utf-8")
    audio = tmp_path / "unknown.wav"
    audio.touch()
    query = tmp_path / "query.tsv"
    query.write_text(f"location\tidentity\n{audio.name}\tunknown\n", encoding="utf-8")
    query_map = tmp_path / "query.yaml"
    query_map.write_text(yaml.safe_dump(dict(columns=dict(uttid="identity", audio_path="location"))), encoding="utf-8")
    model = FeatureModel()
    checkpoint = fixture_checkpoint(tmp_path / "training/checkpoints/step_0800.pt", model, config)
    args = cli.build_parser().parse_args(["--mode", "unlabeled", "--support-manifest", str(support),
                        "--mapping", str(support_map), "--query-manifest", str(query), "--query-mapping", str(query_map),
                        "--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "result"),
                        "--set", "runtime.eval_num_workers=0"])
    actual = cli.ExternalDataset
    monkeypatch.setattr(cli, "ExternalDataset", lambda *a, **kw: FeatureDataset(actual(*a, **kw)))
    monkeypatch.setattr(cli, "FSLPRF", lambda cfg: model)
    report = cli.run(args)
    assert report["summary"] == {"prediction_count": 1}
    assert set(report["input_mappings"]) == {"support", "query"}
    assert report["label_mapping"] == {"H": 0, "AI": 1}
    with (tmp_path / "result/external_predictions.csv").open() as handle:
        assert "label" not in csv.DictReader(handle).fieldnames


def test_custom_query_labels_cannot_be_silently_used(tmp_path, wav):
    path = tmp_path / "query.csv"
    path.write_text(f"id,file,target\nq,{wav},F\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not provide labels"):
        read_manifest(path, labeled=False, input_mapping=dict(columns=dict(uttid="id", audio_path="file", label="target")))


def test_cli_rejects_competing_label_definitions(layouts, tmp_path, cli):
    _, items = layouts
    args = cli.build_parser().parse_args(["--mode", "labeled", "--manifest", str(items[0]["source"]),
                     "--mapping", str(items[0]["mapping"]), "--label-map", "bonafide-spoof",
                     "--checkpoint", "unused.pt", "--output-dir", str(tmp_path / "unused")])
    with pytest.raises(ValueError, match="Conflicting"):
        cli.prepare_inputs(args)


def test_json_mapping_and_explicit_text_encoding(tmp_path, wav):
    source = tmp_path / "non_utf8.table"
    source.write_text(f"编号;文件;真值\n示例;{wav.name};真人\n", encoding="utf-16")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(dict(format=dict(delimiter=";", header=True, encoding="utf-16"),
                       columns=dict(uttid="编号", audio_path="文件", label="真值"),
                       labels=dict(real=["真人"], fake=["合成"]))), encoding="utf-8")
    row = read_manifest(source, input_mapping=load_input_mapping(mapping))[0]
    assert row["uttid"] == "示例" and row["label"] == 0


def test_unhashable_mapping_keys_have_clear_error(tmp_path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("? [format, columns]\n: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be scalar"):
        load_input_mapping(mapping)
