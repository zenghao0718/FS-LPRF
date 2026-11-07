"""Generic external 10-shot evaluation; all scoring lives in engine.evaluator."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess
import time
import traceback

import _bootstrap
from data.external_dataset import ExternalDataset
from data.external_manifest import (directory_records, load_input_mapping, load_label_mapping,
                                    read_manifest, resolve_input_mapping)
from engine.checkpoint import checkpoint_sha256, load_checkpoint, resolve_checkpoint
from engine.evaluator import evaluate
from data.evaluation_manifest import DEV_SEEDS
from engine.external_protocol import labeled_protocol, parse_external_seeds, unlabeled_protocol
from models.fs_lprf_model import FSLPRF
from utils.config import common_parser, load_config

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = common_parser(__doc__)
    parser.add_argument("--mode", required=True, choices=["labeled", "unlabeled"])
    parser.add_argument("--manifest")
    parser.add_argument("--mapping", help="YAML/JSON input format, columns, paths and labels (labeled/Support manifest)")
    parser.add_argument("--query-mapping", help="Optional separate input mapping for --query-manifest")
    parser.add_argument("--label-map", help="YAML/JSON path or bonafide-spoof / real-fake / genuine-synthetic")
    parser.add_argument("--group-by", choices=["none", "attack_id"], default="none")
    parser.add_argument("--support-manifest")
    parser.add_argument("--query-manifest")
    parser.add_argument("--real-support-dir")
    parser.add_argument("--fake-support-dir")
    parser.add_argument("--query-dir")
    parser.add_argument("--checkpoint", required=True, help="Explicit checkpoint FILE; never implicit 'best'")
    parser.add_argument("--output-dir", required=True, help="New, separate evaluation directory (must not exist)")
    parser.add_argument("--shot", type=int, choices=[10], default=10)
    parser.add_argument("--seeds", default="42",
                        help="Labeled Support seeds. Development default: 42. "
                             "Official five-seed paper run: 17,42,71,100,123")
    return parser


def prepare_inputs(args, *, mapping_report=None):
    mapping = load_label_mapping(args.label_map) if args.label_map else None
    schema = load_input_mapping(args.mapping) if args.mapping else {}
    query_schema = load_input_mapping(args.query_mapping) if args.query_mapping else {}
    sources = {}
    mapping_report = {} if mapping_report is None else mapping_report

    def read(path, *, labeled, role, input_schema, schema_path):
        source = Path(path).resolve()
        sources[str(source)] = checkpoint_sha256(source)
        options = resolve_input_mapping(source, mapping if labeled else None, labeled=labeled,
                                        input_mapping=input_schema)
        details = dict(manifest=str(source), options=options)
        if schema_path:
            details.update(mapping_file=str(Path(schema_path).resolve()), mapping_sha256=checkpoint_sha256(schema_path))
        if labeled and args.label_map:
            details["label_map_reference"] = args.label_map
            if Path(args.label_map).is_file():
                details["label_map_sha256"] = checkpoint_sha256(args.label_map)
        mapping_report[role] = details
        return read_manifest(source, labeled=labeled, input_mapping=options), options.get("label_mapping")

    if args.mode == "labeled":
        if not args.manifest:
            raise ValueError("Labeled mode requires --manifest and explicit labels in --mapping or --label-map")
        if any((args.support_manifest, args.query_manifest, args.query_mapping, args.real_support_dir, args.fake_support_dir, args.query_dir)):
            raise ValueError("Labeled mode samples Support automatically; do not specify Support/Query inputs")
        rows, mapping = read(args.manifest, labeled=True, role="labeled", input_schema=schema, schema_path=args.mapping)
        return rows, mapping, sources, None, None
    if args.manifest or args.group_by != "none":
        raise ValueError("Unlabeled Query requires one Real/Fake task; --manifest/attack_id grouping are not allowed")
    if args.support_manifest:
        if args.real_support_dir or args.fake_support_dir:
            raise ValueError("Support manifest cannot be mixed with Support directories")
        supports, mapping = read(args.support_manifest, labeled=True, role="support", input_schema=schema, schema_path=args.mapping)
    else:
        if not args.real_support_dir or not args.fake_support_dir:
            raise ValueError("10-shot requires 10 labeled Real and 10 labeled Fake Support; no 0-shot fallback")
        if mapping is not None or args.mapping:
            raise ValueError("--mapping/--label-map apply to manifests; directory roles already define Real/Fake")
        supports = (directory_records(args.real_support_dir, "support/real", 0) +
                    directory_records(args.fake_support_dir, "support/fake", 1))
        mapping = {"real": 0, "fake": 1}
        mapping_report["support"] = dict(real_dir=str(Path(args.real_support_dir).resolve()),
                                         fake_dir=str(Path(args.fake_support_dir).resolve()), label_mapping=mapping)
    if bool(args.query_dir) == bool(args.query_manifest):
        raise ValueError("Specify exactly one --query-dir or --query-manifest")
    if args.query_mapping and not args.query_manifest:
        raise ValueError("--query-mapping requires --query-manifest")
    if args.query_manifest:
        queries, _ = read(args.query_manifest, labeled=False, role="query", input_schema=query_schema, schema_path=args.query_mapping)
    else:
        queries = directory_records(args.query_dir, "query")
        mapping_report["query"] = dict(query_dir=str(Path(args.query_dir).resolve()))
    ids = {name: [row["uttid"] for row in supports if row["label"] == label]
           for name, label in (("real", 0), ("fake", 1))}
    return supports + queries, mapping, sources, ids, [row["uttid"] for row in queries]


def git_identity():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    # Commit alone cannot identify local uncommitted code.
    hashes = {path.relative_to(ROOT).as_posix(): checkpoint_sha256(path)
              for folder in ("data", "engine", "methods", "models", "utils", "scripts")
              for path in sorted((ROOT / folder).glob("*.py"))}
    return dict(commit=git("rev-parse", "HEAD"), status=git("status", "--short"), source_sha256=hashes)


def write_inputs(path, records):
    fields = sorted({key for row in records for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
                          for key, value in row.items()} for row in records)


def run(args):
    if args.mode == "labeled":
        seeds = parse_external_seeds(args.seeds)
    elif args.seeds != "42":
        raise ValueError("Unlabeled evaluation uses the supplied Support once; --seeds applies only to labeled sampling")
    else:
        seeds = None
    started = time.perf_counter()
    cfg = load_config(args.config, args.set)
    if args.checkpoint == "best":
        raise ValueError("Pass the explicit Best checkpoint file from the desired training run")
    checkpoint = resolve_checkpoint(args.checkpoint, cfg["runtime"]["output_dir"])
    out = Path(args.output_dir).resolve()
    training_roots = [Path(cfg["runtime"]["output_dir"]).resolve(), checkpoint.parent]
    if checkpoint.parent.name == "checkpoints":
        training_roots.append(checkpoint.parent.parent)
    if any(out == parent or parent in out.parents for parent in training_roots):
        raise ValueError("External output must be isolated from training/checkpoint directories")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", mode=args.mode, started_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256(checkpoint),
                  config=cfg, arguments=vars(args), code=git_identity(), shot=10,
                  score="S_fake - S_real", threshold=0, input_kind="original_audio", cache="none")
    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        (out / "external_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        report["input_mappings"] = {}
        records, mapping, sources, supports, queries = prepare_inputs(args, mapping_report=report["input_mappings"])
        report.update(label_mapping=mapping, manifest_sha256=sources,
                      dataset_identity=str(Path(args.manifest).resolve()) if args.manifest else "user_support_query",
                      protocol=("attack_aware_10shot" if args.group_by == "attack_id" else "binary_10shot"))
        write_inputs(out / "inputs.csv", records)
        report["inputs_sha256"] = checkpoint_sha256(out / "inputs.csv")
        ds = ExternalDataset(records, group_by=args.group_by, identity=report["dataset_identity"])
        protocol_path = out / "episode_manifest.json"
        manifest = (labeled_protocol(ds, protocol_path, seeds) if args.mode == "labeled" else
                    unlabeled_protocol(ds, supports, queries, protocol_path))
        report.update(seeds=manifest["seeds"], episode_manifest_sha256=checkpoint_sha256(protocol_path),
                      official_five_seed=list(manifest["seeds"]) == list(DEV_SEEDS))
        if args.mode == "labeled":
            report["aggregation"] = "single_seed" if len(manifest["seeds"]) == 1 else "mean_std"
        seed_key = str(manifest["seeds"][0])
        manifest_summary = dict(
            event="manifest_ready", utterances=len(records), classes=list(ds.fake_classes),
            class_counts={name: len(indices) for name, indices in ds.class_indices.items()},
            seeds=manifest["seeds"], aggregation=report.get("aggregation", "unlabeled"),
            official_five_seed=report["official_five_seed"])
        if args.mode == "labeled":
            manifest_summary["support_per_class"] = {name: len(manifest["supports"][seed_key][name]) for name in manifest["classes"]}
            manifest_summary["query_per_class"] = {name: len(manifest["queries"][name]) for name in manifest["classes"]}
        else:
            manifest_summary["query_count"] = len(manifest["queries"]["query"])
        print(json.dumps(manifest_summary, ensure_ascii=False), flush=True)
        save()
        model = FSLPRF(cfg)
        payload = load_checkpoint(checkpoint, model)
        report["checkpoint_step"] = payload.get("step")
        del payload
        model.requires_grad_(False)
        model.eval()
        result = evaluate(model, ds, manifest, cfg, out, "external")
        report.update(status="complete", summary=result)
        return report
    except Exception as error:
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        report.update(status="failed", failure=failure)
        (out / "failures.jsonl").write_text(json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    finally:
        if not (out / "failures.jsonl").exists():
            (out / "failures.jsonl").touch()
        save()


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
