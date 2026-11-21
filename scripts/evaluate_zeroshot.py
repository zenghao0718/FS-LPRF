"""Score labeled ASVspoof5 queries against a fixed SpeechFake-BD reference bank."""
from contextlib import closing
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import _bootstrap
import torch

from data.external_dataset import ExternalDataset
from data.external_manifest import (load_input_mapping, load_label_mapping, read_manifest,
                                    resolve_input_mapping)
from engine.checkpoint import checkpoint_sha256, load_checkpoint, resolve_checkpoint
from engine.evaluator import feature_batches
from methods.inference import PrototypeBank, predict, zeroshot_scores
from methods.prototype_bank import REAL_QUOTAS, SAMPLES_PER_CLASS
from models.fs_lprf_model import FSLPRF
from scripts.evaluate_external import git_identity
from utils.config import common_parser, load_config
from utils.metrics import METRICS, binary_metrics, macro_metrics
from utils.provenance import XLSR_SHA256
from utils.seed import seed_all


def build_parser():
    parser = common_parser(__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mapping", help="YAML/JSON input mapping (e.g. configs/external/asvspoof5.yaml)")
    parser.add_argument("--label-map", help="Explicit label mapping file or named preset")
    parser.add_argument("--group-by", choices=["attack_id"], default="attack_id")
    parser.add_argument("--checkpoint", required=True, help="Explicit SpeechFake-BD Dev Best checkpoint file")
    parser.add_argument("--prototype-bank", required=True, help="Reference bank built with the same checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Query waveform view and loader seed")
    parser.add_argument("--output-dir", required=True, help="New evaluation directory (must not exist)")
    return parser


def validate_bank(bank, checkpoint_hash):
    classes = bank.classes
    if (len(classes) != 33 or classes[0] != "real" or
            any(not isinstance(name, str) or not name.strip() or name.strip() != name for name in classes) or
            len(set(classes)) != 33 or list(classes[1:]) != sorted(classes[1:])):
        raise ValueError("Expected bank classes in order: real, then 32 distinct sorted Fake class names")
    for name, shape in (("prototypes", (33, 25, 256)), ("dispersion", (33, 25))):
        tensor = getattr(bank, name)
        if (not torch.is_tensor(tensor) or tensor.shape != shape or
                not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"Bank {name} must be a finite floating-point tensor of shape {shape}")
    if bank.metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Prototype-bank Checkpoint SHA256 does not match the evaluation Checkpoint")
    if (bank.metadata.get("real_quotas") != REAL_QUOTAS or
            bank.metadata.get("samples_per_class") != SAMPLES_PER_CLASS):
        raise ValueError("Prototype bank must use the Appendix A Real quotas and 1024 samples per class")
    sample_ids = bank.metadata.get("sample_ids", {})
    if set(sample_ids) != set(classes) or any(len(ids) != SAMPLES_PER_CLASS for ids in sample_ids.values()):
        raise ValueError("Bank metadata must record the 1024 sample IDs for every class")
    if len({uid for ids in sample_ids.values() for uid in ids}) != len(classes) * SAMPLES_PER_CLASS:
        raise ValueError("Bank metadata contains duplicate sample IDs")
    if type(bank.metadata.get("seed")) is not int:
        raise ValueError("Bank metadata must record the integer sampling seed")
    alpha = bank.alpha
    if (alpha.shape != (25,) or not torch.isfinite(alpha).all() or (alpha < 0).any() or
            not torch.isclose(alpha.sum(), alpha.new_tensor(1.0))):
        raise ValueError("Bank must produce 25 finite, nonnegative layer weights summing to 1")
    return alpha


@torch.inference_mode()
def score_queries(model, records, bank, alpha, cfg, seed):
    # Pass only audio identity to the feature pipeline. Every row uses the same scorer.
    dataset = ExternalDataset([dict(uttid=row["uttid"], audio_path=row["audio_path"], label=None)
                               for row in records])
    bank = replace(bank, prototypes=bank.prototypes.to(model.device))
    alpha = alpha.to(model.device)
    scores = {}
    with closing(feature_batches(model, dataset, dataset.uttids, cfg, seed)) as batches:
        for ids, features in batches:
            if features.shape != (len(ids), 25, 256):
                raise ValueError("Expected one [25,256] feature per query ID")
            values = zeroshot_scores(features.float(), bank, alpha=alpha)
            if not torch.isfinite(values).all():
                raise ValueError(f"Non-finite scores in query batch beginning with {ids[0]}")
            for uid, score, prediction in zip(ids, values.cpu().tolist(), predict(values).cpu().tolist()):
                if uid not in dataset.by_id or uid in scores:
                    raise ValueError(f"Unexpected or repeated Query ID from feature loader: {uid}")
                scores[uid] = (score, prediction)
    if len(scores) != len(records):
        raise ValueError(f"Scored {len(scores)} of {len(records)} Query IDs")
    return scores


def attack_metrics(records, scores):
    real = []
    attacks = {}
    for row in records:
        score = scores[row["uttid"]][0]
        if row["label"] == 0:
            real.append(score)
        else:
            attack = row["attack_id"]
            if not attack or attack == "real":
                raise ValueError(f"Missing or invalid Fake Attack/System ID for {row['uttid']}")
            attacks.setdefault(attack, []).append(score)
    for attack, fake in sorted(attacks.items()):
        row = dict(attack_id=attack, real_count=len(real), fake_count=len(fake))
        if not real:
            yield dict(row, status="invalid", error="Metrics require both Real and Fake Query scores")
            continue
        metrics = binary_metrics([0] * len(real) + [1] * len(fake), real + fake)
        yield dict(row, status="complete", **metrics)


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=checkpoint_sha256(path))


def run(args):
    started = time.perf_counter()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", shot=0, seed=args.seed, threshold=0, group_by=args.group_by,
                  started_at=datetime.now(timezone.utc).isoformat(), arguments=vars(args))

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(out / "zeroshot_run.json", report)

    save()
    failures = out / "failures.jsonl"
    failures.touch()
    try:
        report.update(config_file=file_identity(args.config), code=git_identity())
        cfg = load_config(args.config, args.set)
        report["config"] = cfg
        if args.checkpoint == "best":
            raise ValueError("Pass the explicit Best checkpoint file from the desired training run")
        checkpoint = resolve_checkpoint(args.checkpoint, cfg["runtime"]["output_dir"])
        report["checkpoint"] = file_identity(checkpoint)
        report["frontend"] = dict(path=str(Path(cfg["model"]["frontend_path"]).resolve()), sha256=XLSR_SHA256)
        bank_path = Path(args.prototype_bank).resolve()
        report["prototype_bank"] = file_identity(bank_path)
        bank = PrototypeBank.load(bank_path)
        alpha = validate_bank(bank, report["checkpoint"]["sha256"])
        write_json(out / "prototype_bank_info.json", dict(
            **report["prototype_bank"], classes=bank.classes,
            prototype_shape=list(bank.prototypes.shape), dispersion_shape=list(bank.dispersion.shape),
            layer_weights=alpha.tolist(), metadata=bank.metadata))

        report["manifest"] = file_identity(args.manifest)
        schema = load_input_mapping(args.mapping) if args.mapping else None
        if args.mapping:
            report["mapping_file"] = file_identity(args.mapping)
        label_map = load_label_mapping(args.label_map) if args.label_map else None
        if args.label_map and Path(args.label_map).is_file():
            report["label_map_file"] = file_identity(args.label_map)
        options = resolve_input_mapping(args.manifest, label_map, input_mapping=schema)
        report["input_mapping"] = options
        records = read_manifest(args.manifest, input_mapping=options)
        for row in records:
            del row["metadata"]
        seed_all(args.seed)
        model = FSLPRF(cfg)
        report["checkpoint_step"] = load_checkpoint(checkpoint, model).get("step")
        model.requires_grad_(False).eval()
        save()
        scores = score_queries(model, records, bank, alpha, cfg, args.seed)

        write_csv(out / "zeroshot_predictions.csv",
                  ("uttid", "audio_path", "label", "raw_label", "attack_id", "score", "prediction"),
                  (dict(uttid=row["uttid"], audio_path=row["audio_path"], label=row["label"],
                        raw_label=row["raw_label"], attack_id=row["attack_id"],
                        score=scores[row["uttid"]][0], prediction=scores[row["uttid"]][1]) for row in records))
        rows = list(attack_metrics(records, scores))
        write_csv(out / "zeroshot_per_attack.csv",
                  ("attack_id", "real_count", "fake_count", "status", *METRICS, "eer_threshold", "error"), rows)
        valid = [row for row in rows if row["status"] == "complete"]
        invalid = [row for row in rows if row["status"] == "invalid"]
        if invalid:
            with failures.open("a", encoding="utf-8") as handle:
                for row in invalid:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not valid:
            raise ValueError("No Attack/System has both Real and Fake Query scores")
        real_count = sum(row["label"] == 0 for row in records)
        summary = dict(aggregation="equal_weight_attack_macro", macro=macro_metrics(valid), threshold=0,
                       query_count=len(records), real_count=real_count, fake_count=len(records) - real_count,
                       attack_count=len(rows), valid_attack_count=len(valid),
                       invalid_attacks=[row["attack_id"] for row in invalid])
        write_json(out / "zeroshot_summary.json", summary)
        report.update(status="complete", summary=summary)
        return report
    except (Exception, KeyboardInterrupt) as error:
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        report.update(status="failed", failure=failure)
        with failures.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise
    finally:
        save()


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
