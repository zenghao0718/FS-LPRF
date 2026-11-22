from pathlib import Path
import json
import _bootstrap
from utils.config import common_parser, load_config
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import open_waveform_cache
from data.cache_pipeline import open_train_dataset
from data.loaders import worker_options
from models.fs_lprf_model import FSLPRF
from engine.checkpoint import load_checkpoint, resolve_checkpoint
from methods.inference import PrototypeBank
from methods.prototype_bank import (AUGMENTATION_MODES, build_prototype_bank, dataset_config_for_mode,
                                    select_bank_samples)
from methods.prototype_provenance import write_prototype_info
from scripts.check_codec_assignment import write_manifest
from scripts.evaluate_zeroshot import validate_bank


def build_parser():
    parser = common_parser("Build the offline 33-class reference bank for 0-shot evaluation")
    parser.add_argument("--checkpoint", required=True, help="best, or the already selected Best node path")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    parser.add_argument("--augmentation-mode", choices=AUGMENTATION_MODES, default="clean")
    parser.add_argument("--reference-bank", help="Clean bank whose sample IDs are reused for codec50/full")
    parser.add_argument("--expect-checkpoint-sha256")
    parser.add_argument("--runtime-report", help="Write the resolved DataLoader options before any GPU work")
    return parser


def require_eval_runtime(cfg, path=None):
    options = worker_options(cfg, training=False)
    report = dict(worker_options=options, audio_microbatch_size=cfg["train"]["audio_microbatch_size"],
                  legacy_num_workers=cfg["runtime"].get("num_workers"), training=False,
                  note="num_workers counts DataLoader CPU workers, not GPU processes")
    print("RUNTIME " + json.dumps(report, sort_keys=True), flush=True)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = dict(num_workers=8, persistent_workers=True, prefetch_factor=2, pin_memory=True,
                    worker_torch_num_threads=1)
    if any(options.get(key) != value for key, value in expected.items()) or report["audio_microbatch_size"] != 32:
        raise SystemExit("Refusing to start: expected 8 persistent DataLoader workers and audio_microbatch_size=32")
    if report["legacy_num_workers"] not in (None, 8):
        raise SystemExit("runtime.num_workers is set and is not 8; refusing to ignore it")
    return report


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg["runtime"]["output_dir"])
    if checkpoint != resolve_checkpoint("best", cfg["runtime"]["output_dir"]):
        raise ValueError("0-shot bank construction requires the already selected Best checkpoint")
    if args.expect_checkpoint_sha256:
        from engine.checkpoint import checkpoint_sha256
        digest = checkpoint_sha256(checkpoint)
        if digest != args.expect_checkpoint_sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: {digest}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prototype bank: {output}")
    if args.augmentation_mode == "clean":
        if args.reference_bank:
            raise ValueError("Clean construction samples its own IDs and does not take a reference bank")
        # The bank remains unaugmented; use the same full Train waveforms after WAV removal.
        dataset = SpeechFakeDataset(cfg["data"], "train", train_mode=False)
        select_bank_samples(dataset, args.seed)  # Report metadata shortages before opening caches or the model.
        if cfg["data"]["train_cache_mode"] == "waveform":
            dataset.use_cache(open_waveform_cache(dataset, cfg), "waveform")
        model = FSLPRF(cfg)
        load_checkpoint(checkpoint, model)
        bank = build_prototype_bank(model, dataset, cfg, args.seed, checkpoint)
        trace_rows = None
    else:
        require_eval_runtime(cfg, args.runtime_report)
        if not args.reference_bank:
            raise ValueError("codec50/full must reuse sample IDs from the clean reference bank")
        reference = PrototypeBank.load(args.reference_bank)
        clean = SpeechFakeDataset(cfg["data"], "train", train_mode=False)
        expected = select_bank_samples(clean, args.seed)
        if reference.metadata.get("sample_ids") != expected:
            raise ValueError("Reference bank sample IDs do not match clean sampling at this seed")
        if args.expect_checkpoint_sha256 and reference.metadata.get("checkpoint_sha256") != args.expect_checkpoint_sha256:
            raise ValueError("Reference bank checkpoint does not match the requested checkpoint")
        derived = dataset_config_for_mode(cfg, args.augmentation_mode)
        dataset = open_train_dataset(derived)
        dataset.record_augmentation = True
        model = FSLPRF(cfg)
        load_checkpoint(checkpoint, model)
        trace_rows = []
        bank = build_prototype_bank(model, dataset, cfg, args.seed, checkpoint,
                                    selected_samples=reference.metadata["sample_ids"],
                                    augmentation_mode=args.augmentation_mode, trace_rows=trace_rows)
        if [row["uttid"] for row in trace_rows] != [uid for name in reference.metadata["sample_ids"]
                                                    for uid in reference.metadata["sample_ids"][name]]:
            raise ValueError("Augmentation manifest order does not match the clean reference IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output)
    if trace_rows is not None:
        manifest = output.parent / "augmentation_manifest.csv"
        write_manifest(manifest, trace_rows)
        from engine.checkpoint import checkpoint_sha256
        loaded = PrototypeBank.load(output)
        validate_bank(loaded, checkpoint_sha256(checkpoint))
        if loaded.metadata.get("augmentation_mode") != args.augmentation_mode:
            raise ValueError("Saved bank augmentation mode does not match the requested mode")
        write_prototype_info(output.parent / "prototype_bank_info.json", bank_path=output, dataset=dataset,
                             cfg=derived, mode=args.augmentation_mode, repo=Path(__file__).resolve().parents[1])
        applied = sum(bool(row["codec_applied"]) for row in trace_rows)
        print(f"CODEC_ASSIGNMENT applied={applied}/{len(trace_rows)} fraction={applied / len(trace_rows):.6f}", flush=True)
    print(f"Saved {bank.prototypes.shape} prototype bank to {output}")


if __name__ == "__main__":
    main()
