"""Explicit Train/Dev cache build, integrity check, and storage estimates."""
import json
from pathlib import Path
import sys
import time

import _bootstrap
from utils.config import common_parser, load_config
from utils.seed import preserved_rng
from data.cache import cache_path
from data.cache_pipeline import cache_specs, estimate_cache, build_dataset_cache, require_cache
from data.waveform_cache import open_waveform_cache
from data.speechfake_dataset import SpeechFakeDataset


def confirm_space(estimates, confirmed=False):
    if not any(row["warning"] for row in estimates):
        return
    message = "Estimated cache exceeds the configured disk threshold; review storage before building."
    if confirmed:
        print(message + " Explicit confirmation received.", flush=True)
        return
    if not sys.stdin.isatty() or input(message + " Continue? [y/N] ").strip().lower() != "y":
        raise ValueError(message + " Re-run with --confirm-large-cache after reviewing --dry-run.")


def main():
    parser = common_parser("Build/check Train or Dev caches; Test always uses original WAVs")
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--mode", choices=["waveform"], required=True)
    parser.add_argument("--validation-modes", default="light,full")
    parser.add_argument("--seeds", help="Comma-separated manifest seeds for a small cache-generation smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Preserve existing cache, then regenerate explicitly")
    parser.add_argument("--check", action="store_true", help="Read/validate all records and report cache statistics")
    parser.add_argument("--confirm-large-cache", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    started = time.perf_counter()
    seeds = [int(seed) for seed in args.seeds.split(",")] if args.seeds else None
    with preserved_rng():
        if args.check and args.mode == "waveform":
            dataset = SpeechFakeDataset(cfg["data"], args.split, train_mode=False)
            ids = dataset.uttids[:args.limit] if args.limit is not None else None
            reader = open_waveform_cache(dataset, cfg, uttids=ids,
                                         allow_smoke=args.limit is not None, verify_checksums=True)
            print(json.dumps(dict(status="verified", caches=[reader.verify()]), indent=2))
            return
        dataset, identities = cache_specs(cfg, args.split, args.mode,
                         validation_modes=args.validation_modes.split(","), limit=args.limit, seeds=seeds)
        if args.check:
            reports = [require_cache(cfg, identity, allow_smoke=args.limit is not None).verify() for identity in identities]
            print(json.dumps(dict(status="verified", caches=reports), indent=2))
            return
        model = None
        estimates = [estimate_cache(dataset, cfg, identity) for identity in identities]
        preparation = time.perf_counter() - started
        if estimates:
            total = sum(row["bytes"] for row in estimates)
            total_warning = total / 2**30 > cfg["cache"]["warn_if_estimated_gb_above"]
            print(json.dumps(dict(dry_run=args.dry_run, estimated_total_gib=total / 2**30,
                                  warning=total_warning, caches=estimates), indent=2))
            if args.dry_run:
                return
            confirm_space(estimates + [dict(warning=total_warning)], args.confirm_large_cache)
        reports = []
        for identity in identities:
            creating = args.force or not cache_path(cfg, identity).exists()
            reader = build_dataset_cache(dataset, cfg, identity, model=model, resume=args.resume,
                                         force=args.force, preparation_seconds=preparation if creating else 0)
            if creating:
                preparation = 0
            reports.append(reader.stats())
        print(json.dumps(dict(status="complete", caches=reports, elapsed_seconds=time.perf_counter() - started), indent=2))


if __name__ == "__main__":
    main()
