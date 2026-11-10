"""Explicit offline build/verify. Input is the complete Train Waveform Cache only."""
import copy
import json
from pathlib import Path
import time

import _bootstrap
from data.cache import atomic_json
from data.cache_pipeline import _progress
from data.codec_augmentation import codec_settings
from data.codec_cache import (build_codec_cache, estimate_space, make_codec_identity, open_codec_cache,
                              select_smoke_ids, validate_location)
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import open_waveform_cache
from data.codec_workers import CodecWorkerPool
from data.codec_records import summarize_codec_results
from utils.config import common_parser, load_config


def main():
    parser = common_parser("Build one fixed real codec version per Train cached waveform")
    parser.add_argument("--output-dir", type=Path, help="Exact independent cache directory; never overwritten")
    parser.add_argument("--limit", type=int, help="Build an explicitly marked, class/length stratified smoke cache")
    parser.add_argument("--dry-run", action="store_true", help="Check all backends/presets and disk space; do not build")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Check completed cache identity, IDs and shard/index structure; no PCM scan")
    parser.add_argument("--codec-workers", type=int, default=1, help="Number of CPU encoding workers")
    parser.add_argument("--codec-shard-size", type=int, help="Codec-only shard size (config default 512)")
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, help="Bound outstanding tasks AND out-of-order PCM results")
    parser.add_argument("--encodec-device", default="cpu", help="cpu or cuda:N; CUDA has one dedicated spawn worker")
    parser.add_argument("--reserve-gib", type=float, default=5)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be nonnegative")
    cfg = load_config(args.config, args.set)
    if args.codec_shard_size is not None:
        if args.codec_shard_size <= 0:
            parser.error("--codec-shard-size must be positive")
        cfg["cache"]["codec_shard_size"] = args.codec_shard_size
    settings = codec_settings(cfg["data"])
    target = args.output_dir or Path(settings["offline_cache_root"])
    options = copy.deepcopy(cfg["data"])
    options["codec_augment"]["enabled"] = False
    dataset = SpeechFakeDataset(options, "train", train_mode=False)
    with open_waveform_cache(dataset, cfg) as source:
        validate_location(target, source, cfg)
        if args.verify:
            with open_codec_cache(source, cfg, path=target, allow_smoke=True) as reader:
                print(json.dumps(dict(reader.stats(), codec_results=summarize_codec_results(reader.index)), indent=2))
            return
        ids = source.identity["uttids"] if args.limit is None else select_smoke_ids(dataset, source, args.limit)
        space = estimate_space(source, ids, target, reserve_bytes=int(args.reserve_gib * 1024**3), resume=args.resume)
        print(json.dumps(dict(space=space, smoke=args.limit is not None), indent=2), flush=True)
        if not space["enough_space"]:
            raise RuntimeError("Insufficient disk space including reserve; choose another disk or reduce smoke --limit")
        worker_started = time.perf_counter()
        with CodecWorkerPool(source, settings["codecs"], workers=args.codec_workers,
                             encodec_device=args.encodec_device, threads=args.worker_threads,
                             max_inflight=args.max_inflight) as pool:
            preflight_started = time.perf_counter()
            worker_initialization_seconds = preflight_started - worker_started
            preflight = pool.preflight(source, settings["build_seed"])
            identity = make_codec_identity(source, cfg, pool, uttids=ids, smoke=args.limit is not None,
                                           encodec_device=pool.device)
            preflight_seconds = time.perf_counter() - preflight_started
            print(json.dumps(dict(preflight=preflight, codec_report=identity["codec_report"]), indent=2), flush=True)
            if args.dry_run:
                return
            started = time.perf_counter()
            with build_codec_cache(source, cfg, target, identity, pool=pool,
                                   resume=args.resume, progress=_progress) as reader:
                stats = reader.stats()
                report = dict(stats=stats, invocation_seconds=time.perf_counter()-started, space=space,
                              preflight_seconds=preflight_seconds,
                              worker_initialization_seconds=worker_initialization_seconds,
                              codec_workers=args.codec_workers, encodec_device=pool.device,
                              worker_threads=args.worker_threads, codec_shard_size=cfg["cache"]["codec_shard_size"],
                              peak_inflight=pool.peak_inflight, max_inflight=pool.max_inflight,
                              process_start_method="spawn", checksum_policy="none",
                              codec_results=summarize_codec_results(reader.index),
                              compatibility_boundary=reader.metadata.get("compatibility_boundary"),
                              preflight=preflight, source_samples=len(source.rows),
                              estimated_full_pcm_bytes=sum(row["length"] * 4 for row in source.index),
                              projected_full_build_seconds=stats["build_seconds"] * len(source.rows) / len(ids),
                              projection_note="Estimate from this class/length-stratified subset; not a full-build measurement")
                atomic_json(target / "build_report.json", report)
                print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
