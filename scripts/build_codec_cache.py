"""Explicit offline build/verify. Input is the complete Train Waveform Cache only."""
import copy
import json
from pathlib import Path
import time

import _bootstrap
from data.cache import atomic_json
from data.cache_pipeline import _progress
from data.codec_augmentation import CodecAugmentor, codec_settings
from data.codec_cache import (build_codec_cache, estimate_space, make_codec_identity, open_codec_cache,
                              preflight_waveforms, select_smoke_ids, validate_location)
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import open_waveform_cache
from utils.config import common_parser, load_config
from utils.seed import preserved_rng


def main():
    parser = common_parser("Build one fixed real codec version per Train cached waveform")
    parser.add_argument("--output-dir", type=Path, help="Exact independent cache directory; never overwritten")
    parser.add_argument("--limit", type=int, help="Build an explicitly marked, class/length stratified smoke cache")
    parser.add_argument("--dry-run", action="store_true", help="Check all backends/presets and disk space; do not build")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify completed cache, records, source and all PCM")
    parser.add_argument("--reserve-gib", type=float, default=5)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be nonnegative")
    cfg = load_config(args.config, args.set)
    settings = codec_settings(cfg["data"])
    target = args.output_dir or Path(settings["offline_cache_root"])
    options = copy.deepcopy(cfg["data"])
    options["codec_augment"]["enabled"] = False
    dataset = SpeechFakeDataset(options, "train", train_mode=False)
    with open_waveform_cache(dataset, cfg, verify_checksums=True) as source:
        validate_location(target, source, cfg)
        if args.verify:
            with open_codec_cache(source, cfg, path=target, allow_smoke=True) as reader:
                print(json.dumps(reader.verify(), indent=2))
            return
        ids = source.identity["uttids"] if args.limit is None else select_smoke_ids(dataset, source, args.limit)
        space = estimate_space(source, ids, target, reserve_bytes=int(args.reserve_gib * 1024**3), resume=args.resume)
        print(json.dumps(dict(space=space, smoke=args.limit is not None), indent=2), flush=True)
        if not space["enough_space"]:
            raise RuntimeError("Insufficient disk space including reserve; choose another disk or reduce smoke --limit")
        with preserved_rng():
            preflight_started = time.perf_counter()
            augmentor = CodecAugmentor(probability=1, codecs=settings["codecs"], sample_rate=settings["sample_rate"])
            preflight = preflight_waveforms(source, augmentor)
            identity = make_codec_identity(source, cfg, augmentor, uttids=ids, smoke=args.limit is not None)
            preflight_seconds = time.perf_counter() - preflight_started
            print(json.dumps(dict(preflight=preflight, codec_report=identity["codec_report"]), indent=2), flush=True)
            if args.dry_run:
                return
            started = time.perf_counter()
            with build_codec_cache(source, cfg, target, identity, augmentor,
                                   resume=args.resume, progress=_progress) as reader:
                stats = reader.verify()
                report = dict(stats=stats, invocation_seconds=time.perf_counter()-started, space=space,
                              preflight_seconds=preflight_seconds,
                              preflight=preflight, source_samples=len(source.rows),
                              estimated_full_pcm_bytes=sum(row["length"] * 4 for row in source.index),
                              projected_full_build_seconds=stats["build_seconds"] * len(source.rows) / len(ids),
                              projection_note="Estimate from this class/length-stratified subset; not a full-build measurement")
                atomic_json(target / "build_report.json", report)
                print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
