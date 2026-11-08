"""Opt-in AutoDL benchmark; never invoked by training or the local test suite."""
import copy
import csv
import json
from pathlib import Path
import uuid

import _bootstrap
from utils.config import common_parser, load_config
from utils.seed import preserved_rng, seed_all
from models.fs_lprf_model import FSLPRF
from data.codec_augmentation import offline_codec_estimate
from engine.pipeline_benchmark import (BENCHMARK_MODES, benchmark_train, benchmark_dev,
                                       total_experiment_times, dev_measurement_identity)


def main():
    parser = common_parser("Measure pipeline costs on AutoDL using disposable model/optimizer state")
    parser.add_argument("--modes", default="none,waveform")
    parser.add_argument("--workers", default="0,2,4,8")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--include-dev", action="store_true", help="Explicitly time full Light/Full Dev, never Test")
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--dev-timings", type=Path, help="Reuse a prior benchmark JSON's measured Dev timings")
    parser.add_argument("--recorded-light-seconds", type=float, help="Existing Light Dev seconds, used only for the projection")
    parser.add_argument("--recorded-full-seconds", type=float, help="Existing Full Dev seconds, used only for the projection")
    parser.add_argument("--recorded-dev-cache-seconds", type=float, default=0.0)
    parser.add_argument("--recorded-dev-note", default="")
    parser.add_argument("--output", type=Path, default=Path("outputs/pipeline_benchmark"))
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    modes, workers = args.modes.split(","), [int(value) for value in args.workers.split(",")]
    if set(modes) - set(BENCHMARK_MODES) or not workers or any(value < 0 for value in workers):
        parser.error("Use none/waveform/xlsr_feature/waveform_codec and nonnegative worker counts")
    if args.output.resolve().is_relative_to(Path(cfg["runtime"]["output_dir"]).resolve()):
        parser.error("Benchmark output must be separate from the formal training directory")
    output = args.output.resolve() / ("benchmark_" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    local = copy.deepcopy(cfg)
    local["runtime"]["output_dir"] = str(output / "manifests")
    measurement_identity = dev_measurement_identity(cfg) if (args.include_dev or args.dev_only or args.dev_timings) else None
    dev = {}
    if args.recorded_light_seconds is not None or args.recorded_full_seconds is not None:
        if args.recorded_light_seconds is None or args.recorded_full_seconds is None:
            parser.error("Recorded Dev projection needs both light and full seconds")
        recorded = dict(cache_build_seconds=args.recorded_dev_cache_seconds, light_seconds=args.recorded_light_seconds,
                        full_seconds=args.recorded_full_seconds, source=args.recorded_dev_note or "recorded existing Dev timings")
        dev = {"waveform": recorded, "none": dict(recorded)}
    if args.dev_timings:
        previous = json.loads(args.dev_timings.read_text(encoding="utf-8"))
        if previous.get("dev_measurement_identity") != measurement_identity:
            raise ValueError("Prior Dev timings do not match current data, frontend, validation or performance settings")
        dev = previous["dev"]
    with preserved_rng():
        seed_all(cfg["train"]["seed"])
        model = FSLPRF(local)
        train = [] if args.dev_only else benchmark_train(model, local, modes=modes, workers=workers,
                                                         warmup=args.warmup, steps=args.steps)
        if args.include_dev or args.dev_only:
            seed_all(cfg["train"]["seed"])
            dev = benchmark_dev(model, local, output / "dev")
    offline = None
    latency = next((row.get("codec_latency") for row in train if row.get("codec_latency")), None)
    if latency:
        cache = next(row["cache"] for row in train if row["mode"] == "waveform_codec")
        offline = offline_codec_estimate(latency, utterances=cache["samples"], pcm_bytes=cache["bytes"],
                                         workers=max(workers), probability=0.5)
    report = dict(scope="isolated pipeline benchmark; no formal checkpoint, no Test", config=cfg, train=train, dev=dev,
                  dev_measurement_identity=measurement_identity,
                  total_experiment_times=total_experiment_times(train, dev, cfg),
                  offline_codec_estimate=offline,
                  timing_notes="CUDA events for GPU stages; synchronized step walls; data wait is CPU wall time; "
                               "GPU utilization is NVML during timed steps. waveform_codec forces probability 0.5 "
                               "and waveform forces codec augmentation off.")
    (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [{key: value for key, value in row.items() if key not in ("cache", "raw_steps", "codec_latency", "codec_report")} |
            {f"cache_{key}": row["cache"].get(key) for key in ("bytes", "build_seconds", "dtype", "fingerprint")}
            for row in train]
    if rows:
        with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(report=str(output / "benchmark.json"),
                          total_experiment_times=report["total_experiment_times"]), indent=2))


if __name__ == "__main__":
    main()
