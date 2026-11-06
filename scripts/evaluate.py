from pathlib import Path
import json
import time
import _bootstrap
from utils.config import common_parser, load_config
from data.speechfake_dataset import SpeechFakeDataset
from data.evaluation_manifest import load_or_create_manifest
from models.fs_lprf_model import FSLPRF
from engine.checkpoint import load_checkpoint, resolve_checkpoint
from engine.evaluator import evaluate
from data.cache_pipeline import prepare_final_test_caches


def main():
    parser = common_parser("SpeechFake-BD Test 10-shot, five seeds (AutoDL only)")
    parser.add_argument("--checkpoint", required=True,
                        help="Explicit node checkpoint path, or 'best' from the configured output directory")
    parser.add_argument("--shot", type=int, default=10, choices=[10])
    parser.add_argument("--cache-dry-run", action="store_true", help="Retired flag: Test caches are forbidden")
    parser.add_argument("--confirm-large-cache", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["data"]["test_cache_mode"] != "none" or args.cache_dry_run or args.confirm_large_cache:
        raise ValueError("Test always uses original WAVs; Test cache modes and cache flags are forbidden")
    checkpoint = resolve_checkpoint(args.checkpoint, cfg["runtime"]["output_dir"])
    ds = SpeechFakeDataset(cfg["data"], "test")
    out = Path(cfg["runtime"]["output_dir"]) / "evaluation"
    manifest = load_or_create_manifest(ds, out / "test_episode_manifest.json", args.shot, cfg["test"]["seeds"])
    model = FSLPRF(cfg)
    load_checkpoint(checkpoint, model)
    cache_report = None
    started = time.perf_counter()
    result = evaluate(model, ds, manifest, cfg, out, "test")
    evaluation_seconds = time.perf_counter() - started
    (out / "test_run.json").write_text(json.dumps(dict(checkpoint=str(checkpoint),
                                config=cfg, cache=cache_report, evaluation_seconds=evaluation_seconds,
                                summary=result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
