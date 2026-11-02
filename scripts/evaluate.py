from pathlib import Path
import json
import _bootstrap
from fs_lprf.config import common_parser, load_config
from fs_lprf.data.speechfake_dataset import SpeechFakeDataset
from fs_lprf.data.evaluation_manifest import load_or_create_manifest
from fs_lprf.models.fs_lprf_model import FSLPRF
from fs_lprf.engine.checkpoint import load_checkpoint, resolve_checkpoint
from fs_lprf.engine.evaluator import evaluate


def main():
    parser = common_parser("SpeechFake-BD Test 10-shot, five seeds (AutoDL only)")
    parser.add_argument("--checkpoint", required=True,
                        help="Explicit node checkpoint path, or 'best' from the configured output directory")
    parser.add_argument("--shot", type=int, default=10, choices=[10])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg["runtime"]["output_dir"])
    ds = SpeechFakeDataset(cfg["data"], "test")
    out = Path(cfg["runtime"]["output_dir"]) / "evaluation"
    manifest = load_or_create_manifest(ds, out / "test_episode_manifest.json", args.shot, cfg["test"]["seeds"])
    model = FSLPRF(cfg)
    load_checkpoint(checkpoint, model)
    result = evaluate(model, ds, manifest, cfg, out, "test")
    (out / "test_run.json").write_text(json.dumps(dict(checkpoint=str(checkpoint),
                                config=cfg, summary=result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
