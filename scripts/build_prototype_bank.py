from pathlib import Path
import _bootstrap
from utils.config import common_parser, load_config
from data.speechfake_dataset import SpeechFakeDataset
from models.fs_lprf_model import FSLPRF
from engine.checkpoint import load_checkpoint, resolve_checkpoint
from methods.prototype_bank import build_prototype_bank


def main():
    parser = common_parser("Offline 33-class prototype bank; implemented for later work, not part of Phase 1 experiments")
    parser.add_argument("--checkpoint", required=True, help="best, or the already selected Best node path")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg["runtime"]["output_dir"])
    if checkpoint != resolve_checkpoint("best", cfg["runtime"]["output_dir"]):
        raise ValueError("0-shot bank construction requires the already selected Best checkpoint")
    # Deliberately use unaugmented WAVs even when Train uses a fixed augmented feature view.
    dataset = SpeechFakeDataset(cfg["data"], "train", train_mode=False)
    model = FSLPRF(cfg)
    load_checkpoint(checkpoint, model)
    bank = build_prototype_bank(model, dataset, cfg, args.seed, checkpoint)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bank.save(args.output)
    print(f"Saved {bank.prototypes.shape} prototype bank to {args.output}")


if __name__ == "__main__":
    main()
