from pathlib import Path
import _bootstrap
from fs_lprf.config import common_parser, load_config
from fs_lprf.data.speechfake_dataset import SpeechFakeDataset
from fs_lprf.models.fs_lprf_model import FSLPRF
from fs_lprf.engine.checkpoint import load_checkpoint
from fs_lprf.methods.prototype_bank import build_prototype_bank


def main():
    parser = common_parser("Offline 33-class prototype bank; implemented for later work, not part of Phase 1 experiments")
    parser.add_argument("--checkpoint", required=True, help="Final best.pt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    dataset = SpeechFakeDataset(cfg["data"], "train", train_mode=False)
    model = FSLPRF(cfg)
    load_checkpoint(args.checkpoint, model)
    bank = build_prototype_bank(model, dataset, cfg, args.seed, args.checkpoint)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bank.save(args.output)
    print(f"Saved {bank.prototypes.shape} prototype bank to {args.output}")


if __name__ == "__main__":
    main()
