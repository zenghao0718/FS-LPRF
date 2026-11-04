import _bootstrap
from utils.config import common_parser, load_config
from engine.trainer import train


def main():
    parser = common_parser("FS-LPRF Phase 1: 4000-step SpeechFake-BD training (AutoDL only)")
    parser.add_argument("--resume", help="Latest milestone step_XXXX.pt; requires the original run directory, manifests and full-Dev history")
    args = parser.parse_args()
    train(load_config(args.config, args.set), args.resume)


if __name__ == "__main__":
    main()
