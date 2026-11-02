import _bootstrap
from fs_lprf.config import common_parser, load_config
from fs_lprf.engine.trainer import train


def main():
    parser = common_parser("FS-LPRF Phase 1: 4000-step SpeechFake-BD training (AutoDL only)")
    parser.add_argument("--resume", help="Path to last.pt; requires the original run directory and Dev manifest")
    args = parser.parse_args()
    train(load_config(args.config, args.set), args.resume)


if __name__ == "__main__":
    main()
