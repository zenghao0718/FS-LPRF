"""Authorize exactly the pinned 5f53805 -> clipping fallback v2 continuation."""
import copy
import json
from pathlib import Path

import _bootstrap
from data.codec_compat import authorize_clipping_recovery
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import open_waveform_cache
from utils.config import common_parser, load_config


def main():
    parser = common_parser("Authorize a pinned interrupted v2 prefix without changing PCM or its identity")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Final cache path, without .building suffix")
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--encodec-device", default="cuda:0")
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    options = copy.deepcopy(cfg["data"])
    options["codec_augment"]["enabled"] = False
    dataset = SpeechFakeDataset(options, "train", train_mode=False)
    with open_waveform_cache(dataset, cfg, verify_checksums=False) as source:
        result = authorize_clipping_recovery(args.cache_dir, source, cfg,
                    expected_shards=args.expected_shards, expected_records=args.expected_records,
                    expected_total=args.expected_total, device=args.encodec_device,
                    threads=args.worker_threads, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
