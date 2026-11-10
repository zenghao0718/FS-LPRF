"""Export a few matched cached waveforms for boundary/listening checks; no encoding."""
import copy
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import _bootstrap
from data.cache import atomic_json
from data.codec_augmentation import alignment_diagnostics
from data.codec_cache import open_codec_cache, validate_location
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import open_waveform_cache
from utils.config import common_parser, load_config


def main():
    parser = common_parser("Inspect a small number of matched source/codec PCM waveforms")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.samples <= 100:
        parser.error("--samples must be between 1 and 100")
    cfg = load_config(args.config, args.set)
    options = copy.deepcopy(cfg["data"])
    options["codec_augment"]["enabled"] = False
    dataset = SpeechFakeDataset(options, "train")
    with open_waveform_cache(dataset, cfg) as source:
        with open_codec_cache(source, cfg, path=args.cache_dir, allow_smoke=True) as reader:
            target = validate_location(args.output_dir, source, cfg)
            if target.is_relative_to(reader.path) or reader.path.is_relative_to(target):
                raise ValueError("Inspection output must be independent of the Codec Cache")
            target.mkdir(parents=True, exist_ok=False)
            ordered = sorted(reader.index, key=lambda row: (row["length"], row["uttid"]))
            indices = np.linspace(0, len(ordered)-1, min(args.samples, len(ordered)), dtype=int)
            records = []
            for number, position in enumerate(indices):
                row = ordered[int(position)]
                uid = row["uttid"]
                original = source.get(uid).numpy().reshape(-1)
                encoded = reader.get(uid).numpy().reshape(-1)
                sf.write(target / f"{number:03d}_original.wav", original, 16000, subtype="FLOAT")
                sf.write(target / f"{number:03d}_codec.wav", encoded, 16000, subtype="FLOAT")
                records.append(dict(uttid=uid, label=dataset.uttid2info[uid]["label"], codec=row["codec"],
                                    alignment=alignment_diagnostics(original, encoded)))
            report = dict(cache=reader.stats(), records=records)
            atomic_json(target / "inspection.json", report)
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
