from collections import Counter
from pathlib import Path
import json
import _bootstrap
from utils.config import common_parser, load_config
from data.speechfake_dataset import SpeechFakeDataset, ConfiguredRawBoost
from data.episode_sampler import EpisodeSampler
from data.evaluation_manifest import load_or_create_manifest


def main():
    parser = common_parser("Validate SpeechFake-BD manifests, capacities, paths and fixed evaluation splits")
    parser.add_argument("--read-samples", type=int, default=1, help="Audio records to decode per split; default 1")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"],
                        help="Use --splits train dev for pre-training checks without opening Test")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seen_ids, seen_paths, classes = set(), set(), None
    summaries = {}
    if len(args.splits) != len(set(args.splits)):
        raise ValueError("--splits must not repeat a split")
    for split in args.splits:
        ds = SpeechFakeDataset(cfg["data"], split, train_mode=False)
        if classes is not None and classes != ds.fake_classes:
            raise ValueError("Train/Dev/Test attacker class mappings must match")
        classes = ds.fake_classes
        paths = {str(Path(path).resolve()) for path in ds.uttid2path.values()}
        if seen_ids.intersection(ds.uttids) or seen_paths.intersection(paths):
            raise ValueError(f"{split}: uttid or audio path overlaps an earlier split")
        seen_ids.update(ds.uttids)
        seen_paths.update(paths)
        missing = [uid for uid, path in ds.uttid2path.items() if not Path(path).is_file()]
        if missing:
            raise ValueError(f"{split}: {len(missing)} missing audio files; first: {missing[:5]}")
        for index in range(min(args.read_samples, len(ds))):
            ds[index]
        counts = {name: len(pool) for name, pool in ds.class_indices.items()}
        summaries[split] = dict(class_counts=counts, real_sources=dict(Counter(
            ds.uttid2info[ds.uttids[i]]["origin_ds"] for i in ds.class_indices["real"])))
        if split == "train":
            EpisodeSampler(ds.class_indices, **{key: cfg["train"][key] for key in
                ("way", "support_per_class", "total_query_per_episode", "episodes_per_step")})
        else:
            section = cfg["validation" if split == "dev" else "test"]
            load_or_create_manifest(ds, Path(cfg["runtime"]["output_dir"]) / "evaluation" /
                                    f"{split}_episode_manifest.json", section["shot"], section["seeds"])
    import os
    rir = cfg["data"]["rir_root"] or os.environ.get("FS_LPRF_RIR_ROOT")
    if not rir:
        raise ValueError("Set data.rir_root or FS_LPRF_RIR_ROOT")
    augmentor = ConfiguredRawBoost(rir)
    print(json.dumps(dict(status="passed", rir_files=len(augmentor.rir), splits=summaries), indent=2))


if __name__ == "__main__":
    main()
