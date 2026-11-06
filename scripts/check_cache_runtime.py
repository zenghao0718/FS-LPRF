"""Cache-only Train/Dev input smoke; optional online model forward, no updates."""
import json
import torch

import _bootstrap
from data.cache_pipeline import open_train_dataset, bind_evaluation_caches, view_tokens
from data.loaders import ReusableAudioLoader
from data.speechfake_dataset import SpeechFakeDataset
from utils.config import common_parser, load_config
from utils.seed import preserved_rng


def main():
    parser = common_parser("Smoke Train/Dev waveform inputs after source WAV removal; Test is never opened")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--with-model", action="store_true", help="Also run the unchanged online XLS-R/backend forward")
    args = parser.parse_args()
    if args.batches <= 0 or len(args.splits) != len(set(args.splits)):
        parser.error("Use a positive batch count and unique splits")
    cfg = load_config(args.config, args.set)
    datasets = []
    for split in args.splits:
        if cfg["data"][f"{split}_cache_mode"] != "waveform":
            raise ValueError("Cache-only smoke requires waveform mode")
        if split == "train":
            ds = open_train_dataset(cfg)
        else:
            ds = SpeechFakeDataset(cfg["data"], split, train_mode=False)
            bind_evaluation_caches(ds, cfg, {})
        datasets.append(ds)
    # All cache validation happens before potentially loading large model weights.
    model = None
    if args.with_model:
        from models.fs_lprf_model import FSLPRF, forward_batch
        model = FSLPRF(cfg).eval()
    reports = []
    with preserved_rng(), torch.no_grad():
        for ds in datasets:
            count = min(len(ds), args.batches * cfg["train"]["audio_microbatch_size"])
            indices = list(range(count))
            seen, process_refs, output_shapes = [], [], []
            with ReusableAudioLoader(ds, cfg, training=ds.split == "train", seed=42) as reusable:
                stream = (reusable.for_step(indices, 1) if ds.split == "train" else
                          reusable.for_indices(view_tokens(ds, ds.uttids[:count], 42, "full")))
                for batch in stream:
                    if batch["feats"].shape[1:] != (1,64000) or not torch.isfinite(batch["feats"]).all():
                        raise ValueError("Invalid official waveform input")
                    seen.extend(batch["uttids"])
                    if model is not None:
                        features = forward_batch(model,batch)
                        if not torch.isfinite(features).all():
                            raise ValueError("Non-finite online model output")
                        output_shapes.append(list(features.shape))
                iterator = getattr(reusable.loader,"_iterator",None)
                process_refs = list(getattr(iterator,"_workers",[]))
            if seen != ds.uttids[:count] or any(process.is_alive() for process in process_refs):
                raise RuntimeError("Input order or worker cleanup failed")
            reports.append(dict(split=ds.split,input_mode=ds.cache_mode,samples=len(seen),
                                workers_exited=True,online_model_forward=model is not None,
                                output_shapes=output_shapes))
    print(json.dumps(dict(status="passed", optimizer_updates=0, test_initialized=False,
                          hidden_states_saved=False, splits=reports),indent=2))


if __name__ == "__main__":
    main()
