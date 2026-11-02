import csv
from pathlib import Path
import numpy as np
import torch
from fs_lprf.data.loaders import audio_loader
from fs_lprf.methods.inference import kshot_task, kshot_scores, predict
from fs_lprf.utils.metrics import binary_metrics, macro_metrics, mean_std, METRICS
from fs_lprf.utils.logging import write_csv
from fs_lprf.utils.seed import seed_all, preserved_rng


def feature_batches(model, dataset, ids, cfg, seed):
    indices = [dataset.uttid2idx[uid] for uid in ids]
    loader = audio_loader(dataset, indices, cfg["train"]["audio_microbatch_size"],
                          seed, cfg["runtime"]["num_workers"])
    for batch in loader:
        yield batch["uttids"], model({"feats": batch["feats"]})["feats"]


def extract_features(model, dataset, ids, cfg, seed):
    # Support only (or one class during offline bank construction), kept off GPU.
    return torch.cat([features.cpu() for _, features in feature_batches(model, dataset, ids, cfg, seed)])


@torch.no_grad()
def evaluate(model, dataset, manifest, cfg, output_dir, split, step=None):
    if dataset.do_augment:
        raise ValueError("Evaluation must not augment audio")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    rows, macros = [], []
    prediction_path = output_dir / (f"dev_predictions_step{step}.csv" if split == "dev" else "test_predictions.csv")
    fields = ["uttid", "fake_class", "seed", "label", "score", "prediction"]
    # Restore training RNG even if audio decoding/metrics fail.
    try:
        with preserved_rng(), prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            model.eval()
            for seed in manifest["seeds"]:
                seed_all(seed)
                supports = manifest["supports"][str(seed)]
                real_support = extract_features(model, dataset, supports["real"], cfg, seed).to(model.device)
                tasks = {}
                # Encode shared Real Support once; each Fake task still builds its own alpha.
                for name in dataset.fake_classes:
                    fake_support = extract_features(model, dataset, supports[name], cfg, seed).to(model.device)
                    features = torch.cat([real_support, fake_support])
                    labels = torch.tensor([0] * len(real_support) + [1] * len(fake_support), device=model.device)
                    tasks[name] = kshot_task(features, labels)
                real_scores = {name: [] for name in dataset.fake_classes}

                def record(ids, features, name, label):
                    scores = kshot_scores(features, *tasks[name]).cpu()
                    predictions = predict(scores)
                    writer.writerows(dict(uttid=uid, fake_class=name, seed=seed,
                                           label=label, score=float(score), prediction=int(prediction))
                                     for uid, score, prediction in zip(ids, scores, predictions))
                    return scores.numpy()

                for ids, features in feature_batches(model, dataset, manifest["queries"]["real"], cfg, seed):
                    for name in dataset.fake_classes:
                        real_scores[name].append(record(ids, features, name, 0))
                seed_rows = []
                for name in dataset.fake_classes:
                    fake_scores = [record(ids, features, name, 1) for ids, features in
                                   feature_batches(model, dataset, manifest["queries"][name], cfg, seed)]
                    real = np.concatenate(real_scores[name])
                    fake = np.concatenate(fake_scores)
                    values = binary_metrics(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
                    row = dict(seed=seed, fake_class=name, real_query_count=len(real), fake_query_count=len(fake), **values)
                    if step is not None:
                        row = dict(step=step, **row)
                    rows.append(row)
                    seed_rows.append(row)
                macros.append(dict(seed=seed, **macro_metrics(seed_rows)))
                if split == "test":
                    write_csv(output_dir / f"test_per_class_seed{seed}.csv", seed_rows)
            if split == "dev":
                write_csv(output_dir / "dev_per_class.csv", rows, append=True)
            summary = mean_std(macros)
            write_csv(output_dir / f"{split}_macro_per_seed.csv",
                      [dict(step=step, **row) if step is not None else row for row in macros], append=split == "dev")
            write_csv(output_dir / f"{split}_macro_mean_std.csv",
                      [dict(step=step, metric=name, **summary[name]) if step is not None
                       else dict(metric=name, **summary[name]) for name in METRICS], append=split == "dev")
            return dict(mean_eer=summary["eer"]["mean"], std_eer=summary["eer"]["std"],
                        metrics=summary, per_seed=macros)
    finally:
        model.train(was_training)
