import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import numpy as np
import torch
from data.loaders import audio_loader
from methods.inference import kshot_task, kshot_scores, predict
from utils.metrics import binary_metrics, macro_metrics, mean_std, METRICS
from utils.logging import write_csv
from utils.seed import seed_all, preserved_rng


def feature_batches(model, dataset, ids, cfg, seed):
    indices = [dataset.uttid2idx[uid] for uid in ids]
    loader = audio_loader(dataset, indices, cfg["train"]["audio_microbatch_size"],
                          seed, cfg["runtime"]["num_workers"])
    for batch in loader:
        yield batch["uttids"], model({"feats": batch["feats"]})["feats"]


def extract_features(model, dataset, ids, cfg, seed):
    # Support only (or one class during offline bank construction), kept off GPU.
    return torch.cat([features.cpu() for _, features in feature_batches(model, dataset, ids, cfg, seed)])


def _log(logger, message, *args):
    if logger is None:
        print(message % args if args else message, flush=True)
    else:
        logger.info(message, *args)


def _validate_dev_mode(dataset, manifest, cfg, mode, step):
    if mode not in ("light", "full"):
        raise ValueError("Dev mode must be light or full")
    if dataset.split != "dev" or manifest["split"] != "dev":
        raise ValueError("Training validation modes require Dev, never Train or Test")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("Step-specific Dev reports require a positive integer step")
    if len(dataset.fake_classes) != 32 or manifest["classes"] != ["real"] + dataset.fake_classes:
        raise ValueError("Dev requires all 32 Fake classes in the fixed order")
    expected_seeds = [42] if mode == "light" else list(cfg["validation"]["seeds"])
    if manifest["seeds"] != expected_seeds or manifest["shot"] != 10:
        raise ValueError("Dev mode does not match its fixed seed/10-shot protocol")
    for name in manifest["classes"]:
        query = manifest["queries"][name]
        if not query or len(query) != len(set(query)):
            raise ValueError(f"Invalid or empty Dev Query for {name}")
        for seed in manifest["seeds"]:
            support = manifest["supports"][str(seed)][name]
            if len(support) != 10 or len(set(support)) != 10 or set(support) & set(query):
                raise ValueError(f"Invalid Dev Support/Query for {name}, seed={seed}")


def _class_summary(rows, fake_classes, step, mode):
    summaries = []
    for name in fake_classes:
        class_rows = [row for row in rows if row["fake_class"] == name]
        metrics = mean_std(class_rows)
        summary = dict(step=step, mode=mode, fake_class=name, seed_count=len(class_rows),
                       real_query_count_min=min(row["real_query_count"] for row in class_rows),
                       real_query_count_max=max(row["real_query_count"] for row in class_rows),
                       fake_query_count_min=min(row["fake_query_count"] for row in class_rows),
                       fake_query_count_max=max(row["fake_query_count"] for row in class_rows))
        for metric in METRICS:
            summary[f"{metric}_mean"] = metrics[metric]["mean"]
            summary[f"{metric}_std"] = metrics[metric]["std"]
        summaries.append(summary)
    return summaries


def _log_dev_summary(logger, prefix, class_summaries, summary):
    def metric_text(values):
        return " ".join(f"{name.upper()}={values[name]['mean'] * 100:.4f}±"
                        f"{values[name]['std'] * 100:.4f}%" for name in METRICS)
    for row in class_summaries:
        values = {name: dict(mean=row[f"{name}_mean"], std=row[f"{name}_std"]) for name in METRICS}
        real = row["real_query_count_min"]
        fake = row["fake_query_count_min"]
        if row["real_query_count_max"] != real:
            real = f"{real}..{row['real_query_count_max']}"
        if row["fake_query_count_max"] != fake:
            fake = f"{fake}..{row['fake_query_count_max']}"
        _log(logger, "%s %s %s n_real=%s n_fake=%s", prefix, row["fake_class"],
             metric_text(values), real, fake)
    _log(logger, "%s Macro %s classes=%d", prefix, metric_text(summary), len(class_summaries))


@torch.no_grad()
def evaluate(model, dataset, manifest, cfg, output_dir, split, step=None, *, mode=None, logger=None):
    """One scoring route for legacy/Test, lightweight Dev and full five-seed Dev.

    Explicit Dev modes write isolated per-step reports. Only the caller can mark a
    completed full validation or choose Best after all report files are committed.
    """
    if dataset.do_augment:
        raise ValueError("Evaluation must not augment audio")
    if mode is not None:
        if split != "dev":
            raise ValueError("light/full modes are restricted to Dev")
        _validate_dev_mode(dataset, manifest, cfg, mode, step)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    rows, macros = [], []
    started = time.perf_counter()
    report_temp = None
    prefix = f"[{mode.upper()} DEV step={step:04d}]" if mode is not None else None
    query_counts = {name: dict(real_query_count=len(manifest["queries"]["real"]),
                               fake_query_count=len(manifest["queries"][name]))
                    for name in dataset.fake_classes}
    if mode is None:
        prediction_path = output_dir / (f"dev_predictions_step{step}.csv" if split == "dev" else "test_predictions.csv")
    else:
        destination = output_dir / mode
        destination.mkdir(parents=True, exist_ok=True)
        report_temp = TemporaryDirectory(prefix=f".dev_{mode}_step{step:04d}_", dir=destination)
        stage_dir = Path(report_temp.name)
        suffix = f"step{step:04d}.csv"
        paths = {name: stage_dir / f"dev_{mode}_{name}_{suffix}" for name in
                 ("per_class_seed", "per_class_mean_std", "macro_per_seed", "macro_mean_std", "predictions")}
        prediction_path = paths["predictions"]
        _log(logger, "%s START seeds=%s classes=%d n_real=%d n_fake_min=%d n_fake_max=%d report_dir=%s",
             prefix, manifest["seeds"], len(dataset.fake_classes), len(manifest["queries"]["real"]),
             min(value["fake_query_count"] for value in query_counts.values()),
             max(value["fake_query_count"] for value in query_counts.values()), destination)
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
                    if mode is not None:
                        row = dict(mode=mode, **row)
                    rows.append(row)
                    seed_rows.append(row)
                macros.append(dict(seed=seed, **macro_metrics(seed_rows)))
                if split == "test":
                    write_csv(output_dir / f"test_per_class_seed{seed}.csv", seed_rows)
            if split == "dev" and mode is None:
                write_csv(output_dir / "dev_per_class.csv", rows, append=True)
            summary = mean_std(macros)
            result = dict(mean_eer=summary["eer"]["mean"], std_eer=summary["eer"]["std"],
                          metrics=summary, per_seed=macros)
            if mode is None:
                write_csv(output_dir / f"{split}_macro_per_seed.csv",
                          [dict(step=step, **row) if step is not None else row for row in macros], append=split == "dev")
                write_csv(output_dir / f"{split}_macro_mean_std.csv",
                          [dict(step=step, metric=name, **summary[name]) if step is not None
                           else dict(metric=name, **summary[name]) for name in METRICS], append=split == "dev")
                return result
            class_summaries = _class_summary(rows, dataset.fake_classes, step, mode)
            write_csv(paths["per_class_seed"], rows)
            write_csv(paths["per_class_mean_std"], class_summaries)
            write_csv(paths["macro_per_seed"], [dict(step=step, mode=mode, **row) for row in macros])
            write_csv(paths["macro_mean_std"], [dict(step=step, mode=mode, metric=name, **summary[name]) for name in METRICS])
        # Close the prediction file before committing reports, including on Windows.
        reports = []
        for path in paths.values():
            final = destination / path.name
            path.replace(final)
            reports.append(str(final.resolve()))
        elapsed = time.perf_counter() - started
        _log_dev_summary(logger, prefix, class_summaries, summary)
        _log(logger, "%s END elapsed_seconds=%.6f reports=%s", prefix, elapsed, reports)
        result.update(per_class_summary=class_summaries, reports=reports, elapsed_seconds=elapsed,
                      query_counts=query_counts, mode=mode, step=step, report_schema_version=1)
        return result
    except Exception:
        if mode is not None:
            _log(logger, "%s FAILED elapsed_seconds=%.6f report_dir=%s", prefix,
                 time.perf_counter() - started, destination)
        raise
    finally:
        model.train(was_training)
        if report_temp is not None:
            report_temp.cleanup()
