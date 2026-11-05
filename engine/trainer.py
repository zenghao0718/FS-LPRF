import json
import math
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from utils.config import save_config
from data.speechfake_dataset import SpeechFakeDataset
from data.episode_sampler import EpisodeSampler
from data.evaluation_manifest import load_or_create_manifest, load_or_create_light_manifest, dataset_fingerprint
from data.loaders import ReusableAudioLoader
from data.cache import atomic_json
from data.cache_pipeline import open_train_dataset, bind_evaluation_caches, cache_bindings
from models.fs_lprf_model import forward_batch
from methods.loss import step_loss
from utils.seed import seed_all, preserved_rng
from utils.logging import configure_logging
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluator import evaluate
from .validation_schedule import (ValidationHistory, check_new_run,
                                  file_sha256, make_run_identity, validation_mode)


def make_optimizer(model, cfg):
    train = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=train["learning_rate"], betas=tuple(train["betas"]),
                      weight_decay=train["weight_decay"])
    def factor(index):
        update = index + 1
        if update <= train["warmup_steps"]:
            return update / train["warmup_steps"]
        progress = min(1.0, (update - train["warmup_steps"]) /
                       (train["total_steps"] - train["warmup_steps"]))
        minimum = train["min_learning_rate"] / train["learning_rate"]
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return optimizer, LambdaLR(optimizer, factor)


def optimize_step(model, dataset, episodes, cfg, optimizer, step, *, loader=None, timings=None):
    """One backward and one update after all independent episodes and microbatches."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = [index for episode in episodes for index in episode.indices]
    from contextlib import nullcontext
    with nullcontext(loader) if loader is not None else ReusableAudioLoader(dataset, cfg) as step_loader:
        features = torch.cat([forward_batch(model, batch) for batch in step_loader.for_step(indices, step)])
    offset, tasks = 0, []
    for episode in episodes:
        ns, nq = len(episode.support_indices), len(episode.query_indices)
        support = features[offset:offset + ns]
        query = features[offset + ns:offset + ns + nq]
        tasks.append((support, torch.tensor(episode.support_labels, device=features.device),
                      query, torch.tensor(episode.query_labels, device=features.device)))
        offset += ns + nq
    loss, diagnostics = step_loss(tasks)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite Step loss at {step}")
    with timings.measure("backward_optimizer_seconds") if timings else nullcontext():
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             cfg["train"]["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
    return float(loss.detach()), float(norm), diagnostics


def record_validation(writer, result, mode, step):
    for name in ("eer", "auc", "ap", "acc"):
        if mode == "light":
            writer.add_scalar(f"dev/light_macro_{name}", result["metrics"][name]["mean"], step)
        else:
            for statistic in ("mean", "std"):
                writer.add_scalar(f"dev/full_macro_{name}_{statistic}",
                                  result["metrics"][name][statistic], step)
    writer.flush()


def full_validation(model, dev_ds, manifest, cfg, history, step, writer, logger):
    with preserved_rng():
        result = evaluate(model, dev_ds, manifest, cfg, history.out / "evaluation", "dev",
                          step, mode="full", logger=logger)
    history.complete(step, result)
    record_validation(writer, result, "full", step)
    best = history.best
    logger.info("[FULL DEV step=%04d] best_step=%04d best_macro_eer=%.6f reports=%s",
                step, best["step"], best["summary"]["mean_eer"], result["reports"])


def run_steps(model, train_ds, dev_ds, sampler, cfg, optimizer, scheduler,
              full_manifest, light_manifest, history, manifest_hash, train_fingerprint,
              start, writer, logger, episode_log, *, loader=None):
    # A node is durable before validation. Recover validation first if its transaction
    # was interrupted; never skip that node's opportunity to become Best.
    if start:
        completed = next((row for row in history.records if row["step"] == start), None)
        if completed is None:
            logger.info("Recovering incomplete full Dev at milestone %04d before step %04d", start, start + 1)
            full_validation(model, dev_ds, full_manifest, cfg, history, start, writer, logger)
        else:
            # The completion marker may have committed just before a crash that lost
            # TensorBoard's queued events. Replay its summary, never its inference.
            record_validation(writer, completed["summary"], "full", start)
    for step in range(start + 1, cfg["train"]["total_steps"] + 1):
        episodes = sampler.sample_step()
        for index, episode in enumerate(episodes):
            episode_log.write(json.dumps(dict(step=step, episode=index, classes=episode.classes,
                support_ids=[train_ds.uttids[i] for i in episode.support_indices],
                query_ids=[train_ds.uttids[i] for i in episode.query_indices])) + "\n")
        lr = optimizer.param_groups[0]["lr"]
        if loader is None:
            loss, norm, _ = optimize_step(model, train_ds, episodes, cfg, optimizer, step)
        else:
            loss, norm, _ = optimize_step(model, train_ds, episodes, cfg, optimizer, step, loader=loader)
        scheduler.step()
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/learning_rate", lr, step)
        writer.add_scalar("train/gradient_norm", norm, step)
        logger.info("step=%d loss=%.6f lr=%.8g gradient_norm=%.6f", step, loss, lr, norm)
        episode_log.flush()
        mode = validation_mode(step, cfg["train"])
        if mode == "full":
            best = history.best
            save_checkpoint(history.checkpoint(step), model=model, optimizer=optimizer,
                            scheduler=scheduler, config=cfg, step=step,
                            dev_metrics=best["summary"] if best else None,
                            best_eer=best["summary"]["mean_eer"] if best else float("inf"),
                            manifest_hash=manifest_hash, train_fingerprint=train_fingerprint,
                            run_identity=history.identity)
            logger.info("Saved milestone checkpoint before full Dev: %s", history.checkpoint(step))
            full_validation(model, dev_ds, full_manifest, cfg, history, step, writer, logger)
        elif mode == "light":
            with preserved_rng():
                result = evaluate(model, dev_ds, light_manifest, cfg, history.out / "evaluation", "dev",
                                  step, mode="light", logger=logger)
            record_validation(writer, result, "light", step)


def train(cfg, resume=None):
    from models.fs_lprf_model import FSLPRF
    from .checkpoint import resolve_checkpoint
    from contextlib import ExitStack
    import uuid
    seed_all(cfg["train"]["seed"])
    out = Path(cfg["runtime"]["output_dir"]).resolve()
    if not resume:
        check_new_run(out)
    out.mkdir(parents=True, exist_ok=True)
    train_ds = open_train_dataset(cfg)
    dev_ds = SpeechFakeDataset(cfg["data"], "dev")
    if train_ds.fake_classes != dev_ds.fake_classes:
        raise ValueError("Train/Dev attacker class mappings differ")
    manifest_path = out / "evaluation" / "dev_episode_manifest.json"
    light_path = out / "evaluation" / "dev_light_manifest.json"
    if resume and (not manifest_path.exists() or not light_path.exists()):
        raise ValueError("Resume requires the original full and light Dev manifests")
    manifest = load_or_create_manifest(dev_ds, manifest_path, cfg["validation"]["shot"], cfg["validation"]["seeds"])
    light = load_or_create_light_manifest(dev_ds, manifest, manifest_path, light_path,
                cfg["validation"]["shot"], cfg["validation"]["light_seed"],
                cfg["validation"]["light_max_query_per_class"])
    manifest_hash, train_fingerprint = file_sha256(manifest_path), dataset_fingerprint(train_ds)
    bind_evaluation_caches(dev_ds, cfg, {"full": (manifest, manifest_path), "light": (light, light_path)})
    bindings, cache_metadata = cache_bindings(train_ds, dev_ds)
    identity = make_run_identity(cfg, manifest_hash, file_sha256(light_path), train_fingerprint, bindings)
    history = ValidationHistory(out, cfg, identity, manifest_hash, train_fingerprint)
    start, resume_path = 0, None
    if resume:
        resume_path = resolve_checkpoint(resume, out)
        metadata = torch.load(resume_path, map_location="cpu", weights_only=True)
        start = metadata["step"]
        if resume_path != history.checkpoint(start).resolve() or metadata.get("run_identity") != identity:
            raise ValueError("Resume must use a matching milestone in the original output directory")
        if metadata["dev_manifest_sha256"] != manifest_hash or metadata["train_fingerprint"] != train_fingerprint:
            raise ValueError("Resume data or full Dev manifest changed")
        history.initialize(start)
    else:
        history.initialize()
        save_config(cfg, out / "config_resolved.yaml")
        atomic_json(out / "cache_bindings.json", bindings)
        for fingerprint, metadata in cache_metadata.items():
            atomic_json(out / "cache_metadata" / f"{fingerprint}.json", metadata)
    logger = configure_logging(out)
    model = FSLPRF(cfg)
    optimizer, scheduler = make_optimizer(model, cfg)
    if resume_path:
        load_checkpoint(resume_path, model, optimizer, scheduler, restore_random=True,
                        expected_run_identity=identity)
        logger.info("RESUME completed_step=%04d device=%s; replay unsaved steps; retaining previous attempt logs",
                    start, cfg["runtime"]["device"])
    sampler = EpisodeSampler(train_ds.class_indices, **{key: cfg["train"][key] for key in
                             ("way", "support_per_class", "total_query_per_episode", "episodes_per_step")})
    # Keep the previous attempt's episode log. TensorBoard discards events after the
    # restored step, then receives a continuous sequence of replayed training steps.
    episode_path = out / (f"episodes_resume_step{start:04d}_{uuid.uuid4().hex[:8]}.jsonl" if resume else "episodes.jsonl")
    with ExitStack() as stack:
        with preserved_rng():
            writer = stack.enter_context(SummaryWriter(out / "tensorboard", flush_secs=10,
                                                       purge_step=start + 1 if resume else None))
        episode_log = stack.enter_context(episode_path.open("x", encoding="utf-8"))
        loader = stack.enter_context(ReusableAudioLoader(train_ds, cfg))
        run_steps(model, train_ds, dev_ds, sampler, cfg, optimizer, scheduler,
                  manifest, light, history, manifest_hash, train_fingerprint,
                  start, writer, logger, episode_log, loader=loader)
