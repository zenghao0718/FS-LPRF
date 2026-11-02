import hashlib
import json
import math
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from fs_lprf.config import save_config
from fs_lprf.data.speechfake_dataset import SpeechFakeDataset
from fs_lprf.data.episode_sampler import EpisodeSampler
from fs_lprf.data.evaluation_manifest import load_or_create_manifest, dataset_fingerprint
from fs_lprf.data.loaders import audio_loader
from fs_lprf.methods.loss import step_loss
from fs_lprf.utils.seed import seed_all
from fs_lprf.utils.logging import configure_logging
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluator import evaluate


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


def optimize_step(model, dataset, episodes, cfg, optimizer, step):
    """One backward and one update after all independent episodes and microbatches."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = [index for episode in episodes for index in episode.indices]
    loader = audio_loader(dataset, indices, cfg["train"]["audio_microbatch_size"],
                          cfg["train"]["seed"] + step, cfg["runtime"]["num_workers"])
    features = torch.cat([model({"feats": batch["feats"]})["feats"] for batch in loader])
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
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         cfg["train"]["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm), diagnostics


def train(cfg, resume=None):
    from fs_lprf.models.fs_lprf_model import FSLPRF
    seed_all(cfg["train"]["seed"])
    out = Path(cfg["runtime"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if not resume and (out / "checkpoints" / "last.pt").exists():
        raise ValueError("Run already exists: use --resume or a new output_dir")
    logger = configure_logging(out)
    train_ds = SpeechFakeDataset(cfg["data"], "train", train_mode=True)
    dev_ds = SpeechFakeDataset(cfg["data"], "dev")
    if train_ds.fake_classes != dev_ds.fake_classes:
        raise ValueError("Train/Dev attacker class mappings differ")
    # Generate before any training update, then validate and reuse unchanged on resume.
    manifest_path = out / "evaluation" / "dev_episode_manifest.json"
    if resume and not manifest_path.exists():
        raise ValueError("Resume requires the original saved Dev episode manifest")
    manifest = load_or_create_manifest(dev_ds, manifest_path, cfg["validation"]["shot"], cfg["validation"]["seeds"])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    train_fingerprint = dataset_fingerprint(train_ds)
    model = FSLPRF(cfg)
    optimizer, scheduler = make_optimizer(model, cfg)
    sampler = EpisodeSampler(train_ds.class_indices, **{key: cfg["train"][key] for key in
                             ("way", "support_per_class", "total_query_per_episode", "episodes_per_step")})
    start, best, dev_metrics = 0, float("inf"), None
    if resume:
        payload = load_checkpoint(resume, model, optimizer, scheduler, restore_random=True)
        if payload["config"]["train"] != cfg["train"] or payload["config"]["runtime"]["num_workers"] != cfg["runtime"]["num_workers"]:
            raise ValueError("Resume must retain the training configuration and worker count")
        if payload["dev_manifest_sha256"] != manifest_hash or payload["train_fingerprint"] != train_fingerprint:
            raise ValueError("Resume data or saved Dev manifest changed")
        start, best, dev_metrics = payload["step"], payload["best_eer"], payload["dev_metrics"]
    save_config(cfg, out / "config_resolved.yaml")
    episode_path = out / "episodes.jsonl"
    with SummaryWriter(out / "tensorboard") as writer, episode_path.open("a", encoding="utf-8") as episode_log:
        for step in range(start + 1, cfg["train"]["total_steps"] + 1):
            episodes = sampler.sample_step()
            for index, episode in enumerate(episodes):
                episode_log.write(json.dumps(dict(step=step, episode=index, classes=episode.classes,
                    support_ids=[train_ds.uttids[i] for i in episode.support_indices],
                    query_ids=[train_ds.uttids[i] for i in episode.query_indices])) + "\n")
            lr = optimizer.param_groups[0]["lr"]
            loss, norm, _ = optimize_step(model, train_ds, episodes, cfg, optimizer, step)
            scheduler.step()
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/learning_rate", lr, step)
            writer.add_scalar("train/gradient_norm", norm, step)
            logger.info("step=%d loss=%.6f lr=%.8g gradient_norm=%.6f", step, loss, lr, norm)
            improved = False
            if step % cfg["train"]["validation_interval_steps"] == 0 or step == cfg["train"]["total_steps"]:
                dev_metrics = evaluate(model, dev_ds, manifest, cfg, out / "evaluation", "dev", step)
                improved = dev_metrics["mean_eer"] < best
                best = min(best, dev_metrics["mean_eer"])
                writer.add_scalar("dev/macro_eer_mean", dev_metrics["mean_eer"], step)
                writer.add_scalar("dev/macro_eer_std", dev_metrics["std_eer"], step)
                logger.info("dev step=%d macro_eer=%.6f std=%.6f best=%.6f", step,
                            dev_metrics["mean_eer"], dev_metrics["std_eer"], best)
            kwargs = dict(model=model, optimizer=optimizer, scheduler=scheduler, config=cfg,
                          step=step, dev_metrics=dev_metrics, best_eer=best,
                          manifest_hash=manifest_hash, train_fingerprint=train_fingerprint)
            save_checkpoint(out / "checkpoints" / "last.pt", **kwargs)
            if improved:
                save_checkpoint(out / "checkpoints" / "best.pt", **kwargs)
            episode_log.flush()
