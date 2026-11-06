import argparse
import copy
from pathlib import Path
import yaml

SEEDS = [17, 42, 71, 100, 123]

CACHE_DEFAULTS = dict(root="cache/fs_lprf", schema_version=1, feature_dtype="float32",
                      feature_build_batch_size=8, shard_size=64, verify_checksums=True,
                      train_feature_seed=None, warn_if_estimated_gb_above=100)


def add_pipeline_defaults(cfg):
    for split, default in (("train", "waveform"), ("dev", "waveform"), ("test", "none")):
        cfg["data"].setdefault(f"{split}_cache_mode", default)
    cache = cfg.setdefault("cache", {})
    for key, value in CACHE_DEFAULTS.items():
        cache.setdefault(key, value)
    for key, value in dict(train_num_workers=4, eval_num_workers=2, persistent_train_workers=True, persistent_eval_workers=True,
                           prefetch_factor=2, pin_memory=True, worker_torch_num_threads=1).items():
        cfg["runtime"].setdefault(key, value)


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    add_pipeline_defaults(cfg)
    for assignment in overrides or []:
        key, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError("Overrides must be section.key=value")
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target and key != "runtime.num_workers":
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    train = cfg["train"]
    if "validation_interval_steps" in train:
        raise ValueError("Replace train.validation_interval_steps with light_validation_interval_steps and full_validation_interval_steps")
    for key in ("total_query_per_episode", "episodes_per_step", "audio_microbatch_size",
                "light_validation_interval_steps", "full_validation_interval_steps"):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"train.{key} must be a positive integer")
    if train["way"] == 3 and train["total_query_per_episode"] % train["way"]:
        raise ValueError("total_query_per_episode must be divisible by way")
    fixed = {
        "data": dict(max_samples=64000, fix_len=True, trim_silence=True,
                     do_augment=True, augment_type="rawboost"),
        "model": dict(frontend="XLSR", frontend_dim=1024, num_layers=25,
                      layer_attn_head=4, reduce_dim=False, attn_cfg="default",
                      have_padding_mask=False, freeze_frontend=True, bottleneck_dim=256),
        "train": dict(seed=42, way=3, support_per_class=5, total_steps=4000,
                      total_query_per_episode=15, episodes_per_step=8,
                      optimizer="AdamW", learning_rate=1e-4, betas=[0.9, 0.999],
                      weight_decay=1e-4, warmup_steps=100, min_learning_rate=1e-6,
                      gradient_clip=1.0),
        "validation": dict(shot=10, seeds=SEEDS, light_seed=42),
        "test": dict(shot=10, seeds=SEEDS),
    }
    for section, values in fixed.items():
        for key, expected in values.items():
            if cfg[section][key] != expected:
                raise ValueError(f"Phase 1 protocol requires {section}.{key}={expected!r}")
    light, full = train["light_validation_interval_steps"], train["full_validation_interval_steps"]
    if full % light:
        raise ValueError("full_validation_interval_steps must be divisible by light_validation_interval_steps")
    if train["total_steps"] % full:
        raise ValueError("total_steps must be divisible by full_validation_interval_steps")
    cap = cfg["validation"]["light_max_query_per_class"]
    if type(cap) is not int or not 1 <= cap <= 1000:
        raise ValueError("validation.light_max_query_per_class must be an integer between 1 and 1000")
    if type(cfg["validation"]["light_seed"]) is not int:
        raise ValueError("Phase 1 protocol requires validation.light_seed=42 as an integer")
    for split in ("train", "dev", "test"):
        allowed = ("none",) if split == "test" else ("none", "waveform")
        if cfg["data"][f"{split}_cache_mode"] not in allowed:
            raise ValueError(f"Invalid data.{split}_cache_mode")
    runtime, cache = cfg["runtime"], cfg["cache"]
    for key in ("train_num_workers", "eval_num_workers", "num_workers"):
        if key in runtime and (type(runtime[key]) is not int or runtime[key] < 0):
            raise ValueError(f"runtime.{key} must be a nonnegative integer")
    for section, keys in ((runtime, ("prefetch_factor", "worker_torch_num_threads")),
                           (cache, ("feature_build_batch_size", "shard_size"))):
        if any(type(section[key]) is not int or section[key] <= 0 for key in keys):
            raise ValueError("Worker threads, prefetch, shard and feature build batch size must be positive integers")
    for section, keys in ((runtime, ("persistent_train_workers", "persistent_eval_workers", "pin_memory")), (cache, ("verify_checksums",))):
        if any(type(section[key]) is not bool for key in keys):
            raise ValueError("Persistent workers, pin_memory and verify_checksums must be booleans")
    if cache["schema_version"] != 1 or type(cache["schema_version"]) is not int:
        raise ValueError("Only cache.schema_version=1 is supported")
    if cache["feature_dtype"] not in ("float32", "float16"):
        raise ValueError("cache.feature_dtype must be float32 or float16")
    if cache["train_feature_seed"] is not None and (type(cache["train_feature_seed"]) is not int or cache["train_feature_seed"] < 0):
        raise ValueError("cache.train_feature_seed must be null or a nonnegative integer")
    if not isinstance(cache["root"], str) or not cache["root"]:
        raise ValueError("cache.root must be a nonempty path")
    root, project = Path(cache["root"]).resolve(), Path(__file__).resolve().parents[1]
    if root.is_relative_to(project) and (root == project or root.relative_to(project).parts[0] not in ("cache", "outputs", ".cache", ".tmp")):
        raise ValueError("In-repository cache.root must be under a Git-ignored cache/output directory")
    if type(cache["warn_if_estimated_gb_above"]) not in (int, float) or cache["warn_if_estimated_gb_above"] <= 0:
        raise ValueError("cache.warn_if_estimated_gb_above must be positive")


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/speechfake.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    return parser


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
