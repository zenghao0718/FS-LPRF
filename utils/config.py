import argparse
import copy
from pathlib import Path
import yaml

SEEDS = [17, 42, 71, 100, 123]


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    for assignment in overrides or []:
        key, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError("Overrides must be section.key=value")
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
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
    if type(cfg["runtime"]["num_workers"]) is not int or cfg["runtime"]["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/speechfake.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    return parser


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
