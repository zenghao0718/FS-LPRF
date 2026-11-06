"""Opt-in pipeline measurements using disposable backend/optimizer state."""
from collections import defaultdict
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch

from data.cache_pipeline import open_train_dataset, dev_manifests, bind_evaluation_caches
from data.cache import source_digest, file_sha, canonical_sha, PREPROCESSING
from data.waveform_cache import open_waveform_cache
from data.randomness import RNG_SCHEMA_VERSION
from data.loaders import ReusableAudioLoader, worker_options
from data.speechfake_dataset import SpeechFakeDataset
from data.episode_sampler import EpisodeSampler
from utils.seed import preserved_rng, seed_all
from .trainer import make_optimizer, optimize_step
from .evaluator import evaluate
from .validation_schedule import validation_mode


def process_usage(pids):
    """Linux AutoDL process+worker RSS and CPU counters; no extra dependency."""
    if not Path("/proc/self/stat").is_file():
        return None
    cpu, rss = 0., 0
    for pid in pids:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(cpu_seconds=cpu, rss_bytes=rss)


def loader_pids(reusable):
    iterator = getattr(reusable.loader, "_iterator", None)
    return [os.getpid()] + [worker.pid for worker in getattr(iterator, "_workers", [])]


def dev_measurement_identity(cfg):
    dataset = SpeechFakeDataset(cfg["data"], "dev")
    root = Path(cfg["data"]["data_path"])
    source = (open_waveform_cache(dataset, cfg).identity["source_audio_sha256"]
              if cfg["data"]["dev_cache_mode"] == "waveform" else source_digest(dataset, dataset.uttids))
    return canonical_sha(dict(source_audio_sha256=source,
             manifests={suffix: file_sha(root / f"dev.{suffix}") for suffix in ("tsv", "txt")},
             frontend_sha256=file_sha(cfg["model"]["frontend_path"]), validation=cfg["validation"],
             preprocessing=PREPROCESSING, rng_schema_version=RNG_SCHEMA_VERSION,
             microbatch=cfg["train"]["audio_microbatch_size"], workers=worker_options(cfg),
             cache={key: cfg["cache"][key] for key in ("schema_version", "feature_dtype", "shard_size", "verify_checksums")}))


class StageTimings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds, self.events = defaultdict(float), []

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if self.device.type == "cuda" and name != "data_wait_seconds":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                self.events.append((name, start, end))
        else:
            started = time.perf_counter()
            try:
                yield
            finally:
                self.seconds[name] += time.perf_counter() - started

    def finish(self):
        self.synchronize()
        for name, start, end in self.events:
            self.seconds[name] += start.elapsed_time(end) / 1000
        return dict(self.seconds)


class TimedLoader:
    def __init__(self, loader, timings):
        self.loader, self.timings = loader, timings

    def for_step(self, indices, step):
        iterator = iter(self.loader.for_step(indices, step))
        while True:
            try:
                with self.timings.measure("data_wait_seconds"):
                    batch = next(iterator)
            except StopIteration:
                return
            yield batch


@contextmanager
def timed_backend(model, timings):
    originals = []
    targets = [(model, "extract_features", "xlsr_forward_seconds"),
               (model, "tnd_forward", "attention_bottleneck_seconds")]
    if hasattr(model, "bottleneck_proj"):
        targets.append((model.bottleneck_proj, "forward", "attention_bottleneck_seconds"))
    if not hasattr(model, "tnd_forward"):
        targets.append((model, "forward", "attention_bottleneck_seconds"))
    try:
        for owner, name, phase in targets:
            if not hasattr(owner, name):
                continue
            original = getattr(owner, name)
            def wrapped(*args, _original=original, _phase=phase, **kwargs):
                with timings.measure(_phase):
                    return _original(*args, **kwargs)
            originals.append((owner, name, original))
            setattr(owner, name, wrapped)
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def benchmark_train(model, cfg, *, modes=("none", "waveform"), workers=(0, 2, 4, 8),
                    warmup=5, steps=20, dataset_factory=open_train_dataset):
    if warmup < 0 or steps <= 0:
        raise ValueError("Benchmark warmup must be nonnegative and timed steps positive")
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    episodes = None
    reference_ids = None
    results = []
    with preserved_rng():
        for mode in modes:
            for count in workers:
                local = copy.deepcopy(cfg)
                local["data"]["train_cache_mode"] = mode
                local["runtime"].pop("num_workers", None)
                local["runtime"]["train_num_workers"] = count
                dataset = dataset_factory(local)
                ids = getattr(dataset, "uttids", None)
                if episodes is None:
                    seed_all(cfg["train"]["seed"])
                    sampler = EpisodeSampler(dataset.class_indices, **{key: cfg["train"][key] for key in
                        ("way", "support_per_class", "total_query_per_episode", "episodes_per_step")})
                    episodes = [sampler.sample_step() for _ in range(warmup + steps)]
                    reference_ids = ids
                elif ids != reference_ids:
                    raise ValueError("Benchmark modes must use identical sample/index order")
                model.load_state_dict(initial, strict=False)
                seed_all(cfg["train"]["seed"])
                optimizer, scheduler = make_optimizer(model, local)
                records = []
                with ReusableAudioLoader(dataset, local) as reusable:
                    for index, batch_episodes in enumerate(episodes):
                        timer = StageTimings(model.device)
                        timer.synchronize()
                        if index == warmup and timer.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(timer.device)
                        usage_before = process_usage(loader_pids(reusable))
                        started = time.perf_counter()
                        with timed_backend(model, timer):
                            optimize_step(model, dataset, batch_episodes, local, optimizer, index + 1,
                                          loader=TimedLoader(reusable, timer), timings=timer)
                        scheduler.step()
                        timing = timer.finish()
                        timing["step_seconds"] = time.perf_counter() - started
                        usage_after = process_usage(loader_pids(reusable))
                        timing["observed_rss_bytes"] = usage_after["rss_bytes"] if usage_after else None
                        timing["cpu_percent_all_processes"] = (100 * (usage_after["cpu_seconds"] - usage_before["cpu_seconds"]) /
                                            timing["step_seconds"]) if usage_before and usage_after else None
                        timing["samples"] = sum(len(episode.indices) for episode in batch_episodes)
                        if index >= warmup:
                            records.append(timing)
                durations = [row["step_seconds"] for row in records]
                reader = getattr(dataset, "cache_reader", None)
                cache = reader.stats() if reader is not None else dict(bytes=0, build_seconds=0, dtype="float32", fingerprint=None)
                result = dict(mode=mode, num_workers=count, warmup_steps=warmup, timed_steps=steps,
                              audio_microbatch_size=cfg["train"]["audio_microbatch_size"],
                              episodes_per_step=cfg["train"]["episodes_per_step"], samples_per_step=records[0]["samples"],
                              samples_per_second=sum(row["samples"] for row in records) / sum(durations),
                              mean_step_seconds=float(np.mean(durations)), p50_step_seconds=float(np.percentile(durations, 50)),
                              p90_step_seconds=float(np.percentile(durations, 90)), cache=cache,
                              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else 0,
                              prefetch_factor=cfg["runtime"]["prefetch_factor"] if count else None,
                              persistent_workers=bool(count and cfg["runtime"]["persistent_train_workers"]),
                              pin_memory=cfg["runtime"]["pin_memory"], raw_steps=records)
                result["peak_observed_rss_bytes"] = max((row["observed_rss_bytes"] for row in records
                                                        if row["observed_rss_bytes"] is not None), default=None)
                cpu_values = [row["cpu_percent_all_processes"] for row in records if row["cpu_percent_all_processes"] is not None]
                result["mean_cpu_percent_all_processes"] = float(np.mean(cpu_values)) if cpu_values else None
                for phase in ("data_wait_seconds", "xlsr_forward_seconds", "attention_bottleneck_seconds", "backward_optimizer_seconds"):
                    result[phase] = float(np.mean([row.get(phase, 0) for row in records]))
                results.append(result)
                print(f"{mode} workers={count}: {result['samples_per_second']:.2f} samples/s; "
                      f"mean step={result['mean_step_seconds']:.4f}s", flush=True)
    model.load_state_dict(initial, strict=False)
    return results


def benchmark_dev(model, cfg, output_dir):
    """Explicit AutoDL-only measurement; never creates/reads Test."""
    dataset = SpeechFakeDataset(cfg["data"], "dev")
    manifests = dev_manifests(dataset, cfg)
    rows = {}
    for cache_mode in ("none", "waveform"):
        local = copy.deepcopy(cfg)
        local["data"]["dev_cache_mode"] = cache_mode
        readers = bind_evaluation_caches(dataset, local, manifests)
        row = dict(cache_build_seconds=sum(reader.metadata["build_seconds"] for reader in readers),
                   cache_bytes=sum(reader.stats()["bytes"] for reader in readers),
                   cache_fingerprints=[reader.metadata["fingerprint"] for reader in readers])
        for mode, step in (("light", 200), ("full", 800)):
            timer = StageTimings(model.device)
            timer.synchronize()
            started = time.perf_counter()
            evaluate(model, dataset, manifests[mode][0], local, output_dir / cache_mode, "dev", step, mode=mode)
            timer.synchronize()
            row[f"{mode}_seconds"] = time.perf_counter() - started
            row[f"{mode}_cache_build_seconds"] = sum(reader.metadata["build_seconds"] for reader in readers
                                                        if reader.identity["validation_mode"] in (None, mode))
        rows[cache_mode] = row
    return rows


def total_experiment_times(train_rows, dev_rows, cfg):
    totals = []
    light = sum(validation_mode(step, cfg["train"]) == "light" for step in range(1, cfg["train"]["total_steps"] + 1))
    full = sum(validation_mode(step, cfg["train"]) == "full" for step in range(1, cfg["train"]["total_steps"] + 1))
    for row in train_rows:
        dev = dev_rows.get("none" if row["mode"] == "none" else "waveform")
        total = None if dev is None else (row["cache"]["build_seconds"] + dev["cache_build_seconds"] +
                    cfg["train"]["total_steps"] * row["mean_step_seconds"] + light * dev["light_seconds"] + full * dev["full_seconds"])
        totals.append(dict(mode=row["mode"], num_workers=row["num_workers"], total_seconds=total,
                           light_validations=light, full_validations=full,
                           status="measured components, projected full experiment" if dev else "unmeasured Dev timings; no total estimate"))
    return totals
