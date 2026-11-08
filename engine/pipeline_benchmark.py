"""Opt-in pipeline measurements using disposable backend/optimizer state."""
from collections import defaultdict
from contextlib import contextmanager
import copy
import ctypes
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch

from data.cache_pipeline import open_train_dataset, dev_manifests, bind_evaluation_caches
from data.codec_augmentation import measure_codec_latency
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


CODEC_TRAIN_MODE = "waveform_codec"
BENCHMARK_MODES = ("none", "waveform", "xlsr_feature", CODEC_TRAIN_MODE)


def configure_benchmark_mode(local, mode):
    """Waveform mode is the unaugmented-codec baseline. waveform_codec enables the 50% round-trip."""
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    local["data"].setdefault("codec_augment", {})
    if mode == CODEC_TRAIN_MODE:
        local["data"]["train_cache_mode"] = "waveform"
        local["data"]["codec_augment"]["enabled"] = True
        local["data"]["codec_augment"]["probability"] = 0.5
    else:
        local["data"]["train_cache_mode"] = mode
        local["data"]["codec_augment"]["enabled"] = False
    return local


class _Nvml:
    def __init__(self):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError("NVML init failed")
        self.handle = ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("NVML device 0 is unavailable")

        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        self._utilization = Utilization
        self.library.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(Utilization)]
        self.library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int

    def sample(self):
        utilization = self._utilization()
        if self.library.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)) != 0:
            return None
        return int(utilization.gpu), int(utilization.memory)


class UtilizationSampler:
    """Sample GPU utilization during timed steps. CPU-only runs record a null summary."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples, self.error = [], None
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None

    def start(self, device):
        if getattr(device, "type", None) != "cuda" or self._thread is not None:
            return
        try:
            self._nvml = _Nvml()
        except Exception as error:
            self.error = str(error)
            return
        self._thread = threading.Thread(target=self._run, name="gpu-utilization", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            sample = self._nvml.sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.samples:
            return dict(mean_gpu_util_percent=None, p50_gpu_util_percent=None,
                        mean_gpu_memory_util_percent=None, gpu_util_samples=0, gpu_util_error=self.error)
        gpu = [item[0] for item in self.samples]
        memory = [item[1] for item in self.samples]
        return dict(mean_gpu_util_percent=float(np.mean(gpu)), p50_gpu_util_percent=float(np.percentile(gpu, 50)),
                    mean_gpu_memory_util_percent=float(np.mean(memory)), gpu_util_samples=len(self.samples),
                    gpu_util_error=self.error)


def benchmark_train(model, cfg, *, modes=("none", "waveform"), workers=(0, 2, 4, 8),
                    warmup=5, steps=20, dataset_factory=open_train_dataset):
    if warmup < 0 or steps <= 0:
        raise ValueError("Benchmark warmup must be nonnegative and timed steps positive")
    if any(mode not in BENCHMARK_MODES for mode in modes):
        raise ValueError(f"Benchmark modes must be chosen from {BENCHMARK_MODES}")
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    episodes = None
    reference_ids = None
    codec_latency = None
    results = []
    with preserved_rng():
        for mode in modes:
            for count in workers:
                local = configure_benchmark_mode(copy.deepcopy(cfg), mode)
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
                if mode == CODEC_TRAIN_MODE and codec_latency is None and getattr(dataset, "codec_augmentor", None) is not None:
                    codec_latency = measure_codec_latency(dataset, dataset.codec_augmentor, seed=cfg["train"]["seed"])
                model.load_state_dict(initial, strict=False)
                seed_all(cfg["train"]["seed"])
                optimizer, scheduler = make_optimizer(model, local)
                records = []
                gpu_sampler = UtilizationSampler()
                with ReusableAudioLoader(dataset, local) as reusable:
                    try:
                        for index, batch_episodes in enumerate(episodes):
                            timer = StageTimings(model.device)
                            timer.synchronize()
                            if index == warmup:
                                if timer.device.type == "cuda":
                                    torch.cuda.reset_peak_memory_stats(timer.device)
                                gpu_sampler.start(timer.device)
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
                    finally:
                        gpu_summary = gpu_sampler.stop()
                durations = [row["step_seconds"] for row in records]
                waits = [row.get("data_wait_seconds", 0) for row in records]
                reader = getattr(dataset, "cache_reader", None)
                cache = reader.stats() if reader is not None else dict(bytes=0, build_seconds=0, dtype="float32", fingerprint=None)
                codec = local["data"].get("codec_augment") or {}
                result = dict(mode=mode, num_workers=count, warmup_steps=warmup, timed_steps=steps,
                              audio_microbatch_size=cfg["train"]["audio_microbatch_size"],
                              episodes_per_step=cfg["train"]["episodes_per_step"], samples_per_step=records[0]["samples"],
                              samples_per_second=sum(row["samples"] for row in records) / sum(durations),
                              mean_step_seconds=float(np.mean(durations)), p50_step_seconds=float(np.percentile(durations, 50)),
                              p90_step_seconds=float(np.percentile(durations, 90)), cache=cache,
                              codec_enabled=bool(codec.get("enabled", False)),
                              codec_probability=float(codec.get("probability", 0) or 0),
                              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else 0,
                              prefetch_factor=cfg["runtime"]["prefetch_factor"] if count else None,
                              persistent_workers=bool(count and cfg["runtime"]["persistent_train_workers"]),
                              pin_memory=cfg["runtime"]["pin_memory"], raw_steps=records)
                result.update(gpu_summary)
                if mode == CODEC_TRAIN_MODE:
                    result["codec_latency"] = codec_latency
                    augmentor = getattr(dataset, "codec_augmentor", None)
                    result["codec_report"] = augmentor.report() if augmentor is not None else None
                result["peak_observed_rss_bytes"] = max((row["observed_rss_bytes"] for row in records
                                                        if row["observed_rss_bytes"] is not None), default=None)
                cpu_values = [row["cpu_percent_all_processes"] for row in records if row["cpu_percent_all_processes"] is not None]
                result["mean_cpu_percent_all_processes"] = float(np.mean(cpu_values)) if cpu_values else None
                for phase in ("data_wait_seconds", "xlsr_forward_seconds", "attention_bottleneck_seconds", "backward_optimizer_seconds"):
                    result[phase] = float(np.mean([row.get(phase, 0) for row in records]))
                result["p50_data_wait_seconds"] = float(np.percentile(waits, 50))
                results.append(result)
                print(f"{mode} workers={count}: {result['samples_per_second']:.2f} samples/s; "
                      f"mean step={result['mean_step_seconds']:.4f}s median={result['p50_step_seconds']:.4f}s "
                      f"data_wait={result['data_wait_seconds']:.4f}s", flush=True)
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
