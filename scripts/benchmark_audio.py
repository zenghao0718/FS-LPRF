"""Real-only data-loading benchmark. No model, formal manifests or feature cache."""
import _bootstrap

from collections import defaultdict
import copy
import ctypes as ct
from ctypes import wintypes
import hashlib
import inspect
import io
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
import torch

from data.audio import FADTrim, get_audio
from data.audio_benchmark import (RealAudioBenchmarkDataset, scan_real_audio,
                                 distribution, waveform_digest)
from data.cache import atomic_json, file_sha
from data.cache_pipeline import view_tokens
from data.loaders import AudioDataLoader, ReusableAudioLoader, PreprocessedWaveformDataset
from data.randomness import sample_rng, sample_seed
from utils.config import common_parser, load_config


class ProcessUsage:
    """Sample parent + actual multiprocessing children, without new dependencies."""
    def __init__(self):
        self.handles = {}
        if os.name == "nt":
            class Memory(ct.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (name, ct.c_size_t) for name in
                    ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                     "nonpaged", "pagefile", "peak_pagefile")]
            self.Memory = Memory
            self.kernel = ct.WinDLL("kernel32", use_last_error=True)
            self.psapi = ct.WinDLL("psapi", use_last_error=True)
            self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ct.POINTER(ct.c_uint64)] * 4
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ct.POINTER(Memory), wintypes.DWORD]

    def read(self, pid):
        if os.name == "nt":
            if pid not in self.handles:
                handle = self.kernel.OpenProcess(0x1000 | 0x0010, False, pid)
                if not handle:
                    return None
                self.handles[pid] = handle
            handle = self.handles[pid]
            created, exited, kernel, user = (ct.c_uint64() for _ in range(4))
            if not self.kernel.GetProcessTimes(handle, ct.byref(created), ct.byref(exited),
                                               ct.byref(kernel), ct.byref(user)):
                return None
            memory = self.Memory()
            memory.cb = ct.sizeof(memory)
            self.psapi.GetProcessMemoryInfo(handle, ct.byref(memory), ct.sizeof(memory))
            return dict(cpu_seconds=(kernel.value + user.value) / 1e7, rss_bytes=memory.rss)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return dict(cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
                        rss_bytes=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ProcessLookupError):
            return None

    def close(self):
        if os.name == "nt":
            for handle in self.handles.values():
                self.kernel.CloseHandle(handle)
        self.handles.clear()


class UsageMonitor:
    def __enter__(self):
        self.usage, self.cpu, self.pids = ProcessUsage(), {}, {os.getpid()}
        self.baseline = self.usage.read(os.getpid())["cpu_seconds"]
        self.peak_rss = 0
        self.errors = []
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        self.pids.update(process.pid for process in mp.active_children() if process.pid)
        rss = 0
        for pid in self.pids:
            row = self.usage.read(pid)
            if row is not None:
                self.cpu[pid] = max(self.cpu.get(pid, 0), row["cpu_seconds"])
                rss += row["rss_bytes"]
        self.peak_rss = max(self.peak_rss, rss)

    def _run(self):
        while not self.stop.is_set():
            try:
                self._sample()
            except Exception as error:
                self.errors.append(repr(error))
            self.stop.wait(.05)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self._sample()
        elapsed = time.perf_counter() - self.started
        self.result = dict(cpu_percent_one_core=100 * (sum(self.cpu.values()) - self.baseline) / elapsed,
                           peak_sampled_rss_bytes=self.peak_rss, sampling_interval_seconds=.05,
                           observed_pids=sorted(self.pids), measurement_errors=self.errors,
                           note="Parent plus workers; 100% is one core. RSS summed, sampled (not allocation peak).")
        self.usage.close()


def prepare_reference(dataset):
    references, full_hashes, good, failures, changed = {}, {}, [], [], []
    started = time.perf_counter()
    for row in dataset.records:
        uid = row["uttid"]
        try:
            # The original FADTrim path, before any Loader or cache optimization.
            whole = FADTrim.load_preprocessed_waveform(dataset, uid)
            if not torch.isfinite(whole).all():
                raise ValueError("Non-finite full waveform")
            full_hashes[uid] = waveform_digest(whole)
            seed = sample_seed("view", dataset.split, "full", 42, uid)
            with sample_rng(seed):
                first = dataset.sample_segment(whole)
            with sample_rng(seed):
                repeated = dataset.sample_segment(whole)
            if not torch.equal(first, repeated):
                raise AssertionError("Same seed produced different crops")
            with sample_rng(sample_seed("view", dataset.split, "full", 71, uid)):
                alternate = dataset.sample_segment(whole)
            changed.append(not torch.equal(first, alternate))
            references[uid] = waveform_digest(first)
            good.append(row)
        except Exception as error:
            failures.append(dict(uttid=uid, path=row["path"], error=repr(error)))
    return good, references, full_hashes, dict(
        files_tested=len(dataset), valid_files=len(good), processing_failures=failures,
        same_seed_crop_exact=True, different_seed_changed_files=sum(changed),
        different_seed_checked_files=len(changed), seconds=time.perf_counter() - started,
        reference="Unchanged FADTrim.load_preprocessed_waveform + FADTrim.sample_segment")


def profile_preprocessing(dataset, count=64, rounds=3):
    """Time lines of the existing function; never reimplement the audio algorithm.

    Disk bytes are read separately, and get_audio decodes the same bytes in memory.
    This separates file I/O from libsndfile decoding, which normally interleave.
    """
    lines, base = inspect.getsourcelines(get_audio)
    markers = [("    with sf.SoundFile", "decode"),
               ("    if data.ndim != 1:", "mono"),
               ("    if sample_rate !=", "resample"),
               ("    if to_mono:", "mono"),
               ("    if trim_sil:", "silence_trim"),
               ("    if norm:", "rms_normalize"),
               ("    data = data.reshape", "tensor_conversion")]
    boundaries = sorted((base + next(i for i, line in enumerate(lines) if line.startswith(marker)), name)
                        for marker, name in markers)
    def phase(line):
        return next((name for start, name in reversed(boundaries) if line >= start), "decode")
    selected = dataset.records[:count]
    # Warm every selected source/rate before instrumented rounds.
    for row in selected[:8]:
        get_audio(row["path"])
    records = []
    for repeat in range(rounds):
        totals = defaultdict(float)
        for row in selected:
            tick = time.perf_counter()
            raw = Path(row["path"]).read_bytes()
            totals["file_read"] += time.perf_counter() - tick
            last = [None, None]
            def tracer(frame, event, arg):
                if frame.f_code is not get_audio.__code__:
                    return None
                now = time.perf_counter()
                if last[0] is not None:
                    totals[last[0]] += now - last[1]
                if event == "return":
                    last[:] = [None, None]
                else:
                    last[:] = [phase(frame.f_lineno), time.perf_counter()]
                return tracer
            previous = sys.gettrace()
            try:
                sys.settrace(tracer)
                whole, _ = get_audio(io.BytesIO(raw))
            finally:
                sys.settrace(previous)
            with sample_rng(42):
                tick = time.perf_counter()
                dataset.sample_segment(whole)
                totals["fixed_length_crop"] += time.perf_counter() - tick
        records.append(dict(repeat=repeat, warmup=repeat == 0, **totals))
    means = {name: float(np.mean([row.get(name, 0) for row in records[1:]])) / len(selected)
             for name in ("file_read", "decode", "resample", "mono", "silence_trim",
                          "rms_normalize", "fixed_length_crop", "tensor_conversion")}
    return dict(files=len(selected), rounds=rounds, seconds_per_file=means, raw_rounds=records,
                resampling_executed_files=sum(row["sample_rate"] != 16000 for row in selected),
                mono_conversion_executed_files=sum(row["channels"] > 1 for row in selected),
                note="Original get_audio on BytesIO plus line timing; includes tracing overhead. "
                     "File read uses OS cache. These stage timings are diagnostic, not throughput measurements.")


def benchmark_case(dataset, cfg, references, *, workers, persistent, prefetch, strategy, rounds):
    local = copy.deepcopy(cfg)
    local["runtime"].pop("num_workers", None)
    local["runtime"].update(eval_num_workers=workers, persistent_eval_workers=persistent,
                            prefetch_factor=prefetch)
    processes, starts, records = {}, [], []
    original_iter = AudioDataLoader.__iter__
    def observed_iter(owner):
        tick = time.perf_counter()
        iterator = original_iter(owner)
        starts.append(time.perf_counter() - tick)
        processes.update({worker.pid: worker for worker in getattr(iterator, "_workers", [])})
        return iterator
    reusable = None
    total_started = time.perf_counter()
    with UsageMonitor() as monitor, patch.object(AudioDataLoader, "__iter__", observed_iter):
        try:
            for repeat in range(rounds):
                round_started = time.perf_counter()
                created = 0.
                if reusable is None:
                    tick = time.perf_counter()
                    reusable = ReusableAudioLoader(dataset, local, training=False, seed=42)
                    created = time.perf_counter() - tick
                indices = list(range(len(dataset)))
                shift = (repeat * 17) % len(indices)
                indices = indices[shift:] + indices[:shift]
                if repeat % 2:
                    indices.reverse()
                expected = [dataset.uttids[index] for index in indices]
                tokens = view_tokens(dataset, expected, 42, "full")
                stream = reusable.for_indices(tokens)
                load_started = time.perf_counter()
                waits, seen, last_size = [], [], 0
                before_pids = set(processes)
                while True:
                    tick = time.perf_counter()
                    try:
                        batch = next(stream)
                    except StopIteration:
                        break
                    waits.append(time.perf_counter() - tick)
                    seen.extend(batch["uttids"])
                    last_size = len(batch["uttids"])
                    if batch["feats"].shape != (last_size, 1, 64000) or batch["labels"].tolist() != [0] * last_size:
                        raise AssertionError("Waveform/label mismatch")
                    if repeat == 0:
                        for uid, waveform in zip(batch["uttids"], batch["feats"]):
                            if waveform_digest(waveform) != references[uid]:
                                raise AssertionError(f"Original/optimized crop mismatch: {uid}")
                load_seconds = time.perf_counter() - load_started
                if seen != expected:
                    raise AssertionError("Missing, duplicate or reordered benchmark IDs")
                rss = monitor.usage.read(os.getpid())["rss_bytes"]
                row = dict(repeat=repeat, warmup=repeat == 0, creation_seconds=created,
                           iterator_start_seconds=starts[-1], first_batch_wait_seconds=waits[0],
                           steady_batch_seconds=float(np.mean(waits[1:])) if len(waits) > 1 else None,
                           steady_batch_p50_seconds=float(np.median(waits[1:])) if len(waits) > 1 else None,
                           load_seconds=load_seconds, samples=len(seen), batches=len(waits),
                           new_worker_pids=sorted(set(processes) - before_pids),
                           parent_rss_after_round_bytes=rss)
                if strategy == "recreate":
                    tick = time.perf_counter()
                    reusable.close()
                    row["shutdown_seconds"] = time.perf_counter() - tick
                    reusable = None
                row["round_total_seconds"] = time.perf_counter() - round_started
                records.append(row)
        finally:
            tick = time.perf_counter()
            if reusable is not None:
                reusable.close()
            final_shutdown = time.perf_counter() - tick
    alive = [pid for pid, process in processes.items() if process.is_alive()]
    if alive:
        raise AssertionError(f"Workers did not exit: {alive}")
    measured = records[1:]
    def mean(name):
        values = [row[name] for row in measured if row[name] is not None]
        return float(np.mean(values)) if values else None
    return dict(workers=workers, persistent_workers=bool(workers and persistent),
                prefetch_factor=prefetch if workers else None, strategy=strategy,
                batch_size=local["train"]["audio_microbatch_size"], rounds=records,
                measured_rounds=len(measured),
                samples_per_second=sum(row["samples"] for row in measured) / sum(row["load_seconds"] for row in measured),
                first_batch_wait_seconds=mean("first_batch_wait_seconds"),
                steady_batch_seconds=mean("steady_batch_seconds"),
                mean_load_seconds=mean("load_seconds"), mean_creation_seconds=mean("creation_seconds"),
                mean_iterator_start_seconds=mean("iterator_start_seconds"),
                timed_round_total_seconds=sum(row["round_total_seconds"] for row in measured),
                multi_round_total_seconds=time.perf_counter() - total_started,
                final_shutdown_seconds=final_shutdown, workers_exited=True, exact_crop_match=True,
                **monitor.result)


class BenchmarkWaveformReader:
    """Disposable format with an explicit benchmark identity, outside formal caches."""
    def __init__(self, root):
        self.root = Path(root)
        self.state = json.loads((self.root / "benchmark_index.json").read_text(encoding="utf-8"))
        if self.state["purpose"] != "real-audio-benchmark-only" or not self.state["complete"]:
            raise ValueError("Not a complete benchmark-only waveform cache")
        for shard in self.state["shards"]:
            if file_sha(self.root / shard["file"]) != shard["sha256"]:
                raise ValueError("Benchmark shard checksum mismatch")
        self.rows = {row["uttid"]: row for row in self.state["rows"]}
        self.maps = {}

    def __getstate__(self):
        return dict(self.__dict__, maps={})

    def get(self, uid):
        row = self.rows[uid]
        name = row["file"]
        if name not in self.maps:
            if len(self.maps) >= 4:
                self.maps.clear()
            self.maps[name] = np.memmap(self.root / name, dtype="uint8", mode="r")
        array = np.ndarray(row["shape"], dtype="float32", buffer=self.maps[name], offset=row["offset"])
        return torch.from_numpy(array.copy())

    def close(self):
        self.maps.clear()


def build_benchmark_waveforms(dataset, cfg, root, workers, *, resume=False, stop_after=None):
    """Exercise the production full-waveform loader with disposable binary shards."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = root / "benchmark_index.json"
    identity = dict(purpose="real-audio-benchmark-only", uttids=dataset.uttids,
                    audio_implementation_sha256=file_sha(Path(_bootstrap.ROOT) / "data/audio.py"))
    state = dict(**identity, complete=False, rows=[], shards=[])
    if index.exists():
        if not resume:
            raise FileExistsError("Benchmark cache exists; explicit resume required")
        state = json.loads(index.read_text(encoding="utf-8"))
        if any(state[key] != value for key, value in identity.items()):
            raise ValueError("Benchmark resume identity mismatch")
        for shard in state["shards"]:
            if file_sha(root / shard["file"]) != shard["sha256"]:
                raise ValueError("Cannot resume corrupt benchmark shard")
        if [row["uttid"] for row in state["rows"]] != dataset.uttids[:len(state["rows"])]:
            raise ValueError("Benchmark resume prefix mismatch")
    local = copy.deepcopy(cfg)
    local["runtime"].pop("num_workers", None)
    local["runtime"].update(eval_num_workers=workers, persistent_eval_workers=True)
    waveforms = PreprocessedWaveformDataset(dataset)
    current, pending = None, []
    started = time.perf_counter()
    def flush():
        nonlocal current, pending
        if current is None:
            return
        current.flush()
        os.fsync(current.fileno())
        current.close()
        name = pending[0]["file"]
        state["shards"].append(dict(file=name, bytes=(root/name).stat().st_size, sha256=file_sha(root/name)))
        state["rows"].extend(pending)
        current, pending = None, []
        atomic_json(index, state)
    try:
        with ReusableAudioLoader(waveforms, local, training=False) as loader:
            stream = loader.for_indices(range(len(state["rows"]), len(dataset)))
            try:
                for batch in stream:
                    for uid, whole in batch:
                        if current is None:
                            name = f"benchmark_shard_{len(state['shards']):05d}.bin"
                            current = (root / name).open("xb")
                        if uid != dataset.uttids[len(state["rows"]) + len(pending)]:
                            raise ValueError("Benchmark waveform order mismatch")
                        array = whole.numpy()
                        pending.append(dict(uttid=uid, file=name, offset=current.tell(), shape=list(array.shape)))
                        current.write(array.tobytes(order="C"))
                        if len(pending) == cfg["cache"]["shard_size"]:
                            flush()
                            if stop_after is not None and len(state["rows"]) >= stop_after:
                                return state, time.perf_counter() - started
                flush()
            finally:
                stream.close()
        state["complete"] = len(state["rows"]) == len(dataset)
        atomic_json(index, state)
        return state, time.perf_counter() - started
    finally:
        if current is not None:
            current.close()


def benchmark_cache(dataset, cfg, references, full_hashes, output, workers, rounds, best):
    results = dict(builds=[], reads=[], resume=None,
                   note="Disposable benchmark-only shards; no formal split, identity or cache directory is changed.")
    for count in workers:
        records = []
        for repeat in range(rounds):
            with tempfile.TemporaryDirectory(prefix=".benchmark_waveforms_", dir=output) as temporary:
                with UsageMonitor() as monitor:
                    state, seconds = build_benchmark_waveforms(dataset, cfg, temporary, count)
                reader = BenchmarkWaveformReader(temporary)
                for uid in dataset.uttids:
                    if waveform_digest(reader.get(uid)) != full_hashes[uid]:
                        raise AssertionError("Full cached waveform differs from original")
                records.append(dict(repeat=repeat, warmup=repeat == 0, seconds=seconds,
                                    samples_per_second=len(dataset) / seconds,
                                    bytes=sum(row["bytes"] for row in state["shards"]),
                                    shards=len(state["shards"]), **monitor.result))
                if count == workers[0] and repeat == rounds - 1:
                    cached = dataset.with_cache(reader)
                    for read_workers in sorted({0, best["workers"]}):
                        row = benchmark_case(cached, cfg, references, workers=read_workers,
                                             persistent=True, prefetch=best["prefetch_factor"] or 2,
                                             strategy="reuse", rounds=rounds)
                        results["reads"].append(row)
                reader.close()
        result = dict(workers=count, rounds=records,
                      mean_build_seconds=float(np.mean([row["seconds"] for row in records[1:]])),
                      exact_full_waveform_match=True)
        results["builds"].append(result)
        print(f"cache workers={count}: {result['mean_build_seconds']:.3f}s", flush=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark_resume_", dir=output) as temporary:
        cutoff = cfg["cache"]["shard_size"]
        partial, _ = build_benchmark_waveforms(dataset, cfg, temporary, best["workers"], stop_after=cutoff)
        if len(dataset) > cutoff and partial["complete"]:
            raise AssertionError("Interrupted benchmark cache unexpectedly marked complete")
        saved = [row["sha256"] for row in partial["shards"]]
        resumed, seconds = build_benchmark_waveforms(dataset, cfg, temporary, best["workers"], resume=True)
        reader = BenchmarkWaveformReader(temporary)
        for uid in dataset.uttids:
            if waveform_digest(reader.get(uid)) != full_hashes[uid]:
                raise AssertionError("Resumed benchmark waveform differs")
        results["resume"] = dict(verified_prefix_records=len(partial["rows"]),
                                 resumed_records=len(resumed["rows"]), resume_seconds=seconds,
                                 prefix_shards_unchanged=saved == [row["sha256"] for row in resumed["shards"][:len(saved)]],
                                 exact_full_waveform_match=True)
        reader.close()
    return results


def main():
    parser = common_parser("Benchmark actual Real audio without models or SpeechFake manifests")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", default="0,2,4,8")
    parser.add_argument("--prefetch", default="2,4")
    parser.add_argument("--strategies", default="recreate,reuse")
    parser.add_argument("--cache-workers", default="0,2,4,8")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--output-dir", default="outputs/audio_benchmark")
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("At least three rounds are required: one warmup and two measured")
    cfg = load_config(args.config, args.set)
    output = Path(args.output_dir).resolve()
    project = Path(_bootstrap.ROOT).resolve()
    if output.is_relative_to(project) and (output == project or output.relative_to(project).parts[0] not in (".tmp", "outputs", ".cache")):
        parser.error("Benchmark outputs must use a Git-ignored output directory or an external directory")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        parser.error("Choose a new output directory; existing benchmark results are preserved")
    torch.set_num_threads(cfg["runtime"]["worker_torch_num_threads"])
    records, inventory = scan_real_audio(args.audio_root, args.limit)
    if not records:
        raise ValueError("No readable real audio files found")
    dataset = RealAudioBenchmarkDataset(args.audio_root, records)
    good, references, full_hashes, correctness = prepare_reference(dataset)
    if not good:
        raise ValueError("All selected audio files failed the official preprocessing path")
    dataset = RealAudioBenchmarkDataset(args.audio_root, good)
    original = project / ".tmp/local_audio_optimization/baseline/data/audio.py"
    report = dict(purpose="Real-only data pipeline benchmark, not FS-LPRF classification results",
                  host=dict(platform=platform.platform(), python=sys.version, torch=torch.__version__,
                            logical_cpus=os.cpu_count(), parent_torch_threads=torch.get_num_threads(),
                            start_method=mp.get_context().get_start_method()),
                  inventory=inventory, correctness=correctness,
                  audio_source_sha256=file_sha(project/"data/audio.py"),
                  original_audio_source_unchanged=(file_sha(original) == file_sha(project/"data/audio.py")) if original.exists() else None,
                  profile=None, loaders=[], cache=None)
    atomic_json(result_path, report)
    print(f"Selected {len(dataset)} distinct real files from {inventory['total_audio_files']}; failures={len(correctness['processing_failures'])}", flush=True)
    if not args.skip_profile:
        report["profile"] = profile_preprocessing(dataset, rounds=args.rounds)
        atomic_json(result_path, report)
        print("Stage profiling complete", flush=True)
    for workers in [int(value) for value in args.workers.split(",")]:
        combinations = [(False, 2)] if workers == 0 else itertools.product((False, True), [int(value) for value in args.prefetch.split(",")])
        for persistent, prefetch in combinations:
            for strategy in args.strategies.split(","):
                if strategy not in ("recreate", "reuse"):
                    parser.error("Strategies must be recreate,reuse")
                row = benchmark_case(dataset, cfg, references, workers=workers, persistent=persistent,
                                     prefetch=prefetch, strategy=strategy, rounds=args.rounds)
                report["loaders"].append(row)
                atomic_json(result_path, report)
                print(f"workers={workers} persistent={persistent} prefetch={prefetch} {strategy}: "
                      f"{row['samples_per_second']:.2f}/s first={row['first_batch_wait_seconds']:.3f}s "
                      f"total={row['multi_round_total_seconds']:.3f}s", flush=True)
    best = max((row for row in report["loaders"] if row["strategy"] == "reuse"), key=lambda row: row["samples_per_second"], default=None)
    report["recommended_measured_loader"] = best
    if not args.skip_cache:
        if best is None:
            raise ValueError("Cache benchmark requires a measured reusable loader")
        report["cache"] = benchmark_cache(dataset, cfg, references, full_hashes, output,
                                          [int(value) for value in args.cache_workers.split(",")], args.rounds, best)
    report["complete"] = True
    atomic_json(result_path, report)
    print(f"Complete: {result_path}", flush=True)


if __name__ == "__main__":
    main()
