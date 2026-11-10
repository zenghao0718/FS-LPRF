"""Bounded spawn workers: UID-only input, ordered output, one optional CUDA owner."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
import time

import torch

from .cache import CacheReader
from .codec_augmentation import CodecAugmentor, alignment_diagnostics
from .loaders import _worker_import_environment
from .randomness import sample_rng, sample_seed

_SOURCE = None
_AUGMENTOR = None


def _initialize_worker(source_path, families, device, threads, factory):
    global _SOURCE, _AUGMENTOR
    from . import codec_augmentation
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    codec_augmentation._FFMPEG_THREADS = threads
    _AUGMENTOR = factory(probability=1, codecs=families, sample_rate=16000, encodec_device=device)
    _SOURCE = CacheReader(source_path, verify_checksums=False)


def _worker_report():
    return _AUGMENTOR.report()


def _encode_task(uid, spec, seed):
    started = time.perf_counter()
    with sample_rng(sample_seed("codec_encode", seed, uid)):
        audio, info = _AUGMENTOR.apply_array(_SOURCE.get(uid).numpy().reshape(-1), spec)
    record = {key: value for key, value in info.items() if key != "payload"}
    record.update(uttid=uid, seconds=time.perf_counter()-started)
    return audio, record


class CodecTaskError(RuntimeError):
    def __init__(self, uid, spec, error):
        self.uid, self.spec = uid, spec
        super().__init__(f"Codec worker failed: uttid={uid}, spec={spec}: {error}")


class CodecWorkerPool:
    def __init__(self, source, families, *, workers=1, encodec_device="cpu", threads=1,
                 max_inflight=None, factory=CodecAugmentor, executor_factory=ProcessPoolExecutor):
        device = torch.device(encodec_device)
        if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
            raise ValueError("--encodec-device must be cpu or cuda:N")
        if type(workers) is not int or workers <= 0 or type(threads) is not int or threads <= 0:
            raise ValueError("Codec workers/threads must be positive integers")
        self.device = "cuda:0" if str(device) == "cuda" else str(device)
        self.source_path, self.families = str(source.path), list(families)
        self.workers, self.threads = workers, threads
        self.max_inflight = 2 * (workers + int(device.type == "cuda")) if max_inflight is None else max_inflight
        if type(self.max_inflight) is not int or self.max_inflight <= 0:
            raise ValueError("--max-inflight must be positive")
        self.factory, self.executor_factory = factory, executor_factory
        self.cpu_pool, self.gpu_pool, self._pools = None, None, []
        self._report = None
        self.peak_inflight = 0

    def _pool(self, workers, families, device):
        pool = self.executor_factory(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(self.source_path, families, device, self.threads, self.factory))
        self._pools.append(pool)
        # Trigger spawn under the bounded thread environment, once per worker.
        ready = [pool.submit(_worker_report) for _ in range(workers)]
        reports = [future.result() for future in ready]
        return pool, reports[0]

    def __enter__(self):
        self._environment = _worker_import_environment(self.threads)
        self._environment.__enter__()
        try:
            gpu = self.device.startswith("cuda") and "encodec" in self.families
            cpu_families = [name for name in self.families if not (gpu and name == "encodec")]
            reports = []
            if cpu_families:
                self.cpu_pool, report = self._pool(self.workers, cpu_families, "cpu")
                reports.append(report)
            if gpu:
                self.gpu_pool, report = self._pool(1, ["encodec"], self.device)
                reports.append(report)
            self._report = self._merge_reports(reports)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _merge_reports(self, reports):
        result = copy.deepcopy(reports[0])
        result.update(requested=self.families, available_families=self.families, skipped={},
                      choices={}, implementations={}, leads={}, environment={})
        for report in reports:
            for key in ("choices", "implementations", "leads"):
                result[key].update(report[key])
            for key, value in report["environment"].items():
                if value is not None and value != []:
                    result["environment"][key] = value
        result["execution"] = dict(encodec_device=self.device, worker_threads=self.threads)
        return result

    def report(self):
        return copy.deepcopy(self._report)

    def executor_for(self, spec):
        return self.gpu_pool if spec[0] == "encodec" and self.gpu_pool is not None else self.cpu_pool

    def map(self, tasks):
        """Bound futures (including completed out-of-order PCM) and yield manifest order."""
        tasks, pending = iter(tasks), deque()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < self.max_inflight:
                try:
                    uid, spec, seed = next(tasks)
                except StopIteration:
                    exhausted = True
                    break
                future = self.executor_for(spec).submit(_encode_task, uid, spec, seed)
                pending.append((uid, spec, future))
                self.peak_inflight = max(self.peak_inflight, len(pending))
            if pending:
                uid, spec, future = pending.popleft()
                try:
                    audio, record = future.result()
                except Exception as error:
                    raise CodecTaskError(uid, spec, error) from error
                yield uid, audio, record

    def preflight(self, source, seed=42):
        ordered = sorted(source.index, key=lambda row: row["length"])
        ids = list(dict.fromkeys(ordered[i]["uttid"] for i in (0, len(ordered)//2, len(ordered)-1)))
        tasks = ((uid, (family, row["variant"], row["param"]), seed)
                 for family in self.families for row in self._report["choices"][family] for uid in ids)
        rows = []
        for uid, audio, record in self.map(tasks):
            rows.append(dict(record, alignment=alignment_diagnostics(source.get(uid).numpy().reshape(-1), audio)))
        return rows

    def __exit__(self, *args):
        try:
            for pool in self._pools:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self._pools.clear()
            self._environment.__exit__(*args)
