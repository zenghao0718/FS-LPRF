"""Ordered microbatches and reusable multiprocessing workers."""
from functools import partial
from contextlib import contextmanager
import os
import threading
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from utils.seed import seed_worker
from .randomness import sample_rng, sample_seed

_WORKER_ENV_LOCK = threading.RLock()
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextmanager
def _worker_import_environment(threads):
    # Spawned workers import numerical libraries before worker_init_fn runs.
    # Bound their import-time pools too; never leave these settings in the parent.
    with _WORKER_ENV_LOCK:
        previous = {key: os.environ.get(key) for key in _THREAD_VARIABLES}
        try:
            for key in _THREAD_VARIABLES:
                os.environ[key] = str(threads)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class AudioDataLoader(DataLoader):
    def __init__(self, *args, worker_torch_num_threads=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_torch_num_threads = worker_torch_num_threads

    def __iter__(self):
        if self.num_workers == 0:
            return super().__iter__()
        with _worker_import_environment(self.worker_torch_num_threads):
            return super().__iter__()


def worker_options(cfg, training=False):
    runtime = cfg["runtime"]
    legacy = runtime.get("num_workers")
    if legacy is not None:
        warnings.warn("runtime.num_workers is deprecated; use train_num_workers/eval_num_workers "
                      "(the legacy value is used as a compatibility override)", FutureWarning, stacklevel=2)
    workers = legacy if legacy is not None else runtime.get(
        "train_num_workers" if training else "eval_num_workers", 4 if training else 2)
    return dict(num_workers=workers, persistent_workers=training and runtime.get("persistent_train_workers", True),
                prefetch_factor=runtime.get("prefetch_factor", 2), pin_memory=runtime.get("pin_memory", True),
                worker_torch_num_threads=runtime.get("worker_torch_num_threads", 1))


def _init_worker(worker_id, torch_threads):
    torch.set_num_threads(torch_threads)
    seed_worker(worker_id)


class SeededDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, token):
        index, seed = token
        with sample_rng(seed):
            return self.dataset[index]


def _tokens(dataset, indices, seed, step=None):
    ids = getattr(dataset, "uttids", None)
    for position, index in enumerate(indices):
        if isinstance(index, tuple):
            yield index
        else:
            uid = ids[index] if ids is not None else str(index)
            yield index, sample_seed("train" if step is not None else "loader", seed, step, position, uid)


def audio_loader(dataset, indices, batch_size, seed, num_workers=0, *, persistent_workers=False,
                 prefetch_factor=2, pin_memory=None, worker_torch_num_threads=1):
    generator = torch.Generator().manual_seed(seed)
    options = dict(num_workers=num_workers, collate_fn=dataset.collate_fn,
                   worker_init_fn=partial(_init_worker, torch_threads=worker_torch_num_threads),
                   generator=generator, persistent_workers=bool(persistent_workers and num_workers > 0),
                   pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory)
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
    sampler = indices if isinstance(indices, StepSampler) else list(_tokens(dataset, indices, seed))
    return AudioDataLoader(SeededDataset(dataset), batch_size=batch_size, sampler=sampler,
                           worker_torch_num_threads=worker_torch_num_threads, **options)


class StepSampler(Sampler):
    def __init__(self):
        self.tokens = []

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)


class ReusableAudioLoader:
    """One loader per run; only its main-process sampler changes between steps."""
    def __init__(self, dataset, cfg):
        self.dataset, self.seed = dataset, cfg["train"]["seed"]
        self.sampler = StepSampler()
        self.loader = audio_loader(dataset, self.sampler, cfg["train"]["audio_microbatch_size"],
                                   self.seed, **worker_options(cfg, training=True))
        self.busy, self.incomplete, self.closed = False, False, False

    def for_step(self, indices, step):
        if self.busy or self.incomplete or self.closed:
            raise RuntimeError("Fully consume the current step before changing sampler indices")
        self.sampler.tokens = list(_tokens(self.dataset, indices, self.seed, step))
        self.busy = True
        complete = False
        try:
            for batch in self.loader:
                yield batch
            complete = True
        finally:
            self.busy, self.incomplete = False, not complete

    def close(self):
        iterator = getattr(self.loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
