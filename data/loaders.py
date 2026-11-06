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
    return dict(num_workers=workers, persistent_workers=runtime.get(
                    "persistent_train_workers" if training else "persistent_eval_workers", True),
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
    """Reuse a sampler only after its previous stream is exhausted.

    Training retains its original per-step tokens. Evaluation supplies explicit
    view tokens, so merging classes never changes any sample's crop seed.
    """
    def __init__(self, dataset, cfg, *, training=True, seed=None, batch_size=None):
        self.dataset = dataset
        self.seed = cfg["train"]["seed"] if seed is None else seed
        self.sampler = StepSampler()
        self.loader = audio_loader(dataset, self.sampler,
                                   cfg["train"]["audio_microbatch_size"] if batch_size is None else batch_size,
                                   self.seed, **worker_options(cfg, training=training))
        self.busy, self.incomplete, self.closed = False, False, False
        self._active_iterator = None

    def for_step(self, indices, step):
        return self._for_tokens(_tokens(self.dataset, indices, self.seed, step))

    def for_indices(self, indices):
        """Read ordered indices or explicit (index, seed) tokens with this loader."""
        return self._for_tokens(_tokens(self.dataset, indices, self.seed))

    def _for_tokens(self, tokens):
        if self.busy or self.incomplete or self.closed:
            raise RuntimeError("Fully consume the current stream before changing sampler indices; closed loaders cannot be reused")
        self.sampler.tokens = list(tokens)
        self.busy = True
        complete = False
        try:
            self._active_iterator = iter(self.loader)
            for batch in self._active_iterator:
                yield batch
            complete = True
        finally:
            self.busy, self.incomplete = False, not complete
            if not complete:
                # Reject stale sampler resets and release prefetch on early exit.
                self.close()
            self._active_iterator = None

    def close(self):
        self.closed = True
        active, persistent = self._active_iterator, getattr(self.loader, "_iterator", None)
        try:
            for iterator in (active, persistent if persistent is not active else None):
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if shutdown is not None:
                    shutdown()
        finally:
            self.loader._iterator = None
            self._active_iterator = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Cannot reopen a closed audio loader")
        return self

    def __exit__(self, *args):
        self.close()


class PreprocessedWaveformDataset(Dataset):
    """Full official waveforms for cache builds: no crop, padding or augmentation."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.uttids = dataset.uttids

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, index):
        uid = self.uttids[index]
        return uid, self.dataset.load_preprocessed_waveform(uid)

    @staticmethod
    def collate_fn(batch):
        # Variable lengths must remain exact; never pad the cached waveforms.
        return batch
