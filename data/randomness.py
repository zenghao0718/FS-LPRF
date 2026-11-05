"""Versioned, worker-independent sample RNG without touching CUDA in workers."""
from contextlib import contextmanager
import hashlib
import json
import random

import numpy as np
import torch

RNG_SCHEMA_VERSION = 1


def sample_seed(*parts):
    material = json.dumps([RNG_SCHEMA_VERSION, *parts], ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


@contextmanager
def sample_rng(seed):
    """Isolate Python, NumPy and CPU Torch draws, including with zero workers."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
