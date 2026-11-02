from contextlib import contextmanager
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng():
    # Store NumPy's uint32 state as a tensor so our checkpoints remain weights-only loadable.
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (ns[0], torch.from_numpy(ns[1].astype(np.int64)), ns[2], ns[3], ns[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], ns[1].cpu().numpy().astype(np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def preserved_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
