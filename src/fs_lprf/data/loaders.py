import torch
from torch.utils.data import DataLoader
from fs_lprf.utils.seed import seed_worker


def audio_loader(dataset, indices, batch_size, seed, num_workers=0):
    generator = torch.Generator().manual_seed(seed)
    # A fresh nonpersistent loader fixes both worker RNG and task ordering per seed/checkpoint.
    return DataLoader(dataset, batch_size=batch_size, sampler=list(indices),
                      num_workers=num_workers, collate_fn=dataset.collate_fn,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False, pin_memory=torch.cuda.is_available())
