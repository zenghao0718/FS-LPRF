from collections import defaultdict
from pathlib import Path
import random
import torch
from engine.evaluator import extract_features
from methods.inference import PrototypeBank
from methods.layer_weights import METHOD_VERSION
from utils.seed import preserved_rng, seed_all
from data.evaluation_manifest import dataset_fingerprint
from utils.provenance import UPSTREAM_COMMIT

REAL_QUOTAS = {"AISHELL-1": 256, "AISHELL-3": 256, "LibriTTS": 256, "VCTK": 256}


def select_bank_samples(dataset, seed):
    if dataset.split != "train" or dataset.do_augment:
        raise ValueError("Prototype bank requires unaugmented SpeechFake-BD Train")
    if getattr(dataset, "cache_mode", "none") == "xlsr_feature":
        raise ValueError("Prototype bank cannot reuse the augmented Train feature-cache view")
    rng = random.Random(seed)
    sources = defaultdict(list)
    for index in dataset.class_indices["real"]:
        uid = dataset.uttids[index]
        sources[dataset.uttid2info[uid]["origin_ds"]].append(uid)
    shortages = {name: dict(required=count, available=len(sources[name]))
                 for name, count in REAL_QUOTAS.items() if len(sources[name]) < count}
    shortages.update({name: dict(required=1024, available=len(dataset.class_indices[name]))
                      for name in dataset.fake_classes if len(dataset.class_indices[name]) < 1024})
    if shortages:
        raise ValueError(f"Prototype-bank capacity mismatch (no repeated audio permitted): {shortages}")
    selected = {"real": [uid for source, count in REAL_QUOTAS.items() for uid in rng.sample(sources[source], count)]}
    selected.update({name: rng.sample([dataset.uttids[i] for i in dataset.class_indices[name]], 1024)
                     for name in dataset.fake_classes})
    return selected


@torch.no_grad()
def build_prototype_bank(model, dataset, cfg, seed, checkpoint):
    selected = select_bank_samples(dataset, seed)
    prototypes = []
    classes = ["real"] + dataset.fake_classes
    was_training = model.training
    try:
        with preserved_rng():
            seed_all(seed)
            model.eval()
            for name in classes:
                # Hold at most one class (1024 x 25 x 256) in CPU memory.
                features = extract_features(model, dataset, selected[name], cfg, seed)
                prototype = features.float().mean(0, keepdim=True)
                prototypes.append(prototype[0])
    finally:
        model.train(was_training)
    from models.frontend_compat import sha256_file
    metadata = dict(version=2, method_version=METHOD_VERSION, upstream_commit=UPSTREAM_COMMIT,
                    checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
                    dataset_fingerprint=dataset_fingerprint(dataset), seed=seed, samples_per_class=1024,
                    real_quotas=REAL_QUOTAS, sample_ids=selected)
    return PrototypeBank(classes, torch.stack(prototypes), model.layer_logits.detach().cpu().clone(), metadata)
