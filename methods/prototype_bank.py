from collections import defaultdict
from pathlib import Path
import random
import torch
from engine.evaluator import extract_features
from methods.inference import PrototypeBank
from methods.prototypes import class_dispersion
from utils.seed import preserved_rng, seed_all
from data.evaluation_manifest import dataset_fingerprint
from utils.provenance import UPSTREAM_COMMIT

REAL_QUOTAS = {"AISHELL-1": 256, "AISHELL-3": 256, "LibriTTS": 256, "VCTK": 256}
REAL_SOURCE_ALIASES = {"Aishell": "AISHELL-1", "Aishell3": "AISHELL-3", "VCTK-Corpus": "VCTK"}
SAMPLES_PER_CLASS = 1024


def select_bank_samples(dataset, seed):
    if dataset.split != "train" or dataset.do_augment or getattr(dataset, "do_codec_augment", False):
        raise ValueError("Prototype bank requires unaugmented SpeechFake-BD Train")
    if getattr(dataset, "cache_mode", "none") == "xlsr_feature":
        raise ValueError("Prototype bank cannot reuse the augmented Train feature-cache view")
    fake_classes = sorted(dataset.fake_classes)
    if len(fake_classes) != 32 or len(set(fake_classes)) != 32 or "real" in fake_classes:
        raise ValueError("Prototype bank requires exactly 32 distinct Fake classes from metadata")
    rng = random.Random(seed)
    sources = defaultdict(list)
    source_counts = defaultdict(int)
    for index in dataset.class_indices["real"]:
        uid = dataset.uttids[index]
        raw_source = dataset.uttid2info[uid]["origin_ds"]
        source_counts[raw_source] += 1
        sources[REAL_SOURCE_ALIASES.get(raw_source, raw_source)].append(uid)
    shortages = {name: dict(required=count, available=len(sources[name]))
                 for name, count in REAL_QUOTAS.items() if len(sources[name]) < count}
    fake_shortages = {name: dict(required=SAMPLES_PER_CLASS, available=len(dataset.class_indices[name]))
                      for name in fake_classes if len(dataset.class_indices[name]) < SAMPLES_PER_CLASS}
    if shortages or fake_shortages:
        raise ValueError(f"Prototype-bank capacity mismatch: Real shortages={shortages}; "
                         f"available metadata Real sources={dict(sorted(source_counts.items()))}; "
                         f"Fake shortages={fake_shortages}")
    selected = {"real": [uid for source, count in REAL_QUOTAS.items() for uid in rng.sample(sources[source], count)]}
    selected.update({name: rng.sample([dataset.uttids[i] for i in dataset.class_indices[name]], SAMPLES_PER_CLASS)
                     for name in fake_classes})
    if len({uid for ids in selected.values() for uid in ids}) != len(selected) * SAMPLES_PER_CLASS:
        raise ValueError("Prototype bank requires unique sample IDs across all classes")
    return selected


@torch.no_grad()
def build_prototype_bank(model, dataset, cfg, seed, checkpoint):
    selected = select_bank_samples(dataset, seed)
    prototypes, dispersions = [], []
    classes = list(selected)
    was_training = model.training
    try:
        with preserved_rng():
            seed_all(seed)
            model.eval()
            for name in classes:
                # Hold at most one class (1024 x 25 x 256) in CPU memory.
                features = extract_features(model, dataset, selected[name], cfg, seed)
                if features.shape != (len(selected[name]), 25, 256) or not torch.isfinite(features).all():
                    raise ValueError(f"Expected finite [25,256] features for every reference sample in {name}")
                features = features.float()
                prototype = features.mean(0, keepdim=True)
                labels = torch.zeros(len(features), dtype=torch.long)
                prototypes.append(prototype[0])
                dispersions.append(class_dispersion(features, labels, prototype)[0])
    finally:
        model.train(was_training)
    from models.frontend_compat import sha256_file
    selected_real_sources = defaultdict(int)
    for uid in selected["real"]:
        selected_real_sources[dataset.uttid2info[uid]["origin_ds"]] += 1
    metadata = dict(version=1, method_version="2.4", upstream_commit=UPSTREAM_COMMIT,
                    checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
                    dataset_fingerprint=dataset_fingerprint(dataset), seed=seed, samples_per_class=SAMPLES_PER_CLASS,
                    real_quotas=REAL_QUOTAS, selected_real_sources=dict(selected_real_sources), sample_ids=selected)
    return PrototypeBank(classes, torch.stack(prototypes), torch.stack(dispersions), metadata)
