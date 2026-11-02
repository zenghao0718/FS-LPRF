import hashlib
import json
from pathlib import Path
import random


def dataset_fingerprint(dataset):
    digest = hashlib.sha256()
    for uttid in dataset.uttids:
        info = dataset.uttid2info[uttid]
        row = [uttid, str(dataset.uttid2path[uttid]), info["origin_ds"], info["speaker"],
               info["attacker"], info["label"]]
        digest.update((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_or_create_manifest(dataset, path, shot, seeds):
    """Choose supports once, exclude their union, and reuse common queries forever."""
    path = Path(path)
    fingerprint = dataset_fingerprint(dataset)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = (dataset.split, shot, list(seeds), fingerprint)
        actual = (manifest["split"], manifest["shot"], manifest["seeds"], manifest["dataset_fingerprint"])
        if actual != expected:
            raise ValueError("Saved evaluation manifest does not match split/shot/seeds/data; use a separate run directory")
        validate_episode_manifest(dataset, manifest)
        return manifest
    classes = ["real"] + dataset.fake_classes
    pools = {c: [dataset.uttids[i] for i in dataset.class_indices[c]] for c in classes}
    supports = {}
    for seed in seeds:
        rng = random.Random(seed)
        supports[str(seed)] = {}
        for name in classes:
            if len(pools[name]) < shot:
                raise ValueError(f"{name}: fewer than {shot} Support candidates")
            supports[str(seed)][name] = rng.sample(pools[name], shot)
    queries = {}
    for name in classes:
        excluded = {uid for seed in seeds for uid in supports[str(seed)][name]}
        queries[name] = [uid for uid in pools[name] if uid not in excluded]
        if not queries[name]:
            raise ValueError(f"{name}: no Query remains after excluding five Support sets")
    manifest = dict(version=1, split=dataset.split, shot=shot, seeds=list(seeds),
                    dataset_fingerprint=fingerprint, classes=classes,
                    supports=supports, queries=queries,
                    loader_order="all supports (Real then sorted Fake); Real Query; sorted Fake Query")
    validate_episode_manifest(dataset, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False)
    return manifest


def validate_episode_manifest(dataset, manifest):
    classes = ["real"] + dataset.fake_classes
    if manifest["classes"] != classes:
        raise ValueError("Manifest class mapping changed")
    for name in classes:
        pool = {dataset.uttids[i] for i in dataset.class_indices[name]}
        query = manifest["queries"][name]
        if not query or len(query) != len(set(query)):
            raise ValueError(f"Invalid Query for {name}")
        union = set()
        for seed in manifest["seeds"]:
            support = manifest["supports"][str(seed)][name]
            if len(support) != manifest["shot"] or len(set(support)) != len(support):
                raise ValueError(f"Invalid Support for {name}, seed={seed}")
            union.update(support)
        if union & set(query) or union | set(query) != pool:
            raise ValueError(f"Support/Query mismatch for {name}")
