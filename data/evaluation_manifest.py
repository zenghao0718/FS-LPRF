import hashlib
import json
from pathlib import Path
import random
import copy
import os
import tempfile

DEV_SEEDS = [17, 42, 71, 100, 123]


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
            raise ValueError(f"{name}: no Query remains after excluding configured Support sets")
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


def _expected_light_manifest(dataset, full_manifest, full_manifest_path, shot, seed,
                             max_query_per_class):
    """Derive the monitor subset without consuming any training/global RNG state."""
    if dataset.split != "dev" or full_manifest.get("split") != "dev":
        raise ValueError("Light validation may only use the original Dev manifest")
    if type(shot) is not int or type(seed) is not int or shot != 10 or seed != 42 or full_manifest.get("shot") != shot:
        raise ValueError("Light Dev requires 10-shot and the original seed 42 Support")
    if type(max_query_per_class) is not int or not 1 <= max_query_per_class <= 1000:
        raise ValueError("Light Query cap must be an integer between 1 and 1000")
    if full_manifest.get("seeds") not in ([42], DEV_SEEDS) or len(dataset.fake_classes) != 32:
        raise ValueError("Light Dev requires a [42] or official five-seed, 32-Fake-class manifest")
    source_bytes = Path(full_manifest_path).read_bytes()
    if json.loads(source_bytes) != full_manifest:
        raise ValueError("Source Dev manifest on disk does not match the supplied manifest")
    fingerprint = dataset_fingerprint(dataset)
    if full_manifest.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Source Dev manifest data fingerprint does not match the dataset")
    try:
        validate_episode_manifest(dataset, full_manifest)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid source Dev Support/Query manifest") from error
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    classes = ["real"] + dataset.fake_classes
    queries, quotas = {}, {}
    for name in classes:
        candidates = sorted(full_manifest["queries"][name])
        quota = min(max_query_per_class, len(candidates))
        # The original validator rejects empty classes; small nonempty classes use all IDs.
        quotas[name] = quota
        if len(candidates) > quota:
            material = json.dumps([source_sha, seed, name], ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")
            class_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
            candidates = sorted(random.Random(class_seed).sample(candidates, quota))
        queries[name] = candidates
    return dict(version=1, kind="dev_light", split="dev", shot=shot, seed=seed,
                seeds=[seed], max_query_per_class=max_query_per_class,
                source_manifest_sha256=source_sha, dataset_fingerprint=fingerprint,
                classes=classes, supports={str(seed): copy.deepcopy(full_manifest["supports"][str(seed)])},
                query_quotas=quotas, queries=queries, loader_order=full_manifest["loader_order"])


def _validate_light_payload(manifest, expected, full_manifest):
    if manifest != expected:
        raise ValueError("Saved light Dev manifest does not match its source SHA, data, light configuration or deterministic selection")
    for name in expected["classes"]:
        queries = manifest["queries"][name]
        support_union = {uid for value in full_manifest["supports"].values() for uid in value[name]}
        if not queries or len(queries) != len(set(queries)) or set(queries) & support_union:
            raise ValueError(f"Invalid light Dev Query for {name}")
        if not set(queries) <= set(full_manifest["queries"][name]):
            raise ValueError(f"Light Dev Query is outside the original Dev Query for {name}")


def validate_light_manifest(dataset, manifest, full_manifest, full_manifest_path,
                            shot=10, seed=42, max_query_per_class=1000):
    expected = _expected_light_manifest(dataset, full_manifest, full_manifest_path,
                                        shot, seed, max_query_per_class)
    _validate_light_payload(manifest, expected, full_manifest)


def load_or_create_light_manifest(dataset, full_manifest, full_manifest_path, path,
                                   shot=10, seed=42, max_query_per_class=1000):
    """Persist a fixed monitoring subset; never resample Support or overwrite a saved subset."""
    expected = _expected_light_manifest(dataset, full_manifest, full_manifest_path,
                                        shot, seed, max_query_per_class)
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _validate_light_payload(manifest, expected, full_manifest)
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(expected, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        # A fully-written same-filesystem hard link publishes atomically and refuses overwrite.
        try:
            os.link(temporary, path)
        except FileExistsError:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            _validate_light_payload(manifest, expected, full_manifest)
            return manifest
        return expected
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
