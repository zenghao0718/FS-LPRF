"""Cache preparation and runtime binding, with no implicit Test or training builds."""
import copy
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from .cache import CacheReader, CacheWriter, cache_path, make_cache_identity, source_digest, file_sha
from .evaluation_manifest import load_or_create_manifest, load_or_create_light_manifest
from .loaders import audio_loader, worker_options, ReusableAudioLoader, PreprocessedWaveformDataset
from .randomness import RNG_SCHEMA_VERSION, sample_seed
from .waveform_cache import open_waveform_cache, publish_waveform_reference
from .speechfake_dataset import SpeechFakeDataset
from utils.seed import preserved_rng


def train_view_seed(cfg):
    configured = cfg["cache"]["train_feature_seed"]
    return cfg["train"]["seed"] if configured is None else configured


def view_tokens(dataset, ids, seed, mode):
    return [(dataset.uttid2idx[uid], sample_seed("view", dataset.split, mode, seed, uid)) for uid in ids]


def evaluation_ids(manifest, seed):
    ids = []
    for name in manifest["classes"]:
        ids.extend(manifest["supports"][str(seed)][name])
    for name in manifest["classes"]:
        ids.extend(manifest["queries"][name])
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation cache requires unique, disjoint Support/Query IDs")
    return ids


def evaluation_identities(dataset, cfg, manifest, path, mode, *, limit=None, seeds=None):
    identities = []
    selected_seeds = manifest["seeds"] if seeds is None else [seed for seed in manifest["seeds"] if seed in seeds]
    for seed in selected_seeds:
        ids = evaluation_ids(manifest, seed)
        if limit is not None:
            ids = ids[:limit]
        identities.append(make_cache_identity(dataset, cfg, "xlsr_feature", validation_mode=mode,
                          seed=seed, manifest=manifest, manifest_path=path, uttids=ids, smoke=limit is not None))
    return identities


def dev_manifests(dataset, cfg):
    if dataset.split != "dev":
        raise ValueError("Dev manifest preparation must never initialize Test")
    root = Path(cfg["runtime"]["output_dir"]) / "evaluation"
    full_path, light_path = root / "dev_episode_manifest.json", root / "dev_light_manifest.json"
    full = load_or_create_manifest(dataset, full_path, cfg["validation"]["shot"], cfg["validation"]["seeds"])
    light = load_or_create_light_manifest(dataset, full, full_path, light_path,
                  cfg["validation"]["shot"], cfg["validation"]["light_seed"],
                  cfg["validation"]["light_max_query_per_class"])
    return {"full": (full, full_path), "light": (light, light_path)}


def cache_specs(cfg, split, mode, *, validation_modes=("light", "full"), limit=None, seeds=None):
    if split not in ("train", "dev"):
        raise ValueError("build_cache accepts Train/Dev only; Test must read original WAVs")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    dataset = SpeechFakeDataset(cfg["data"], split, train_mode=split == "train" and mode == "xlsr_feature")
    if split == "train" or mode == "waveform":
        ids = dataset.uttids if limit is None else dataset.uttids[:limit]
        identity = make_cache_identity(dataset, cfg, mode, seed=train_view_seed(cfg) if mode == "xlsr_feature" else None,
                                        uttids=ids, smoke=limit is not None)
        return dataset, [identity]
    if mode != "xlsr_feature" or not validation_modes or set(validation_modes) - {"light", "full"}:
        raise ValueError("Dev caches require xlsr_feature and light/full validation modes")
    manifests = dev_manifests(dataset, cfg)
    identities = []
    for validation_mode in validation_modes:
        manifest, path = manifests[validation_mode]
        identities.extend(evaluation_identities(dataset, cfg, manifest, path, validation_mode, limit=limit, seeds=seeds))
    if not identities:
        raise ValueError("No requested cache seed occurs in the validated manifests")
    return dataset, identities


def require_cache(cfg, identity, *, allow_smoke=False):
    path = cache_path(cfg, identity)
    if not path.is_dir():
        split = identity["split"]
        command = (f"python scripts/build_cache.py --config <your-config.yaml> --split {split} "
                   f"--mode {identity['mode']}")
        if split == "dev":
            command += " --validation-modes light,full"
        raise ValueError(f"Required cache missing or fingerprint changed: {path}. Build explicitly: {command}")
    return CacheReader(path, identity, verify_checksums=cfg["cache"]["verify_checksums"], allow_smoke=allow_smoke)


def open_train_dataset(cfg, *, allow_codec_smoke=False):
    mode = cfg["data"]["train_cache_mode"]
    dataset = SpeechFakeDataset(cfg["data"], "train", train_mode=mode != "xlsr_feature")
    offline = getattr(dataset, "do_codec_augment", False) and getattr(dataset, "codec_mode", "online") == "offline"
    if mode == "waveform":
        reader = open_waveform_cache(dataset, cfg)
        dataset.use_cache(reader, mode)
    elif mode != "none":
        identity = make_cache_identity(dataset, cfg, mode, seed=train_view_seed(cfg))
        dataset.use_cache(require_cache(cfg, identity), mode)
    if offline:
        if mode != "waveform":
            raise ValueError("Offline Codec training requires data.train_cache_mode=waveform")
        from .codec_cache import open_codec_cache
        dataset.codec_cache_reader = open_codec_cache(dataset.cache_reader, cfg, allow_smoke=allow_codec_smoke)
    return dataset


def bind_evaluation_caches(dataset, cfg, manifests, *, build_missing=False, model=None):
    mode = cfg["data"][f"{dataset.split}_cache_mode"]
    if dataset.split == "test":
        if mode != "none" or getattr(dataset, "cache_mode", "none") != "none":
            raise ValueError("Test must read original WAVs without any cache")
        return []
    if mode == "none":
        return []
    if mode == "waveform":
        if dataset.split != "dev" or dataset.do_augment or build_missing:
            raise ValueError("Dev Waveform Cache must be built explicitly without augmentation")
        reader = open_waveform_cache(dataset, cfg)
        dataset.use_cache(reader, "waveform")
        return [reader]
    if dataset.split not in ("dev", "test") or dataset.do_augment:
        raise ValueError("Evaluation caches require unaugmented Dev/Test")
    if build_missing and dataset.split != "test":
        raise ValueError("Training must never silently generate Dev caches")
    readers = []
    for validation_mode, (manifest, path) in manifests.items():
        for identity in evaluation_identities(dataset, cfg, manifest, path, validation_mode):
            if build_missing:
                reader = build_dataset_cache(dataset, cfg, identity, model=model)
            else:
                reader = require_cache(cfg, identity)
            dataset.feature_caches[(validation_mode, identity["seed"])] = reader
            readers.append(reader)
    return readers


def evaluation_view(dataset, cfg, mode, seed):
    view = copy.copy(dataset)
    view.validation_mode = mode
    if dataset.split == "test" and (
            cfg["data"].get("test_cache_mode", "none") != "none" or getattr(dataset, "cache_mode", "none") != "none"):
        raise ValueError("Test must read original WAVs without any cache")
    if cfg["data"].get(f"{dataset.split}_cache_mode", "none") == "xlsr_feature":
        try:
            reader = dataset.feature_caches[(mode, seed)]
        except KeyError as error:
            raise ValueError(f"Required {dataset.split}/{mode}/seed_{seed:03d} cache is not bound") from error
        view.feature_caches = {}
        view.use_cache(reader, "xlsr_feature")
    return view


def cache_bindings(train_dataset, dev_dataset):
    result = dict(rng_schema_version=RNG_SCHEMA_VERSION, train_mode=train_dataset.cache_mode)
    if getattr(train_dataset, "do_codec_augment", False) and getattr(train_dataset, "codec_mode", "online") == "online":
        # Persist/bind executable, library, preset and weight provenance for online runs too.
        result["online_codec"] = dict(report=train_dataset.codec_augmentor.report(),
                                     implementation_sha256=file_sha(Path(__file__).with_name("codec_augmentation.py")))
    readers = ([train_dataset.cache_reader] if train_dataset.cache_reader else [])
    codec_reader = getattr(train_dataset, "codec_cache_reader", None)
    if codec_reader is not None:
        readers.append(codec_reader)
    readers += ([dev_dataset.cache_reader] if dev_dataset.cache_reader else [])
    readers += list(dev_dataset.feature_caches.values())
    result["caches"] = [dict(fingerprint=reader.metadata["fingerprint"],
                             content_fingerprint=reader.metadata["content_fingerprint"],
                             schema_version=reader.identity["schema_version"], split=reader.identity["split"],
                             mode=reader.identity["mode"], validation_mode=reader.identity["validation_mode"],
                             seed=reader.identity["seed"]) for reader in readers]
    if train_dataset.cache_mode == "none":
        result["train_source_sha256"] = source_digest(train_dataset, train_dataset.uttids)
    if dev_dataset.cache_mode == "none" and not dev_dataset.feature_caches:
        result["dev_source_sha256"] = source_digest(dev_dataset, dev_dataset.uttids)
    return result, {reader.metadata["fingerprint"]: reader.metadata for reader in readers}


def _progress(done, total, elapsed):
    rate = done / max(elapsed, 1e-9)
    print(f"cache {done}/{total} samples {rate:.2f}/s elapsed={elapsed:.1f}s ETA={(total-done)/max(rate,1e-9):.1f}s", flush=True)


def build_dataset_cache(dataset, cfg, identity, *, model=None, resume=False, force=False, preparation_seconds=0):
    if dataset.split == "test" or identity["split"] == "test":
        raise ValueError("Test caches are forbidden; keep original Test WAVs")
    target = cache_path(cfg, identity)
    if target.exists() and not force:
        reader = CacheReader(target, identity, verify_checksums=cfg["cache"]["verify_checksums"], allow_smoke=identity["smoke"])
        return publish_waveform_reference(reader, cfg)
    if identity["mode"] == "xlsr_feature" and model is None:
        raise ValueError("Feature cache generation requires the frozen frontend model")
    if dataset.cache_mode != "none":
        raise ValueError("Build caches from the original WAV/preprocessing path")
    if identity["mode"] == "waveform" and dataset.do_augment:
        raise ValueError("Waveform cache generation must never initialize augmentation")
    codec = cfg["data"].get("codec_augment") or {}
    if identity["mode"] == "xlsr_feature" and isinstance(codec, dict) and codec.get("enabled"):
        raise ValueError("Feature-cache generation cannot apply Codec Augmentation; disable data.codec_augment.enabled")
    was_training = model.training if model is not None else None
    try:
        with preserved_rng(), CacheWriter(target, identity, cfg["cache"]["shard_size"], resume=resume,
                                          force=force, progress=_progress,
                                          verify_checksums=cfg["cache"]["verify_checksums"]) as writer:
            writer.previous_seconds += preparation_seconds
            ids = identity["uttids"][len(writer.index):]
            if identity["mode"] == "waveform":
                waveforms = PreprocessedWaveformDataset(dataset)
                indices = [dataset.uttid2idx[uid] for uid in ids]
                # Workers preprocess; only the parent commits and resumes shards.
                with ReusableAudioLoader(waveforms, cfg, training=False,
                                         batch_size=min(cfg["train"]["audio_microbatch_size"],
                                                        cfg["cache"]["shard_size"])) as loader:
                    for batch in loader.for_indices(indices):
                        for uid, waveform in batch:
                            writer.add(uid, waveform)
            else:
                model.eval()
                tokens = view_tokens(dataset, ids, identity["seed"], identity["validation_mode"] or "train")
                loader = audio_loader(dataset, tokens, cfg["cache"]["feature_build_batch_size"],
                                      identity["seed"], **worker_options(cfg))
                with torch.no_grad():
                    for batch in loader:
                        states = model.extract_frontend_features(batch["feats"], to_cpu=True)
                        for uid, features in zip(batch["uttids"], states):
                            writer.add(uid, features)
            return publish_waveform_reference(writer.finish(), cfg)
    finally:
        if model is not None:
            model.train(was_training)


def estimate_cache(dataset, cfg, identity, model=None):
    ids = identity["uttids"]
    frames = None
    if identity["mode"] == "waveform":
        infos = {path: sf.info(path) for path in set(dataset.uttid2path[uid] for uid in ids)}
        elements = sum(int(np.ceil(infos[dataset.uttid2path[uid]].frames * 16000 /
                                  infos[dataset.uttid2path[uid]].samplerate)) for uid in ids)
        note = "upper bound before silence trimming; index/metadata and filesystem overhead excluded"
    else:
        if model is None:
            raise ValueError("Feature dry-run measures T from a real frontend output; provide existing XLS-R weights")
        was_training = model.training
        try:
            with preserved_rng(), torch.no_grad():
                model.eval()
                sample = model.extract_frontend_features(torch.zeros(1, 1, cfg["data"]["max_samples"]), to_cpu=True)
                frames = sample.shape[2]
        finally:
            model.train(was_training)
        elements = len(ids) * 25 * frames * 1024
        note = "T measured from the frontend; index/metadata and filesystem overhead excluded"
    size = elements * np.dtype(identity["dtype"]).itemsize
    return dict(samples=len(ids), frames=frames, dtype=identity["dtype"], bytes=size, gib=size / 2**30,
                fingerprint=cache_path(cfg, identity).name, path=str(cache_path(cfg, identity)),
                warning=size / 2**30 > cfg["cache"]["warn_if_estimated_gb_above"], note=note)


def prepare_final_test_caches(dataset, cfg, manifest, path, model, *, dry_run=False, confirm_large_cache=False):
    """Compatibility guard: final Test is always the original online WAV path."""
    raise ValueError("Test caches are forbidden; keep original WAVs and use data.test_cache_mode=none")
