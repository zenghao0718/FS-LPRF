from collections import defaultdict
import copy
from pathlib import Path
import random
import time
import torch
from engine.evaluator import extract_features, set_feature_progress
from methods.inference import PrototypeBank
from methods.prototypes import class_dispersion
from utils.seed import preserved_rng, seed_all
from data.evaluation_manifest import dataset_fingerprint
from data.loaders import ReusableAudioLoader
from data.randomness import sample_seed
from utils.provenance import UPSTREAM_COMMIT

AUGMENTATION_MODES = ("clean", "codec50", "full")

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


def dataset_config_for_mode(cfg, mode):
    """Copy a validated training snapshot into a prototype-bank view.

    Clean keeps the caller's config. codec50/full require the snapshot's offline
    codec settings and only change whether RIR/RawBoost run. Phase-1 validation
    still sees the original do_augment=true snapshot.
    """
    if mode not in AUGMENTATION_MODES:
        raise ValueError(f"Unknown prototype augmentation mode: {mode}")
    derived = copy.deepcopy(cfg)
    if mode == "clean":
        return derived
    codec = derived["data"]["codec_augment"]
    if (codec.get("enabled") is not True or codec.get("mode") != "offline" or float(codec.get("probability")) != 0.5
            or list(codec.get("codecs")) != ["mp3", "opus", "amr", "encodec"] or codec.get("sample_rate") != 16000
            or codec.get("build_seed") != 42):
        raise ValueError("codec50/full require the codec_offline_v1 offline cache at probability 0.5")
    if derived["data"].get("train_cache_mode") != "waveform":
        raise ValueError("Offline codec prototype banks require the Train waveform cache")
    if mode == "full" and not derived["data"].get("rir_root"):
        raise ValueError("full prototype augmentation requires data.rir_root; RIR is not optional")
    derived["data"]["do_augment"] = mode == "full"
    return derived


def validate_augmented_reference(dataset, selected, seed, augmentation_mode):
    """Reuse an existing clean bank's IDs. This function never draws a new sample."""
    if augmentation_mode not in ("codec50", "full"):
        raise ValueError("Augmented reference checks apply to codec50 and full only")
    if type(seed) is not int:
        raise ValueError("Prototype seed must be an integer")
    if dataset.split != "train":
        raise ValueError("Augmented prototype banks require SpeechFake-BD Train")
    if not getattr(dataset, "do_codec_augment", False) or getattr(dataset, "codec_mode", None) != "offline":
        raise ValueError("codec50/full require offline codec augmentation")
    if float(getattr(dataset, "codec_probability", None)) != 0.5:
        raise ValueError("Codec probability must stay at the training value 0.5")
    if getattr(dataset, "codec_cache_reader", None) is None or getattr(dataset, "cache_mode", None) != "waveform":
        raise ValueError("Offline Codec requires bound original and codec waveform caches")
    if augmentation_mode == "codec50":
        if dataset.do_augment:
            raise ValueError("codec50 disables RIR and RawBoost")
    else:
        augmentor = getattr(dataset, "augmentor", None)
        if not dataset.do_augment or augmentor is None:
            raise ValueError("full mode requires RIR and RawBoost; refusing to continue with them disabled")
        if float(augmentor.rir_prob) != 1 / 3 or float(augmentor.noise_prob) != 1 / 5 or augmentor.rawboost_args.algo != 5:
            raise ValueError("full mode requires RIR probability 1/3 and RawBoost Algorithm 5 at probability 1/5")
    if getattr(dataset, "cache_mode", "none") == "xlsr_feature":
        raise ValueError("Prototype bank cannot reuse the augmented Train feature-cache view")
    fake_classes = sorted(dataset.fake_classes)
    if len(fake_classes) != 32 or len(set(fake_classes)) != 32 or "real" in fake_classes:
        raise ValueError("Prototype bank requires exactly 32 distinct Fake classes from metadata")
    if not isinstance(selected, dict) or list(selected) != ["real", *fake_classes]:
        raise ValueError("Reference sample IDs must follow Real, then the sorted Fake class order")
    seen = []
    for name, group in selected.items():
        if not isinstance(group, (list, tuple)) or len(group) != SAMPLES_PER_CLASS or len(set(group)) != SAMPLES_PER_CLASS:
            raise ValueError(f"{name} must contribute exactly {SAMPLES_PER_CLASS} unique sample IDs")
        for uid in group:
            info = dataset.uttid2info.get(uid)
            if info is None or uid not in dataset.uttid2idx:
                raise ValueError(f"Reference sample is not in the SpeechFake-BD Train manifest: {uid}")
            if info["attacker"] != name:
                raise ValueError(f"Reference sample {uid} does not belong to {name}")
        seen.extend(group)
    if len(set(seen)) != len(selected) * SAMPLES_PER_CLASS:
        raise ValueError("Prototype bank requires unique sample IDs across all classes")
    sources = defaultdict(int)
    for uid in selected["real"]:
        raw = dataset.uttid2info[uid]["origin_ds"]
        canonical = REAL_SOURCE_ALIASES.get(raw, raw)
        if canonical == "CommonVoice" or raw == "CommonVoice":
            raise ValueError("CommonVoice is excluded from the Real prototype")
        sources[canonical] += 1
    if dict(sources) != REAL_QUOTAS:
        raise ValueError(f"Real reference quotas must stay {REAL_QUOTAS}; got {dict(sources)}")
    return selected


def _manifest_row(dataset, seed, augmentation_mode, trace):
    uid = trace["uttid"]
    if augmentation_mode == "full":
        rir_probability, rawboost_probability = float(dataset.augmentor.rir_prob), float(dataset.augmentor.noise_prob)
    else:
        rir_probability, rawboost_probability = 0.0, 0.0
    view = getattr(dataset, "validation_mode", "full")
    return dict(uttid=uid, class_name=dataset.uttid2info[uid]["attacker"],
                origin_ds=dataset.uttid2info[uid]["origin_ds"],
                sample_seed=sample_seed("view", dataset.split, view, seed, uid),
                codec_applied=bool(trace["codec_applied"]), codec_family=trace["codec_family"],
                codec_variant=trace["codec_variant"], codec_param=trace["codec_param"],
                codec_status=trace["codec_status"], rir_probability=rir_probability,
                rawboost_probability=rawboost_probability, augmentation_mode=augmentation_mode,
                rir_applied=bool(trace["rir_applied"]), rir_path=trace["rir_path"],
                rawboost_applied=bool(trace["rawboost_applied"]), evaluation_view_mode=view)


@torch.no_grad()
def build_prototype_bank(model, dataset, cfg, seed, checkpoint, *, selected_samples=None,
                         augmentation_mode="clean", trace_rows=None):
    if augmentation_mode == "clean" and selected_samples is None and trace_rows is None:
        selected = select_bank_samples(dataset, seed)
    elif augmentation_mode == "clean":
        raise ValueError("Clean prototype construction keeps the original unaugmented sampling path")
    else:
        selected = validate_augmented_reference(dataset, selected_samples, seed, augmentation_mode)
        if trace_rows is None or not getattr(dataset, "record_augmentation", False):
            raise ValueError("Augmented prototype construction requires augmentation tracing")
        dataset.augmentation_trace_sink = []
    prototypes, dispersions = [], []
    classes = list(selected)
    was_training = model.training
    completed = 0
    total = len(classes) * SAMPLES_PER_CLASS
    started = time.perf_counter()
    # One persistent loader keeps per-sample view tokens unchanged and avoids
    # respawning workers for every class. Clean builds keep their original path.
    loader = ReusableAudioLoader(dataset, cfg, training=False, seed=seed) if (
        augmentation_mode != "clean" and isinstance(cfg, dict) and "runtime" in cfg) else None
    try:
        with preserved_rng():
            seed_all(seed)
            model.eval()
            for name in classes:
                if augmentation_mode != "clean":
                    dataset.augmentation_trace_sink.clear()

                def progress(done, class_total, elapsed, class_name=name, finished=completed):
                    overall = finished + done
                    rate = overall / max(time.perf_counter() - started, 1e-9)
                    eta = (total - overall) / rate if rate else float("inf")
                    print(f"PROGRESS mode={augmentation_mode} class={class_name} class_audio={done}/{class_total} "
                          f"overall={overall}/{total} rate={rate:.2f}/s eta_seconds={eta:.0f}", flush=True)

                set_feature_progress(progress if augmentation_mode != "clean" else None)
                # Hold at most one class (1024 x 25 x 256) in CPU memory.
                if loader is None:
                    features = extract_features(model, dataset, selected[name], cfg, seed)
                else:
                    features = extract_features(model, dataset, selected[name], cfg, seed, loader=loader)
                if features.shape != (len(selected[name]), 25, 256) or not torch.isfinite(features).all():
                    raise ValueError(f"Expected finite [25,256] features for every reference sample in {name}")
                features = features.float()
                prototype = features.mean(0, keepdim=True)
                labels = torch.zeros(len(features), dtype=torch.long)
                prototypes.append(prototype[0])
                dispersions.append(class_dispersion(features, labels, prototype)[0])
                if augmentation_mode != "clean":
                    traces = list(dataset.augmentation_trace_sink)
                    if [row["uttid"] for row in traces] != list(selected[name]):
                        raise ValueError(f"Augmentation trace lost or reordered reference IDs for {name}")
                    trace_rows.extend(_manifest_row(dataset, seed, augmentation_mode, row) for row in traces)
                completed += len(selected[name])
    finally:
        set_feature_progress(None)
        if loader is not None:
            loader.close()
        model.train(was_training)
    from models.frontend_compat import sha256_file
    selected_real_sources = defaultdict(int)
    for uid in selected["real"]:
        selected_real_sources[dataset.uttid2info[uid]["origin_ds"]] += 1
    metadata = dict(version=1, method_version="2.4", upstream_commit=UPSTREAM_COMMIT,
                    checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
                    dataset_fingerprint=dataset_fingerprint(dataset), seed=seed, samples_per_class=SAMPLES_PER_CLASS,
                    real_quotas=REAL_QUOTAS, selected_real_sources=dict(selected_real_sources), sample_ids=selected)
    if augmentation_mode != "clean":
        metadata["augmentation_mode"] = augmentation_mode
        metadata["codec_probability"] = 0.5
        metadata["rir_probability"] = 1 / 3 if augmentation_mode == "full" else 0.0
        metadata["rawboost_probability"] = 1 / 5 if augmentation_mode == "full" else 0.0
        metadata["rawboost_algorithm"] = 5 if augmentation_mode == "full" else None
        metadata["augmentation_order"] = ["preprocess", "codec_or_original", "crop", "rir", "rawboost"]
    return PrototypeBank(classes, torch.stack(prototypes), torch.stack(dispersions), metadata)
