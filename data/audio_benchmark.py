"""Real-directory benchmark inputs; never a SpeechFake train/dev/test split."""
from argparse import Namespace
from collections import Counter, defaultdict
import copy
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch

from .audio import FADTrim, GeneralFAD
from .speechfake_dataset import SpeechFakeDataset, validate_manifest

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".au", ".m4a"}


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {}
    return dict(min=float(values.min()), p25=float(np.percentile(values, 25)),
                median=float(np.median(values)), p75=float(np.percentile(values, 75)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def scan_real_audio(root, limit=512, seed=71):
    """Sample distinct paths across source folders without inventing any classes."""
    root = Path(root).resolve()
    if not root.is_dir() or limit <= 0:
        raise ValueError("A real audio directory and positive sample limit are required")
    pools, extensions = defaultdict(list), Counter()
    scan_errors = []
    def onerror(error):
        scan_errors.append(dict(path=str(error.filename), error=str(error)))
    for directory, dirs, names in os.walk(root, onerror=onerror):
        dirs.sort()
        for name in sorted(names):
            path = Path(directory) / name
            extensions[path.suffix.lower()] += 1
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                relative = path.relative_to(root)
                source = relative.parts[0] if len(relative.parts) > 1 else "(root)"
                pools[source].append(path)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    offsets = Counter()
    while len(selected) < limit:
        added = False
        for source in sorted(pools):
            if offsets[source] < len(pools[source]) and len(selected) < limit:
                selected.append(pools[source][offsets[source]])
                offsets[source] += 1
                added = True
        if not added:
            break
    rates, channels, formats, sources = Counter(), Counter(), Counter(), Counter()
    records, unreadable = [], []
    for path in selected:
        try:
            info = sf.info(path)
            relative = path.relative_to(root)
            source = relative.parts[0] if len(relative.parts) > 1 else "(root)"
            record = dict(path=str(path), uttid=relative.as_posix(),
                          source=source, bytes=path.stat().st_size,
                          sample_rate=info.samplerate, channels=info.channels, frames=info.frames,
                          format=info.format, subtype=info.subtype, duration=info.duration)
            records.append(record)
            rates[str(info.samplerate)] += 1
            channels[str(info.channels)] += 1
            formats[f"{info.format}/{info.subtype}"] += 1
            sources[record["source"]] += 1
        except Exception as error:
            unreadable.append(dict(path=str(path), error=repr(error)))
    report = dict(root=str(root), total_audio_files=sum(map(len, pools.values())),
                  all_file_extensions=dict(extensions), source_audio_counts={k: len(v) for k, v in pools.items()},
                  requested_samples=limit, selected_files=len(selected), readable_files=len(records),
                  selection_seed=seed, selection="round-robin source folders, deterministic without replacement",
                  sample_rate_distribution=dict(rates), channel_distribution=dict(channels),
                  format_distribution=dict(formats), sampled_source_counts=dict(sources),
                  file_size_bytes=distribution([row["bytes"] for row in records]),
                  duration_seconds=distribution([row["duration"] for row in records]),
                  unreadable_files=unreadable, scan_errors=scan_errors, files=records)
    return records, report


class RealAudioBenchmarkDataset(FADTrim):
    """Use the production reader, preprocessing, crop and collate unchanged."""
    def __init__(self, root, records):
        self.root = Path(root).resolve()
        self.records = records
        self.benchmark_cache = None
        super().__init__(Namespace(max_samples=64000, fix_len=True, trim_silence=True,
                                   do_augment=False, return_audio=False),
                         split="benchmark", train_mode=False)

    def _load_both(self, args):
        self.uttid2path = {row["uttid"]: row["path"] for row in self.records}
        self.uttid2info = {row["uttid"]: dict(origin_ds=row["source"], speaker="benchmark-only",
                                          attacker="real", label=0) for row in self.records}

    def load_preprocessed_waveform(self, uttid):
        if self.benchmark_cache is not None:
            return self.benchmark_cache.get(uttid)
        return super().load_preprocessed_waveform(uttid)

    def with_cache(self, reader):
        result = copy.copy(self)
        result.benchmark_cache = reader
        return result

    def __getitem__(self, index):
        sample = self._getitem_impl(index)
        if sample["feats"].shape != (1, 64000) or not torch.isfinite(sample["feats"]).all():
            raise ValueError(f"Invalid official benchmark waveform: {self.uttids[index]}")
        return sample


def waveform_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes(order="C")).hexdigest()


class BenchmarkManifestDataset(GeneralFAD):
    """Real-only temporary Train/Dev/Test fixture, never a formal SpeechFake split.

    Formal SpeechFakeDataset still enforces all 32 Fake classes. This adapter
    only exercises cache identity/runtime mechanics with explicitly temporary
    manifests and calls the original FADTrim audio functions unchanged.
    """
    use_cache = SpeechFakeDataset.use_cache
    __getstate__ = SpeechFakeDataset.__getstate__

    def __init__(self, options, split, train_mode=False):
        validate_manifest(options["data_path"], split)
        args = dict(options, do_augment=False)
        super().__init__(Namespace(**args), split, train_mode=False)
        if any(info["label"] != 0 or info["attacker"] != "real" for info in self.uttid2info.values()):
            raise ValueError("Benchmark manifests accept actual Real files only")
        self.cache_mode, self.cache_reader = "none", None
        self.feature_caches = {}
        self.fake_classes = []
        self.class_indices = {"real": list(range(len(self.uttids)))}

    def load_preprocessed_waveform(self, uttid):
        if self.cache_mode == "waveform":
            return self.cache_reader.get(uttid)
        return super().load_preprocessed_waveform(uttid)

    def __getitem__(self, index):
        sample = self._getitem_impl(index)
        if sample["feats"].shape != (1, 64000) or not torch.isfinite(sample["feats"]).all():
            raise ValueError(f"Invalid official fixture waveform: {self.uttids[index]}")
        return sample
