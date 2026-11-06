from argparse import Namespace
from pathlib import Path
import copy
import csv
import os
import torch
from .audio import GeneralFAD
from .augmentation import RawBoostAudioAugmentation


class ConfiguredRawBoost(RawBoostAudioAugmentation):
    """Use RIR-only candidates from the configured OpenSLR 28 root."""
    def __init__(self, rir_root):
        self.rir_root = Path(rir_root).resolve()
        super().__init__()

    def load_rir_noise(self):
        self.rir = self._get_all_rir_paths(self.rir_root)


LEGACY_SPEECHFAKE_PREFIX = "/root/autodl-tmp/SpeechFake_Data/"


def locate_audio_file(dataset, stored):
    """Resolve a Test file inside the processed dataset without altering identity strings.

    Manifests keep the absolute paths recorded in the waveform cache fingerprint.
    Train and Dev waveform reads do not use this path. Test uses a sibling
    ``test/`` directory when the packaged file is present, and otherwise keeps
    the manifest path so an unmoved source tree still opens.
    """
    stored = str(stored)
    prefix = LEGACY_SPEECHFAKE_PREFIX
    if getattr(dataset, "split", None) != "test" or not stored.startswith(prefix):
        return stored
    relative = Path(stored[len(prefix):])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return stored
    packaged = Path(dataset.args.data_path).resolve().parent / "test" / relative
    if packaged.is_file() and not packaged.is_symlink():
        return str(packaged)
    return stored


def validate_manifest(data_path, split):
    """Validate the GeneralFAD contract before its permissive dictionaries are populated."""
    ids = []
    root = Path(data_path)
    for suffix, columns in (("tsv", 2), ("txt", 5)):
        seen = set()
        with open(root / f"{split}.{suffix}", encoding="utf-8") as handle:
            for line, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(row) != columns or any(not field or field != field.strip() for field in row):
                    raise ValueError(f"{split}.{suffix}:{line}: expected {columns} nonempty Tab-separated fields, no header")
                uttid = row[0]
                if uttid == "uttid" or uttid in seen:
                    raise ValueError(f"{split}.{suffix}:{line}: header or duplicate uttid {uttid}")
                seen.add(uttid)
                if suffix == "tsv" and not Path(row[1]).is_absolute():
                    raise ValueError(f"{uttid}: audio_path must be absolute")
                if suffix == "txt":
                    if row[4] not in ("bonafide", "spoof"):
                        raise ValueError(f"{uttid}: label must be bonafide or spoof")
                    if (row[4] == "bonafide") != (row[3] == "real"):
                        raise ValueError(f"{uttid}: bonafide must use attacker=real; spoof must have its own task/model class")
        ids.append(seen)
    if not ids[0] or ids[0] != ids[1]:
        raise ValueError(f"{split}: TSV/TXT must have the same nonempty set of unique uttids")


class SpeechFakeDataset(GeneralFAD):
    def __init__(self, data_config, split, train_mode=False):
        if split not in ("train", "dev", "test"):
            raise ValueError("Only SpeechFake-BD train/dev/test are supported")
        if train_mode and split != "train":
            raise ValueError("Augmentation is restricted to Train")
        options = copy.deepcopy(data_config)
        if not options.get("data_path"):
            raise ValueError("Set data.data_path to the directory containing the six manifests")
        options["data_path"] = str(Path(options["data_path"]).resolve())
        validate_manifest(options["data_path"], split)
        # Attach the configured RIR augmentor after initializing the waveform dataset.
        options["do_augment"] = False
        super().__init__(Namespace(**options), split, train_mode=train_mode)
        self.do_augment = train_mode and data_config["do_augment"]
        self.cache_mode, self.cache_reader = "none", None
        self.feature_caches = {}
        if self.do_augment:
            rir_root = data_config.get("rir_root") or os.environ.get("FS_LPRF_RIR_ROOT")
            if not rir_root:
                raise ValueError("Set data.rir_root (OpenSLR 28 RIRS_NOISES)")
            self.augmentor = ConfiguredRawBoost(rir_root)
        self.class_indices = {"real": []}
        for index, uttid in enumerate(self.uttids):
            attacker = self.uttid2info[uttid]["attacker"]
            self.class_indices.setdefault(attacker, []).append(index)
        self.fake_classes = sorted(set(self.class_indices) - {"real"})
        if len(self.fake_classes) != 32 or not self.class_indices["real"]:
            raise ValueError(f"SpeechFake-BD requires Real and exactly 32 distinct Fake attacker classes; got {len(self.fake_classes)}")

    def use_cache(self, reader, mode):
        if self.split == "test":
            raise ValueError("Test must read original WAVs without any cache")
        if mode not in ("waveform", "xlsr_feature") or reader.identity["split"] != self.split:
            raise ValueError("Cache mode/split does not match the dataset")
        if mode == "xlsr_feature" and self.do_augment:
            raise ValueError("Cached features must not be augmented again")
        self.cache_mode, self.cache_reader = mode, reader
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache_source_hashes", None)
        return state

    def load_preprocessed_waveform(self, uttid):
        if self.cache_mode == "waveform":
            return self.cache_reader.get(uttid)
        stored = self.uttid2path[uttid]
        located = locate_audio_file(self, stored)
        if located == stored:
            return super().load_preprocessed_waveform(uttid)
        self.uttid2path[uttid] = located
        try:
            return super().load_preprocessed_waveform(uttid)
        finally:
            self.uttid2path[uttid] = stored

    @staticmethod
    def collate_fn(batch):
        if batch and batch[0].get("input_kind") == "xlsr_feature":
            if any(sample.get("input_kind") != "xlsr_feature" or sample["feats"].shape != batch[0]["feats"].shape for sample in batch):
                raise ValueError("Cannot mix cached hidden-state shapes or waveform/feature inputs")
            return dict(uttids=[sample["uttid"] for sample in batch],
                        feats=torch.stack([sample["feats"] for sample in batch]), input_kind="xlsr_feature",
                        labels=torch.tensor([sample["label"] for sample in batch]))
        return GeneralFAD.collate_fn(batch)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            from .randomness import sample_rng
            with sample_rng(index[1]):
                return self[index[0]]
        # Preserve all official audio processing but never silently substitute another class.
        uttid = self.uttids[index]
        try:
            if self.cache_mode == "xlsr_feature":
                feats = self.cache_reader.get(uttid)
                if feats.ndim != 3 or feats.shape[0] != 25 or feats.shape[-1] != 1024:
                    raise ValueError("Expected complete cached frontend states [25,T,1024]")
                return dict(uttid=uttid, feats=feats, input_kind="xlsr_feature", **self.uttid2info[uttid])
            result = self._getitem_impl(index)
            if result is None or result["feats"].shape != (1, 64000):
                raise ValueError("Expected official fixed-length mono waveform [1,64000]")
            if not torch.isfinite(result["feats"]).all():
                raise ValueError("Audio contains non-finite values after official preprocessing")
            return result
        except Exception as error:
            raise RuntimeError(f"Audio read failed: split={self.split}, uttid={uttid}, path={self.uttid2path[uttid]}") from error
