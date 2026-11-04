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

    def __getitem__(self, index):
        # Preserve all official audio processing but never silently substitute another class.
        uttid = self.uttids[index]
        try:
            result = self._getitem_impl(index)
            if result is None or result["feats"].shape != (1, 64000):
                raise ValueError("Expected official fixed-length mono waveform [1,64000]")
            if not torch.isfinite(result["feats"]).all():
                raise ValueError("Audio contains non-finite values after official preprocessing")
            return result
        except Exception as error:
            raise RuntimeError(f"Audio read failed: split={self.split}, uttid={uttid}, path={self.uttid2path[uttid]}") from error
