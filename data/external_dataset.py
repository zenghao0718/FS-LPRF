"""External metadata adapter over the original Layer-Wise waveform pipeline."""
from argparse import Namespace
import io

import soundfile as sf
import torch

from .audio import FADTrim, get_audio


class ExternalDataset(FADTrim):
    def __init__(self, records, *, group_by="none", identity="external"):
        if group_by not in ("none", "attack_id"):
            raise ValueError("group_by must be none or attack_id")
        self.records = records
        self.identity = identity
        self.group_by = group_by
        self.by_id = {row["uttid"]: row for row in records}
        if len(self.by_id) != len(records) or len({row["audio_path"] for row in records}) != len(records):
            raise ValueError("External inputs require unique uttids and disjoint resolved audio paths")
        super().__init__(Namespace(max_samples=64000, fix_len=True, trim_silence=True,
                                   do_augment=False, return_audio=False), "external", train_mode=False)
        self.cache_mode = "none"
        self.class_indices = {"real": []}
        for i, row in enumerate(records):
            if row["label"] is not None:
                self.class_indices.setdefault(self.uttid2info[row["uttid"]]["attacker"], []).append(i)
        self.fake_classes = sorted(set(self.class_indices) - {"real"})

    def _load_both(self, args):
        self.uttid2path = {row["uttid"]: row["audio_path"] for row in self.records}
        self.uttid2info = {}
        for row in self.records:
            label = row["label"]
            attack = row.get("attack_id", "")
            if self.group_by == "attack_id" and label == 1 and (not attack or attack == "real"):
                raise ValueError("Every Fake needs a nonempty attack_id distinct from 'real' for attack-aware evaluation")
            name = "real" if label == 0 else (attack if self.group_by == "attack_id" else "fake")
            self.uttid2info[row["uttid"]] = dict(origin_ds=self.identity, speaker=row.get("speaker", row.get("metadata", {}).get("speaker", "")),
                                                attacker=name, label=label, raw_label=row.get("raw_label", ""))

    def load_preprocessed_waveform(self, uttid):
        try:
            return super().load_preprocessed_waveform(uttid)
        except NotImplementedError:
            # The legacy reader rejects multichannel resampling. Keep its first-channel
            # convention and pass an in-memory mono stream through the SAME reader.
            audio, rate = sf.read(self.uttid2path[uttid], dtype="float32", always_2d=True)
            buffer = io.BytesIO()
            sf.write(buffer, audio[:, 0], rate, format="WAV", subtype="FLOAT")
            buffer.seek(0)
            return get_audio(buffer, to_mono=True, trim_sil=self.trim_silence)[0]

    def __getitem__(self, index):
        uid = self.uttids[index]
        try:
            audio = self.get_audio(uid)
            if audio.shape != (1, 64000) or not torch.isfinite(audio).all():
                raise ValueError("Expected finite mono waveform [1,64000]")
            # Unknown Query labels remain None; they never become Real or Fake.
            return dict(uttid=uid, feats=audio, **self.uttid2info[uid])
        except Exception as error:
            raise RuntimeError(f"Audio read failed: uttid={uid}, path={self.uttid2path[uid]}") from error

    @staticmethod
    def collate_fn(batch):
        return dict(uttids=[row["uttid"] for row in batch],
                    feats=torch.stack([row["feats"] for row in batch]))
