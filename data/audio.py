"""Layer-Wise audio reading and GeneralFAD processing used by SpeechFake-BD.

Source: XIAOYixuan/tomatoDD, 88efe1376dfc5f5b9b62d016d33a7381c158d8af,
tomato/data/{audio_util,fad_trim,base}.py; author: Yixuan.
audio_util was adapted from piotrkawa/deepfake-whisper-features.
See docs/CODE_PROVENANCE.md. Processing order and RNG calls are preserved.
"""
import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLING_RATE = 16_000

def trim_silence(audio: np.ndarray,
                 threshold: float = 30,
                 win_length: int = 1024,
                 shift_length: int = 256) -> np.ndarray:
    """
    Trims the leading and trailing silence of the given audio.

    :param audio: A numpy.ndarray representing the audio signal.
    :param threshold: The threshold in decibels used for silence detection. Default is 30 dB.
    :param win_length: The window length for analysis in points. Default is 1024.
    :param shift_length: The shift length for the analysis window in points. Default is 256.
    :return: A numpy.ndarray of the trimmed audio.
    """
    # Use librosa's trim function to remove leading and trailing silence based on the specified parameters.
    trimmed_audio, _ = librosa.effects.trim(
        y=audio,
        top_db=threshold,
        frame_length=win_length,
        hop_length=shift_length
    )

    return trimmed_audio


def get_audio(
    audio_path,
    to_mono=True,
    norm=True,
    trim_sil=True,
    target_rms=-20,
    frame_offset=0,
    num_frames=-1
):
    """
    Load an audio file using soundfile, optionally resample with librosa,
    optionally convert to mono, and optionally RMS-normalize to target_rms dB.

    :param audio_path: Path to the audio file.
    :param to_mono: If True, keep only the first channel (as in original code).
    :param norm: If True, apply RMS-based normalization to target_rms dB.
    :param trim_sil: If True, trim leading and trailing silence.
    :param target_rms: Target RMS level in dB (default -20 dB).
    :param frame_offset: Number of frames to skip from the beginning.
    :param num_frames: How many frames to read. None or negative => read all.
    :return: A tuple (waveform, sample_rate),
             where waveform is a NumPy array of shape (num_samples,) or (channels, num_samples).
    """
    with sf.SoundFile(audio_path, 'r') as f:
        if frame_offset > 0:
            f.seek(frame_offset)
        data = f.read(frames=num_frames, dtype='float32', always_2d=False)
        sample_rate = f.samplerate

    if data.ndim != 1:
        data = data.T

    if sample_rate != SAMPLING_RATE:
        if data.ndim == 1:
            data = librosa.resample(y=data, orig_sr=sample_rate, target_sr=SAMPLING_RATE)
        else:
            raise NotImplementedError

    if to_mono:
        if data.ndim > 1 and data.shape[0] > 1:
            data = data[0]

    if trim_sil:
        data = trim_silence(data)

    if norm:
        rms = np.sqrt(np.mean(np.square(data)))
        target_rms_linear = 10 ** (target_rms / 20.0)
        scaling_factor = target_rms_linear / rms
        data = data * scaling_factor
        data = np.clip(data, -1.0, 1.0)

    data = data.reshape(1, -1)
    data_tensor = torch.from_numpy(data)
    return data_tensor, sample_rate


class FADTrim(Dataset):
    """The fixed-length waveform subset of Layer-Wise FADTrim."""
    def __init__(self, args, split, train_mode=True):
        super().__init__()
        self.args = args
        self.split = split
        self.train_mode = train_mode
        self.fast_forward_mode = False
        self.sample_rate = SAMPLING_RATE
        self.max_len = getattr(args, "max_samples", -1)
        if self.max_len < 0:
            self.max_len = getattr(args, "max_len", 4) * self.sample_rate
        self.fix_len = getattr(args, "fix_len", True)
        self.return_audio = getattr(args, "return_audio", False)
        self.trim_silence = getattr(args, "trim_silence", True)
        self._load_both(args)
        assert len(self.uttid2path) == len(self.uttid2info)
        self.uttids = list(self.uttid2path)
        self.uttid2idx = {uttid: idx for idx, uttid in enumerate(self.uttids)}
        # SpeechFakeDataset attaches its configured RIR/RawBoost augmentor.
        self.do_augment = train_mode & getattr(args, "do_augment", False)

    def _load_both(self, args):
        self.uttid2path = self._load_manifest(args)
        self.uttid2info = self._load_uttinfo(args)

    def __len__(self):
        return len(self.uttids)

    def get_info(self, uttid):
        return self.uttid2info[uttid]

    def get_path(self, uttid):
        return self.uttid2path[uttid]

    def get_idx_by_uttid(self, uttid):
        return self.uttid2idx[uttid]

    def sample_segment(self, feats):
        # feats: [C, T]
        if feats.shape[1] == self.max_len:
            return feats
        elif feats.shape[1] < self.max_len: # max len: 4 secs, 64000
            num_repeats = int(self.max_len / feats.shape[1]) + 1
            feats = feats.repeat(1, num_repeats)
        stt = np.random.randint(feats.shape[1] - self.max_len)
        feats = feats[:, stt:stt+self.max_len]
        return feats

    def load_preprocessed_waveform(self, uttid):
        """Original file/preprocessing path, before sample_segment and augmentation.

        The optional st/dur read branch is deliberately retained. Waveform cache
        builders reject that branch because its random file read is not a full
        deterministic waveform.
        """
        audio_path = self.uttid2path[uttid]
        if not os.path.exists(audio_path):
            raise ValueError(f"File {audio_path} does not exist")

        frame_offset = 0
        num_frames = -1
        info = self.uttid2info[uttid]
        if "st" in info and "dur" in info:
            st = info["st"]
            dur = info["dur"]
            st = int(st * self.sample_rate)
            dur = int(dur * self.sample_rate)
            if dur > self.max_len:
                frame_offset = np.random.randint(dur - self.max_len)
                num_frames = self.max_len
        audio, sample_rate = get_audio(audio_path,
                                                  to_mono=True, trim_sil=self.trim_silence,
                                                  frame_offset=frame_offset, num_frames=num_frames)
        if audio.shape[1] == 0:
            raise ValueError(f"Audio {audio_path} is empty")
        return audio

    def get_audio(self, uttid):
        audio = self.load_preprocessed_waveform(uttid)
        if self.fix_len:
            audio = self.sample_segment(audio)
        else:
            audio = audio[:, :self.max_len]
        if self.do_augment and self.train_mode:
            audio = self.augmentor(audio)

        #self.augmentor = AudioAugmentation()
        #audio, noise = self.augmentor.add_pink_noise(audio, noise_std=10)
        return audio

    def _getitem_impl(self, idx):
        if self.fast_forward_mode:
            return None
        uttid = self.uttids[idx]
        info = self.uttid2info[uttid]
        audio = self.get_audio(uttid)
        result = dict(uttid=uttid, feats=audio, label=info["label"],
                      origin_ds=info["origin_ds"], speaker=info["speaker"], attacker=info["attacker"])
        if self.return_audio:
            result["audio"] = audio
        return result

    def __getitem__(self, idx):
        return self._getitem_impl(idx)

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: list of dict
        """
        # if it's fast forward, then all the samples are None
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None

        uttids = [sample["uttid"] for sample in batch]
        feats = [sample["feats"] for sample in batch]
        labels = [sample["label"] for sample in batch]
        origin_ds = [sample["origin_ds"] for sample in batch]
        speakers = [sample["speaker"] for sample in batch]
        attackers = [sample["attacker"] for sample in batch]
        max_feat_len = max([feat.shape[1] for feat in feats])
        # if all the features are of the same length, then we don't need to pad
        all_fix_len = all([feat.shape[1] == max_feat_len for feat in feats])
        if all_fix_len:
            batch_feats = torch.stack(feats)
        else:
            batch_feats = torch.zeros(len(feats), feats[0].shape[0], max_feat_len) # N, C, T
            for idx, feat in enumerate(feats):
                batch_feats[idx, :, :feat.shape[1]] = feat

            # padding_mask, bool
            batch_padding_mask = torch.zeros_like(batch_feats, dtype=torch.bool)
            for i, feat in enumerate(feats):
                batch_padding_mask[i, :, feat.shape[1]:] = True

        batch_labels = torch.LongTensor(labels)
        #logger.info(f"Batch feats shape: {batch_feats.shape}")
        #logger.info(f"Batch labels shape: {batch_labels.shape}")
        if all_fix_len:
            return {
                "uttids": uttids,
                "feats": batch_feats,
                "labels": batch_labels,
                "origin_ds": origin_ds,
                "speakers": speakers,
                "attackers": attackers,
                "padding_mask": None
            }
        else:
            return {
                "uttids": uttids,
                "feats": batch_feats,
                "labels": batch_labels,
                "origin_ds": origin_ds,
                "speakers": speakers,
                "attackers": attackers,
                "padding_mask": batch_padding_mask
        }


class GeneralFAD(FADTrim):
    def _load_manifest(self, args):
        """ Has two columns
        uttid, path
        """
        self.dataset_name = Path(args.data_path).name
        manifest_path = os.path.join(args.data_path, f"{self.split}.tsv")
        uttid2path = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                uttid, path = line.strip().split("\t")
                uttid2path[uttid] = path
        return uttid2path

    def _load_uttinfo(self, args):
        """ Has five columns
        uttid, origin_ds, speaker, attacker, label
        label: [bonafide, spoof]
        """
        label_path = os.path.join(args.data_path, f"{self.split}.txt")
        uttid2info = {}
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                uttid, origin_ds, speaker, attacker, label = line.strip().split("\t")
                uttid2info[uttid] = {
                    "origin_ds": origin_ds,
                    "speaker": speaker,
                    "attacker": attacker,
                    "label": 0 if label=="bonafide" else 1
                }
        return uttid2info
