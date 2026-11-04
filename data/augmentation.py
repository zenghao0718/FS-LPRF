"""The original RIR then RawBoost Algorithm 5 path, with configured resources.

Layer-Wise source: XIAOYixuan/tomatoDD, commit
88efe1376dfc5f5b9b62d016d33a7381c158d8af, tomato/data/audio_augmentation.py
and tomato/data/RawBoost/*.py. RawBoost authors: Hemlata Tak, Massimiliano Todisco;
RIR implementation based on ESPNet2. See docs/CODE_PROVENANCE.md.
"""
from argparse import Namespace
import copy
from pathlib import Path
import shlex
import numpy as np
import torch
from scipy import signal
from .audio import get_audio

RIR_INDEX_FILES = (
    "simulated_rirs/smallroom/rir_list",
    "simulated_rirs/mediumroom/rir_list",
    "simulated_rirs/largeroom/rir_list",
    "real_rirs_isotropic_noises/rir_list",
)


def _rir_path_from_index(line, index, line_number, root):
    """Parse Kaldi-style metadata plus one file path, without executing commands."""
    location = f"{index}:{line_number}"
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""  # Preserve native Windows paths as well as POSIX paths.
    try:
        fields = iter(list(lexer))
    except ValueError as error:
        raise ValueError(f"{location}: malformed RIR index record: {error}") from error
    paths = []
    for field in fields:
        if field.startswith("--"):
            value = field.partition("=")[2] if "=" in field else next(fields, None)
            if not value or value.startswith("--"):
                raise ValueError(f"{location}: missing metadata value for {field}")
        else:
            paths.append(field)
    if len(paths) != 1 or not paths[0].strip():
        raise ValueError(f"{location}: expected exactly one RIR WAV path")
    reference = Path(paths[0])
    if not reference.parts:
        raise ValueError(f"{location}: expected a WAV file path: {paths[0]}")
    if reference.is_absolute():
        rir = reference
    elif reference.parts[0] in ("RIRS_NOISES", root.name):
        rir = root.joinpath(*reference.parts[1:])
    elif reference.parts[0] in ("simulated_rirs", "real_rirs_isotropic_noises", "pointsource_noises"):
        rir = root / reference
    else:
        rir = index.parent / reference
    rir = rir.resolve()
    if not rir.is_relative_to(root):
        raise ValueError(f"{location}: RIR path escapes resource root: {reference}")
    if {"noise_list", "pointsource_noises"}.intersection(rir.relative_to(root).parts):
        raise ValueError(f"{location}: ordinary-noise path is not a RIR: {reference}")
    if rir.suffix.lower() != ".wav":
        raise ValueError(f"{location}: expected a WAV file path: {reference}")
    if not rir.is_file():
        raise FileNotFoundError(f"{location}: indexed RIR WAV file is missing or not a file: {rir}")
    return rir


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x


def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);

        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0);
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G/20)*b/np.amax(abs(h))
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y


def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


def get_default_args():
    return Namespace(algo=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                     minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5,
                     maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)


def process_Rawboost_feature(feature, sr, args, algo):
    if algo != 5:
        raise ValueError("FS-LPRF uses RawBoost Algorithm 5 only")
    feature = LnL_convolutive_noise(
        feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
        args.minCoeff, args.maxCoeff, args.minG, args.maxG,
        args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
    return ISD_additive_noise(feature, args.P, args.g_sd)


class RawBoostAudioAugmentation:
    def __init__(self):
        self.load_rir_noise()
        self.rawboost_args = get_default_args()
        self.process_Rawboost_feature = process_Rawboost_feature
        self.rir_prob = float(1/3)
        self.noise_prob = float(1/5)

    def load_rir_noise(self):
        raise NotImplementedError("Provide a configured RIR resource root")

    def _get_all_rir_paths(self, path):
        """Read all four OpenSLR 28 text indexes; never scan for WAV candidates."""
        root = Path(path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"RIR resource root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"RIR resource root is not a directory: {root}")
        rirs = set()
        for relative_index in RIR_INDEX_FILES:
            index = root / relative_index
            if not index.resolve().is_relative_to(root):
                raise ValueError(f"RIR index escapes resource root: {index}")
            if not index.is_file():
                raise FileNotFoundError(f"Required RIR index is missing or not a regular text file: {index}")
            entries = 0
            with index.open(encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    rirs.add(_rir_path_from_index(line, index, line_number, root))
                    entries += 1
            if not entries:
                raise ValueError(f"No RIR WAV files: empty RIR index {index}")
        return sorted(rirs)

    def convolve_rir(self, audio):
        rir_path = np.random.choice(self.rir)
        rir, sample_rate = get_audio(rir_path, to_mono=True, trim_sil=False)

        augmented = signal.convolve(audio.numpy(), rir.numpy(),
                                          mode="full")[:, :audio.size(1)]
        # reverse mean power
        augment_power = (augmented**2).mean()
        scale = float(np.sqrt(self.audio_power / max(augment_power, 1e-10)))
        augmented = scale * augmented
        return torch.from_numpy(augmented), rir.squeeze(1)

    def __call__(self, audio):
        self.audio_power = float((audio**2).mean())
        rir_prob = np.random.rand()
        if rir_prob < self.rir_prob:
            #print("convolve_rir")
            audio, rir = self.convolve_rir(audio)
        noise_prob = np.random.rand()
        if noise_prob < self.noise_prob:
            #print("add_noises")
            np_audio = audio.squeeze(0).numpy()
            sr = 16_000
            noisy_audio = self.process_Rawboost_feature(np_audio, sr, self.rawboost_args, self.rawboost_args.algo)
            audio = torch.from_numpy(noisy_audio).unsqueeze(0).to(torch.float32)
        #print(type(audio), audio.shape)
        return audio
