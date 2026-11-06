from pathlib import Path

import numpy as np
import soundfile as sf

from data.evaluation_manifest import dataset_fingerprint
from data.speechfake_dataset import LEGACY_SPEECHFAKE_PREFIX, SpeechFakeDataset, locate_audio_file


def _write_split(root, split, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}.tsv").write_text("".join(f"{uttid}\t{path}\n" for uttid, path, *_ in rows), encoding="utf-8")
    (root / f"{split}.txt").write_text("".join(
        f"{uttid}\t{origin}\t{speaker}\t{attacker}\t{label}\n"
        for uttid, _path, origin, speaker, attacker, label in rows), encoding="utf-8")


def _processed_fixture(tmp_path):
    package = tmp_path / "SpeechFake_Processed"
    manifests = package / "manifests"
    stored = f"{LEGACY_SPEECHFAKE_PREFIX}Real/unit/sample.wav"
    audio = package / "test" / "Real" / "unit" / "sample.wav"
    audio.parent.mkdir(parents=True)
    tone = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
    sf.write(audio, tone, 16000, subtype="FLOAT")
    attackers = ["real"] + [f"fake{index:02d}" for index in range(32)]
    rows = []
    for index, attacker in enumerate(attackers):
        label = "bonafide" if attacker == "real" else "spoof"
        rows.append((f"utt{index:02d}", stored, "unit", f"spk{index:02d}", attacker, label))
    _write_split(manifests, "test", rows)
    return package, stored, audio


def test_test_audio_opens_from_processed_tree_without_rewriting_manifest_paths(tmp_path):
    package, stored, audio = _processed_fixture(tmp_path)
    dataset = SpeechFakeDataset(dict(data_path=str(package / "manifests"), trim_silence=True,
                                     max_samples=64000, fix_len=True, do_augment=False), "test", train_mode=False)
    assert dataset.uttid2path["utt00"] == stored
    assert locate_audio_file(dataset, stored) == str(audio)
    before = dataset_fingerprint(dataset)
    waveform = dataset.load_preprocessed_waveform("utt00")
    assert waveform.shape[0] == 1 and waveform.shape[1] > 0
    assert dataset.uttid2path["utt00"] == stored
    assert dataset_fingerprint(dataset) == before
    assert dataset[0]["feats"].shape == (1, 64000)


def test_train_identity_strings_are_not_mapped_to_the_test_tree(tmp_path):
    package, stored, _audio = _processed_fixture(tmp_path)
    manifests = package / "manifests"
    attackers = ["real"] + [f"fake{index:02d}" for index in range(32)]
    rows = []
    for index, attacker in enumerate(attackers):
        label = "bonafide" if attacker == "real" else "spoof"
        rows.append((f"utt{index:02d}", f"{stored}.{index}", "unit", f"spk{index:02d}", attacker, label))
    _write_split(manifests, "train", rows)
    dataset = SpeechFakeDataset(dict(data_path=str(manifests), trim_silence=True, max_samples=64000,
                                     fix_len=True, do_augment=False), "train", train_mode=False)
    assert locate_audio_file(dataset, dataset.uttid2path["utt00"]) == dataset.uttid2path["utt00"]
    assert dataset.uttid2path["utt00"].startswith(LEGACY_SPEECHFAKE_PREFIX)


def test_packaged_path_rejects_parent_traversal(tmp_path):
    package, stored, _audio = _processed_fixture(tmp_path)
    dataset = SpeechFakeDataset(dict(data_path=str(package / "manifests"), trim_silence=True,
                                     max_samples=64000, fix_len=True, do_augment=False), "test", train_mode=False)
    escaped = LEGACY_SPEECHFAKE_PREFIX + "../secret.wav"
    assert locate_audio_file(dataset, escaped) == escaped
    assert Path(locate_audio_file(dataset, stored)).is_file()
