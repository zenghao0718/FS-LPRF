"""OpenSLR 28 candidate discovery uses directory roles, not WAV filenames."""
import pytest

from fs_lprf.data.speechfake_dataset import ConfiguredRawBoost


def _create_files(root, relative_paths):
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_rir_candidates_only_use_simulated_and_real_rir_lists_in_sorted_order(tmp_path):
    root = tmp_path / "RIRS_NOISES"
    rirs = [
        "simulated_rirs/rir_list/rir2.wav",
        "real_rirs_isotropic_noises/rir_list/rir3.wav",
        "simulated_rirs/rir_list/rir1.wav",
    ]
    noises = [
        "simulated_rirs/noise_list/noise1.wav",
        "real_rirs_isotropic_noises/noise_list/noise2.wav",
        "pointsource_noises/noise3.wav",
    ]
    _create_files(root, noises + rirs)
    augmentor = ConfiguredRawBoost(root)
    assert len(augmentor.rir) == 3
    assert augmentor.rir == sorted(root / relative_path for relative_path in rirs)
    assert all("rir_list" in path.relative_to(root).parts for path in augmentor.rir)
    assert not set(augmentor.rir) & {root / relative_path for relative_path in noises}
    assert ConfiguredRawBoost(root).rir == augmentor.rir


def test_rir_candidates_accept_only_wav_files_without_filename_heuristics(tmp_path):
    root = tmp_path / "RIRS_NOISES"
    rirs = [
        "simulated_rirs/rir_list/nested/noise_named_rir.wav",
        "real_rirs_isotropic_noises/rir_list/recording.WAV",
    ]
    _create_files(root, rirs + [
        "simulated_rirs/rir_list/not_rir.flac",
        "simulated_rirs/rir_list/not_rir.wav.txt",
        "other/rir_looking_name.wav",
        "rir_in_root.wav",
        "metadata/rir_list",  # A file named rir_list is not an RIR directory.
    ])
    (root / "simulated_rirs/rir_list/directory.wav").mkdir()
    assert ConfiguredRawBoost(root).rir == sorted(root / relative_path for relative_path in rirs)


def test_noise_subtrees_are_excluded_even_when_they_contain_rir_list(tmp_path):
    root = tmp_path / "RIRS_NOISES"
    rir = "simulated_rirs/rir_list/rir.wav"
    _create_files(root, [rir,
        "simulated_rirs/noise_list/rir_list/noise.wav",
        "real_rirs_isotropic_noises/noise_list/rir_list/noise.wav",
        "pointsource_noises/rir_list/noise.wav",
        "simulated_rirs/rir_list/noise_list/noise.wav",
        "simulated_rirs/rir_list/pointsource_noises/noise.wav",
    ])
    assert ConfiguredRawBoost(root).rir == [root / rir]


def test_missing_rir_root_reports_its_path(tmp_path):
    root = tmp_path / "missing_RIRS_NOISES"
    with pytest.raises(FileNotFoundError, match="RIR resource root") as error:
        ConfiguredRawBoost(root)
    assert str(root) in str(error.value)


def test_rir_root_must_be_a_directory(tmp_path):
    root = tmp_path / "RIRS_NOISES"
    root.touch()
    with pytest.raises(NotADirectoryError, match="RIR resource root"):
        ConfiguredRawBoost(root)


@pytest.mark.parametrize("files", [
    [],
    ["simulated_rirs/noise_list/noise1.wav",
     "real_rirs_isotropic_noises/noise_list/noise2.wav",
     "pointsource_noises/noise3.wav"],
    ["simulated_rirs/rir_list/metadata.txt"],
], ids=["empty", "noise_only", "no_wav"])
def test_no_rir_files_report_expected_directory_structure(tmp_path, files):
    root = tmp_path / "RIRS_NOISES"
    root.mkdir()
    _create_files(root, files)
    with pytest.raises(ValueError, match="No RIR WAV files.*rir_list") as error:
        ConfiguredRawBoost(root)
    assert str(root) in str(error.value)
