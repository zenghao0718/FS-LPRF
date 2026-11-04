"""OpenSLR 28 RIR candidates must come from its four text indexes."""
import pytest

from data.speechfake_dataset import ConfiguredRawBoost

INDEX_FILES = (
    "simulated_rirs/smallroom/rir_list",
    "simulated_rirs/mediumroom/rir_list",
    "simulated_rirs/largeroom/rir_list",
    "real_rirs_isotropic_noises/rir_list",
)


def _expected_rirs(root):
    return sorted(root / path for path in (
        "simulated_rirs/smallroom/Room001/a.wav",
        "simulated_rirs/mediumroom/Room001/c.wav",
        "simulated_rirs/largeroom/Room001/d.wav",
        "real_rirs_isotropic_noises/real.wav",
    ))


def _create_files(root, relative_paths):
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_rir_candidates_only_use_simulated_and_real_rir_lists_in_sorted_order(rir_root):
    root = rir_root
    extra = "simulated_rirs/smallroom/Room001/b.wav"
    noises = [
        "simulated_rirs/smallroom/Room001/unindexed.wav",
        "real_rirs_isotropic_noises/noise.wav",
        "pointsource_noises/point.wav",
    ]
    _create_files(root, noises + [extra])
    index = root / INDEX_FILES[0]
    # Duplicate entries and alternate spellings must not duplicate candidates.
    with index.open("a", encoding="utf-8") as handle:
        handle.write(f"RIRS_NOISES/{extra}\nRoom001/a.wav\n./Room001/a.wav\n")
    assert all((root / relative).is_file() for relative in INDEX_FILES)
    augmentor = ConfiguredRawBoost(root)
    assert len(augmentor.rir) == 5
    assert augmentor.rir == sorted(_expected_rirs(root) + [root / extra])
    assert not set(augmentor.rir) & {root / relative_path for relative_path in noises}
    assert ConfiguredRawBoost(root).rir == augmentor.rir


def test_rir_candidates_accept_only_wav_files_without_filename_heuristics(rir_root):
    root = rir_root
    extra = "real_rirs_isotropic_noises/noise_named_rir.WAV"
    _create_files(root, [extra,
        "simulated_rirs/smallroom/not_rir.flac",
        "simulated_rirs/smallroom/not_rir.wav.txt",
        "other/rir_looking_name.wav",
        "rir_in_root.wav",
        "metadata/rir_list",  # Unrelated indexes must not be loaded.
    ])
    (root / "simulated_rirs/smallroom/directory.wav").mkdir()
    with (root / INDEX_FILES[-1]).open("a", encoding="utf-8") as handle:
        handle.write("noise_named_rir.WAV\n")
    assert ConfiguredRawBoost(root).rir == sorted(_expected_rirs(root) + [root / extra])


def test_noise_subtrees_are_excluded_even_when_they_contain_rir_list(rir_root):
    for relative in ("simulated_rirs/noise_list", "real_rirs_isotropic_noises/noise_list", "pointsource_noises"):
        _create_files(rir_root, [f"{relative}/noise.wav"])
        (rir_root / relative / "rir_list").write_text("noise.wav\n", encoding="utf-8")
    assert ConfiguredRawBoost(rir_root).rir == _expected_rirs(rir_root)


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
    ["simulated_rirs/smallroom/metadata.txt"],
], ids=["empty", "noise_only", "no_wav"])
def test_no_rir_files_report_expected_directory_structure(tmp_path, files):
    root = tmp_path / "RIRS_NOISES"
    root.mkdir()
    _create_files(root, INDEX_FILES)
    _create_files(root, files)
    with pytest.raises(ValueError, match="No RIR WAV files.*rir_list") as error:
        ConfiguredRawBoost(root)
    assert str(root) in str(error.value)


@pytest.mark.parametrize("style", [
    "dataset_relative", "root_relative", "index_relative", "absolute",
    "native_absolute", "quoted_absolute", "dot_dataset_relative", "split_options",
])
def test_rir_index_resolves_path_formats_and_kaldi_metadata(rir_root, style):
    target = rir_root / "real_rirs_isotropic_noises/real.wav"
    expected = _expected_rirs(rir_root)
    if style == "quoted_absolute":
        target = target.with_name("real sample.wav")
        target.touch()
        expected = sorted([path for path in expected if path.name != "real.wav"] + [target])
    references = {
        "dataset_relative": "RIRS_NOISES/real_rirs_isotropic_noises/real.wav",
        "root_relative": "real_rirs_isotropic_noises/real.wav",
        "index_relative": "real.wav",
        "absolute": target.as_posix(),
        "native_absolute": str(target),
        "quoted_absolute": f'"{target.as_posix()}"',
        "dot_dataset_relative": "./RIRS_NOISES/real_rirs_isotropic_noises/real.wav",
        "split_options": "real.wav",
    }
    options = "--rir-id=real --room-id=room --probability=1.0 --rt60=0.5"
    if style == "split_options":
        options = "--rir-id real --room-id room --probability 1.0"
    (rir_root / INDEX_FILES[-1]).write_text(f"{options} {references[style]}\n", encoding="utf-8")
    assert ConfiguredRawBoost(rir_root).rir == expected


@pytest.mark.parametrize("relative", INDEX_FILES)
def test_missing_any_rir_index_cannot_return_a_partial_pool(rir_root, relative):
    index = rir_root / relative
    index.unlink()
    with pytest.raises(FileNotFoundError, match="Required RIR index") as error:
        ConfiguredRawBoost(rir_root)
    assert str(index) in str(error.value)


@pytest.mark.parametrize("relative", INDEX_FILES)
def test_empty_any_rir_index_cannot_return_a_partial_pool(rir_root, relative):
    index = rir_root / relative
    index.write_text("\n \t\n# no entries\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty RIR index") as error:
        ConfiguredRawBoost(rir_root)
    assert str(index) in str(error.value)


def test_rir_list_must_be_a_text_file_not_a_directory(rir_root):
    index = rir_root / INDEX_FILES[0]
    index.unlink()
    index.mkdir()
    (index / "unindexed.wav").touch()
    with pytest.raises(FileNotFoundError, match="regular text file"):
        ConfiguredRawBoost(rir_root)


def test_blank_lines_comments_and_utf8_bom_do_not_drop_indexed_rirs(rir_root):
    index = rir_root / INDEX_FILES[0]
    index.write_text("\n \t\n# comment\n" + index.read_text(encoding="utf-8") + "\n",
                     encoding="utf-8-sig")
    assert ConfiguredRawBoost(rir_root).rir == _expected_rirs(rir_root)


@pytest.mark.parametrize("record,error_type,message", [
    ("missing.wav", FileNotFoundError, "indexed RIR WAV file"),
    ("invalid.flac", ValueError, "expected a WAV"),
    ("directory.wav", FileNotFoundError, "not a file"),
    ("../../outside.wav", ValueError, "escapes resource root"),
    ("RIRS_NOISES/../outside.wav", ValueError, "escapes resource root"),
    ("absolute_outside", ValueError, "escapes resource root"),
    ("RIRS_NOISES/pointsource_noises/noise.wav", ValueError, "ordinary-noise"),
    ("--rir-id=missing_path", ValueError, "exactly one"),
    ("real.wav real.wav", ValueError, "exactly one"),
    ('"unclosed.wav', ValueError, "malformed"),
    ('"sox real.wav -t wav - |"', ValueError, "expected a WAV"),
    ('""', ValueError, "exactly one"),
    (".", ValueError, "expected a WAV"),
])
def test_invalid_index_record_reports_index_and_line(rir_root, record, error_type, message):
    index = rir_root / INDEX_FILES[-1]
    (index.parent / "directory.wav").mkdir()
    if record == "absolute_outside":
        record = (rir_root.parent / "outside.wav").as_posix()
    index.write_text(f"\n{record}\n", encoding="utf-8")
    with pytest.raises(error_type, match=message) as error:
        ConfiguredRawBoost(rir_root)
    assert f"{index}:2:" in str(error.value)
