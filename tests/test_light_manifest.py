import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fs_lprf.config import SEEDS, load_config, validate_config
from fs_lprf.data.evaluation_manifest import (
    dataset_fingerprint, load_or_create_light_manifest, validate_episode_manifest,
    validate_light_manifest,
)
from fs_lprf.utils.seed import capture_rng, seed_all


@pytest.fixture
def original_dev(tmp_path):
    """Metadata only: exact 1500- and 80-Query cases with five disjoint supports."""
    names = ["real"] + [f"TTS__fake{index:02d}" for index in range(32)]
    dataset = SimpleNamespace(split="dev", fake_classes=names[1:], uttids=[],
                              uttid2path={}, uttid2info={}, class_indices={})
    supports = {str(seed): {} for seed in SEEDS}
    queries = {}
    for index, name in enumerate(names):
        query_count = 1500 if index in (0, 1) else 80 if index == 2 else index + 3
        ids = [f"{name}_{number:04d}" for number in range(50 + query_count)]
        dataset.class_indices[name] = list(range(len(dataset.uttids), len(dataset.uttids) + len(ids)))
        dataset.uttids.extend(ids)
        for uid in ids:
            dataset.uttid2path[uid] = f"/synthetic/{uid}.wav"
            dataset.uttid2info[uid] = dict(origin_ds="synthetic", speaker=uid,
                                         attacker=name, label=0 if name == "real" else 1)
        for seed_index, seed in enumerate(SEEDS):
            supports[str(seed)][name] = ids[10 * seed_index:10 * (seed_index + 1)]
        queries[name] = ids[50:]
    full = dict(version=1, split="dev", shot=10, seeds=SEEDS,
                dataset_fingerprint=dataset_fingerprint(dataset), classes=names,
                supports=supports, queries=queries,
                loader_order="all supports (Real then sorted Fake); Real Query; sorted Fake Query")
    validate_episode_manifest(dataset, full)
    path = tmp_path / "dev_episode_manifest.json"
    path.write_text(json.dumps(full), encoding="utf-8")
    return dataset, full, path


def assert_rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert torch.equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch"], right["torch"])
    assert len(left["cuda"]) == len(right["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def test_light_manifest_caps_large_classes_keeps_all_small_and_reuses_support(original_dev, tmp_path):
    dataset, full, source = original_dev
    source_bytes = source.read_bytes()
    light = load_or_create_light_manifest(dataset, full, source, tmp_path / "dev_light_manifest.json")
    assert light["seeds"] == [42] and light["shot"] == 10
    assert light["supports"] == {"42": full["supports"]["42"]}
    assert light["source_manifest_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert light["dataset_fingerprint"] == full["dataset_fingerprint"]
    assert len(light["queries"]) == 33
    assert len(light["queries"]["real"]) == 1000
    assert len(light["queries"][dataset.fake_classes[0]]) == 1000
    assert len(light["queries"][dataset.fake_classes[1]]) == 80
    assert light["queries"][dataset.fake_classes[1]] == sorted(full["queries"][dataset.fake_classes[1]])
    for name in light["classes"]:
        query = light["queries"][name]
        assert len(query) == len(set(query)) == light["query_quotas"][name]
        assert set(query) <= set(full["queries"][name])
        for seed in SEEDS:
            assert not set(query) & set(full["supports"][str(seed)][name])
    # No real source manifest or support sampling is changed by the monitor derivation.
    assert source.read_bytes() == source_bytes
    assert full["queries"][dataset.fake_classes[0]] != light["queries"][dataset.fake_classes[0]]
    validate_light_manifest(dataset, light, full, source)


def test_light_selection_is_deterministic_across_runs_and_preserves_all_rng(original_dev, tmp_path):
    dataset, full, source = original_dev
    seed_all(71)
    before = capture_rng()
    path = tmp_path / "dev_light_manifest.json"
    first = load_or_create_light_manifest(dataset, full, source, path)
    assert_rng_equal(before, capture_rng())
    saved = path.read_bytes()
    random.random()
    np.random.rand(20)
    torch.rand(20)
    before_reload = capture_rng()
    assert load_or_create_light_manifest(dataset, full, source, path) == first
    assert_rng_equal(before_reload, capture_rng())
    assert path.read_bytes() == saved
    assert load_or_create_light_manifest(dataset, full, source, tmp_path / "another.json") == first
    assert not list(tmp_path.glob("*.tmp"))


def test_light_manifest_rejects_source_sha_or_cap_change_without_replacing_file(original_dev, tmp_path):
    dataset, full, source = original_dev
    path = tmp_path / "dev_light_manifest.json"
    load_or_create_light_manifest(dataset, full, source, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="source SHA"):
        load_or_create_light_manifest(dataset, full, source, path, max_query_per_class=50)
    source.write_bytes(source.read_bytes() + b"\n")  # Same parsed data, different source bytes.
    with pytest.raises(ValueError, match="source SHA"):
        load_or_create_light_manifest(dataset, full, source, path)
    assert path.read_bytes() == original


def test_light_manifest_rejects_tampering_and_nonselected_seed_support_overlap(original_dev, tmp_path):
    dataset, full, source = original_dev
    path = tmp_path / "dev_light_manifest.json"
    light = load_or_create_light_manifest(dataset, full, source, path)
    light["queries"]["real"][0] = full["supports"]["17"]["real"][0]
    path.write_text(json.dumps(light), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic selection"):
        load_or_create_light_manifest(dataset, full, source, path)
    with pytest.raises(ValueError, match="deterministic selection"):
        validate_light_manifest(dataset, light, full, source)


def test_light_manifest_rejects_mismatched_data_or_source_object(original_dev, tmp_path):
    dataset, full, source = original_dev
    path = tmp_path / "dev_light_manifest.json"
    changed = copy.deepcopy(full)
    changed["queries"]["real"] = list(reversed(changed["queries"]["real"]))
    with pytest.raises(ValueError, match="on disk"):
        load_or_create_light_manifest(dataset, changed, source, path)
    dataset.uttid2info[dataset.uttids[0]]["speaker"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        load_or_create_light_manifest(dataset, full, source, path)


def test_light_manifest_forbids_test_sources_and_empty_source_classes(original_dev, tmp_path):
    dataset, full, source = original_dev
    dataset.split = "test"
    with pytest.raises(ValueError, match="only.*Dev"):
        load_or_create_light_manifest(dataset, full, source, tmp_path / "light.json")
    dataset.split = "dev"
    full["queries"][dataset.fake_classes[1]] = []
    source.write_text(json.dumps(full), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid source"):
        load_or_create_light_manifest(dataset, full, source, tmp_path / "light.json")


@pytest.mark.parametrize("field,value,message", [
    ("train.light_validation_interval_steps", 0, "positive integer"),
    ("train.full_validation_interval_steps", True, "positive integer"),
    ("train.full_validation_interval_steps", 700, "divisible by light"),
    ("train.full_validation_interval_steps", 600, "total_steps.*divisible"),
    ("validation.light_seed", 17, "Phase 1 protocol"),
    ("validation.light_max_query_per_class", 0, "between 1 and 1000"),
    ("validation.light_max_query_per_class", 1001, "between 1 and 1000"),
    ("train.episodes_per_step", 4, "Phase 1 protocol"),
    ("train.total_query_per_episode", 6, "Phase 1 protocol"),
])
def test_new_configuration_rejects_invalid_schedule_or_protocol(config, field, value, message):
    changed = copy.deepcopy(config)
    section, key = field.split(".")
    changed[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(changed)


def test_new_configuration_fields_can_be_overridden_consistently():
    path = Path(__file__).resolve().parents[1] / "configs" / "speechfake.yaml"
    cfg = load_config(path, ["train.light_validation_interval_steps=100",
                            "train.full_validation_interval_steps=400",
                            "validation.light_max_query_per_class=80"])
    assert cfg["train"]["light_validation_interval_steps"] == 100
    assert cfg["train"]["full_validation_interval_steps"] == 400
    assert cfg["validation"]["light_max_query_per_class"] == 80
    legacy = copy.deepcopy(cfg)
    legacy["train"]["validation_interval_steps"] = 200
    with pytest.raises(ValueError, match="Replace train.validation_interval_steps"):
        validate_config(legacy)
