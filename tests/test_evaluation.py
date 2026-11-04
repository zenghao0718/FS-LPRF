import csv
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fs_lprf.data.evaluation_manifest import load_or_create_manifest
from fs_lprf.engine.evaluator import evaluate
from fs_lprf.methods.prototype_bank import select_bank_samples, REAL_QUOTAS
from fs_lprf.utils.seed import seed_all, capture_rng


class FeatureDataset(torch.utils.data.Dataset):
    """Synthetic engine fixture; no dataset or real experimental metrics are used."""
    def __init__(self, metadata):
        self.__dict__.update(metadata.__dict__)
        self.synthetic_features = torch.randn(len(self.uttids), 1, 9, generator=torch.Generator().manual_seed(71))

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, index):
        return dict(uttid=self.uttids[index], feats=self.synthetic_features[index])

    @staticmethod
    def collate_fn(batch):
        return dict(uttids=[item["uttid"] for item in batch], feats=torch.stack([item["feats"] for item in batch]))


class FeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(9, 25 * 7)
        self.device = torch.device("cpu")

    def forward(self, source):
        return dict(feats=self.proj(source["feats"].squeeze(1)).view(-1, 25, 7))


def test_evaluator_fixed_manifest_predictions_macro_and_rng_restoration(dataset, config, tmp_path):
    ds = FeatureDataset(dataset)
    seed_all(42)
    model = FeatureModel().train()
    manifest = load_or_create_manifest(ds, tmp_path / "manifest.json", 10, [17, 42, 71, 100, 123])
    before = capture_rng()
    first = evaluate(model, ds, manifest, config, tmp_path, "dev", step=200)
    after = capture_rng()
    assert model.training
    assert before["python"] == after["python"]
    assert torch.equal(before["torch"], after["torch"])
    assert torch.equal(before["numpy"][1], after["numpy"][1])
    second = evaluate(model, ds, manifest, config, tmp_path, "dev", step=400)
    assert first == second
    assert len(first["per_seed"]) == 5
    assert first["mean_eer"] == np.mean([row["eer"] for row in first["per_seed"]])
    assert (tmp_path / "dev_predictions_step200.csv").read_bytes() == (tmp_path / "dev_predictions_step400.csv").read_bytes()
    with (tmp_path / "dev_per_class.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 5 * 32
    with (tmp_path / "dev_predictions_step200.csv").open() as handle:
        assert set(csv.DictReader(handle).fieldnames) == {"uttid", "fake_class", "seed", "label", "score", "prediction"}


@pytest.fixture
def bank_dataset():
    dataset = SimpleNamespace(split="train", do_augment=False, class_indices={"real": []},
                              uttids=[], uttid2info={}, fake_classes=[f"NV__g{i:02d}" for i in range(32)])
    # Use the four actual BD sources, independently of the production constant.
    for source in ("AISHELL-1", "AISHELL-3", "LibriTTS", "VCTK"):
        for number in range(260):
            uid = f"{source}_{number}"
            dataset.class_indices["real"].append(len(dataset.uttids))
            dataset.uttids.append(uid)
            dataset.uttid2info[uid] = {"origin_ds": source}
    for name in dataset.fake_classes:
        dataset.class_indices[name] = list(range(len(dataset.uttids), len(dataset.uttids) + 1030))
        dataset.uttids.extend(f"{name}_{number}" for number in range(1030))
    return dataset


def test_bank_sampling_uses_exact_real_source_quotas_without_replacement(bank_dataset):
    dataset = bank_dataset
    assert REAL_QUOTAS == {"AISHELL-1": 256, "AISHELL-3": 256, "LibriTTS": 256, "VCTK": 256}
    assert "CommonVoice" not in {info["origin_ds"] for info in dataset.uttid2info.values()}
    selected = select_bank_samples(dataset, 17)
    assert len(selected) == 33
    assert set(selected) == {"real", *dataset.fake_classes}
    assert all(len(ids) == len(set(ids)) == 1024 for ids in selected.values())
    assert len({uid for ids in selected.values() for uid in ids}) == 33 * 1024
    for name in dataset.fake_classes:
        assert set(selected[name]) <= {dataset.uttids[index] for index in dataset.class_indices[name]}
    assert {source: sum(dataset.uttid2info[uid]["origin_ds"] == source for uid in selected["real"])
            for source in REAL_QUOTAS} == REAL_QUOTAS
    assert selected == select_bank_samples(dataset, 17)


@pytest.mark.parametrize("source", ["AISHELL-1", "AISHELL-3", "LibriTTS", "VCTK"])
def test_bank_sampling_reports_each_real_source_shortage(bank_dataset, source):
    indexes = [index for index in bank_dataset.class_indices["real"]
               if bank_dataset.uttid2info[bank_dataset.uttids[index]]["origin_ds"] == source]
    removed = set(indexes[255:])
    bank_dataset.class_indices["real"] = [index for index in bank_dataset.class_indices["real"]
                                           if index not in removed]
    with pytest.raises(ValueError, match="capacity mismatch") as error:
        select_bank_samples(bank_dataset, 17)
    message = str(error.value)
    assert source in message and "'required': 256" in message and "'available': 255" in message
    assert "CommonVoice" not in message


def test_bank_sampling_keeps_fake_capacity_requirement(bank_dataset):
    name = bank_dataset.fake_classes[0]
    bank_dataset.class_indices[name] = bank_dataset.class_indices[name][:1023]
    with pytest.raises(ValueError, match="capacity mismatch") as error:
        select_bank_samples(bank_dataset, 17)
    message = str(error.value)
    assert name in message and "'required': 1024" in message and "'available': 1023" in message


def test_bank_sampling_preserves_seed_determinism_and_global_rng(bank_dataset):
    before = random.getstate()
    first = select_bank_samples(bank_dataset, 17)
    second = select_bank_samples(bank_dataset, 42)
    assert first == select_bank_samples(bank_dataset, 17)
    assert any(first[name] != second[name] for name in first)
    assert random.getstate() == before


@pytest.mark.parametrize("split,augment", [("dev", False), ("train", True)])
def test_bank_sampling_still_requires_unaugmented_train(bank_dataset, split, augment):
    bank_dataset.split = split
    bank_dataset.do_augment = augment
    with pytest.raises(ValueError, match="unaugmented SpeechFake-BD Train"):
        select_bank_samples(bank_dataset, 17)
