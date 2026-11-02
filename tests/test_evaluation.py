import csv
import random
from types import SimpleNamespace
import numpy as np
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


def test_bank_sampling_uses_exact_real_source_quotas_without_replacement():
    dataset = SimpleNamespace(split="train", do_augment=False, class_indices={"real": []},
                              uttids=[], uttid2info={}, fake_classes=[f"NV__g{i:02d}" for i in range(32)])
    for source in REAL_QUOTAS:
        for number in range(206):
            uid = f"{source}_{number}"
            dataset.class_indices["real"].append(len(dataset.uttids))
            dataset.uttids.append(uid)
            dataset.uttid2info[uid] = {"origin_ds": source}
    for name in dataset.fake_classes:
        dataset.class_indices[name] = list(range(len(dataset.uttids), len(dataset.uttids) + 1024))
        dataset.uttids.extend(f"{name}_{number}" for number in range(1024))
    selected = select_bank_samples(dataset, 17)
    assert len(selected) == 33
    assert all(len(ids) == len(set(ids)) == 1024 for ids in selected.values())
    assert {source: sum(dataset.uttid2info[uid]["origin_ds"] == source for uid in selected["real"])
            for source in REAL_QUOTAS} == REAL_QUOTAS
    assert selected == select_bank_samples(dataset, 17)
