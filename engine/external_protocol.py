"""External task organization; sampling stays in the SpeechFake manifest builder."""
import json
from pathlib import Path

from data.evaluation_manifest import DEV_SEEDS, dataset_fingerprint, load_or_create_manifest


def labeled_protocol(dataset, path):
    if not dataset.class_indices["real"] or not dataset.fake_classes:
        raise ValueError("Labeled 10-shot requires both Real and Fake audio")
    if any(row["label"] is None for row in dataset.records):
        raise ValueError("Labeled evaluation requires every label")
    return load_or_create_manifest(dataset, path, 10, DEV_SEEDS)


def unlabeled_protocol(dataset, support_ids, query_ids, path):
    if dataset.group_by != "none" or dataset.fake_classes != ["fake"]:
        raise ValueError("Unlabeled 10-shot requires a single Real/Fake Support task")
    expected = {"real": 0, "fake": 1}
    if set(support_ids) != set(expected):
        raise ValueError("10-shot requires explicit Real and Fake Support")
    for name, label in expected.items():
        ids = support_ids[name]
        if len(ids) != 10 or len(set(ids)) != 10:
            raise ValueError(f"10-shot requires exactly 10 {name} Support audio files")
        if any(dataset.by_id[uid]["label"] != label for uid in ids):
            raise ValueError(f"Incorrect {name} Support labels")
    support = [uid for ids in support_ids.values() for uid in ids]
    if (not query_ids or len(query_ids) != len(set(query_ids)) or set(support) & set(query_ids)
            or set(support) | set(query_ids) != set(dataset.uttids)):
        raise ValueError("Support and nonempty Query must be disjoint and cover all inputs")
    if any(dataset.by_id[uid]["label"] is not None for uid in query_ids):
        raise ValueError("Unlabeled Query must not have labels")
    manifest = dict(version=1, kind="external_unlabeled", split="external", shot=10,
                    seeds=[42], classes=["real", "fake"], supports={"42": support_ids},
                    queries={"query": query_ids}, dataset_fingerprint=dataset_fingerprint(dataset),
                    loader_order="all supports (Real then Fake); unlabeled Query")
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
