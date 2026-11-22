"""Compare prototype-bank augmentation manifests without rereading audio."""
import csv
from collections import defaultdict
from pathlib import Path
import json
import sys


FIELDS = ("uttid", "class_name", "origin_ds", "sample_seed", "codec_applied", "codec_family",
          "codec_variant", "codec_param", "codec_status", "rir_probability", "rawboost_probability",
          "augmentation_mode", "rir_applied", "rir_path", "rawboost_applied", "evaluation_view_mode")


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty augmentation manifest: {path}")
    return rows


def _flag(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Expected a boolean flag, got {value!r}")


def summarize(rows):
    by_class = defaultdict(lambda: dict(count=0, codec=0, rir=0, rawboost=0))
    for row in rows:
        bucket = by_class[row["class_name"]]
        bucket["count"] += 1
        bucket["codec"] += int(_flag(row["codec_applied"]))
        bucket["rir"] += int(_flag(row.get("rir_applied", False)))
        bucket["rawboost"] += int(_flag(row.get("rawboost_applied", False)))
    total = len(rows)
    return dict(rows=total, classes=len(by_class),
                codec_applied=sum(item["codec"] for item in by_class.values()),
                codec_fraction=sum(item["codec"] for item in by_class.values()) / total,
                rir_applied=sum(item["rir"] for item in by_class.values()),
                rawboost_applied=sum(item["rawboost"] for item in by_class.values()),
                per_class={name: dict(item, codec_fraction=item["codec"] / item["count"])
                           for name, item in sorted(by_class.items())})


def compare_codec_assignment(reference_rows, candidate_rows):
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("Augmentation manifests have different lengths")
    examples = []
    mismatches = 0
    for index, (left, right) in enumerate(zip(reference_rows, candidate_rows)):
        reason = None
        if left["uttid"] != right["uttid"] or left["class_name"] != right["class_name"]:
            reason = "identity"
        elif _flag(left["codec_applied"]) != _flag(right["codec_applied"]):
            reason = "codec_applied"
        if reason is None:
            continue
        mismatches += 1
        if len(examples) < 20:
            examples.append(dict(index=index, uttid=left["uttid"], reason=reason))
    return dict(rows=len(reference_rows), mismatches=mismatches, examples=examples, identical=mismatches == 0)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_codec_assignment.py B.csv C.csv report.json")
    left, right = read_manifest(sys.argv[1]), read_manifest(sys.argv[2])
    report = dict(reference=summarize(left), candidate=summarize(right), comparison=compare_codec_assignment(left, right))
    Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(identical=report["comparison"]["identical"], mismatches=report["comparison"]["mismatches"],
                          reference_codec=report["reference"]["codec_applied"],
                          candidate_codec=report["candidate"]["codec_applied"]), sort_keys=True))
    if not report["comparison"]["identical"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
