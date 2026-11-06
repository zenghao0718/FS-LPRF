"""Write source-removal suggestions only; never delete any source file."""
import csv
import json
from pathlib import Path
import uuid

import _bootstrap
from data.cache import atomic_json
from data.deletion_audit import audit_source_deletion
from utils.config import common_parser, load_config


def main():
    parser = common_parser("Read-only Train/Dev source deletion safety audit; preserve all Test originals")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    output = Path(args.output_dir or ("outputs/source_deletion_audit/" + uuid.uuid4().hex)).resolve()
    project = Path(_bootstrap.ROOT).resolve()
    if output.is_relative_to(project) and (
            output == project or output.relative_to(project).parts[0] not in ("outputs", ".tmp", ".cache")):
        raise ValueError("Audit reports must use an ignored output directory or an external directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing audit reports are preserved; choose a new output directory")
    try:
        report = audit_source_deletion(cfg)
    except (ValueError, OSError) as error:
        report = dict(status="blocked", errors=[str(error)], deletable_files=[], retained_files=[],
                      retained_essentials=[], deletion_performed=False,
                      estimated_reclaimable_logical_bytes=0)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"audit.json", report)
    for name, key in (("deletable_wav.csv", "deletable_files"), ("retained_files.csv", "retained_files")):
        with (output/name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "splits", "uttids", "bytes", "reasons"])
            writer.writeheader()
            for row in report[key]:
                writer.writerow({field: "|".join(row.get(field, [])) if field in ("splits", "uttids", "reasons")
                                 else row.get(field, 0) for field in writer.fieldnames})
    print(json.dumps(dict(status=report["status"], output_dir=str(output),
                          deletable_files=len(report["deletable_files"]),
                          estimated_reclaimable_logical_bytes=report["estimated_reclaimable_logical_bytes"],
                          errors=report["errors"], deletion_performed=False), indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
