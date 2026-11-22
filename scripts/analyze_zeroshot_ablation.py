"""CPU-only A/B/C metrics. Query audio is not reloaded and nothing is retrained."""
import csv
import json
from pathlib import Path
import argparse

import _bootstrap
import numpy as np

from utils.metrics import binary_metrics, macro_metrics
from utils.mindcf import pooled_mindcf


def read_predictions(path):
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            label = int(row["label"])
            if label not in (0, 1) or not np.isfinite(score):
                raise ValueError(f"Invalid prediction row: {row['uttid']}")
            rows.append(dict(uttid=row["uttid"], label=label, attack_id=row["attack_id"], score=score,
                             prediction=int(row["prediction"])))
    if not rows or len({row["uttid"] for row in rows}) != len(rows):
        raise ValueError(f"Predictions must contain unique utterance IDs: {path}")
    return rows


def attack_rows(rows):
    real = [row["score"] for row in rows if row["label"] == 0]
    grouped = {}
    for row in rows:
        if row["label"] == 1:
            grouped.setdefault(row["attack_id"], []).append(row["score"])
    table = []
    for attack in sorted(grouped):
        fake = grouped[attack]
        metrics = binary_metrics([0] * len(real) + [1] * len(fake), real + fake)
        table.append(dict(attack_id=attack, real_count=len(real), fake_count=len(fake), **metrics))
    if len(table) != 16:
        raise ValueError(f"Expected 16 attacks, found {len(table)}")
    return table


def summarize(rows):
    per_attack = attack_rows(rows)
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    pooled = binary_metrics(labels, scores)
    mindcf = pooled_mindcf(labels, scores)
    if any(row["prediction"] != int(row["score"] >= 0) for row in rows):
        raise ValueError("Stored predictions do not follow the fixed zero threshold")
    return dict(macro=macro_metrics(per_attack), pooled={key: pooled[key] for key in ("eer", "auc", "ap", "acc")},
                pooled_mindcf=mindcf["mindcf"], mindcf_threshold_metric_only=mindcf["threshold"],
                mindcf_parameters={key: mindcf[key] for key in ("cmiss", "cfa", "pspoof", "beta", "default_cost", "score_direction")},
                query_count=len(rows), real_count=sum(label == 0 for label in labels),
                fake_count=sum(label == 1 for label in labels), per_attack=per_attack)


def percent(value):
    return value * 100


def write_reports(output, names, summaries, paths):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    comparison = []
    for name in names:
        item = summaries[name]
        comparison.append(dict(
            experiment=name, macro_eer=item["macro"]["eer"], pooled_mindcf=item["pooled_mindcf"],
            pooled_eer=item["pooled"]["eer"], macro_auc=item["macro"]["auc"], macro_ap=item["macro"]["ap"],
            macro_acc=item["macro"]["acc"], pooled_auc=item["pooled"]["auc"], pooled_ap=item["pooled"]["ap"],
            pooled_acc=item["pooled"]["acc"], predictions=str(paths[name])))
    (output / "A_B_C_comparison.json").write_text(json.dumps(dict(experiments=comparison), indent=2) + "\n", encoding="utf-8")
    with (output / "A_B_C_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    attacks = sorted({row["attack_id"] for row in summaries[names[0]]["per_attack"]})
    with (output / "A_B_C_per_attack.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["attack_id"]
        for name in names:
            fields.extend([f"{name}_eer", f"{name}_auc", f"{name}_ap", f"{name}_acc"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        indexed = {name: {row["attack_id"]: row for row in summaries[name]["per_attack"]} for name in names}
        for attack in attacks:
            record = dict(attack_id=attack)
            for name in names:
                row = indexed[name][attack]
                record.update({f"{name}_eer": row["eer"], f"{name}_auc": row["auc"], f"{name}_ap": row["ap"],
                               f"{name}_acc": row["acc"]})
            writer.writerow(record)
    return comparison


def _line(name, item):
    macro = item["macro"]
    pooled = item["pooled"]
    return (f"| {name} | {percent(macro['eer']):.4f}% | {item['pooled_mindcf']:.6f} | {percent(pooled['eer']):.4f}% | "
            f"{percent(macro['auc']):.4f}% | {percent(macro['ap']):.4f}% | {percent(macro['acc']):.4f}% |")


def write_markdown(path, comparison_source, summaries):
    lines = ["# FS-LPRF Codec Offline v1 原型增强消融", "",
             "研究问题：在正式模型不变的前提下，原型构建加入 Codec，以及进一步加入 RIR + RawBoost，是否改善 ASVspoof5 未知生成器的 0-shot 检测。",
             "", "分数方向：项目分数 Fake 越高越假，`score >= 0` 判为 Fake。Pooled minDCF 使用 `-project_score`，bona fide 为目标类。",
             "minDCF 阈值只用于计算指标，不改变固定零阈值。", "",
             "| 实验 | Macro EER ↓ | Pooled minDCF ↓ | Pooled EER ↓ | Macro AUC ↑ | Macro AP ↑ | Macro ACC |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    labels = {"A": "A Clean", "B": "B Codec50", "C": "C FullAug"}
    for key, item in summaries.items():
        lines.append(_line(labels[key], item))
    lines.extend(["", "## Pooled AUC / AP / ACC", ""])
    for key, item in summaries.items():
        pooled = item["pooled"]
        lines.append(f"- {labels[key]}：AUC {percent(pooled['auc']):.4f}%，AP {percent(pooled['ap']):.4f}%，ACC {percent(pooled['acc']):.4f}%")
    lines.extend(["", "## 相对变化", ""])
    base, codec, full = summaries["A"], summaries["B"], summaries["C"]
    lines.append(f"- B 相对 A 的 Macro EER 变化：{(codec['macro']['eer'] - base['macro']['eer']) * 100:+.4f} 个百分点；"
                 f"Pooled minDCF 变化：{codec['pooled_mindcf'] - base['pooled_mindcf']:+.6f}。")
    lines.append(f"- C 相对 B 的 Macro EER 变化：{(full['macro']['eer'] - codec['macro']['eer']) * 100:+.4f} 个百分点；"
                 f"Pooled minDCF 变化：{full['pooled_mindcf'] - codec['pooled_mindcf']:+.6f}。")
    eer_down = codec["macro"]["eer"] < base["macro"]["eer"]
    dcf_down = codec["pooled_mindcf"] < base["pooled_mindcf"]
    lines.append("- B 相对 A：" + ("Macro EER 与 Pooled minDCF 都下降。" if eer_down and dcf_down else
                 "Macro EER 与 Pooled minDCF 的方向不一致，不能把单一指标改善说成全面提升。" if eer_down != dcf_down else
                 "Macro EER 与 Pooled minDCF 都没有下降。"))
    c_eer = full["macro"]["eer"] < codec["macro"]["eer"]
    c_dcf = full["pooled_mindcf"] < codec["pooled_mindcf"]
    lines.append("- C 相对 B：" + ("RIR + RawBoost 在两个主要指标上都进一步下降。" if c_eer and c_dcf else
                 "RIR + RawBoost 的 Macro EER 与 Pooled minDCF 方向不一致。" if c_eer != c_dcf else
                 "RIR + RawBoost 没有在两个主要指标上进一步下降。"))
    lines.extend(["", "## 逐 Attack EER（%）", "", "| Attack | A | B | C | B-A | C-B |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    indexed = {name: {row["attack_id"]: row for row in summaries[name]["per_attack"]} for name in summaries}
    for attack in sorted(indexed["A"]):
        values = [indexed[name][attack]["eer"] * 100 for name in ("A", "B", "C")]
        lines.append(f"| {attack} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[1]-values[0]:+.4f} | {values[2]-values[1]:+.4f} |")
    improved = [attack for attack in indexed["A"] if indexed["C"][attack]["eer"] < indexed["A"][attack]["eer"]]
    degraded = [attack for attack in indexed["A"] if indexed["C"][attack]["eer"] > indexed["A"][attack]["eer"]]
    lines.extend(["", f"C 相对 A 的 EER 下降攻击：{', '.join(improved) or '无'}。",
                  f"C 相对 A 的 EER 上升攻击：{', '.join(degraded) or '无'}。", ""])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    paths = {"A": Path(args.a), "B": Path(args.b), "C": Path(args.c)}
    loaded = {name: read_predictions(path) for name, path in paths.items()}
    ids = [row["uttid"] for row in loaded["A"]]
    if any([row["uttid"] for row in loaded[name]] != ids for name in ("B", "C")):
        raise SystemExit("A/B/C query IDs or order differ; refusing to compare")
    summaries = {name: summarize(rows) for name, rows in loaded.items()}
    if round(summaries["A"]["macro"]["eer"] * 100, 4) != 11.0202:
        raise SystemExit(f"Recomputed A Macro EER is not the reported 11.0202%: {summaries['A']['macro']['eer']}")
    root = Path(args.results_root)
    for name, item in summaries.items():
        destination = root / name
        destination.mkdir(parents=True, exist_ok=True)
        payload = dict(item, predictions=str(paths[name].resolve()))
        (destination / "metrics_recomputed.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output = Path(args.output_dir)
    write_reports(output, ("A", "B", "C"), summaries, paths)
    write_markdown(output / "final_report.md", None, summaries)
    print(json.dumps({name: dict(macro_eer=item["macro"]["eer"], pooled_mindcf=item["pooled_mindcf"],
                                 pooled_eer=item["pooled"]["eer"]) for name, item in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
