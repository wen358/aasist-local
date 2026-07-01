import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    bona = scores[labels == 0]
    spoof = scores[labels == 1]
    thresholds = np.sort(np.unique(scores))
    thresholds = np.r_[thresholds[0] - 1e-6, thresholds, thresholds[-1] + 1e-6]
    best_eer = 1.0
    best_threshold = thresholds[0]
    for threshold in thresholds:
        false_reject = (bona < threshold).sum() / bona.size
        false_accept = (spoof >= threshold).sum() / spoof.size
        current = (false_reject + false_accept) / 2.0
        if current < best_eer:
            best_eer = current
            best_threshold = threshold
    return float(best_eer), float(best_threshold)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    eer_value, threshold = compute_eer(labels, scores)
    predicted_bonafide = scores >= 0.0
    predicted_labels = np.where(predicted_bonafide, 0, 1)
    return {
        "eer": eer_value,
        "eer_percent": eer_value * 100.0,
        "eer_threshold": threshold,
        "accuracy_at_0": float((predicted_labels == labels).mean()),
        "mean_score": float(scores.mean()),
    }


def summarize_by_codec(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        groups[row["codec"]].append((int(row["label"]), float(row["score"])))

    result = []
    for codec in sorted(groups):
        values = groups[codec]
        labels = np.asarray([label for label, _ in values], dtype=np.int64)
        scores = np.asarray([score for _, score in values], dtype=np.float64)
        bonafide = int((labels == 0).sum())
        spoof = int((labels == 1).sum())
        if bonafide and spoof:
            group_metrics = metrics(labels, scores)
        else:
            group_metrics = {
                "eer": None,
                "eer_percent": None,
                "eer_threshold": None,
                "accuracy_at_0": None,
                "mean_score": float(scores.mean()),
            }
        result.append(
            {
                "codec": codec,
                "n": len(values),
                "bonafide": bonafide,
                "spoof": spoof,
                **group_metrics,
            }
        )
    result.sort(key=lambda row: -1.0 if row["eer"] is None else row["eer"], reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AASIST score CSV by codec without rerunning inference.")
    parser.add_argument("--scores", type=Path, required=True, help="CSV written by eval_manifest.py --output-scores.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for codec-level metrics.")
    args = parser.parse_args()

    with args.scores.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty score file: {args.scores}")

    required = {"utterance_id", "codec", "label", "score"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"score file is missing columns: {sorted(missing)}")

    summary = summarize_by_codec(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["codec", "n", "bonafide", "spoof", "eer", "eer_percent", "eer_threshold", "accuracy_at_0", "mean_score"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps({"by_codec": summary}, indent=2))
    print(f"wrote codec metrics: {args.output}")


if __name__ == "__main__":
    main()
