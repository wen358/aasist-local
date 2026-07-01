import argparse
import csv
import json
import random
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
    predicted_labels = np.where(scores >= 0.0, 0, 1)
    return {
        "eer": eer_value,
        "eer_percent": eer_value * 100.0,
        "eer_threshold": threshold,
        "accuracy_at_0": float((predicted_labels == labels).mean()),
        "mean_score": float(scores.mean()),
    }


def read_scores(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty score file: {path}")
    required = {"utterance_id", "codec", "label", "score"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"score file is missing columns: {sorted(missing)}")
    return rows


def split_calibration(rows: list[dict], cal_per_class: int, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["codec"], int(row["label"]))].append(row)

    rng = random.Random(seed)
    calibration_ids = set()
    for key, values in grouped.items():
        shuffled = list(values)
        rng.shuffle(shuffled)
        take = min(cal_per_class, len(shuffled) // 2)
        if take == 0:
            print(f"warning: no calibration samples selected for {key}")
        for row in shuffled[:take]:
            calibration_ids.add(row["utterance_id"])

    calibration = [row for row in rows if row["utterance_id"] in calibration_ids]
    test = [row for row in rows if row["utterance_id"] not in calibration_ids]
    return calibration, test


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    codecs = [row["codec"] for row in rows]
    return labels, scores, codecs


def learn_codec_shifts(calibration: list[dict], target_codecs: set[str]) -> dict[str, float]:
    labels, scores, codecs = arrays(calibration)
    _, global_threshold = compute_eer(labels, scores)
    shifts = {}
    for codec in sorted(set(codecs)):
        if target_codecs and codec not in target_codecs:
            shifts[codec] = 0.0
            continue
        indexes = np.asarray([index for index, value in enumerate(codecs) if value == codec], dtype=np.int64)
        codec_labels = labels[indexes]
        codec_scores = scores[indexes]
        if (codec_labels == 0).sum() == 0 or (codec_labels == 1).sum() == 0:
            shifts[codec] = 0.0
            continue
        _, codec_threshold = compute_eer(codec_labels, codec_scores)
        shifts[codec] = float(global_threshold - codec_threshold)
    return shifts


def apply_shifts(rows: list[dict], shifts: dict[str, float]) -> np.ndarray:
    return np.asarray([float(row["score"]) + shifts.get(row["codec"], 0.0) for row in rows], dtype=np.float64)


def metrics_by_codec(rows: list[dict], scores: np.ndarray) -> list[dict]:
    labels, _, codecs = arrays(rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, codec in enumerate(codecs):
        groups[codec].append(index)

    result = []
    for codec in sorted(groups):
        indexes = np.asarray(groups[codec], dtype=np.int64)
        group_labels = labels[indexes]
        group_scores = scores[indexes]
        bonafide = int((group_labels == 0).sum())
        spoof = int((group_labels == 1).sum())
        if bonafide and spoof:
            group_metrics = metrics(group_labels, group_scores)
        else:
            group_metrics = {
                "eer": None,
                "eer_percent": None,
                "eer_threshold": None,
                "accuracy_at_0": None,
                "mean_score": float(group_scores.mean()),
            }
        result.append({"codec": codec, "n": int(indexes.size), "bonafide": bonafide, "spoof": spoof, **group_metrics})
    result.sort(key=lambda row: -1.0 if row["eer"] is None else row["eer"], reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Codec-aware score-shift calibration for existing AASIST scores.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-shifted-scores", type=Path)
    parser.add_argument("--target-codecs", nargs="*", default=[])
    parser.add_argument("--cal-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_scores(args.scores)
    calibration, test = split_calibration(rows, cal_per_class=args.cal_per_class, seed=args.seed)
    target_codecs = set(args.target_codecs)
    shifts = learn_codec_shifts(calibration, target_codecs)

    test_labels, test_scores, _ = arrays(test)
    shifted_scores = apply_shifts(test, shifts)
    result = {
        "scores": str(args.scores),
        "seed": args.seed,
        "cal_per_class": args.cal_per_class,
        "target_codecs": sorted(target_codecs),
        "n_total": len(rows),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "codec_shifts": shifts,
        "before": {
            "overall": metrics(test_labels, test_scores),
            "by_codec": metrics_by_codec(test, test_scores),
        },
        "after": {
            "overall": metrics(test_labels, shifted_scores),
            "by_codec": metrics_by_codec(test, shifted_scores),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote calibration result: {args.output_json}")

    if args.output_shifted_scores:
        args.output_shifted_scores.parent.mkdir(parents=True, exist_ok=True)
        with args.output_shifted_scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["utterance_id", "codec", "label", "score", "calibrated_score", "split"])
            calibration_ids = {row["utterance_id"] for row in calibration}
            for row in rows:
                split = "calibration" if row["utterance_id"] in calibration_ids else "test"
                calibrated = float(row["score"]) + shifts.get(row["codec"], 0.0)
                writer.writerow([row["utterance_id"], row["codec"], row["label"], row["score"], calibrated, split])
        print(f"wrote shifted scores: {args.output_shifted_scores}")


if __name__ == "__main__":
    main()
