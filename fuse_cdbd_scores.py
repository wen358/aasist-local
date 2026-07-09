import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_manifest import metrics, metrics_by_codec


def read_predictions(path: Path, score_column: str) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"utterance_id", "codec", "label", "split", score_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            utterance_id = row["utterance_id"]
            rows[utterance_id] = {
                "utterance_id": utterance_id,
                "codec": row["codec"],
                "label": int(row["label"]),
                "split": row["split"],
                "score": float(row[score_column]),
            }
    return rows


def align_predictions(first: dict[str, dict], second: dict[str, dict]) -> list[dict]:
    common = sorted(set(first) & set(second))
    if not common:
        raise ValueError("no overlapping utterance_id values")
    missing_first = len(second) - len(common)
    missing_second = len(first) - len(common)
    if missing_first or missing_second:
        print(f"warning: using {len(common)} common rows; missing_first={missing_first}, missing_second={missing_second}")

    rows = []
    for utterance_id in common:
        row_a = first[utterance_id]
        row_b = second[utterance_id]
        for field in ["codec", "label", "split"]:
            if row_a[field] != row_b[field]:
                raise ValueError(f"mismatched {field} for {utterance_id}: {row_a[field]} vs {row_b[field]}")
        rows.append(
            {
                "utterance_id": utterance_id,
                "codec": row_a["codec"],
                "label": row_a["label"],
                "split": row_a["split"],
                "score_a": row_a["score"],
                "score_b": row_b["score"],
            }
        )
    return rows


def zscore(scores: np.ndarray, calibration_mask: np.ndarray) -> np.ndarray:
    calibration = scores[calibration_mask]
    return (scores - calibration.mean()) / (calibration.std() + 1e-6)


def fuse_scores(score_a: np.ndarray, score_b: np.ndarray, alpha: float) -> np.ndarray:
    return (alpha * score_b) + ((1.0 - alpha) * score_a)


def alpha_grid(step: float) -> list[float]:
    count = int(round(1.0 / step))
    return [round(index * step, 10) for index in range(count + 1)]


def select_global_alpha(labels: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, calibration_mask: np.ndarray, alphas: list[float]) -> tuple[float, dict]:
    best_alpha = alphas[0]
    best_metrics = None
    best_eer = float("inf")
    for alpha in alphas:
        fused = fuse_scores(score_a, score_b, alpha)
        current = metrics(labels[calibration_mask], fused[calibration_mask])
        if current["eer"] < best_eer:
            best_alpha = alpha
            best_metrics = current
            best_eer = current["eer"]
    return best_alpha, best_metrics or {}


def select_codec_alphas(
    labels: np.ndarray,
    codecs: list[str],
    score_a: np.ndarray,
    score_b: np.ndarray,
    calibration_mask: np.ndarray,
    alphas: list[float],
) -> tuple[dict[str, float], dict[str, dict]]:
    codec_alphas = {}
    codec_calibration_metrics = {}
    for codec in sorted(set(codecs)):
        codec_mask = np.asarray([item == codec for item in codecs], dtype=bool)
        mask = calibration_mask & codec_mask
        if int(mask.sum()) == 0 or len(set(labels[mask].tolist())) < 2:
            codec_alphas[codec] = 0.5
            codec_calibration_metrics[codec] = {}
            continue
        best_alpha, best_metrics = select_global_alpha(labels, score_a, score_b, mask, alphas)
        codec_alphas[codec] = best_alpha
        codec_calibration_metrics[codec] = best_metrics
    return codec_alphas, codec_calibration_metrics


def apply_codec_alphas(score_a: np.ndarray, score_b: np.ndarray, codecs: list[str], codec_alphas: dict[str, float]) -> np.ndarray:
    fused = np.zeros_like(score_a, dtype=np.float64)
    for index, codec in enumerate(codecs):
        fused[index] = fuse_scores(score_a[index], score_b[index], codec_alphas[codec])
    return fused


def write_scores(path: Path, rows: list[dict], score_a_z: np.ndarray, score_b_z: np.ndarray, global_fused: np.ndarray, codec_fused: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", "codec", "label", "split", "score_a_z", "score_b_z", "global_fused", "codec_fused"])
        for index, row in enumerate(rows):
            writer.writerow(
                [
                    row["utterance_id"],
                    row["codec"],
                    row["label"],
                    row["split"],
                    float(score_a_z[index]),
                    float(score_b_z[index]),
                    float(global_fused[index]),
                    float(codec_fused[index]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse two CDBD prediction CSV files on the shared calibration/test split.")
    parser.add_argument("--score-a", type=Path, required=True)
    parser.add_argument("--score-b", type=Path, required=True)
    parser.add_argument("--name-a", default="aasist")
    parser.add_argument("--name-b", default="ssl")
    parser.add_argument("--score-column", default="cdbd_score")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-scores", type=Path, required=True)
    args = parser.parse_args()

    rows = align_predictions(
        read_predictions(args.score_a, args.score_column),
        read_predictions(args.score_b, args.score_column),
    )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    codecs = [row["codec"] for row in rows]
    splits = [row["split"] for row in rows]
    score_a = np.asarray([row["score_a"] for row in rows], dtype=np.float64)
    score_b = np.asarray([row["score_b"] for row in rows], dtype=np.float64)
    calibration_mask = np.asarray([split == "calibration" for split in splits], dtype=bool)
    test_mask = np.asarray([split == "test" for split in splits], dtype=bool)

    score_a_z = zscore(score_a, calibration_mask)
    score_b_z = zscore(score_b, calibration_mask)
    alphas = alpha_grid(args.alpha_step)
    global_alpha, global_calibration = select_global_alpha(labels, score_a_z, score_b_z, calibration_mask, alphas)
    global_fused = fuse_scores(score_a_z, score_b_z, global_alpha)
    codec_alphas, codec_calibration = select_codec_alphas(labels, codecs, score_a_z, score_b_z, calibration_mask, alphas)
    codec_fused = apply_codec_alphas(score_a_z, score_b_z, codecs, codec_alphas)
    test_codecs = [codec for codec, is_test in zip(codecs, test_mask) if is_test]

    result = {
        "score_a": str(args.score_a),
        "score_b": str(args.score_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "score_column": args.score_column,
        "n_total": int(labels.size),
        "n_calibration": int(calibration_mask.sum()),
        "n_test": int(test_mask.sum()),
        "alpha_grid": alphas,
        "single_model": {
            args.name_a: {
                "overall": metrics(labels[test_mask], score_a_z[test_mask]),
                "by_codec": metrics_by_codec(labels[test_mask], score_a_z[test_mask], test_codecs),
            },
            args.name_b: {
                "overall": metrics(labels[test_mask], score_b_z[test_mask]),
                "by_codec": metrics_by_codec(labels[test_mask], score_b_z[test_mask], test_codecs),
            },
        },
        "global_fusion": {
            "formula": f"alpha * {args.name_b} + (1 - alpha) * {args.name_a}",
            "selected_alpha": global_alpha,
            "calibration": global_calibration,
            "overall": metrics(labels[test_mask], global_fused[test_mask]),
            "by_codec": metrics_by_codec(labels[test_mask], global_fused[test_mask], test_codecs),
        },
        "codec_aware_fusion": {
            "formula": f"codec_alpha * {args.name_b} + (1 - codec_alpha) * {args.name_a}",
            "codec_alphas": codec_alphas,
            "calibration_by_codec": codec_calibration,
            "overall": metrics(labels[test_mask], codec_fused[test_mask]),
            "by_codec": metrics_by_codec(labels[test_mask], codec_fused[test_mask], test_codecs),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_scores(args.output_scores, rows, score_a_z, score_b_z, global_fused, codec_fused)
    print(json.dumps(result, indent=2))
    print(f"wrote result: {args.output_json}")
    print(f"wrote scores: {args.output_scores}")


if __name__ == "__main__":
    main()
