import argparse
import csv
import glob
import json
import statistics
from pathlib import Path


def metric(result: dict, section: str, name: str) -> float:
    return float(result[section]["overall"][name])


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if values else float("nan")


def summarize_group(group: str, paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        baseline_eer = metric(result, "baseline_original_score", "eer_percent")
        cdbd_eer = metric(result, "cdbd_probe_logreg", "eer_percent")
        baseline_acc = metric(result, "baseline_original_score", "accuracy_at_0")
        cdbd_acc = metric(result, "cdbd_probe_logreg", "accuracy_at_0")
        rows.append(
            {
                "path": str(path),
                "baseline_eer": baseline_eer,
                "cdbd_eer": cdbd_eer,
                "delta_eer": cdbd_eer - baseline_eer,
                "baseline_acc": baseline_acc,
                "cdbd_acc": cdbd_acc,
                "delta_acc": cdbd_acc - baseline_acc,
            }
        )

    return {
        "group": group,
        "n_seeds": len(rows),
        "baseline_eer_mean": mean([row["baseline_eer"] for row in rows]),
        "baseline_eer_std": pstdev([row["baseline_eer"] for row in rows]),
        "cdbd_eer_mean": mean([row["cdbd_eer"] for row in rows]),
        "cdbd_eer_std": pstdev([row["cdbd_eer"] for row in rows]),
        "delta_eer_mean": mean([row["delta_eer"] for row in rows]),
        "delta_eer_std": pstdev([row["delta_eer"] for row in rows]),
        "baseline_acc_mean": mean([row["baseline_acc"] for row in rows]),
        "cdbd_acc_mean": mean([row["cdbd_acc"] for row in rows]),
        "delta_acc_mean": mean([row["delta_acc"] for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CDBD probe experiment JSON results.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("groups", nargs="+", help="Format: group_name=glob_pattern")
    args = parser.parse_args()

    summaries = []
    for item in args.groups:
        if "=" not in item:
            raise ValueError(f"group must use group_name=glob_pattern format: {item}")
        group, pattern = item.split("=", 1)
        paths = [Path(value) for value in sorted(glob.glob(pattern))]
        if not paths:
            print(f"warning: no files matched {group}: {pattern}")
            continue
        summaries.append(summarize_group(group, paths))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "n_seeds",
        "baseline_eer_mean",
        "baseline_eer_std",
        "cdbd_eer_mean",
        "cdbd_eer_std",
        "delta_eer_mean",
        "delta_eer_std",
        "baseline_acc_mean",
        "cdbd_acc_mean",
        "delta_acc_mean",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps(summaries, indent=2))
    print(f"wrote summary: {args.output}")


if __name__ == "__main__":
    main()
