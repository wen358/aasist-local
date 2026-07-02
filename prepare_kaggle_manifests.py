import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


LOCAL_2021_DF_ROOT = "D:/audio_antispoofing/data/ASVspoof 2021_DF"
KAGGLE_2021_DF_ROOT = "/kaggle/input/datasets/pankajsomkuwar/asvspoof-2021-df"


def kaggle_path(local_path: str) -> str:
    normalized = local_path.replace("\\", "/")
    if not normalized.startswith(LOCAL_2021_DF_ROOT):
        raise ValueError(f"unexpected 2021 DF path: {local_path}")
    return normalized.replace(LOCAL_2021_DF_ROOT, KAGGLE_2021_DF_ROOT, 1)


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "utterance_id", "codec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {path}")


def convert_paths(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        converted.append(
            {
                "path": kaggle_path(row["path"]),
                "label": row["label"],
                "utterance_id": row["utterance_id"],
                "codec": row["codec"],
            }
        )
    return converted


def balanced_by_label(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["label"])].append(row)

    rng = random.Random(seed)
    sampled = []
    for label in [0, 1]:
        values = list(groups[label])
        rng.shuffle(values)
        if len(values) < per_class:
            raise ValueError(f"label {label} has only {len(values)} rows, need {per_class}")
        sampled.extend(values[:per_class])
    return sampled


def balanced_by_codec(rows: list[dict], per_codec_per_class: int, seed: int) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    codecs = sorted({row["codec"] for row in rows})
    for row in rows:
        groups[(row["codec"], int(row["label"]))].append(row)

    rng = random.Random(seed)
    sampled = []
    for codec in codecs:
        for label in [0, 1]:
            values = list(groups[(codec, label)])
            rng.shuffle(values)
            take = min(per_codec_per_class, len(values))
            sampled.extend(values[:take])
            print(f"{codec} label={label} available={len(values)} take={take}")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Kaggle-path ASVspoof 2021 DF manifests from a local manifest.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-per-class", type=int, default=50)
    parser.add_argument("--codec-per-class", type=int, default=2513)
    parser.add_argument("--write-full", action="store_true")
    args = parser.parse_args()

    rows = convert_paths(read_manifest(args.source))
    if args.write_full:
        write_manifest(args.output_dir / "kaggle_2021_df_eval_full.csv", rows)
    write_manifest(
        args.output_dir / f"kaggle_2021_df_direct_{args.smoke_per_class}_each.csv",
        balanced_by_label(rows, per_class=args.smoke_per_class, seed=args.seed),
    )
    write_manifest(
        args.output_dir / "kaggle_2021_df_codec_balanced_full.csv",
        balanced_by_codec(rows, per_codec_per_class=args.codec_per_class, seed=args.seed),
    )


if __name__ == "__main__":
    main()
