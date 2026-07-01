import argparse
import csv
import json
import subprocess
import tempfile
import wave
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def pad(x: np.ndarray, max_len: int) -> np.ndarray:
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        return load_wav(path)
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "decoded.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-codec:a",
                "pcm_s16le",
                "-ac",
                "1",
                str(wav_path),
            ],
            check=True,
        )
        return load_wav(wav_path)


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "path": Path(row["path"]),
                    "label": int(row["label"]),
                    "utterance_id": row["utterance_id"],
                    "codec": row.get("codec") or "clean",
                }
            )
    return rows


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, cut: int):
        self.rows = read_manifest(manifest)
        self.cut = cut

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, _ = load_audio(row["path"])
        return torch.tensor(pad(audio, self.cut), dtype=torch.float32), row["label"], row["utterance_id"], row["codec"]


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
    # AASIST official scores use larger values for bonafide, so threshold predicts bonafide.
    predicted_bonafide = scores >= 0.0
    predicted_labels = np.where(predicted_bonafide, 0, 1)
    return {
        "eer": eer_value,
        "eer_percent": eer_value * 100.0,
        "eer_threshold": threshold,
        "accuracy_at_0": float((predicted_labels == labels).mean()),
        "mean_score": float(scores.mean()),
    }


def metrics_by_codec(labels: np.ndarray, scores: np.ndarray, codecs: list[str]) -> list[dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, codec in enumerate(codecs):
        groups[codec].append(index)

    rows = []
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
        rows.append(
            {
                "codec": codec,
                "n": int(indexes.size),
                "bonafide": bonafide,
                "spoof": spoof,
                **group_metrics,
            }
        )
    rows.sort(key=lambda row: -1.0 if row["eer"] is None else row["eer"], reverse=True)
    return rows


def load_model(config_path: Path, weights_path: Path, device: torch.device):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model_config"]
    module = import_module(f"models.{model_config['architecture']}")
    model = getattr(module, "Model")(model_config).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, int(model_config["nb_samp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AASIST on a CSV manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/AASIST.conf"))
    parser.add_argument("--weights", type=Path, default=Path("models/weights/AASIST.pth"))
    parser.add_argument("--output-scores", type=Path)
    parser.add_argument("--output-codec-metrics", type=Path)
    parser.add_argument("--group-by-codec", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cut = load_model(args.config, args.weights, device)
    dataset = ManifestDataset(args.manifest, cut=cut)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    total = len(dataset)

    labels = []
    scores = []
    codecs_seen = []
    output_rows = []
    processed = 0
    next_progress = args.progress_every
    with torch.no_grad():
        for batch_x, batch_y, utt_ids, codecs in loader:
            batch_x = batch_x.to(device)
            _, logits = model(batch_x)
            batch_scores = logits[:, 1].detach().cpu().numpy()
            labels.extend(batch_y.numpy().tolist())
            scores.extend(batch_scores.tolist())
            codecs_seen.extend(list(codecs))
            for utt_id, codec, label, score in zip(utt_ids, codecs, batch_y.numpy().tolist(), batch_scores.tolist()):
                output_rows.append([utt_id, codec, int(label), float(score)])
            processed += len(batch_y)
            if args.progress_every > 0 and (processed >= next_progress or processed == total):
                print(f"processed {processed}/{total}", flush=True)
                while next_progress <= processed:
                    next_progress += args.progress_every

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    result = metrics(label_array, score_array)
    print(json.dumps(result, indent=2))

    codec_rows = []
    if args.group_by_codec or args.output_codec_metrics:
        codec_rows = metrics_by_codec(label_array, score_array, codecs_seen)
        print(json.dumps({"by_codec": codec_rows}, indent=2))

    if args.output_scores:
        args.output_scores.parent.mkdir(parents=True, exist_ok=True)
        with args.output_scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["utterance_id", "codec", "label", "score"])
            writer.writerows(output_rows)
        print(f"wrote scores: {args.output_scores}")

    if args.output_codec_metrics:
        args.output_codec_metrics.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["codec", "n", "bonafide", "spoof", "eer", "eer_percent", "eer_threshold", "accuracy_at_0", "mean_score"]
        with args.output_codec_metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(codec_rows)
        print(f"wrote codec metrics: {args.output_codec_metrics}")


if __name__ == "__main__":
    main()
