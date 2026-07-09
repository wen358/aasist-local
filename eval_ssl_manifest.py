import argparse
import csv
import json
import subprocess
import tempfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 64000


def pad_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    if x.shape[0] >= max_len:
        return x[:max_len]
    repeats = int(max_len / max(1, x.shape[0])) + 1
    return np.tile(x, repeats)[:max_len]


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


def decode_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        audio, sr = load_wav(path)
        if sr == sample_rate:
            return audio, sr
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
                "-ar",
                str(sample_rate),
                str(wav_path),
            ],
            check=True,
        )
        return load_wav(wav_path)


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {key.strip(): value for key, value in row.items() if key is not None}
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
    def __init__(self, manifest: Path, target_samples: int):
        self.rows = read_manifest(manifest)
        self.target_samples = target_samples

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, _ = decode_audio(row["path"])
        audio = pad_or_trim(audio, self.target_samples)
        return torch.tensor(audio, dtype=torch.float32), row["label"], row["utterance_id"], row["codec"]


class Wav2Vec2CM(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'transformers'. Install it with: "
                "pip install transformers"
            ) from exc

        config = Wav2Vec2Config()
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.encoder = Wav2Vec2Model(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.register_buffer("class_weights", torch.ones(2))

    def forward(self, batch):
        frames = batch["frames"] if isinstance(batch, dict) else batch
        outputs = self.encoder(frames, attention_mask=None)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        return {"logits": logits, "embedding": pooled}


def load_checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, checkpoint
    raise TypeError(f"unsupported checkpoint type: {type(checkpoint)}")


def inspect_checkpoint(path: Path) -> None:
    state, checkpoint = load_checkpoint_state(path)
    print("checkpoint_type:", type(checkpoint).__name__)
    if isinstance(checkpoint, dict):
        print("top_level_keys:", list(checkpoint.keys()))
    print("state_dict_len:", len(state))
    for key, value in list(state.items())[:40]:
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        print(f"{key}: {shape}")
    print("...")
    for key, value in list(state.items())[-20:]:
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        print(f"{key}: {shape}")


def load_model(weights: Path, device: torch.device, hidden_dim: int, dropout: float):
    model = Wav2Vec2CM(hidden_dim=hidden_dim, dropout=dropout)
    state, _ = load_checkpoint_state(weights)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


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


def score_logits(logits: torch.Tensor, mode: str) -> np.ndarray:
    if mode == "bonafide_minus_spoof":
        scores = logits[:, 0] - logits[:, 1]
    elif mode == "bonafide_logit":
        scores = logits[:, 0]
    elif mode == "negative_spoof_logit":
        scores = -logits[:, 1]
    else:
        raise ValueError(f"unknown score mode: {mode}")
    return scores.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSL wav2vec2 checkpoint on a CSV manifest.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-scores", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-codec-metrics", type=Path)
    parser.add_argument("--group-by-codec", action="store_true")
    parser.add_argument("--inspect-checkpoint", action="store_true")
    parser.add_argument("--score-mode", choices=["bonafide_minus_spoof", "bonafide_logit", "negative_spoof_logit"], default="bonafide_minus_spoof")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-samples", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.inspect_checkpoint:
        inspect_checkpoint(args.weights)
        if args.manifest is None:
            return

    if args.manifest is None:
        raise SystemExit("--manifest is required unless --inspect-checkpoint is used alone")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device=device, hidden_dim=args.hidden_dim, dropout=args.dropout)
    dataset = ManifestDataset(args.manifest, target_samples=args.target_samples)
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
            outputs = model({"frames": batch_x})
            batch_scores = score_logits(outputs["logits"], args.score_mode)
            batch_labels = batch_y.numpy().tolist()
            labels.extend(batch_labels)
            scores.extend(batch_scores.tolist())
            codecs_seen.extend(list(codecs))
            for utt_id, codec, label, score in zip(utt_ids, codecs, batch_labels, batch_scores.tolist()):
                output_rows.append([utt_id, codec, int(label), float(score)])
            processed += len(batch_y)
            if args.progress_every > 0 and (processed >= next_progress or processed == total):
                print(f"processed {processed}/{total}", flush=True)
                while next_progress <= processed:
                    next_progress += args.progress_every

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    result = metrics(label_array, score_array)
    result_payload = {
        "manifest": str(args.manifest),
        "weights": str(args.weights),
        "score_mode": args.score_mode,
        "n_total": int(label_array.size),
        "bonafide": int((label_array == 0).sum()),
        "spoof": int((label_array == 1).sum()),
        "overall": result,
    }
    print(json.dumps(result, indent=2))

    codec_rows = []
    if args.group_by_codec or args.output_codec_metrics:
        codec_rows = metrics_by_codec(label_array, score_array, codecs_seen)
        result_payload["by_codec"] = codec_rows
        print(json.dumps({"by_codec": codec_rows}, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        print(f"wrote result: {args.output_json}")

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
