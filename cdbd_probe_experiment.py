import argparse
import csv
import hashlib
import json
import random
import subprocess
import tempfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_manifest import load_audio, load_model, metrics, metrics_by_codec, pad, read_manifest


DEFAULT_PROBES = [
    "original",
    "mp3_64k",
    "aac_64k",
    "ogg_q4",
    "resample_8k",
    "resample_22050",
    "mulaw",
    "alaw",
    "noise_20db",
    "gain_0.9",
]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def ffmpeg_decode(input_path: Path, output_path: Path, sample_rate: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-codec:a",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output_path),
        ],
        check=True,
    )


def codec_roundtrip(audio: np.ndarray, sample_rate: int, probe: str, temp_dir: Path) -> np.ndarray:
    source = temp_dir / "source.wav"
    write_wav(source, audio, sample_rate)
    if probe == "mp3_64k":
        encoded = temp_dir / "encoded.mp3"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-codec:a", "libmp3lame", "-b:a", "64k", str(encoded)]
    elif probe == "aac_64k":
        encoded = temp_dir / "encoded.m4a"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-codec:a", "aac", "-b:a", "64k", str(encoded)]
    elif probe == "ogg_q4":
        encoded = temp_dir / "encoded.ogg"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-codec:a", "libvorbis", "-q:a", "4", str(encoded)]
    elif probe == "mulaw":
        encoded = temp_dir / "encoded.wav"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-codec:a", "pcm_mulaw", "-ac", "1", str(encoded)]
    elif probe == "alaw":
        encoded = temp_dir / "encoded.wav"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-codec:a", "pcm_alaw", "-ac", "1", str(encoded)]
    else:
        raise ValueError(f"unsupported codec probe: {probe}")

    subprocess.run(command, check=True)
    decoded = temp_dir / "decoded.wav"
    ffmpeg_decode(encoded, decoded, sample_rate)
    decoded_audio, _ = load_audio(decoded)
    return decoded_audio


def ffmpeg_filter(audio: np.ndarray, sample_rate: int, probe: str, temp_dir: Path) -> np.ndarray:
    source = temp_dir / "source.wav"
    write_wav(source, audio, sample_rate)
    output = temp_dir / "filtered.wav"
    if probe == "resample_8k":
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-ar", "8000", "-ac", "1", str(output)]
        subprocess.run(command, check=True)
        back = temp_dir / "filtered_back.wav"
        ffmpeg_decode(output, back, sample_rate)
        decoded_audio, _ = load_audio(back)
        return decoded_audio
    if probe == "resample_22050":
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-ar", "22050", "-ac", "1", str(output)]
        subprocess.run(command, check=True)
        back = temp_dir / "filtered_back.wav"
        ffmpeg_decode(output, back, sample_rate)
        decoded_audio, _ = load_audio(back)
        return decoded_audio
    if probe == "atempo_0.95":
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-filter:a", "atempo=0.95", "-ac", "1", str(output)]
        subprocess.run(command, check=True)
        decoded_audio, _ = load_audio(output)
        return decoded_audio
    raise ValueError(f"unsupported filter probe: {probe}")


def seeded_noise(audio: np.ndarray, utterance_id: str, probe: str) -> np.ndarray:
    digest = hashlib.sha256(f"{utterance_id}:{probe}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))) + 1e-8)
    target_noise_rms = signal_rms / 10.0
    return audio + noise * (target_noise_rms / noise_rms)


def apply_probe(audio: np.ndarray, sample_rate: int, probe: str, utterance_id: str) -> np.ndarray:
    if probe == "original":
        return audio
    if probe == "gain_0.9":
        return audio * 0.9
    if probe == "noise_20db":
        return seeded_noise(audio, utterance_id, probe)
    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)
        if probe in {"mp3_64k", "aac_64k", "ogg_q4", "mulaw", "alaw"}:
            return codec_roundtrip(audio, sample_rate, probe, temp_dir)
        if probe in {"resample_8k", "resample_22050", "atempo_0.95"}:
            return ffmpeg_filter(audio, sample_rate, probe, temp_dir)
    raise ValueError(f"unknown probe: {probe}")


def sample_rows(rows: list[dict], max_per_codec_class: int | None, seed: int) -> list[dict]:
    if max_per_codec_class is None:
        return rows
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["codec"], row["label"])].append(row)

    rng = random.Random(seed)
    sampled = []
    for key in sorted(grouped):
        values = list(grouped[key])
        rng.shuffle(values)
        sampled.extend(values[:max_per_codec_class])
    sampled.sort(key=lambda row: row["utterance_id"])
    return sampled


def encode_views(model, device: torch.device, cut: int, views: list[np.ndarray], batch_size: int) -> tuple[list[float], np.ndarray]:
    scores = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(views), batch_size):
            batch = [pad(view, cut) for view in views[start : start + batch_size]]
            tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)
            hidden, logits = model(tensor)
            scores.extend(logits[:, 1].detach().cpu().numpy().astype(float).tolist())
            embeddings.append(hidden.detach().cpu().numpy().astype(np.float64))
    return scores, np.concatenate(embeddings, axis=0)


def score_response_features(score_map: dict[str, float], probes: list[str]) -> list[float]:
    original = score_map["original"]
    probe_scores = np.asarray([score_map[name] for name in probes if name != "original"], dtype=np.float64)
    deltas = probe_scores - original
    return [
        original,
        float(probe_scores.mean()),
        float(probe_scores.std()),
        float(probe_scores.min()),
        float(probe_scores.max()),
        float(probe_scores.max() - probe_scores.min()),
        float(deltas.mean()),
        float(deltas.std()),
        float(np.abs(deltas).max()),
        *[float(score_map[name]) for name in probes if name != "original"],
        *[float(score_map[name] - original) for name in probes if name != "original"],
    ]


def score_feature_names(probes: list[str]) -> list[str]:
    probe_names = [name for name in probes if name != "original"]
    return [
        "score_original",
        "probe_mean",
        "probe_std",
        "probe_min",
        "probe_max",
        "probe_range",
        "delta_mean",
        "delta_std",
        "delta_abs_max",
        *[f"score_{name}" for name in probe_names],
        *[f"delta_{name}" for name in probe_names],
    ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denominator)


def embedding_response_features(embedding_map: dict[str, np.ndarray], probes: list[str]) -> list[float]:
    original = embedding_map["original"]
    probe_embeddings = np.stack([embedding_map[name] for name in probes if name != "original"])
    deltas = probe_embeddings - original
    mean_probe = probe_embeddings.mean(axis=0)
    mean_delta = deltas.mean(axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    cosines = np.asarray([cosine_similarity(original, embedding) for embedding in probe_embeddings], dtype=np.float64)
    return [
        float(np.linalg.norm(original)),
        float(np.linalg.norm(mean_probe)),
        float(np.linalg.norm(mean_delta)),
        float(delta_norms.mean()),
        float(delta_norms.std()),
        float(delta_norms.max()),
        float(cosines.mean()),
        float(cosines.std()),
        *original.astype(float).tolist(),
        *mean_delta.astype(float).tolist(),
    ]


def embedding_feature_names(embedding_dim: int) -> list[str]:
    return [
        "embedding_original_norm",
        "embedding_probe_mean_norm",
        "embedding_mean_delta_norm",
        "embedding_delta_norm_mean",
        "embedding_delta_norm_std",
        "embedding_delta_norm_max",
        "embedding_cosine_mean",
        "embedding_cosine_std",
        *[f"embedding_original_{index}" for index in range(embedding_dim)],
        *[f"embedding_mean_delta_{index}" for index in range(embedding_dim)],
    ]


def response_features(
    score_map: dict[str, float],
    embedding_map: dict[str, np.ndarray],
    probes: list[str],
    feature_level: str,
) -> list[float]:
    score_features = score_response_features(score_map, probes)
    embedding_features = embedding_response_features(embedding_map, probes)
    if feature_level == "score":
        return score_features
    if feature_level == "embedding":
        return [score_map["original"], *embedding_features]
    if feature_level == "score_embedding":
        return [*score_features, *embedding_features]
    raise ValueError(f"unsupported feature level: {feature_level}")


def feature_names(probes: list[str], embedding_dim: int, feature_level: str) -> list[str]:
    score_names = score_feature_names(probes)
    embedding_names = embedding_feature_names(embedding_dim)
    if feature_level == "score":
        return score_names
    if feature_level == "embedding":
        return ["score_original", *embedding_names]
    if feature_level == "score_embedding":
        return [*score_names, *embedding_names]
    raise ValueError(f"unsupported feature level: {feature_level}")


def split_calibration(rows: list[dict], cal_per_codec_class: int, seed: int) -> tuple[list[int], list[int]]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["codec"], row["label"])].append(index)

    rng = random.Random(seed)
    calibration = set()
    for key, indexes in grouped.items():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        take = min(cal_per_codec_class, len(shuffled) // 2)
        calibration.update(shuffled[:take])
    test = [index for index in range(len(rows)) if index not in calibration]
    return sorted(calibration), test


def fit_logistic_regression(features: np.ndarray, labels: np.ndarray, steps: int, lr: float, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = (labels == 0).astype(np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    x = (features - mean) / std
    x = np.c_[np.ones(x.shape[0]), x]
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(x @ weights, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x.T @ (probs - targets)) / x.shape[0]
        gradient[1:] += l2 * weights[1:]
        weights -= lr * gradient
    return weights, mean, std


def predict_logistic(features: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (features - mean) / std
    x = np.c_[np.ones(x.shape[0]), x]
    return x @ weights


def main() -> None:
    parser = argparse.ArgumentParser(description="CDBD-style multi-probe response experiment on top of frozen AASIST.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("local_configs/AASIST_manifest_eval.conf"))
    parser.add_argument("--weights", type=Path, default=Path("models/weights/AASIST.pth"))
    parser.add_argument("--output-probe-scores", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--probes", nargs="+", default=DEFAULT_PROBES)
    parser.add_argument("--max-per-codec-class", type=int)
    parser.add_argument("--cal-per-codec-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--feature-level", choices=["score", "embedding", "score_embedding"], default="score")
    parser.add_argument("--logreg-steps", type=int, default=2000)
    parser.add_argument("--logreg-lr", type=float, default=0.05)
    parser.add_argument("--logreg-l2", type=float, default=1e-3)
    args = parser.parse_args()

    if "original" not in args.probes:
        args.probes = ["original", *args.probes]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cut = load_model(args.config, args.weights, device)
    rows = sample_rows(read_manifest(args.manifest), args.max_per_codec_class, args.seed)
    names = None

    probe_score_rows = []
    feature_rows = []
    features = []
    labels = []
    codecs = []
    utterance_ids = []

    for index, row in enumerate(rows, start=1):
        audio, sample_rate = load_audio(row["path"])
        views = [apply_probe(audio, sample_rate, probe, row["utterance_id"]) for probe in args.probes]
        scores, embeddings = encode_views(model, device, cut, views, args.batch_size)
        score_map = dict(zip(args.probes, scores))
        embedding_map = dict(zip(args.probes, embeddings))
        if names is None:
            names = feature_names(args.probes, embeddings.shape[1], args.feature_level)
        row_features = response_features(score_map, embedding_map, args.probes, args.feature_level)

        for probe in args.probes:
            probe_score_rows.append([row["utterance_id"], row["codec"], row["label"], probe, score_map[probe]])
        feature_rows.append([row["utterance_id"], row["codec"], row["label"], *row_features])
        features.append(row_features)
        labels.append(row["label"])
        codecs.append(row["codec"])
        utterance_ids.append(row["utterance_id"])

        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
            print(f"processed {index}/{len(rows)}", flush=True)

    feature_array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if names is None:
        raise ValueError("no rows were processed")
    original_scores = feature_array[:, names.index("score_original")]
    cal_indexes, test_indexes = split_calibration(rows, args.cal_per_codec_class, args.seed)

    weights, mean, std = fit_logistic_regression(
        feature_array[cal_indexes],
        label_array[cal_indexes],
        steps=args.logreg_steps,
        lr=args.logreg_lr,
        l2=args.logreg_l2,
    )
    cdbd_scores = predict_logistic(feature_array, weights, mean, std)

    test_labels = label_array[test_indexes]
    test_original = original_scores[test_indexes]
    test_cdbd = cdbd_scores[test_indexes]
    test_codecs = [codecs[index] for index in test_indexes]
    result = {
        "manifest": str(args.manifest),
        "n_total": len(rows),
        "n_calibration": len(cal_indexes),
        "n_test": len(test_indexes),
        "seed": args.seed,
        "probes": args.probes,
        "feature_level": args.feature_level,
        "feature_names": names,
        "baseline_original_score": {
            "overall": metrics(test_labels, test_original),
            "by_codec": metrics_by_codec(test_labels, test_original, test_codecs),
        },
        "cdbd_probe_logreg": {
            "overall": metrics(test_labels, test_cdbd),
            "by_codec": metrics_by_codec(test_labels, test_cdbd, test_codecs),
        },
    }

    args.output_probe_scores.parent.mkdir(parents=True, exist_ok=True)
    with args.output_probe_scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", "codec", "label", "probe", "score"])
        writer.writerows(probe_score_rows)

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    with args.output_features.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", "codec", "label", *names])
        writer.writerows(feature_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote probe scores: {args.output_probe_scores}")
    print(f"wrote features: {args.output_features}")
    print(f"wrote result: {args.output_json}")


if __name__ == "__main__":
    main()
