import argparse
import csv
import hashlib
import json
import random
import subprocess
import tempfile
import time
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

LIGHT_QUANTIZE_PROBES = ["original", "noise_20db", "gain_0.9", "mulaw", "alaw"]
CODEC_TARGET_PROBES = ["original", "noise_20db", "gain_0.9", "mp3_64k", "aac_64k", "ogg_q4"]
MIXED_TARGET_PROBES = ["original", "noise_20db", "gain_0.9", "mulaw", "alaw", "mp3_64k", "ogg_q4", "resample_8k"]
CODEC_ADAPTIVE_PROBES = {
    "oggm4a": CODEC_TARGET_PROBES,
    "low_mp3": MIXED_TARGET_PROBES,
}


def probes_for_codec(codec: str, default_probes: list[str], probe_policy: str) -> list[str]:
    if probe_policy == "fixed":
        return default_probes
    if probe_policy == "codec_adaptive":
        return CODEC_ADAPTIVE_PROBES.get(codec, LIGHT_QUANTIZE_PROBES)
    raise ValueError(f"unsupported probe policy: {probe_policy}")


def run_with_retry(command: list[str], attempts: int = 3) -> None:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(0.5 * attempt)
    raise last_error


def load_audio_with_retry(path: Path, attempts: int = 3) -> tuple[np.ndarray, int]:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return load_audio(path)
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(0.5 * attempt)
    raise last_error


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def ffmpeg_decode(input_path: Path, output_path: Path, sample_rate: int) -> None:
    run_with_retry(
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
        ]
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

    run_with_retry(command)
    decoded = temp_dir / "decoded.wav"
    ffmpeg_decode(encoded, decoded, sample_rate)
    decoded_audio, _ = load_audio_with_retry(decoded)
    return decoded_audio


def ffmpeg_filter(audio: np.ndarray, sample_rate: int, probe: str, temp_dir: Path) -> np.ndarray:
    source = temp_dir / "source.wav"
    write_wav(source, audio, sample_rate)
    output = temp_dir / "filtered.wav"
    if probe == "resample_8k":
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-ar", "8000", "-ac", "1", str(output)]
        run_with_retry(command)
        back = temp_dir / "filtered_back.wav"
        ffmpeg_decode(output, back, sample_rate)
        decoded_audio, _ = load_audio_with_retry(back)
        return decoded_audio
    if probe == "resample_22050":
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-ar", "22050", "-ac", "1", str(output)]
        run_with_retry(command)
        back = temp_dir / "filtered_back.wav"
        ffmpeg_decode(output, back, sample_rate)
        decoded_audio, _ = load_audio_with_retry(back)
        return decoded_audio
    if probe == "atempo_0.95":
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-filter:a", "atempo=0.95", "-ac", "1", str(output)]
        run_with_retry(command)
        decoded_audio, _ = load_audio_with_retry(output)
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


def codec_onehot(codec: str, codec_names: list[str]) -> list[float]:
    return [1.0 if codec == name else 0.0 for name in codec_names]


def codec_feature_names(codec_names: list[str]) -> list[str]:
    return [f"codec_{name}" for name in codec_names]


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


def zscore_by_calibration(scores: np.ndarray, calibration_indexes: list[int]) -> np.ndarray:
    calibration_scores = scores[calibration_indexes]
    return (scores - calibration_scores.mean()) / (calibration_scores.std() + 1e-6)


def fit_score_fusion(
    labels: np.ndarray,
    original_scores: np.ndarray,
    cdbd_scores: np.ndarray,
    calibration_indexes: list[int],
    alpha_values: list[float],
) -> tuple[float, np.ndarray, dict]:
    original_z = zscore_by_calibration(original_scores, calibration_indexes)
    cdbd_z = zscore_by_calibration(cdbd_scores, calibration_indexes)
    calibration_labels = labels[calibration_indexes]
    best_alpha = alpha_values[0]
    best_metrics = None
    best_eer = float("inf")
    for alpha in alpha_values:
        fused = (alpha * cdbd_z) + ((1.0 - alpha) * original_z)
        current_metrics = metrics(calibration_labels, fused[calibration_indexes])
        current_eer = float(current_metrics["eer"])
        if current_eer < best_eer:
            best_eer = current_eer
            best_alpha = alpha
            best_metrics = current_metrics
    fused_scores = (best_alpha * cdbd_z) + ((1.0 - best_alpha) * original_z)
    return best_alpha, fused_scores, best_metrics


class MLPBackend(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class ProbeAttentionBackend(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((embedding_dim * 2) + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, original: torch.Tensor, deltas: torch.Tensor, original_score: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(deltas).squeeze(-1), dim=1)
        pooled_delta = torch.sum(deltas * weights.unsqueeze(-1), dim=1)
        features = torch.cat([original, pooled_delta, original_score.unsqueeze(1)], dim=1)
        return self.classifier(features).squeeze(1)


class HybridAttentionMLPBackend(torch.nn.Module):
    def __init__(self, base_dim: int, embedding_dim: int, num_probes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(base_dim + embedding_dim + num_probes + 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, base_features: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(deltas).squeeze(-1), dim=1)
        pooled_delta = torch.sum(deltas * weights.unsqueeze(-1), dim=1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1, keepdim=True)
        weight_max = weights.max(dim=1, keepdim=True).values
        weight_std = weights.std(dim=1, keepdim=True, unbiased=False)
        features = torch.cat([base_features, pooled_delta, weights, entropy, weight_max, weight_std], dim=1)
        return self.net(features).squeeze(1)


def fit_predict_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    all_features: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    targets = (train_labels == 0).astype(np.float32)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-6
    train_x = ((train_features - mean) / std).astype(np.float32)
    all_x = ((all_features - mean) / std).astype(np.float32)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = MLPBackend(train_x.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(targets))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, all_x.shape[0], batch_size):
            batch_x = torch.from_numpy(all_x[start : start + batch_size]).to(device)
            scores.append(model(batch_x).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float64)


def fit_predict_hybrid_attention_mlp(
    train_base_features: np.ndarray,
    train_deltas: np.ndarray,
    train_labels: np.ndarray,
    all_base_features: np.ndarray,
    all_deltas: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    targets = (train_labels == 0).astype(np.float32)
    base_mean = train_base_features.mean(axis=0)
    base_std = train_base_features.std(axis=0) + 1e-6
    delta_mean = train_deltas.reshape(-1, train_deltas.shape[-1]).mean(axis=0)
    delta_std = train_deltas.reshape(-1, train_deltas.shape[-1]).std(axis=0) + 1e-6

    train_base_x = ((train_base_features - base_mean) / base_std).astype(np.float32)
    train_delta_x = ((train_deltas - delta_mean) / delta_std).astype(np.float32)
    all_base_x = ((all_base_features - base_mean) / base_std).astype(np.float32)
    all_delta_x = ((all_deltas - delta_mean) / delta_std).astype(np.float32)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = HybridAttentionMLPBackend(
        train_base_x.shape[1],
        train_delta_x.shape[-1],
        train_delta_x.shape[1],
        hidden_dim,
        dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_base_x),
        torch.from_numpy(train_delta_x),
        torch.from_numpy(targets),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model.train()
    for _ in range(epochs):
        for batch_base, batch_delta, batch_y in loader:
            batch_base = batch_base.to(device)
            batch_delta = batch_delta.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_base, batch_delta), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, all_base_x.shape[0], batch_size):
            batch_base = torch.from_numpy(all_base_x[start : start + batch_size]).to(device)
            batch_delta = torch.from_numpy(all_delta_x[start : start + batch_size]).to(device)
            scores.append(model(batch_base, batch_delta).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float64)


def fit_predict_probe_attention(
    train_originals: np.ndarray,
    train_deltas: np.ndarray,
    train_original_scores: np.ndarray,
    train_labels: np.ndarray,
    all_originals: np.ndarray,
    all_deltas: np.ndarray,
    all_original_scores: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    targets = (train_labels == 0).astype(np.float32)
    original_mean = train_originals.mean(axis=0)
    original_std = train_originals.std(axis=0) + 1e-6
    delta_mean = train_deltas.reshape(-1, train_deltas.shape[-1]).mean(axis=0)
    delta_std = train_deltas.reshape(-1, train_deltas.shape[-1]).std(axis=0) + 1e-6
    score_mean = train_original_scores.mean()
    score_std = train_original_scores.std() + 1e-6

    train_original_x = ((train_originals - original_mean) / original_std).astype(np.float32)
    train_delta_x = ((train_deltas - delta_mean) / delta_std).astype(np.float32)
    train_score_x = ((train_original_scores - score_mean) / score_std).astype(np.float32)
    all_original_x = ((all_originals - original_mean) / original_std).astype(np.float32)
    all_delta_x = ((all_deltas - delta_mean) / delta_std).astype(np.float32)
    all_score_x = ((all_original_scores - score_mean) / score_std).astype(np.float32)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = ProbeAttentionBackend(train_original_x.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_original_x),
        torch.from_numpy(train_delta_x),
        torch.from_numpy(train_score_x),
        torch.from_numpy(targets),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model.train()
    for _ in range(epochs):
        for batch_original, batch_delta, batch_score, batch_y in loader:
            batch_original = batch_original.to(device)
            batch_delta = batch_delta.to(device)
            batch_score = batch_score.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_original, batch_delta, batch_score), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, all_original_x.shape[0], batch_size):
            batch_original = torch.from_numpy(all_original_x[start : start + batch_size]).to(device)
            batch_delta = torch.from_numpy(all_delta_x[start : start + batch_size]).to(device)
            batch_score = torch.from_numpy(all_score_x[start : start + batch_size]).to(device)
            scores.append(model(batch_original, batch_delta, batch_score).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="CDBD-style multi-probe response experiment on top of frozen AASIST.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("local_configs/AASIST_manifest_eval.conf"))
    parser.add_argument("--weights", type=Path, default=Path("models/weights/AASIST.pth"))
    parser.add_argument("--output-probe-scores", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--probes", nargs="+", default=DEFAULT_PROBES)
    parser.add_argument("--probe-policy", choices=["fixed", "codec_adaptive"], default="fixed")
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
    parser.add_argument("--classifier", choices=["logreg", "mlp", "probe_attention", "hybrid_attention_mlp"], default="logreg")
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--use-codec-onehot", action="store_true")
    args = parser.parse_args()

    if "original" not in args.probes:
        args.probes = ["original", *args.probes]
    if args.probe_policy == "codec_adaptive" and args.feature_level != "embedding":
        raise ValueError("--probe-policy codec_adaptive currently requires --feature-level embedding")
    if args.probe_policy == "codec_adaptive" and args.classifier in {"probe_attention", "hybrid_attention_mlp"}:
        raise ValueError("--probe-policy codec_adaptive currently supports logreg/mlp classifiers only")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cut = load_model(args.config, args.weights, device)
    rows = sample_rows(read_manifest(args.manifest), args.max_per_codec_class, args.seed)
    codec_names = sorted({row["codec"] for row in rows})
    names = None

    probe_score_rows = []
    feature_rows = []
    features = []
    attention_originals = []
    attention_deltas = []
    attention_original_scores = []
    labels = []
    codecs = []
    utterance_ids = []
    selected_probe_counts: dict[str, int] = defaultdict(int)

    for index, row in enumerate(rows, start=1):
        audio, sample_rate = load_audio_with_retry(row["path"])
        row_probes = probes_for_codec(row["codec"], args.probes, args.probe_policy)
        selected_probe_counts[" ".join(row_probes)] += 1
        views = [apply_probe(audio, sample_rate, probe, row["utterance_id"]) for probe in row_probes]
        scores, embeddings = encode_views(model, device, cut, views, args.batch_size)
        score_map = dict(zip(row_probes, scores))
        embedding_map = dict(zip(row_probes, embeddings))
        if names is None:
            names = feature_names(row_probes, embeddings.shape[1], args.feature_level)
            if args.use_codec_onehot:
                names = [*names, *codec_feature_names(codec_names)]
        row_features = response_features(score_map, embedding_map, row_probes, args.feature_level)
        if args.use_codec_onehot:
            row_features = [*row_features, *codec_onehot(row["codec"], codec_names)]
        original_embedding = embedding_map["original"].astype(np.float64)
        probe_embeddings = np.stack([embedding_map[name] for name in row_probes if name != "original"]).astype(np.float64)
        attention_originals.append(original_embedding)
        attention_deltas.append(probe_embeddings - original_embedding)
        attention_original_scores.append(float(score_map["original"]))

        for probe in row_probes:
            probe_score_rows.append([row["utterance_id"], row["codec"], row["label"], probe, score_map[probe]])
        feature_rows.append([row["utterance_id"], row["codec"], row["label"], *row_features])
        features.append(row_features)
        labels.append(row["label"])
        codecs.append(row["codec"])
        utterance_ids.append(row["utterance_id"])

        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
            print(f"processed {index}/{len(rows)}", flush=True)

    feature_array = np.asarray(features, dtype=np.float64)
    if args.classifier in {"probe_attention", "hybrid_attention_mlp"}:
        attention_original_array = np.asarray(attention_originals, dtype=np.float64)
        attention_delta_array = np.asarray(attention_deltas, dtype=np.float64)
        attention_original_score_array = np.asarray(attention_original_scores, dtype=np.float64)
    else:
        attention_original_array = None
        attention_delta_array = None
        attention_original_score_array = None
    label_array = np.asarray(labels, dtype=np.int64)
    if names is None:
        raise ValueError("no rows were processed")
    original_scores = feature_array[:, names.index("score_original")]
    cal_indexes, test_indexes = split_calibration(rows, args.cal_per_codec_class, args.seed)

    if args.classifier == "logreg":
        weights, mean, std = fit_logistic_regression(
            feature_array[cal_indexes],
            label_array[cal_indexes],
            steps=args.logreg_steps,
            lr=args.logreg_lr,
            l2=args.logreg_l2,
        )
        cdbd_scores = predict_logistic(feature_array, weights, mean, std)
    elif args.classifier == "mlp":
        cdbd_scores = fit_predict_mlp(
            feature_array[cal_indexes],
            label_array[cal_indexes],
            feature_array,
            seed=args.seed,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            dropout=args.mlp_dropout,
            batch_size=args.mlp_batch_size,
            device=device,
        )
    elif args.classifier == "probe_attention":
        cdbd_scores = fit_predict_probe_attention(
            attention_original_array[cal_indexes],
            attention_delta_array[cal_indexes],
            attention_original_score_array[cal_indexes],
            label_array[cal_indexes],
            attention_original_array,
            attention_delta_array,
            attention_original_score_array,
            seed=args.seed,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            dropout=args.mlp_dropout,
            batch_size=args.mlp_batch_size,
            device=device,
        )
    else:
        cdbd_scores = fit_predict_hybrid_attention_mlp(
            feature_array[cal_indexes],
            attention_delta_array[cal_indexes],
            label_array[cal_indexes],
            feature_array,
            attention_delta_array,
            seed=args.seed,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            dropout=args.mlp_dropout,
            batch_size=args.mlp_batch_size,
            device=device,
        )

    test_labels = label_array[test_indexes]
    test_original = original_scores[test_indexes]
    test_cdbd = cdbd_scores[test_indexes]
    test_codecs = [codecs[index] for index in test_indexes]
    fusion_alpha_values = [round(index / 20.0, 2) for index in range(21)]
    fusion_alpha, fusion_scores, fusion_calibration_metrics = fit_score_fusion(
        label_array,
        original_scores,
        cdbd_scores,
        cal_indexes,
        fusion_alpha_values,
    )
    test_fusion = fusion_scores[test_indexes]
    result = {
        "manifest": str(args.manifest),
        "n_total": len(rows),
        "n_calibration": len(cal_indexes),
        "n_test": len(test_indexes),
        "seed": args.seed,
        "probes": args.probes,
        "probe_policy": args.probe_policy,
        "codec_adaptive_probes": CODEC_ADAPTIVE_PROBES if args.probe_policy == "codec_adaptive" else {},
        "selected_probe_counts": dict(sorted(selected_probe_counts.items())),
        "feature_level": args.feature_level,
        "classifier": args.classifier,
        "use_codec_onehot": args.use_codec_onehot,
        "codec_onehot_names": codec_names if args.use_codec_onehot else [],
        "feature_names": names,
        "baseline_original_score": {
            "overall": metrics(test_labels, test_original),
            "by_codec": metrics_by_codec(test_labels, test_original, test_codecs),
        },
        f"cdbd_probe_{args.classifier}": {
            "overall": metrics(test_labels, test_cdbd),
            "by_codec": metrics_by_codec(test_labels, test_cdbd, test_codecs),
        },
        "score_fusion": {
            "formula": "alpha * zscore(cdbd_score) + (1 - alpha) * zscore(original_score)",
            "alpha_grid": fusion_alpha_values,
            "selected_alpha": fusion_alpha,
            "calibration": fusion_calibration_metrics,
            "overall": metrics(test_labels, test_fusion),
            "by_codec": metrics_by_codec(test_labels, test_fusion, test_codecs),
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
