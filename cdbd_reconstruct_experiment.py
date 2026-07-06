import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cdbd_probe_experiment import apply_probe, load_audio_with_retry, sample_rows
from eval_manifest import load_model, metrics, metrics_by_codec, pad, read_manifest


DEFAULT_PROBES = ["original", "noise_20db", "gain_0.9", "mulaw", "alaw"]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def softmax_entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def encode_views(model, device: torch.device, cut: int, views: list[np.ndarray], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(views), batch_size):
            batch = [pad(view, cut) for view in views[start : start + batch_size]]
            tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)
            hidden, output = model(tensor)
            logits.append(output.detach().cpu().numpy().astype(np.float64))
            embeddings.append(hidden.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(logits, axis=0), np.concatenate(embeddings, axis=0)


def classifier_scores_from_embeddings(model, embeddings: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            batch = torch.tensor(embeddings[start : start + batch_size], dtype=torch.float32).to(device)
            logits = model.out_layer(batch)
            scores.append(logits[:, 1].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(scores, axis=0)


def unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def db_score_and_reconstruct(
    original: np.ndarray,
    selected: np.ndarray,
    tau: float,
    lambda_value: float,
) -> tuple[float, np.ndarray]:
    deltas = selected - original
    mean_delta = deltas.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta))
    if mean_norm < 1e-8:
        return 0.0, selected.mean(axis=0)

    direction = mean_delta / mean_norm
    displacement_dirs = unit_vectors(deltas)
    db_score = float((displacement_dirs @ direction).mean())
    if db_score > tau:
        return db_score, original + lambda_value * mean_norm * direction
    return db_score, selected.mean(axis=0)


def result_block(labels: np.ndarray, scores: np.ndarray, codecs: list[str]) -> dict:
    return {
        "overall": metrics(labels, scores),
        "by_codec": metrics_by_codec(labels, scores, codecs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-free CDBD-style test-time reconstruction for frozen AASIST.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("local_configs/AASIST_manifest_eval.conf"))
    parser.add_argument("--weights", type=Path, default=Path("models/weights/AASIST.pth"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-scores", type=Path, required=True)
    parser.add_argument("--probes", nargs="+", default=DEFAULT_PROBES)
    parser.add_argument("--max-per-codec-class", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--topk-values", default="1,2,3,4")
    parser.add_argument("--tau-values", default="0.3,0.5,0.7")
    parser.add_argument("--lambda-values", default="0.5,1.0,1.5,2.0")
    args = parser.parse_args()

    if "original" not in args.probes:
        args.probes = ["original", *args.probes]
    probe_names = [probe for probe in args.probes if probe != "original"]
    if not probe_names:
        raise ValueError("at least one non-original probe is required")

    topk_values = parse_int_list(args.topk_values)
    tau_values = parse_float_list(args.tau_values)
    lambda_values = parse_float_list(args.lambda_values)
    max_topk = len(probe_names)
    for topk in topk_values:
        if topk < 1 or topk > max_topk:
            raise ValueError(f"top-k must be between 1 and {max_topk}: {topk}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cut = load_model(args.config, args.weights, device)
    rows = sample_rows(read_manifest(args.manifest), args.max_per_codec_class, args.seed)

    labels = []
    codecs = []
    utterance_ids = []
    original_scores = []
    mean_all_embeddings = []
    topk_mean_embeddings: dict[int, list[np.ndarray]] = {topk: [] for topk in topk_values}
    dbd_embeddings: dict[str, list[np.ndarray]] = {}
    dbd_scores_meta: dict[str, list[float]] = {}
    for topk in topk_values:
        for tau in tau_values:
            for lambda_value in lambda_values:
                key = f"dbd_k{topk}_tau{tau:g}_lambda{lambda_value:g}"
                dbd_embeddings[key] = []
                dbd_scores_meta[key] = []

    for index, row in enumerate(rows, start=1):
        audio, sample_rate = load_audio_with_retry(row["path"])
        views = [apply_probe(audio, sample_rate, probe, row["utterance_id"]) for probe in args.probes]
        logits, embeddings = encode_views(model, device, cut, views, args.batch_size)

        original_embedding = embeddings[0]
        probe_embeddings = embeddings[1:]
        probe_entropies = softmax_entropy(logits[1:])
        entropy_order = np.argsort(probe_entropies)

        labels.append(row["label"])
        codecs.append(row["codec"])
        utterance_ids.append(row["utterance_id"])
        original_scores.append(float(logits[0, 1]))
        mean_all_embeddings.append(embeddings.mean(axis=0))

        for topk in topk_values:
            selected = probe_embeddings[entropy_order[:topk]]
            topk_mean_embeddings[topk].append(selected.mean(axis=0))
            for tau in tau_values:
                for lambda_value in lambda_values:
                    key = f"dbd_k{topk}_tau{tau:g}_lambda{lambda_value:g}"
                    db_score, reconstructed = db_score_and_reconstruct(original_embedding, selected, tau, lambda_value)
                    dbd_embeddings[key].append(reconstructed)
                    dbd_scores_meta[key].append(db_score)

        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
            print(f"processed {index}/{len(rows)}", flush=True)

    label_array = np.asarray(labels, dtype=np.int64)
    original_score_array = np.asarray(original_scores, dtype=np.float64)
    method_scores: dict[str, np.ndarray] = {
        "original": original_score_array,
        "mean_all": classifier_scores_from_embeddings(model, np.stack(mean_all_embeddings), device, args.batch_size),
    }
    for topk, embeddings_list in topk_mean_embeddings.items():
        method_scores[f"entropy_mean_k{topk}"] = classifier_scores_from_embeddings(model, np.stack(embeddings_list), device, args.batch_size)
    for key, embeddings_list in dbd_embeddings.items():
        method_scores[key] = classifier_scores_from_embeddings(model, np.stack(embeddings_list), device, args.batch_size)

    results = {name: result_block(label_array, scores, codecs) for name, scores in method_scores.items()}
    ranking = sorted(
        (
            {
                "method": name,
                "eer_percent": block["overall"]["eer_percent"],
                "accuracy_at_0": block["overall"]["accuracy_at_0"],
            }
            for name, block in results.items()
        ),
        key=lambda item: item["eer_percent"],
    )
    db_score_summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for key, values in dbd_scores_meta.items()
    }
    output = {
        "manifest": str(args.manifest),
        "n_total": len(rows),
        "seed": args.seed,
        "probes": args.probes,
        "topk_values": topk_values,
        "tau_values": tau_values,
        "lambda_values": lambda_values,
        "training_free": True,
        "ranking": ranking,
        "results": results,
        "db_score_summary": db_score_summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    args.output_scores.parent.mkdir(parents=True, exist_ok=True)
    with args.output_scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", "codec", "label", *method_scores.keys()])
        for row_index, utterance_id in enumerate(utterance_ids):
            writer.writerow(
                [
                    utterance_id,
                    codecs[row_index],
                    labels[row_index],
                    *[float(scores[row_index]) for scores in method_scores.values()],
                ]
            )

    print(json.dumps({"ranking": ranking[:10]}, indent=2))
    print(f"wrote result: {args.output_json}")
    print(f"wrote scores: {args.output_scores}")


if __name__ == "__main__":
    main()
