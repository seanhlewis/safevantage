#!/usr/bin/env python3
"""Train and apply scene-disjoint selective decision heads for v0.4."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

FPR_CAP = float(os.environ.get("SAFEVANTAGE_FPR_CAP", "0.01"))


SUPPORT_THRESHOLDS = (0.05, 0.10, 0.30, 0.50, 0.70)
PAIR_THRESHOLDS = (0.10, 0.30, 0.50)
PAIR_RADII = (0.25, 0.50, 1.00)
YES_PROBABILITIES = (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
NO_PROBABILITIES = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
L2 = 0.10
FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_fold(scene_id: int) -> int:
    digest = hashlib.sha256(f"safevantage-selective-v0.4::{scene_id}".encode()).hexdigest()
    return int(digest[:16], 16) % FOLDS


def sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def pair_distance(first: dict, second: dict) -> float | None:
    if int(first["position_id"]) == int(second["position_id"]):
        return None
    a, b = first.get("target_world"), second.get("target_world")
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a[:2], dtype=float) - np.asarray(b[:2], dtype=float)))


def row_features(row: dict, categories: tuple[str, ...]) -> tuple[list[float], dict[tuple[float, float], bool]]:
    history = row["history"]
    position_scores: dict[int, float] = defaultdict(float)
    for item in history:
        position = int(item["position_id"])
        position_scores[position] = max(position_scores[position], float(item["score"]))
    ordered = sorted(position_scores.values(), reverse=True)
    top = (ordered + [0.0, 0.0, 0.0])[:3]
    grounded = [item for item in history if item.get("target_world") is not None]
    distances = [
        distance
        for index, first in enumerate(grounded)
        for second in grounded[index + 1 :]
        if (distance := pair_distance(first, second)) is not None
    ]
    context_max = max(
        [float(score) for item in history for score in item.get("context_scores", {}).values()] or [0.0]
    )
    numeric = [
        top[0],
        top[1],
        top[2],
        float(np.mean(top)),
        *[float(sum(score >= threshold for score in ordered)) for threshold in SUPPORT_THRESHOLDS],
        len(position_scores) / max(1.0, float(row.get("total_positions", 40))),
        float(len({int(item["room_id"]) for item in history})),
        float(len(history)),
        len(grounded) / max(1.0, float(len(history))),
        min(5.0, min(distances)) if distances else 5.0,
        context_max,
    ]
    category = str(row["category"])
    numeric.extend(float(category == name) for name in categories)
    pairs = {}
    for threshold, radius in itertools.product(PAIR_THRESHOLDS, PAIR_RADII):
        support = [item for item in history if float(item["score"]) >= threshold]
        pairs[(threshold, radius)] = any(
            distance is not None and distance <= radius
            for index, first in enumerate(support)
            for second in support[index + 1 :]
            for distance in [pair_distance(first, second)]
        )
    return numeric, pairs


def feature_names(categories: tuple[str, ...]) -> list[str]:
    return [
        "position_score_1",
        "position_score_2",
        "position_score_3",
        "position_score_top3_mean",
        *[f"support_positions_ge_{threshold:.2f}" for threshold in SUPPORT_THRESHOLDS],
        "position_coverage",
        "distinct_rooms",
        "observations",
        "grounded_fraction",
        "minimum_grounded_pair_distance_clipped_5m",
        "maximum_context_score",
        *[f"category={name}" for name in categories],
    ]


def build_matrix(rows: list[dict], categories: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    values = []
    pair_values = {pair: [] for pair in itertools.product(PAIR_THRESHOLDS, PAIR_RADII)}
    for row in rows:
        features, pairs = row_features(row, categories)
        values.append(features)
        for pair, present in pairs.items():
            pair_values[pair].append(present)
    return np.asarray(values, dtype=float), {
        pair: np.asarray(present, dtype=bool) for pair, present in pair_values.items()
    }


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    # Every deployable feature is standardized from the training-fold
    # statistics only. Constant columns retain unit scale.
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    design = np.column_stack([np.ones(len(z)), z])

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ weights
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        loss += 0.5 * L2 * float(np.sum(weights[1:] ** 2))
        gradient = design.T @ (sigmoid(logits) - y) / len(y)
        gradient[1:] += L2 * weights[1:]
        return loss, gradient

    result = minimize(
        lambda weights: objective(weights)[0],
        np.zeros(design.shape[1], dtype=float),
        jac=lambda weights: objective(weights)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return mean, std, float(result.x[0]), np.asarray(result.x[1:], dtype=float)


def predict(x: np.ndarray, mean: np.ndarray, std: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    return sigmoid(intercept + ((x - mean) / std) @ coefficients)


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def decision_metrics(truth: np.ndarray, probability: np.ndarray, pair: np.ndarray, travel: np.ndarray, yes_p: float, no_p: float) -> dict:
    yes = (probability >= yes_p) & pair
    no = (probability <= no_p) & ~yes
    wait = ~(yes | no)
    yes_tp = int(np.sum(yes & truth))
    yes_fp = int(np.sum(yes & ~truth))
    yes_fn = int(np.sum(~yes & truth))
    no_tp = int(np.sum(no & ~truth))
    no_fp = int(np.sum(no & truth))
    no_fn = int(np.sum(~no & ~truth))
    answered = yes | no
    risk = 5.0 * (yes & ~truth) + 2.0 * (no & truth) + 0.5 * wait + 0.001 * travel
    return {
        "episodes": int(len(truth)),
        "mean_risk": float(np.mean(risk)),
        "macro_f1": 0.5 * (f1(yes_tp, yes_fp, yes_fn) + f1(no_tp, no_fp, no_fn)),
        "false_positive_rate": yes_fp / max(1, int(np.sum(~truth))),
        "answer_rate": float(np.mean(answered)),
        "answered_accuracy": float(np.mean((yes == truth)[answered])) if np.any(answered) else None,
        "mean_geodesic_m": float(np.mean(travel)),
        "predictions": {"yes": int(np.sum(yes)), "no": int(np.sum(no)), "wait": int(np.sum(wait))},
    }


def rank_key(record: dict) -> tuple[float, float, float, float]:
    metrics = record["metrics"]
    fpr_excess = max(0.0, metrics["false_positive_rate"] - FPR_CAP)
    return (fpr_excess, metrics["mean_risk"], -metrics["macro_f1"], metrics["mean_geodesic_m"])


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_method(rows: list[dict], probabilities: np.ndarray, pairs: dict) -> dict:
    truth = np.asarray([bool(row["truth"]) for row in rows])
    travel = np.asarray([float(row["geodesic_m"]) for row in rows])
    candidates = []
    for pair_key, pair in pairs.items():
        for yes_p, no_p in itertools.product(YES_PROBABILITIES, NO_PROBABILITIES):
            candidates.append({
                "decision": {
                    "yes_probability": yes_p,
                    "no_probability": no_p,
                    "pair_support_threshold": pair_key[0],
                    "pair_radius_m": pair_key[1],
                },
                "metrics": decision_metrics(truth, probabilities, pair, travel, yes_p, no_p),
            })
    candidates.sort(key=rank_key)
    return {"best": candidates[0], "top_10": candidates[:10], "evaluated_decisions": len(candidates)}


def train(args: argparse.Namespace) -> None:
    all_rows = load_rows(args.rows)
    methods = sorted({str(row["method"]) for row in all_rows})
    categories = tuple(sorted({str(row["category"]) for row in all_rows}))
    models = {}
    for method in methods:
        rows = [row for row in all_rows if row["method"] == method]
        x, pairs = build_matrix(rows, categories)
        truth = np.asarray([bool(row["truth"]) for row in rows], dtype=float)
        folds = np.asarray([scene_fold(int(row["scene_id"])) for row in rows])
        probabilities = np.zeros(len(rows), dtype=float)
        for fold in range(FOLDS):
            train_mask = folds != fold
            test_mask = folds == fold
            if not np.any(test_mask):
                raise RuntimeError(f"empty scene fold {fold} for {method}")
            mean, std, intercept, coefficients = fit_logistic(x[train_mask], truth[train_mask])
            probabilities[test_mask] = predict(x[test_mask], mean, std, intercept, coefficients)
        selection = summarize_method(rows, probabilities, pairs)
        mean, std, intercept, coefficients = fit_logistic(x, truth)
        models[method] = {
            "categories": categories,
            "feature_names": feature_names(categories),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "intercept": intercept,
            "coefficients": coefficients.tolist(),
            "decision": selection["best"]["decision"],
            "oof_metrics": selection["best"]["metrics"],
            "top_10_oof": selection["top_10"],
            "folds": FOLDS,
            "l2": L2,
        }

    baseline_methods = [method for method in methods if method != args.safe_method]
    if args.safe_method not in models or not baseline_methods:
        raise RuntimeError("safe method or baselines missing from training rows")
    strongest = min(baseline_methods, key=lambda method: rank_key({"metrics": models[method]["oof_metrics"]}))
    safe_metrics = models[args.safe_method]["oof_metrics"]
    baseline_metrics = models[strongest]["oof_metrics"]
    payload = {
        "schema": "safevantage_selective_head_v0.4",
        "selection_split": "train_oof",
        "source_rows": str(args.rows),
        "source_rows_sha256": sha256(args.rows),
        "safe_method": args.safe_method,
        "strongest_baseline": strongest,
        "deltas_safe_minus_strongest": {
            key: float(safe_metrics[key]) - float(baseline_metrics[key])
            for key in ("mean_risk", "macro_f1", "false_positive_rate", "mean_geodesic_m")
        },
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def apply(args: argparse.Namespace) -> None:
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    all_rows = load_rows(args.rows)
    output_rows = []
    summaries = {}
    for method, model in payload["models"].items():
        rows = [row for row in all_rows if row["method"] == method]
        if not rows:
            raise RuntimeError(f"validation rows missing method {method}")
        categories = tuple(model["categories"])
        x, pairs = build_matrix(rows, categories)
        probabilities = predict(
            x,
            np.asarray(model["mean"]),
            np.asarray(model["std"]),
            float(model["intercept"]),
            np.asarray(model["coefficients"]),
        )
        decision = model["decision"]
        pair = pairs[(float(decision["pair_support_threshold"]), float(decision["pair_radius_m"]))]
        truth = np.asarray([bool(row["truth"]) for row in rows])
        yes = (probabilities >= float(decision["yes_probability"])) & pair
        no = (probabilities <= float(decision["no_probability"])) & ~yes
        wait = ~(yes | no)
        travel = np.asarray([float(row["geodesic_m"]) for row in rows])
        summaries[method] = decision_metrics(
            truth, probabilities, pair, travel, float(decision["yes_probability"]), float(decision["no_probability"])
        )
        for row, probability, is_yes, is_no, is_wait in zip(rows, probabilities, yes, no, wait):
            prediction = "yes" if is_yes else "no" if is_no else "wait"
            fp = int(is_yes and not row["truth"])
            fn = int(is_no and row["truth"])
            output_rows.append({
                key: row[key] for key in (
                    "scene_id", "category", "truth", "start_position_id", "method", "sensor",
                    "action_budget", "actions_taken", "geodesic_m"
                )
            } | {
                "prediction": prediction,
                "presence_probability": float(probability),
                "false_positive": fp,
                "false_negative": fn,
                "abstain": int(is_wait),
                "risk": 5.0 * fp + 2.0 * fn + 0.5 * int(is_wait) + 0.001 * float(row["geodesic_m"]),
            })
    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    args.output_rows.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    result = {
        "schema": "safevantage_selective_head_application_v0.4",
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "source_rows": str(args.rows),
        "source_rows_sha256": sha256(args.rows),
        "methods": summaries,
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--rows", type=Path, required=True)
    train_parser.add_argument("--safe-method", default="safevantage_active")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.set_defaults(function=train)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--model", type=Path, required=True)
    apply_parser.add_argument("--rows", type=Path, required=True)
    apply_parser.add_argument("--output-rows", type=Path, required=True)
    apply_parser.add_argument("--summary", type=Path, required=True)
    apply_parser.set_defaults(function=apply)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

