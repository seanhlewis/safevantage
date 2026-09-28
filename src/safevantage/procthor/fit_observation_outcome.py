#!/usr/bin/env python3
"""Fit the SafeVantage observation-outcome model used during evidence selection.

Training is restricted to the training split. Hidden visibility is used only
as the training target; evaluation receives frozen coefficients and must not
read hidden scene labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .evaluate_procthor import controlled_score, controlled_target_estimate
except ImportError:  # Direct script invocation from the reproduction commands.
    from evaluate_procthor import controlled_score, controlled_target_estimate


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_logistic(features: np.ndarray, labels: np.ndarray, steps: int = 80, reg: float = 1e-2):
    x = np.column_stack([np.ones(len(features)), features])
    weights = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        probabilities = sigmoid(x @ weights)
        gradient = (x.T @ (probabilities - labels)) / max(len(labels), 1) + reg * weights
        hessian = (x.T * (probabilities * (1.0 - probabilities))) @ x / max(len(labels), 1)
        hessian += reg * np.eye(x.shape[1])
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian) @ gradient
        weights -= delta
        if float(np.max(np.abs(delta))) < 1e-6:
            break
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    scene_dirs = [
        path for path in sorted((args.scene_root / "train").glob("scene_*"))
        if (path / "COMPLETE.json").exists()
    ]
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[:args.max_scenes]
    visibility_features = []
    visibility_labels = []
    score_rows = []
    false_positive_scores = []
    absent_score_count = 0
    for scene_dir in scene_dirs:
        public = json.loads((scene_dir / "public.json").read_text(encoding="utf-8"))
        hidden = json.loads((scene_dir / "hidden.json").read_text(encoding="utf-8"))
        views = {int(row["view_id"]): row for row in public["views"]}
        geodesic = np.asarray(public["geodesic_position_cost_m"], dtype=float)
        for category, present in hidden["vocabulary_presence"].items():
            category = str(category)
            target = controlled_target_estimate(public, hidden, category) if present else None
            scores = []
            for row in hidden["views"]:
                view_id = int(row["view_id"])
                score = controlled_score(
                    int(hidden["scene_id"]), category, view_id,
                    float(row["visible_fraction"].get(category, 0.0)),
                )
                scores.append(score)
                if not present:
                    absent_score_count += 1
                if not present and score >= 0.55:
                    false_positive_scores.append(score)
                if target is not None:
                    candidate = views[view_id]
                    point = np.asarray(candidate["position_world"][:2], dtype=float)
                    delta = np.asarray(target[:2], dtype=float) - point
                    distance = float(np.linalg.norm(delta))
                    direction = delta / distance if distance > 1e-6 else np.zeros(2)
                    heading = np.asarray([math.cos(float(candidate["yaw_rad"])), math.sin(float(candidate["yaw_rad"]))])
                    cosine = float(np.dot(heading, direction)) if distance > 1e-6 else 0.0
                    visibility_features.append([distance, cosine, float(geodesic[0, int(candidate["position_id"])]), 1.0])
                    visibility_labels.append(float(row["visible_fraction"].get(category, 0.0) >= 1e-4))
            score_rows.append((max(scores), float(bool(present))))
    if not visibility_features:
        raise RuntimeError("no positive training targets found")
    features = np.asarray(visibility_features, dtype=float)
    labels = np.asarray(visibility_labels, dtype=float)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    coefficients = fit_logistic(normalized, labels)
    score_x = np.asarray([[row[0]] for row in score_rows], dtype=float)
    score_y = np.asarray([row[1] for row in score_rows], dtype=float)
    belief_coeff = fit_logistic(score_x, score_y, reg=1e-3)
    payload = {
        "version": "counterevidence-outcome-model-v0.1",
        "training_split": "train",
        "training_scenes": len(scene_dirs),
        "visibility_examples": len(labels),
        "visibility_positive_rate": float(labels.mean()),
        "feature_names": ["distance_to_target_m", "heading_cosine", "baseline_from_first_position_m", "independent_position"],
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "bias": float(coefficients[0]),
        "weights": coefficients[1:].tolist(),
        "false_positive_rate": float(len(false_positive_scores) / max(absent_score_count, 1)),
        "belief_calibration": {"bias": float(belief_coeff[0]), "slope": float(belief_coeff[1])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
