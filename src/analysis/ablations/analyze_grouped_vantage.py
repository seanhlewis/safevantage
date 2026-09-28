#!/usr/bin/env python3
"""Analyze grouped vantage-awareness ablations on train OOF and validation.

This is supporting analysis code, not the SafeVantage policy implementation.
The script deliberately does not accept a locked-test path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MASKED_PREFIXES = ("support_positions_ge_",)
MASKED_NAMES = {
    "position_score_2",
    "position_score_3",
    "position_score_top3_mean",
    "grounded_fraction",
    "minimum_grounded_pair_distance_clipped_5m",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sv_base_head", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(row: dict) -> tuple:
    return (
        int(row["scene_id"]),
        str(row["category"]),
        int(row["start_position_id"]),
        int(row["action_budget"]),
    )


def configure_features(base, profile: str):
    original = base.row_features

    def profiled(row: dict, categories: tuple[str, ...]):
        values, pairs = original(row, categories)
        if profile == "no_vantage":
            names = base.feature_names(categories)
            for index, name in enumerate(names):
                if name in MASKED_NAMES or name.startswith(MASKED_PREFIXES):
                    values[index] = 0.0
        return values, pairs

    return profiled


def matrix(base, rows: list[dict], categories: tuple[str, ...], profile: str):
    original = base.row_features
    base.row_features = configure_features(base, profile)
    try:
        return base.build_matrix(rows, categories)
    finally:
        base.row_features = original


def train_model(base, rows: list[dict], profile: str, pair_gate_mode: str) -> dict:
    categories = tuple(sorted({str(row["category"]) for row in rows}))
    x, pairs = matrix(base, rows, categories, profile)
    truth = np.asarray([bool(row["truth"]) for row in rows], dtype=float)
    folds = np.asarray([base.scene_fold(int(row["scene_id"])) for row in rows])
    probabilities = np.zeros(len(rows), dtype=float)
    for fold in range(base.FOLDS):
        train_mask, test_mask = folds != fold, folds == fold
        if not np.any(test_mask):
            raise RuntimeError(f"empty scene fold {fold}")
        mean, std, intercept, coefficients = base.fit_logistic(x[train_mask], truth[train_mask])
        probabilities[test_mask] = base.predict(x[test_mask], mean, std, intercept, coefficients)
    selection = base.summarize_method(rows, probabilities, pairs, pair_gate_mode)
    mean, std, intercept, coefficients = base.fit_logistic(x, truth)
    return {
        "categories": categories,
        "feature_names": base.feature_names(categories),
        "feature_profile": profile,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "intercept": intercept,
        "coefficients": coefficients.tolist(),
        "decision": selection["best"]["decision"],
        "pair_gate_mode": pair_gate_mode,
        "oof_metrics": selection["best"]["metrics"],
        "folds": base.FOLDS,
        "l2": base.L2,
    }


def predictions(base, rows: list[dict], model: dict, *, oof: bool) -> dict[tuple, dict]:
    categories = tuple(model["categories"])
    x, pairs = matrix(base, rows, categories, model["feature_profile"])
    truth = np.asarray([bool(row["truth"]) for row in rows])
    if oof:
        folds = np.asarray([base.scene_fold(int(row["scene_id"])) for row in rows])
        probability = np.zeros(len(rows), dtype=float)
        for fold in range(base.FOLDS):
            train_mask, test_mask = folds != fold, folds == fold
            mean, std, intercept, coefficients = base.fit_logistic(
                x[train_mask], truth[train_mask].astype(float)
            )
            probability[test_mask] = base.predict(x[test_mask], mean, std, intercept, coefficients)
    else:
        probability = base.predict(
            x,
            np.asarray(model["mean"]),
            np.asarray(model["std"]),
            float(model["intercept"]),
            np.asarray(model["coefficients"]),
        )
    decision = model["decision"]
    if model["pair_gate_mode"] == "disabled":
        pair = np.ones(len(rows), dtype=bool)
    else:
        pair = pairs[(float(decision["pair_support_threshold"]), float(decision["pair_radius_m"]))]
    yes = (probability >= float(decision["yes_probability"])) & pair
    no = (probability <= float(decision["no_probability"])) & ~yes
    output = {}
    for row, is_yes, is_no, prob in zip(rows, yes, no, probability):
        output[row_key(row)] = {
            "scene": int(row["scene_id"]),
            "truth": bool(row["truth"]),
            "prediction": 1 if is_yes else 0 if is_no else -1,
            "probability": float(prob),
            "travel": float(row["geodesic_m"]),
        }
    return output


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def metrics(rows: list[dict]) -> dict[str, float]:
    truth = np.asarray([row["truth"] for row in rows], dtype=bool)
    pred = np.asarray([row["prediction"] for row in rows], dtype=np.int8)
    travel = np.asarray([row["travel"] for row in rows], dtype=float)
    yes, no, wait = pred == 1, pred == 0, pred == -1
    answered = ~wait
    yes_tp, yes_fp = int(np.sum(yes & truth)), int(np.sum(yes & ~truth))
    no_tp, no_fp = int(np.sum(no & ~truth)), int(np.sum(no & truth))
    risk = 5.0 * (yes & ~truth) + 2.0 * (no & truth) + 0.5 * wait + 0.001 * travel
    return {
        "mean_risk": float(np.mean(risk)),
        "macro_f1": 0.5 * (
            f1(yes_tp, yes_fp, int(np.sum(~yes & truth)))
            + f1(no_tp, no_fp, int(np.sum(~no & ~truth)))
        ),
        "false_positive_rate": yes_fp / max(1, int(np.sum(~truth))),
        "answer_rate": float(np.mean(answered)),
        "answered_accuracy": float(np.mean((yes == truth)[answered])) if np.any(answered) else None,
        "mean_geodesic_m": float(np.mean(travel)),
    }


STAT_COLUMNS = {
    "episodes": 0,
    "yes_tp": 1,
    "yes_fp": 2,
    "no_tp": 3,
    "no_fp": 4,
    "positives": 5,
    "negatives": 6,
    "risk_sum": 7,
    "travel_sum": 8,
    "answered": 9,
    "answered_correct": 10,
}


def scene_sufficient_statistics(predictions_by_key: dict, scenes: list[int]) -> np.ndarray:
    """Return additive per-scene statistics for exact paired bootstrapping."""
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    stats = np.zeros((len(scenes), len(STAT_COLUMNS)), dtype=float)
    for row in predictions_by_key.values():
        target = stats[scene_index[int(row["scene"])]]
        truth = bool(row["truth"])
        prediction = int(row["prediction"])
        yes, no, wait = prediction == 1, prediction == 0, prediction == -1
        target[STAT_COLUMNS["episodes"]] += 1
        target[STAT_COLUMNS["yes_tp"]] += yes and truth
        target[STAT_COLUMNS["yes_fp"]] += yes and not truth
        target[STAT_COLUMNS["no_tp"]] += no and not truth
        target[STAT_COLUMNS["no_fp"]] += no and truth
        target[STAT_COLUMNS["positives"]] += truth
        target[STAT_COLUMNS["negatives"]] += not truth
        target[STAT_COLUMNS["risk_sum"]] += (
            5.0 * (yes and not truth)
            + 2.0 * (no and truth)
            + 0.5 * wait
            + 0.001 * float(row["travel"])
        )
        target[STAT_COLUMNS["travel_sum"]] += float(row["travel"])
        target[STAT_COLUMNS["answered"]] += not wait
        target[STAT_COLUMNS["answered_correct"]] += (yes == truth) and not wait
    return stats


def metrics_from_sufficient_statistics(stats: np.ndarray) -> dict[str, np.ndarray]:
    episodes = stats[:, STAT_COLUMNS["episodes"]]
    yes_tp = stats[:, STAT_COLUMNS["yes_tp"]]
    yes_fp = stats[:, STAT_COLUMNS["yes_fp"]]
    no_tp = stats[:, STAT_COLUMNS["no_tp"]]
    no_fp = stats[:, STAT_COLUMNS["no_fp"]]
    positives = stats[:, STAT_COLUMNS["positives"]]
    negatives = stats[:, STAT_COLUMNS["negatives"]]
    yes_denominator = 2 * yes_tp + yes_fp + (positives - yes_tp)
    no_denominator = 2 * no_tp + no_fp + (negatives - no_tp)
    yes_f1 = np.divide(2 * yes_tp, yes_denominator, out=np.zeros_like(yes_tp), where=yes_denominator > 0)
    no_f1 = np.divide(2 * no_tp, no_denominator, out=np.zeros_like(no_tp), where=no_denominator > 0)
    answered = stats[:, STAT_COLUMNS["answered"]]
    answered_accuracy = np.divide(
        stats[:, STAT_COLUMNS["answered_correct"]],
        answered,
        out=np.full_like(answered, np.nan),
        where=answered > 0,
    )
    return {
        "mean_risk": stats[:, STAT_COLUMNS["risk_sum"]] / episodes,
        "macro_f1": 0.5 * (yes_f1 + no_f1),
        "false_positive_rate": np.divide(yes_fp, negatives, out=np.zeros_like(yes_fp), where=negatives > 0),
        "answer_rate": answered / episodes,
        "answered_accuracy": answered_accuracy,
        "mean_geodesic_m": stats[:, STAT_COLUMNS["travel_sum"]] / episodes,
    }


def bootstrap(first: dict, second: dict, draws: int, seed: int) -> dict:
    keys = sorted(set(first) & set(second))
    if len(keys) != len(first) or len(keys) != len(second) or not keys:
        raise RuntimeError(f"unaligned rows: first={len(first)} second={len(second)} paired={len(keys)}")
    by_scene: dict[int, list[tuple]] = defaultdict(list)
    for item in keys:
        by_scene[item[0]].append(item)
    scenes = sorted(by_scene)
    first_point = metrics([first[item] for item in keys])
    second_point = metrics([second[item] for item in keys])
    rng = np.random.default_rng(seed)
    sampled_scene_indices = rng.integers(0, len(scenes), size=(draws, len(scenes)))
    first_scene_stats = scene_sufficient_statistics(first, scenes)
    second_scene_stats = scene_sufficient_statistics(second, scenes)
    first_draws = metrics_from_sufficient_statistics(first_scene_stats[sampled_scene_indices].sum(axis=1))
    second_draws = metrics_from_sufficient_statistics(second_scene_stats[sampled_scene_indices].sum(axis=1))
    samples = {name: first_draws[name] - second_draws[name] for name in first_point}
    return {
        "scenes": len(scenes),
        "paired_episodes": len(keys),
        "first": first_point,
        "second": second_point,
        "delta_first_minus_second": {
            name: {
                "estimate": first_point[name] - second_point[name],
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
                "probability_favorable": float(np.mean(values < 0))
                if name in {"mean_risk", "false_positive_rate", "mean_geodesic_m"}
                else float(np.mean(values > 0)),
            }
            for name, values in samples.items()
        },
    }


def merge_shards(root: Path, split: str, budget: int, expected_scenes: int) -> list[dict]:
    paths = [
        root / "eval" / split / f"a{budget}" / "no_vantage_acquisition" / f"shard_{index}" / "rows.jsonl"
        for index in range(8)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing {split}/a{budget} shards: {missing}")
    rows = [row for path in paths for row in load_jsonl(path)]
    keys = [row_key(row) for row in rows]
    scenes = {int(row["scene_id"]) for row in rows}
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate keys in {split}/a{budget}")
    if len(scenes) != expected_scenes:
        raise RuntimeError(f"expected {expected_scenes} scenes in {split}/a{budget}, found {len(scenes)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-head", type=Path, required=True)
    parser.add_argument("--full-train-a8", type=Path, required=True)
    parser.add_argument("--full-train-a12", type=Path, required=True)
    parser.add_argument("--full-validation-a8", type=Path, required=True)
    parser.add_argument("--full-validation-a12", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if "test" in str(args.full_train_a8).lower() or "test" in str(args.full_validation_a8).lower():
        raise RuntimeError("locked-test paths are forbidden")

    base = load_module(args.base_head)
    result = {
        "schema": "safevantage_grouped_vantage_ablation_v0.1",
        "locked_test_accessed": False,
        "draws": args.draws,
        "budgets": {},
    }
    for budget, full_train_path, full_val_path in (
        (8, args.full_train_a8, args.full_validation_a8),
        (12, args.full_train_a12, args.full_validation_a12),
    ):
        full_train = load_jsonl(full_train_path)
        full_val = load_jsonl(full_val_path)
        no_train = merge_shards(args.root, "train", budget, 116)
        no_val = merge_shards(args.root, "validation", budget, 96)
        merged_train_path = args.root / "analysis" / "rows" / f"a{budget}" / "no_vantage_train.jsonl"
        merged_val_path = args.root / "analysis" / "rows" / f"a{budget}" / "no_vantage_validation.jsonl"
        write_jsonl(merged_train_path, no_train)
        write_jsonl(merged_val_path, no_val)

        models = {
            "full_required": train_model(base, full_train, "full", "required"),
            "no_vantage_acquisition_required": train_model(base, no_train, "full", "required"),
            "full_gate_free": train_model(base, full_train, "full", "disabled"),
            "no_vantage_acquisition_gate_free": train_model(base, no_train, "full", "disabled"),
            "no_vantage_end_to_end_gate_free": train_model(base, no_train, "no_vantage", "disabled"),
        }
        model_path = args.root / "analysis" / "models" / f"a{budget}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        train_predictions = {
            name: predictions(base, full_train if name.startswith("full") else no_train, model, oof=True)
            for name, model in models.items()
        }
        val_predictions = {
            name: predictions(base, full_val if name.startswith("full") else no_val, model, oof=False)
            for name, model in models.items()
        }
        contrasts = {
            "original_contract": ("full_required", "no_vantage_acquisition_required"),
            "gate_free_acquisition": ("full_gate_free", "no_vantage_acquisition_gate_free"),
            "gate_free_end_to_end": ("full_gate_free", "no_vantage_end_to_end_gate_free"),
            "pair_gate_control": ("full_required", "full_gate_free"),
        }
        budget_result = {
            "accounting": {
                "train_scenes": len({row["scene_id"] for row in no_train}),
                "train_episodes": len(no_train),
                "validation_scenes": len({row["scene_id"] for row in no_val}),
                "validation_episodes": len(no_val),
                "no_vantage_train_sha256": digest(merged_train_path),
                "no_vantage_validation_sha256": digest(merged_val_path),
            },
            "models": models,
            "contrasts": {},
        }
        for offset, (label, (first_name, second_name)) in enumerate(contrasts.items()):
            budget_result["contrasts"][label] = {
                "first": first_name,
                "second": second_name,
                "train_oof": bootstrap(
                    train_predictions[first_name], train_predictions[second_name], args.draws,
                    args.seed + budget * 100 + offset,
                ),
                "train_frozen_validation": bootstrap(
                    val_predictions[first_name], val_predictions[second_name], args.draws,
                    args.seed + budget * 1000 + offset,
                ),
            }
        result["budgets"][str(budget)] = budget_result

    output = args.root / "analysis" / "RESULT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
