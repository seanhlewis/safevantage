#!/usr/bin/env python3
"""Recompute pooled metrics and paired scene-bootstrap intervals from CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    total = sum(int(row["episodes"]) for row in rows)
    if total == 0:
        raise ValueError("cannot calculate metrics from an empty group")
    yes_tp = sum(int(row["pred_yes_truth_yes"]) for row in rows)
    yes_fp = sum(int(row["pred_yes_truth_no"]) for row in rows)
    yes_fn = sum(int(row["truth_yes"]) - int(row["pred_yes_truth_yes"]) for row in rows)
    no_tp = sum(int(row["pred_no_truth_no"]) for row in rows)
    no_fp = sum(int(row["pred_no_truth_yes"]) for row in rows)
    no_fn = sum(int(row["truth_no"]) - int(row["pred_no_truth_no"]) for row in rows)
    answered = yes_tp + yes_fp + no_tp + no_fp
    correct = yes_tp + no_tp
    negatives = sum(int(row["truth_no"]) for row in rows)
    return {
        "episodes": total,
        "mean_risk": sum(float(row["risk_sum"]) for row in rows) / total,
        "macro_f1": 0.5 * (f1(yes_tp, yes_fp, yes_fn) + f1(no_tp, no_fp, no_fn)),
        "false_positive_rate": yes_fp / negatives if negatives else 0.0,
        "answer_rate": answered / total,
        "answered_accuracy": correct / answered if answered else 0.0,
        "mean_geodesic_m": sum(float(row["travel_m_sum"]) for row in rows) / total,
        "mean_actions": sum(int(row["actions_sum"]) for row in rows) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="paired scene metrics CSV")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", default="safevantage_active")
    parser.add_argument("--proposed", default="safevantage_dc_full")
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[(int(row["action_budget"]), row["method"])].append(row)

    output: dict[str, object] = {
        "bootstrap": {"unit": "scene_group_id", "resamples": args.resamples, "seed": args.seed,
                      "interval": "percentile 95% (2.5th, 97.5th)",
                      "delta_direction": f"{args.proposed} minus comparison method"},
        "budgets": {},
    }
    for budget in sorted({key[0] for key in grouped}):
        methods = sorted(method for b, method in grouped if b == budget)
        by_method = {method: grouped[(budget, method)] for method in methods}
        scene_ids = sorted({row["scene_group_id"] for row in by_method[methods[0]]})
        for method, rows in by_method.items():
            if {row["scene_group_id"] for row in rows} != set(scene_ids):
                parser.error(f"scene groups do not pair across methods at budget {budget}")
            if len({row["scene_group_id"] for row in rows}) != len(rows):
                parser.error(f"duplicate scene-method row for {method}, budget {budget}")
        budget_result: dict[str, object] = {
            "scenes": len(scene_ids),
            "point_metrics": {method: metrics(by_method[method]) for method in methods},
            "paired_deltas": {},
        }
        if args.reference not in by_method or args.proposed not in by_method:
            parser.error(f"requested methods missing at budget {budget}: {args.reference}, {args.proposed}")
        index = {method: {row["scene_group_id"]: row for row in rows}
                 for method, rows in by_method.items()}
        # Historical intervals were computed with a fresh seed-0 generator
        # for each action budget; preserve that convention for exact matching.
        rng = np.random.default_rng(args.seed)

        def sampled_values(method: str, sampled: list[str]) -> tuple[float, float]:
            stats = metrics([index[method][scene] for scene in sampled])
            return float(stats["mean_risk"]), float(stats["macro_f1"])

        delta_samples = np.empty((args.resamples, 2), dtype=float)
        for sample_idx in range(args.resamples):
            sampled = [scene_ids[i] for i in rng.integers(0, len(scene_ids), len(scene_ids))]
            proposed = sampled_values(args.proposed, sampled)
            reference = sampled_values(args.reference, sampled)
            delta_samples[sample_idx] = np.subtract(proposed, reference)
        delta_mean = delta_samples.mean(axis=0)
        delta_ci = np.quantile(delta_samples, [0.025, 0.975], axis=0)
        budget_result["paired_deltas"] = {
            "comparison": args.reference,
            "risk": {"mean": float(delta_mean[0]), "ci95": delta_ci[:, 0].tolist()},
            "macro_f1": {"mean": float(delta_mean[1]), "ci95": delta_ci[:, 1].tolist()},
        }
        output["budgets"][str(budget)] = budget_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote bootstrap metrics to {args.output}")


if __name__ == "__main__":
    main()
