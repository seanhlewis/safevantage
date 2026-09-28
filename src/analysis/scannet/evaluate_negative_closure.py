#!/usr/bin/env python3
"""Frozen explicit coverage-certificate audit on four-state ScanNet episodes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


EPISTEMIC_STATES = {"observed_absent", "observed_present", "unobserved", "view_ambiguous"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: each JSONL record must be an object")
        required = {"split", "scene_id", "epistemic_state", "max_p_clear", "coverage_fraction", "source_coverage_fraction"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{path}:{line_no}: missing required fields: {', '.join(missing)}")
        if row["split"] not in {"dev", "test"}:
            raise ValueError(f"{path}:{line_no}: split must be 'dev' or 'test'")
        if row["epistemic_state"] not in EPISTEMIC_STATES:
            raise ValueError(f"{path}:{line_no}: epistemic_state must be one of {sorted(EPISTEMIC_STATES)}")
        for field in ("max_p_clear", "coverage_fraction", "source_coverage_fraction"):
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {field} must be numeric") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}:{line_no}: {field} must be finite and between 0 and 1")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no episode records found")
    splits = {split: {str(row["scene_id"]) for row in rows if row["split"] == split} for split in ("dev", "test")}
    if len(splits["dev"]) != 6 or len(splits["test"]) != 12:
        raise ValueError(f"expected 6 development and 12 test scenes; found {len(splits['dev'])} and {len(splits['test'])}")
    if splits["dev"] & splits["test"]:
        raise ValueError("development and test scene IDs must be disjoint")
    return rows


def metrics(rows: list[dict[str, Any]], rule: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    predicted = [rule(row) for row in rows]
    absent = [row["epistemic_state"] == "observed_absent" for row in rows]
    true_count = sum(absent); non_count = len(rows) - true_count
    tp = sum(p and y for p, y in zip(predicted, absent)); fp = sum(p and not y for p, y in zip(predicted, absent))
    by_state = {}
    for state in sorted({row["epistemic_state"] for row in rows}):
        keep = [p for p, row in zip(predicted, rows) if row["epistemic_state"] == state]
        by_state[state] = {"n": len(keep), "negative_commit_rate": sum(keep) / len(keep) if keep else 0.0}
    return {"n": len(rows), "observed_absent_recall": tp / true_count if true_count else 0.0,
            "false_negative_commit_rate": fp / non_count if non_count else 0.0,
            "negative_commit_precision": tp / (tp + fp) if tp + fp else 0.0,
            "negative_commits": tp + fp, "by_state": by_state}


def choose(rows: list[dict[str, Any]], family: str, max_false: float) -> dict[str, Any]:
    clear_values = sorted({float(row["max_p_clear"]) for row in rows})
    coverage_values = sorted({min(float(row["coverage_fraction"]), float(row["source_coverage_fraction"])) for row in rows})
    options = []
    if family == "score_only":
        for clear_max in clear_values:
            rule = lambda row, p=clear_max: float(row["max_p_clear"]) <= p
            options.append(({"clear_max": clear_max}, metrics(rows, rule)))
    else:
        for coverage_min in coverage_values:
            for clear_max in clear_values:
                rule = lambda row, c=coverage_min, p=clear_max: (
                    min(float(row["coverage_fraction"]), float(row["source_coverage_fraction"])) >= c
                    and float(row["max_p_clear"]) <= p
                )
                options.append(({"coverage_min": coverage_min, "clear_max": clear_max}, metrics(rows, rule)))
    feasible = [item for item in options if item[1]["false_negative_commit_rate"] <= max_false + 1e-12]
    if not feasible:
        raise ValueError(f"no feasible {family} rule")
    params, score = max(feasible, key=lambda item: (item[1]["observed_absent_recall"], item[1]["negative_commit_precision"], item[1]["negative_commits"], str(item[0])))
    return {"params": params, "development": score}


def make_rule(family: str, params: dict[str, float]) -> Callable[[dict[str, Any]], bool]:
    if family == "score_only":
        return lambda row: float(row["max_p_clear"]) <= params["clear_max"]
    return lambda row: (
        min(float(row["coverage_fraction"]), float(row["source_coverage_fraction"])) >= params["coverage_min"]
        and float(row["max_p_clear"]) <= params["clear_max"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen ScanNet four-state episode audit; this command consumes prepared episodes and does not process raw scans.",
        epilog=("Input is JSONL, one episode object per line. Required fields: split (dev/test), scene_id, epistemic_state "
                "(observed_absent/observed_present/unobserved/view_ambiguous), max_p_clear, coverage_fraction, and "
                "source_coverage_fraction. The frozen split has 6 development and 12 disjoint test scenes. See "
                "data/schemas/scannet_negative_closure.schema.json and data/analysis_input_guide.json."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=Path, required=True, help="Prepared four-state ScanNet episode JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Path for the resulting JSON summary")
    parser.add_argument("--development-max-false", type=float, default=.05,
                        help="Maximum development false-negative-commit rate used to select each rule (default: 0.05)")
    args = parser.parse_args()
    rows = read_jsonl(args.episodes); dev = [row for row in rows if row["split"] == "dev"]; test = [row for row in rows if row["split"] == "test"]
    result: dict[str, Any] = {"schema": "safevantage_negative_closure_v0_1", "data_status": "immutable-consumed-ScanNet-four-state",
                              "fresh_test_touched": False, "development_scenes": len({r["scene_id"] for r in dev}), "test_scenes": len({r["scene_id"] for r in test}), "methods": {}}
    for family in ("score_only", "coverage_certificate"):
        selected = choose(dev, family, args.development_max_false)
        selected["test"] = metrics(test, make_rule(family, selected["params"]))
        result["methods"][family] = selected
    closure = result["methods"]["coverage_certificate"]["test"]; scalar = result["methods"]["score_only"]["test"]
    result["gates"] = {"false_negative_commit_at_most_0_10": closure["false_negative_commit_rate"] <= .10,
                       "observed_absent_recall_at_least_0_80": closure["observed_absent_recall"] >= .80,
                       "recall_gain_over_score_only_at_least_0_20": closure["observed_absent_recall"] - scalar["observed_absent_recall"] >= .20}
    result["pass"] = all(result["gates"].values())
    result["warning"] = "Constructed observability-closure unit test; not evidence that arbitrary trajectories exhaust a scene."
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
