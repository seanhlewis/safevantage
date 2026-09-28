#!/usr/bin/env python3
"""Frozen multi-split, matched-coverage, and calibration-selected AURC audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


CANONICAL_METHOD_FIELDS = {
    "safevantage_full": "support_count",
    "conceptgraphs_tau1": "confidence",
    "vlmaps_tau1": "confidence",
}


def read_jsonl(path: Path, kind: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: each JSONL record must be an object")
        if kind == "canonical":
            required = {"case_id", "scene_id", "gt_present", "method"}
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_no}: missing required fields: {', '.join(missing)}")
            method = row["method"]
            if method not in CANONICAL_METHOD_FIELDS:
                continue
            score_field = CANONICAL_METHOD_FIELDS[method]
            if score_field not in row:
                raise ValueError(f"{path}:{line_no}: method {method} requires {score_field}")
            score = row[score_field]
            if row["gt_present"] not in (0, 1, False, True):
                raise ValueError(f"{path}:{line_no}: gt_present must be 0 or 1")
        else:
            missing = sorted({"case_id", "score"} - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_no}: missing required fields: {', '.join(missing)}")
            score = row["score"]
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: score must be numeric") from exc
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_no}: score must be finite")
        identity = (row.get("method"), str(row["case_id"])) if kind == "canonical" else str(row["case_id"])
        if identity in seen:
            raise ValueError(f"{path}:{line_no}: duplicate input row for {identity}")
        seen.add(identity)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no usable {kind} records")
    return rows


def score_rows(canonical: list[dict[str, Any]], caption: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    specs = {
        "SafeVantage": ("safevantage_full", "support_count"),
        "ConceptGraphs": ("conceptgraphs_tau1", "confidence"),
        "VLMaps": ("vlmaps_tau1", "confidence"),
    }
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for name, (method, field) in specs.items():
        output[name] = {
            str(row["case_id"]): {
                "case_id": str(row["case_id"]), "scene_id": str(row["scene_id"]),
                "truth": int(row["gt_present"]), "score": float(row[field]),
            }
            for row in canonical if row.get("method") == method
        }
    canonical_truth = output["SafeVantage"]
    output["caption-RAG"] = {
        str(row["case_id"]): {
            "case_id": str(row["case_id"]),
            "scene_id": str(row.get("scene_id", canonical_truth[str(row["case_id"])]["scene_id"])),
            "truth": int(row.get("gt_present", canonical_truth[str(row["case_id"])]["truth"])),
            "score": float(row["score"]),
        }
        for row in caption if str(row["case_id"]) in canonical_truth
    }
    common = set.intersection(*(set(rows) for rows in output.values()))
    if len(common) < 1200:
        raise ValueError(f"only {len(common)} four-way matched cases")
    for case_id in common:
        reference = output["SafeVantage"][case_id]
        for name, mapping in output.items():
            row = mapping[case_id]
            if row["scene_id"] != reference["scene_id"] or row["truth"] != reference["truth"]:
                raise ValueError(f"case_id {case_id} has inconsistent scene_id or gt_present in {name}")
    return {name: {case: rows[case] for case in sorted(common)} for name, rows in output.items()}


def cuts(rows: list[dict[str, Any]]) -> list[tuple[int, int, float | None, float]]:
    ordered = sorted(rows, key=lambda row: (row["score"], row["case_id"]))
    scores = [row["score"] for row in ordered]
    positions = [0] + [i for i in range(1, len(rows)) if scores[i] != scores[i - 1]] + [len(rows)]
    result = []
    for b in positions:
        yes_threshold = scores[b] if b < len(rows) else math.inf
        for a in positions:
            if a > b:
                break
            no_threshold = None if a == 0 else scores[a - 1]
            result.append((a, b, no_threshold, yes_threshold))
    return result


def decide(score: float, no_threshold: float | None, yes_threshold: float) -> str:
    if score >= yes_threshold:
        return "yes"
    if no_threshold is not None and score <= no_threshold:
        return "no"
    return "abstain"


def evaluate(rows: list[dict[str, Any]], threshold: tuple[float | None, float], costs: dict[str, float]) -> dict[str, Any]:
    no_threshold, yes_threshold = threshold
    per_case, answered, correct, fp, negatives, recall, positives = [], 0, 0, 0, 0, 0, 0
    for row in rows:
        truth = int(row["truth"]); decision = decide(float(row["score"]), no_threshold, yes_threshold)
        cost = costs["view"] * 160
        if truth == 0:
            negatives += 1
        else:
            positives += 1
        if decision == "yes":
            answered += 1; correct += int(truth == 1); recall += int(truth == 1)
            cost += costs["false_positive"] if truth == 0 else 0.0
            fp += int(truth == 0)
        elif decision == "no":
            answered += 1; correct += int(truth == 0)
            cost += costs["false_negative"] if truth == 1 else 0.0
        else:
            cost += costs["abstain"]
        per_case.append({"case_id": row["case_id"], "scene_id": row["scene_id"], "cost": cost})
    n = len(rows)
    return {
        "risk": sum(row["cost"] for row in per_case) / n,
        "coverage": answered / n,
        "answered_accuracy": correct / answered if answered else 0.0,
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "present_recall": recall / positives if positives else 0.0,
        "per_case": per_case,
    }


def candidates(rows: list[dict[str, Any]], costs: dict[str, float]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["score"], row["case_id"]))
    out = []
    for _a, _b, no_threshold, yes_threshold in cuts(rows):
        metric = evaluate(ordered, (no_threshold, yes_threshold), costs)
        out.append({"threshold": (no_threshold, yes_threshold), **{k: v for k, v in metric.items() if k != "per_case"}})
    return out


def choose_minrisk(options: list[dict[str, Any]]) -> dict[str, Any]:
    return min(options, key=lambda x: (round(x["risk"], 12), -round(x["coverage"], 12), round(x["false_positive_rate"], 12), str(x["threshold"])))


def choose_coverage(options: list[dict[str, Any]], target: float) -> dict[str, Any]:
    return min(options, key=lambda x: (round(abs(x["coverage"] - target), 12), round(x["risk"], 12), round(x["false_positive_rate"], 12), str(x["threshold"])))


def auc(points: list[dict[str, float]]) -> float:
    ordered = sorted(points, key=lambda x: x["target_coverage"])
    return sum((b["target_coverage"] - a["target_coverage"]) * (a["risk"] + b["risk"]) / 2 for a, b in zip(ordered, ordered[1:])) / (ordered[-1]["target_coverage"] - ordered[0]["target_coverage"])


def bootstrap(per_case_delta: dict[str, tuple[str, float]], samples: int, seed: int) -> dict[str, Any]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for _case, (scene, delta) in per_case_delta.items():
        by_scene[scene].append(delta)
    scenes = sorted(by_scene); rng = random.Random(seed); draws = []
    for _ in range(samples):
        selected = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        values = [value for scene in selected for value in by_scene[scene]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {"mean": sum(v for values in by_scene.values() for v in values) / sum(map(len, by_scene.values())),
            "ci95": [draws[int(.025 * samples)], draws[int(.975 * samples)]],
            "probability_positive": sum(v > 0 for v in draws) / samples,
            "scenes": len(scenes)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen HM3D scene-level score streams; this command does not generate model scores or read raw scans.",
        epilog=("JSONL formats: canonical rows require case_id, scene_id, gt_present, method, and support_count for "
                "safevantage_full or confidence for conceptgraphs_tau1/vlmaps_tau1. Caption rows require case_id and score; "
                "scene_id and gt_present may be omitted. See data/schemas/hm3d_*.schema.json and "
                "data/analysis_input_guide.json."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--canonical", type=Path, required=True, help="JSONL frozen SafeVantage, ConceptGraphs, and VLMaps scores")
    parser.add_argument("--caption-rows", type=Path, required=True, help="JSONL frozen caption-RAG scores")
    parser.add_argument("--config", type=Path, required=True, help="Frozen HM3D multi-split JSON configuration")
    parser.add_argument("--output", type=Path, required=True, help="Path for the resulting JSON summary")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")); costs = cfg["costs"]
    data = score_rows(read_jsonl(args.canonical, "canonical"), read_jsonl(args.caption_rows, "caption"))
    scenes = sorted({row["scene_id"] for row in data["SafeVantage"].values()}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    folds = [scenes[i::cfg["scene_folds"]] for i in range(cfg["scene_folds"])]
    if any(len(fold) != cfg["calibration_scenes_per_fold"] for fold in folds):
        raise ValueError(f"unexpected fold sizes: {[len(fold) for fold in folds]}")
    result: dict[str, Any] = {"schema": cfg["schema"], "data_status": cfg["data_status"], "fresh_test_touched": False,
                              "matched_cases": len(data["SafeVantage"]), "scenes": len(scenes), "folds": [], "methods": {}}
    repeated_costs: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in data}
    repeated_matched: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in data}
    for fold_index, calibration_scenes in enumerate(folds):
        calibration_set = set(calibration_scenes); fold_result = {"fold": fold_index, "calibration_scenes": calibration_scenes, "methods": {}}
        safe_target_coverage = None
        cached: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for name, mapping in data.items():
            cal = [row for row in mapping.values() if row["scene_id"] in calibration_set]
            test = [row for row in mapping.values() if row["scene_id"] not in calibration_set]
            opts = candidates(cal, costs); chosen = choose_minrisk(opts)
            if name == "SafeVantage": safe_target_coverage = chosen["coverage"]
            cached[name] = (cal, test, opts)
            metric = evaluate(test, tuple(chosen["threshold"]), costs)
            for row in metric["per_case"]: repeated_costs[name][row["case_id"]].append(row["cost"])
            curve = []
            for target in cfg["coverage_grid"]:
                selected = choose_coverage(opts, float(target)); test_point = evaluate(test, tuple(selected["threshold"]), costs)
                curve.append({"target_coverage": float(target), "calibration_coverage": selected["coverage"], "risk": test_point["risk"], "coverage": test_point["coverage"]})
            fold_result["methods"][name] = {"selected": {k: v for k, v in metric.items() if k != "per_case"}, "threshold": chosen["threshold"], "aurc": auc(curve), "curve": curve}
        assert safe_target_coverage is not None
        for name, (_cal, test, opts) in cached.items():
            selected = choose_coverage(opts, safe_target_coverage); metric = evaluate(test, tuple(selected["threshold"]), costs)
            for row in metric["per_case"]: repeated_matched[name][row["case_id"]].append(row["cost"])
            fold_result["methods"][name]["matched_coverage"] = {k: v for k, v in metric.items() if k != "per_case"}
            fold_result["methods"][name]["matched_threshold"] = selected["threshold"]
        result["folds"].append(fold_result)
    for name in data:
        selected_risks = [fold["methods"][name]["selected"]["risk"] for fold in result["folds"]]
        matched_risks = [fold["methods"][name]["matched_coverage"]["risk"] for fold in result["folds"]]
        aurcs = [fold["methods"][name]["aurc"] for fold in result["folds"]]
        result["methods"][name] = {"selected_risk_mean": sum(selected_risks) / len(selected_risks), "selected_risk_range": [min(selected_risks), max(selected_risks)],
                                    "matched_coverage_risk_mean": sum(matched_risks) / len(matched_risks), "aurc_mean": sum(aurcs) / len(aurcs), "aurc_range": [min(aurcs), max(aurcs)]}
    comparisons = {}
    for index, baseline in enumerate(name for name in data if name != "SafeVantage"):
        deltas = {}
        for case_id, safe_values in repeated_costs["SafeVantage"].items():
            base_values = repeated_costs[baseline][case_id]
            deltas[case_id] = (data["SafeVantage"][case_id]["scene_id"], sum(base_values) / len(base_values) - sum(safe_values) / len(safe_values))
        matched_deltas = {}
        for case_id, safe_values in repeated_matched["SafeVantage"].items():
            base_values = repeated_matched[baseline][case_id]
            matched_deltas[case_id] = (data["SafeVantage"][case_id]["scene_id"], sum(base_values) / len(base_values) - sum(safe_values) / len(safe_values))
        comparisons[baseline] = {"selected": bootstrap(deltas, cfg["bootstrap_samples"], cfg["bootstrap_seed"] + index),
                                 "matched_coverage": bootstrap(matched_deltas, cfg["bootstrap_samples"], cfg["bootstrap_seed"] + 100 + index)}
    result["comparisons_vs_safevantage"] = comparisons
    result["gates"] = {
        "lower_selected_risk_all": all(result["methods"]["SafeVantage"]["selected_risk_mean"] < result["methods"][b]["selected_risk_mean"] for b in comparisons),
        "lower_matched_coverage_risk_all": all(result["methods"]["SafeVantage"]["matched_coverage_risk_mean"] < result["methods"][b]["matched_coverage_risk_mean"] for b in comparisons),
        "selected_ci_above_zero_all": all(v["selected"]["ci95"][0] > 0 for v in comparisons.values()),
        "lower_aurc_all": all(result["methods"]["SafeVantage"]["aurc_mean"] < result["methods"][b]["aurc_mean"] for b in comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"methods": result["methods"], "comparisons": comparisons, "gates": result["gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
