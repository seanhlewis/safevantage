#!/usr/bin/env python3
"""Reduce episode predictions to paired, scene-level sufficient statistics.

The output intentionally omits object/category labels, sensor traces, view
histories, positions, and benchmark asset paths. It retains the counts and
sums needed to recompute the reported pooled metrics and paired scene
bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


COUNT_FIELDS = (
    "episodes",
    "truth_yes",
    "truth_no",
    "pred_yes_truth_yes",
    "pred_yes_truth_no",
    "pred_no_truth_yes",
    "pred_no_truth_no",
    "pred_wait_truth_yes",
    "pred_wait_truth_no",
)
SUM_FIELDS = ("risk_sum", "travel_m_sum", "actions_sum")
FIELDS = (
    "split",
    "scene_group_id",
    "action_budget",
    "method",
    *COUNT_FIELDS,
    *SUM_FIELDS,
)


def parse_input(value: str) -> tuple[int, Path]:
    try:
        budget_text, path_text = value.split("=", 1)
        budget = int(budget_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use ACTION_BUDGET=PATH, e.g. 8=rows8.jsonl") from exc
    return budget, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True,
                        metavar="BUDGET=JSONL", help="applied-head episode JSONL; repeat per budget")
    parser.add_argument("--output", type=Path, required=True, help="output CSV path")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    grouped: dict[tuple[int, int, str], dict[str, float | int]] = defaultdict(
        lambda: {**{field: 0 for field in COUNT_FIELDS}, **{field: 0.0 for field in SUM_FIELDS}}
    )
    scene_ids_by_budget: dict[int, set[int]] = {}
    for budget, path in args.input:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
        seen_scenes: set[int] = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    scene_id = int(row["scene_id"])
                    method = str(row["method"])
                    prediction = str(row["prediction"])
                    truth = bool(row["truth"])
                    if prediction not in {"yes", "no", "wait"}:
                        raise ValueError(f"unknown prediction {prediction!r}")
                    if int(row["action_budget"]) != budget:
                        raise ValueError("row action_budget does not match --input budget")
                    if row.get("sensor") != "controlled":
                        raise ValueError("expected the controlled-sensor evaluation rows")
                    risk = float(row["risk"])
                    travel = float(row["geodesic_m"])
                    actions = int(row["actions_taken"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    parser.error(f"{path}:{line_number}: invalid applied-head row: {exc}")

                seen_scenes.add(scene_id)
                values = grouped[(budget, scene_id, method)]
                values["episodes"] += 1
                values["truth_yes" if truth else "truth_no"] += 1
                values[f"pred_{prediction}_truth_{'yes' if truth else 'no'}"] += 1
                values["risk_sum"] += risk
                values["travel_m_sum"] += travel
                values["actions_sum"] += actions

        previous = scene_ids_by_budget.setdefault(budget, seen_scenes)
        if previous != seen_scenes:
            parser.error(f"scene set differs between multiple inputs for action budget {budget}")

    if not grouped:
        parser.error("no episode rows found")
    budget_scene_sets = [scene_ids_by_budget[b] for b, _ in args.input]
    if any(scene_set != budget_scene_sets[0] for scene_set in budget_scene_sets[1:]):
        parser.error("action-budget inputs do not contain the same scene set")

    # Stable opaque group IDs preserve pairing while omitting benchmark IDs.
    ordered_scene_ids = sorted(budget_scene_sets[0])
    group_id = {scene_id: f"scene_{i:04d}" for i, scene_id in enumerate(ordered_scene_ids, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for (budget, scene_id, method), values in sorted(grouped.items()):
            writer.writerow({
                "split": args.split,
                "scene_group_id": group_id[scene_id],
                "action_budget": budget,
                "method": method,
                **{key: values[key] for key in COUNT_FIELDS},
                **{key: f"{values[key]:.17g}" for key in SUM_FIELDS},
            })
    print(f"Wrote {len(grouped)} scene-method-budget rows for {len(ordered_scene_ids)} scenes to {args.output}")


if __name__ == "__main__":
    main()
