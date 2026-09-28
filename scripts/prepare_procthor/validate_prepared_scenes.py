#!/usr/bin/env python3
"""Check that prepared scene records match the SafeVantage evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(manifest_path: Path, scene_root: Path, split: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["splits"][split]
    errors: list[str] = []
    completed = 0
    ineligible = 0
    for entry in expected:
        scene_id = int(entry["scene_id"])
        folder = scene_root / split / f"scene_{scene_id:05d}"
        if (folder / "INELIGIBLE.json").is_file():
            ineligible += 1
            continue
        try:
            complete = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
            public = json.loads((folder / "public.json").read_text(encoding="utf-8"))
            hidden = json.loads((folder / "hidden.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"scene {scene_id}: cannot read completed records ({exc})")
            continue
        checks = {
            "completion id": complete.get("scene_id") == scene_id,
            "completion split": complete.get("split") == split,
            "public schema": public.get("schema") == "safevantage_public_scene_v1",
            "hidden schema": hidden.get("schema") == "safevantage_hidden_scene_v1",
            "scene ids": public.get("scene_id") == scene_id and hidden.get("scene_id") == scene_id,
            "public split": public.get("split") == split,
            "public views": len(public.get("views", [])) == 160,
            "hidden views": len(hidden.get("views", [])) == 160,
            "candidate positions": complete.get("positions") == 40,
            "start positions": len(public.get("start_position_ids", [])) == 4,
            "vocabulary labels": isinstance(hidden.get("vocabulary_presence"), dict),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"scene {scene_id}: " + ", ".join(failed))
        else:
            completed += 1

    report = {
        "split": split,
        "manifest_scenes": len(expected),
        "completed_scenes": completed,
        "ineligible_scenes": ineligible,
        "errors": errors,
        "passed": not errors and completed + ineligible == len(expected),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.manifest, args.scene_root, args.split)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
