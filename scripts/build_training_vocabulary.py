#!/usr/bin/env python3
"""List the categories represented in the training scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_root = args.scene_root / "train"
    scene_dirs = [
        path for path in sorted(train_root.glob("scene_*"))
        if path.is_dir() and (path / "COMPLETE.json").is_file()
    ]
    if not scene_dirs:
        parser.error(f"no completed training scenes found under {train_root}")

    scenes = []
    vocabulary: set[str] = set()
    for scene_dir in scene_dirs:
        path = scene_dir / "hidden.json"
        if not path.is_file():
            parser.error(f"training scene has no hidden.json: {scene_dir}")
        try:
            hidden = json.loads(path.read_text(encoding="utf-8"))
            presence = hidden["vocabulary_presence"]
            if not isinstance(presence, dict) or not all(isinstance(value, bool) for value in presence.values()):
                raise ValueError("vocabulary_presence must map categories to booleans")
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid training-scene vocabulary in {path}: {exc}")
        categories = sorted(str(category) for category, present in presence.items() if present)
        vocabulary.update(categories)
        scenes.append({"categories": categories})

    result = {
        "schema": "safevantage_training_vocabulary_v1",
        "fit_split": "train",
        "scene_count": len(scenes),
        "vocabulary": sorted(vocabulary),
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote training-only vocabulary with {len(vocabulary)} categories from {len(scenes)} scenes")


if __name__ == "__main__":
    main()
