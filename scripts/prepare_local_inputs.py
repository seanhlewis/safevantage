#!/usr/bin/env python3
"""Stage separately obtained benchmark inputs under the ignored local data tree.

This utility never downloads third-party content and never changes its source.
Directory links are the default to avoid duplicating large assets; use --mode
copy to make a separate local copy when required by the filesystem or workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def validate_procthor(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_root = root / split
        scenes = sorted(path for path in split_root.glob("scene_*") if path.is_dir())
        if not scenes:
            raise ValueError(f"no scene_* directories found under {split_root}")
        eligible = []
        for scene in scenes:
            if (scene / "INELIGIBLE.json").is_file():
                continue
            required = [scene / "COMPLETE.json", scene / "public.json", scene / "hidden.json"]
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise ValueError(f"{scene}: missing {', '.join(missing)}")
            eligible.append(scene)
        if not eligible:
            raise ValueError(f"no completed eligible scenes found under {split_root}")
        counts[split] = len(eligible)
        # Check the first scene in each split against the evaluator interface.
        public = json.loads((eligible[0] / "public.json").read_text(encoding="utf-8"))
        hidden = json.loads((eligible[0] / "hidden.json").read_text(encoding="utf-8"))
        if not isinstance(public.get("start_position_ids"), list) or not isinstance(public.get("views"), list):
            raise ValueError(f"{eligible[0]}: public.json lacks start_position_ids or views arrays")
        if not isinstance(hidden.get("scene_id"), int) or not isinstance(hidden.get("views"), list):
            raise ValueError(f"{eligible[0]}: hidden.json lacks integer scene_id or views array")
        if not isinstance(hidden.get("vocabulary_presence"), dict):
            raise ValueError(f"{eligible[0]}: hidden.json lacks vocabulary_presence object")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("procthor", "hm3d"), required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory already obtained under the upstream dataset terms")
    parser.add_argument("--destination", type=Path,
                        help="defaults to data/local_inputs/<dataset>")
    parser.add_argument("--mode", choices=("link", "copy"), default="link")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    counts = validate_procthor(source) if args.dataset == "procthor" else None
    if args.dataset == "hm3d" and not any(source.iterdir()):
        parser.error(f"HM3D source directory is empty: {source}")

    default_destination = Path(__file__).resolve().parents[1] / "data" / "local_inputs" / args.dataset
    destination = (args.destination or default_destination).expanduser().resolve()
    if destination == source or source in destination.parents:
        parser.error("destination must not be the source directory or a folder inside it")
    if destination.exists() or destination.is_symlink():
        parser.error(f"destination already exists; choose an unused path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "link":
        try:
            os.symlink(source, destination, target_is_directory=True)
        except OSError as exc:
            parser.error(f"could not create directory link ({exc}); rerun with --mode copy")
    else:
        shutil.copytree(source, destination, copy_function=shutil.copy2)

    manifest_dir = destination.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "source": str(source),
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "procthor_split_scene_counts": counts,
    }
    (manifest_dir / f"{args.dataset}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Staged {args.dataset} input at {destination} ({args.mode})")
    if counts:
        print("Scene directories by split: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
