#!/usr/bin/env python3
"""Freeze official ProcTHOR IDs and materialize one held-out scene locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

NAMESPACE = "fresh-msproc-active-v0.1"
VERSIONS = {
    "train": "20251122_with_occupancy",
    "val": "20251217_with_occupancy",
    "test": "20251121_with_occupancy",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def portable_relpath(path: Path, assets_root: Path) -> str:
    """Serialize a resource path without retaining node-local cache prefixes."""
    return str(Path(path).resolve().relative_to(assets_root.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True,
                        help="MolmoSpaces checkout at the pinned revision")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()

    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.repo.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if actual_commit != config["molmospaces_commit"]:
        raise RuntimeError(
            f"MolmoSpaces checkout is at {actual_commit}; "
            f"expected {config['molmospaces_commit']}"
        )
    sys.path.insert(0, str(args.repo.resolve()))

    os.environ["MLSPACES_CACHE_DIR"] = str(args.cache)
    os.environ["MLSPACES_ASSETS_DIR"] = str(args.assets)

    import molmo_spaces.molmo_spaces_constants as constants

    constants.DATA_TYPE_TO_SOURCE_TO_VERSION = {
        "scenes": {
            "procthor-10k-train": VERSIONS["train"],
            "procthor-10k-val": VERSIONS["val"],
            "procthor-10k-test": VERSIONS["test"],
        },
        "objects": {"thor": "20251117"},
    }

    split_specs = {
        "train": ("train", int(config["scene_counts"]["train"])),
        "validation": ("val", int(config["scene_counts"]["validation"])),
        "test": ("test", int(config["scene_counts"]["test"])),
    }
    selected: dict[str, list[dict[str, object]]] = {}
    available_counts: dict[str, int] = {}

    for output_name, (split, count) in split_specs.items():
        scenes, version = constants.get_scenes(
            "procthor-10k", split, return_version=True
        )
        available = []
        for scene_id, variants in scenes[split].items():
            if not isinstance(variants, dict) or variants.get("base") is None:
                continue
            available.append((int(scene_id), variants))
        available_counts[output_name] = len(available)
        ordered = sorted(
            available,
            key=lambda item: digest(f"{NAMESPACE}::{split}::{item[0]}"),
        )
        if len(ordered) < count:
            raise RuntimeError(
                f"{split} exposes {len(ordered)} official scenes, need {count}"
            )
        selected[output_name] = [
            {
                "scene_id": scene_id,
                "scene_relpath": portable_relpath(Path(variants["base"]), args.assets),
                "ceiling_relpath": portable_relpath(Path(variants["ceiling"]), args.assets),
                "map_relpath": portable_relpath(Path(variants["map"]), args.assets),
                "selection_hash": digest(
                    f"{NAMESPACE}::{split}::{scene_id}"
                ),
                "resource_version": version,
            }
            for scene_id, variants in ordered[:count]
        ]

    manifest = {
        "schema": "safevantage_scene_manifest_v1",
        "selection_seed": NAMESPACE,
        "molmospaces_commit": config["molmospaces_commit"],
        "resource_versions": VERSIONS,
        "available_scene_counts": available_counts,
        "splits": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Materialize exactly one validation bundle as an infrastructure smoke test.
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    smoke = selected["validation"][0]
    # This navigation benchmark never evaluates manipulation or grasping.  Fetch
    # the scene and every referenced visual object, but deliberately request no
    # grasp packages; keeping them out of the frozen smoke path prevents an
    # irrelevant manipulation dependency from changing benchmark availability.
    install_scene_with_objects_and_grasps_from_path(
        args.assets / str(smoke["scene_relpath"]),
        grasp_sources=(),
        exclude_thor=False,
    )
    checked = {}
    for key in ("scene_relpath", "ceiling_relpath", "map_relpath"):
        path = args.assets / str(smoke[key])
        checked[key] = {"exists": path.exists(), "bytes": path.stat().st_size}
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"smoke asset missing after install: {path}")

    print(
        json.dumps(
            {
                "molmospaces_commit": manifest["molmospaces_commit"],
                "available_scene_counts": available_counts,
                "selected_counts": {k: len(v) for k, v in selected.items()},
                "smoke_scene": smoke["scene_id"],
                "smoke_bundle": checked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
