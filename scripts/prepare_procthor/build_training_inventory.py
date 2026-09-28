#!/usr/bin/env python3
"""Select the semantic vocabulary from frozen training scenes only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


VERSIONS = {
    "train": "20251122_with_occupancy",
    "val": "20251217_with_occupancy",
    "test": "20251121_with_occupancy",
}
STRUCTURAL = {
    "world", "room", "wall", "floor", "ceiling", "doorway", "doorframe",
    "light", "camera", "visual", "collision",
}


def normalize_body_name(name: str) -> str | None:
    token = name.split("|", 1)[0].strip()
    # ProcTHOR generated bodies use
    # ``<semantic-type> <stable-asset-uuid> <articulation flags...>``.
    # Only the first field is a category; the remaining fields identify an
    # instance/configuration and must never enter the vocabulary.
    token = token.split(maxsplit=1)[0] if token else ""
    token = re.sub(r"(?:[_-][0-9a-f]{8,}|[_-]\d+)$", "", token, flags=re.I)
    token = token.replace("_", " ").replace("-", " ").strip().lower()
    token = re.sub(r"\s+", " ", token)
    if not token or token in STRUCTURAL or token.startswith(("room ", "wall ", "world ")):
        return None
    if len(token) < 3 or not any(character.isalpha() for character in token):
        return None
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--vocabulary-size", type=int, default=24)
    args = parser.parse_args()

    os.environ["MLSPACES_CACHE_DIR"] = str(args.cache)
    os.environ["MLSPACES_ASSETS_DIR"] = str(args.assets)
    import molmo_spaces.molmo_spaces_constants as constants
    from molmo_spaces.utils.lazy_loading_utils import install_scene_from_path

    constants.DATA_TYPE_TO_SOURCE_TO_VERSION = {
        "scenes": {
            "procthor-10k-train": VERSIONS["train"],
            "procthor-10k-val": VERSIONS["val"],
            "procthor-10k-test": VERSIONS["test"],
        },
        "objects": {"thor": "20251117"},
    }

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenes = manifest["splits"]["train"]
    scene_rows = []
    prevalence: Counter[str] = Counter()
    raw_examples: dict[str, set[str]] = {}
    failures = []
    for index, scene in enumerate(scenes):
        try:
            scene_path = args.assets / scene["scene_relpath"]
            install_scene_from_path(scene_path)
            root = ET.parse(scene_path).getroot()
            categories = set()
            for body in root.iter("body"):
                raw = body.attrib.get("name", "")
                category = normalize_body_name(raw)
                if category is None:
                    continue
                categories.add(category)
                raw_examples.setdefault(category, set()).add(raw)
            for category in categories:
                prevalence[category] += 1
            scene_rows.append({"scene_id": scene["scene_id"], "categories": sorted(categories)})
        except Exception as error:  # recorded, never outcome-conditioned replacement
            failures.append({"scene_id": scene["scene_id"], "error_type": type(error).__name__})
        if (index + 1) % 32 == 0:
            print(f"indexed {index + 1}/{len(scenes)} training scenes", flush=True)

    denominator = len(scene_rows)
    eligible = sorted(
        [
            category
            for category, count in prevalence.items()
            if 0.10 <= count / denominator <= 0.90
        ],
            key=lambda category: hashlib.sha256(
            f"fresh-msproc-active-v0.1::vocabulary::{category}".encode()
        ).hexdigest(),
    )
    vocabulary = eligible[: args.vocabulary_size]
    payload = {
        "schema": "safevantage_training_vocabulary_v1",
        "training_scene_ids": [int(scene["scene_id"]) for scene in scenes],
        "source_split": "train",
        "attempted_scenes": len(scenes),
        "valid_scenes": denominator,
        "failures": failures,
        "prevalence": dict(sorted(prevalence.items())),
        "raw_examples": {key: sorted(value)[:8] for key, value in sorted(raw_examples.items())},
        "eligible_categories": eligible,
        "vocabulary": vocabulary,
        "scenes": scene_rows,
        "admissible": denominator > 0 and len(eligible) >= args.vocabulary_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "attempted": len(scenes),
                "valid": denominator,
                "failures": len(failures),
                "eligible": len(eligible),
                "vocabulary": vocabulary,
            },
            indent=2,
        )
    )
    if not payload["admissible"]:
        raise RuntimeError(
            f"training vocabulary inadmissible: valid={denominator}, "
            f"eligible={len(eligible)}, required={args.vocabulary_size}"
        )


if __name__ == "__main__":
    main()
