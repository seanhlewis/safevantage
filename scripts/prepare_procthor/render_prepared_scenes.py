#!/usr/bin/env python3
"""Render public RGB-D action graphs and separate hidden semantic labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

VERSIONS = {
    "train": "20251122_with_occupancy",
    "val": "20251217_with_occupancy",
    "test": "20251121_with_occupancy",
}
FORBIDDEN_PUBLIC_KEYS = {"visible_pixels", "visible_fraction", "vocabulary_presence"}
STRUCTURAL = {
    "world", "room", "wall", "floor", "ceiling", "doorway", "doorframe",
    "light", "camera", "visual", "collision",
}


def camera_quaternion(mujoco, position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a MuJoCo camera rotation that faces the target."""
    direction = target - position
    direction /= np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(direction, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    up /= np.linalg.norm(up)
    rotation = np.column_stack([right, up, -direction])
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion


def normalize_body_name(name: str) -> str | None:
    """Return the semantic field of a ProcTHOR body name, never its UUID."""
    name = str(name)
    token = name.split("|", 1)[0].strip()
    # MjSpec.compile() serializes ProcTHOR's space-delimited instance fields as
    # underscores, e.g. ``chair_<uuid>_1_0``.  Strip the UUID and every trailing
    # articulation index in one operation before canonicalizing the category.
    token = re.sub(r"[_-][0-9a-f]{16,}(?:[_-]\d+)*$", "", token, flags=re.I)
    token = token.split(maxsplit=1)[0] if token else ""
    token = re.sub(r"(?:[_-]\d+)+$", "", token)
    token = re.sub(r"[^a-z0-9]+", "", token.lower())
    if not token or token in STRUCTURAL:
        return None
    if len(token) < 3 or not any(character.isalpha() for character in token):
        return None
    return token


def assert_public_no_hidden(value: object, path: str = "public") -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if leaked:
            raise RuntimeError(f"hidden key leaked into {path}: {sorted(leaked)}")
        for key, child in value.items():
            assert_public_no_hidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_public_no_hidden(child, f"{path}[{index}]")


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        raise RuntimeError("occupancy map has no free connected component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def candidate_positions(
    occupancy: np.ndarray, scene_id: int, count: int = 40, downsample: int = 5
) -> tuple[np.ndarray, np.ndarray, int]:
    coarse = occupancy[::downsample, ::downsample]
    component = largest_component(coarse)
    free = np.argwhere(component)
    if len(free) < count:
        raise RuntimeError(f"largest component has {len(free)} cells, need {count}")
    seed = int(hashlib.sha256(f"fresh-msproc-active-v0.1::poses::{scene_id}".encode()).hexdigest()[:16], 16)
    selected = [seed % len(free)]
    min_sq = ((free - free[selected[0]]) ** 2).sum(axis=1).astype(float)
    for _ in range(1, count):
        nxt = int(np.argmax(min_sq))
        selected.append(nxt)
        min_sq = np.minimum(min_sq, ((free - free[nxt]) ** 2).sum(axis=1))
    coarse_px = free[selected]
    original_px = coarse_px * downsample
    return original_px, component, downsample


def geodesic_matrix(component: np.ndarray, points_px: np.ndarray, downsample: int, px_per_m: float) -> np.ndarray:
    from skimage.graph import MCP_Geometric

    coarse_points = np.rint(points_px / downsample).astype(int)
    costs_grid = np.where(component, 1.0, np.inf)
    matrix = np.full((len(coarse_points), len(coarse_points)), np.inf, dtype=np.float32)
    solver = MCP_Geometric(costs_grid, fully_connected=True)
    for index, start in enumerate(coarse_points):
        costs, _ = solver.find_costs([tuple(start)])
        matrix[index] = [float(costs[tuple(target)]) * downsample / px_per_m for target in coarse_points]
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("candidate poses are not mutually reachable")
    return matrix


def top_level_categories(model, mujoco) -> tuple[dict[int, str], set[str]]:
    body_category: dict[int, str] = {}
    categories: set[str] = set()
    for body_id in range(1, model.nbody):
        current = body_id
        while current > 0:
            category = normalize_body_name(model.body(current).name or "")
            if category is not None:
                body_category[body_id] = category
                categories.add(category)
                break
            current = int(model.body_parentid[current])
    geom_category = {
        geom_id: body_category.get(int(model.geom_bodyid[geom_id]), "")
        for geom_id in range(model.ngeom)
    }
    return geom_category, categories


def raycast_visible_pixels(
    model,
    data,
    camera_name: str,
    geom_category: dict[int, str],
    vocabulary: set[str],
    mujoco,
    output_width: int,
    output_height: int,
    sample_width: int = 80,
    sample_height: int = 60,
) -> Counter[str]:
    """Estimate category visibility on a deterministic pixel-center ray lattice.

    This avoids MuJoCo's segmentation framebuffer, which loses its EGL context
    for a reproducible subset of geometry-heavy ProcTHOR scenes.  RGB and depth
    remain rendered at full resolution; the hidden visibility oracle uses an
    80x60 lattice and reports equivalent 320x240 pixel-area counts.
    """
    camera_id = int(model.camera(camera_name).id)
    position = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    right = rotation[:, 0]
    up = rotation[:, 1]
    forward = -rotation[:, 2]
    fovy = math.radians(float(model.cam_fovy[camera_id]))
    half_height = math.tan(fovy / 2.0)
    half_width = half_height * (output_width / float(output_height))
    geom_group = np.ones(6, dtype=np.uint8)
    scale = (output_width * output_height) / float(sample_width * sample_height)
    directions = np.empty((sample_width * sample_height, 3), dtype=np.float64)
    ray_index = 0
    for row in range(sample_height):
        y = (1.0 - 2.0 * ((row + 0.5) / sample_height)) * half_height
        for col in range(sample_width):
            x = (2.0 * ((col + 0.5) / sample_width) - 1.0) * half_width
            direction = forward + x * right + y * up
            direction /= np.linalg.norm(direction)
            directions[ray_index] = direction
            ray_index += 1

    hit_geom_ids = np.full(len(directions), -1, dtype=np.int32)
    distances = np.full(len(directions), -1.0, dtype=np.float64)
    used_multi_ray = False
    if hasattr(mujoco, "mj_multiRay"):
        for vectors in (directions, directions.reshape(-1)):
            try:
                mujoco.mj_multiRay(
                    model,
                    data,
                    position,
                    vectors,
                    geom_group,
                    1,
                    -1,
                    hit_geom_ids,
                    distances,
                    len(directions),
                    1.0e6,
                )
                used_multi_ray = True
                break
            except (TypeError, ValueError):
                continue
    if not used_multi_ray:
        geom_id = np.full(1, -1, dtype=np.int32)
        for index, direction in enumerate(directions):
            geom_id[0] = -1
            mujoco.mj_ray(model, data, position, direction, geom_group, 1, -1, geom_id)
            hit_geom_ids[index] = geom_id[0]

    pixels: Counter[str] = Counter()
    for geom_id in hit_geom_ids:
        category = geom_category.get(int(geom_id), "")
        if category in vocabulary:
            pixels[category] += scale
    return Counter({key: int(round(value)) for key, value in pixels.items()})


def render_scene(args, scene: dict, vocabulary: list[str], installer, mujoco) -> dict:
    scene_id = int(scene["scene_id"])
    scene_output = args.output_dir / args.split / f"scene_{scene_id:05d}"
    scene_output.mkdir(parents=True, exist_ok=True)
    complete = scene_output / "COMPLETE.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))

    scene_path = args.assets / scene["scene_relpath"]
    ceiling_path = args.assets / scene["ceiling_relpath"]
    map_path = args.assets / scene["map_relpath"]
    installer(ceiling_path, grasp_sources=(), exclude_thor=False)

    map_image = Image.open(map_path)
    map_rgb = np.asarray(map_image.convert("RGB"))
    occupancy = map_rgb[..., 0] > 0
    map_to_world = np.asarray(json.loads(map_image.info["map_to_world"]), dtype=float)
    px_per_m = float(json.loads(map_image.info["px_per_m"]))
    positions_px, component, downsample = candidate_positions(occupancy, scene_id)
    homogeneous = np.column_stack([positions_px, np.ones(len(positions_px))])
    positions_world = homogeneous @ map_to_world.T
    geodesics = geodesic_matrix(component, positions_px, downsample, px_per_m)

    phase = int(hashlib.sha256(f"fresh-msproc-active-v0.1::yaw::{scene_id}".encode()).hexdigest()[:8], 16) % 4
    yaws = [((phase + offset) % 4) * math.pi / 2 for offset in range(4)]
    spec = mujoco.MjSpec.from_file(str(ceiling_path))
    views = []
    for position_id, world in enumerate(positions_world):
        for yaw_id, yaw in enumerate(yaws):
            view_id = position_id * len(yaws) + yaw_id
            pos = np.array([world[0], world[1], 1.25])
            target = pos + np.array([math.cos(yaw) * 3.0, math.sin(yaw) * 3.0, -0.12])
            name = f"sv_view_{view_id:03d}"
            spec.worldbody.add_camera(
                name=name,
                pos=pos.tolist(),
                quat=camera_quaternion(mujoco, pos, target).tolist(),
                fovy=70.0,
            )
            views.append(
                {
                    "view_id": view_id,
                    "camera": name,
                    "position_id": position_id,
                    "position_px": positions_px[position_id].tolist(),
                    "position_world": positions_world[position_id].tolist(),
                    "yaw_rad": yaw,
                    "room_id": int(map_rgb[positions_px[position_id, 0], positions_px[position_id, 1], 2]),
                }
            )

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_category, scene_categories = top_level_categories(model, mujoco)
    matched_categories = sorted(scene_categories.intersection(vocabulary))
    print(
        f"scene {scene_id}: {len(scene_categories)} categories; "
        f"{len(matched_categories)} match the frozen vocabulary",
        flush=True,
    )
    if not matched_categories:
        if args.mark_zero_vocabulary_ineligible:
            marker = scene_output / "INELIGIBLE.json"
            record = {
                "schema": "safevantage_locked_scene_ineligible_v1",
                "scene_id": scene_id,
                "split": args.split,
                "reason": "zero_matched_frozen_vocabulary_categories",
                "frozen_vocabulary_size": len(vocabulary),
                "normalized_scene_categories": sorted(scene_categories),
            }
            marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return record
        raise RuntimeError("compiled scene has zero matched vocabulary categories")
    segmentation_pixels = [
        raycast_visible_pixels(
            model,
            data,
            view["camera"],
            geom_category,
            set(vocabulary),
            mujoco,
            args.width,
            args.height,
        )
        for view in views
    ]

    hidden_views = [
        {
            "view_id": view["view_id"],
            "visible_pixels": dict(sorted(pixels.items())),
            "visible_fraction": {
                key: value / float(args.width * args.height) for key, value in sorted(pixels.items())
            },
        }
        for view, pixels in zip(views, segmentation_pixels)
    ]
    public_views = [dict(view) for view in views]
    depth_path = None
    if not args.controlled_only:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        rgb_dir = scene_output / "rgb"
        rgb_dir.mkdir(exist_ok=True)
        depth_frames = []
        public_views = []
        for view in views:
            renderer.update_scene(data, camera=view["camera"])
            rgb = renderer.render().copy()
            rgb_path = rgb_dir / f"view_{view['view_id']:03d}.jpg"
            Image.fromarray(rgb).save(rgb_path, quality=92, subsampling=0)

            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=view["camera"])
            depth_frames.append(renderer.render().astype(np.float16))
            renderer.disable_depth_rendering()
            public_views.append({**view, "rgb": str(rgb_path.relative_to(scene_output))})
        renderer.close()
        depth_path = scene_output / "depth_float16.npz"
        np.savez_compressed(depth_path, depth=np.stack(depth_frames))
    map_output = scene_output / "occupancy_map.png"
    map_output.unlink(missing_ok=True)
    shutil.copyfile(map_path, map_output)
    starts = sorted(
        range(len(positions_world)),
        key=lambda idx: hashlib.sha256(f"fresh-msproc-active-v0.1::start::{scene_id}::{idx}".encode()).hexdigest(),
    )[:4]
    public = {
        "schema": "safevantage_public_scene_v1",
        "scene_id": scene_id,
        "split": args.split,
        "views": public_views,
        "occupancy_map": "occupancy_map.png",
        "geodesic_position_cost_m": geodesics.tolist(),
        "start_position_ids": starts,
        "render": {"width": args.width, "height": args.height},
        "controlled_only": bool(args.controlled_only),
    }
    if depth_path is not None:
        public["depth"] = depth_path.name
    hidden = {
        "schema": "safevantage_hidden_scene_v1",
        "scene_id": scene_id,
        "vocabulary_presence": {category: category in scene_categories for category in vocabulary},
        "views": hidden_views,
    }
    assert_public_no_hidden(public)
    public_path = scene_output / "public.json"
    hidden_path = scene_output / "hidden.json"
    public_path.write_text(json.dumps(public, sort_keys=True) + "\n", encoding="utf-8")
    hidden_path.write_text(json.dumps(hidden, sort_keys=True) + "\n", encoding="utf-8")
    record = {
        "scene_id": scene_id,
        "split": args.split,
        "views": len(public_views),
        "positions": len(positions_world),
        "present_vocabulary_categories": sum(hidden["vocabulary_presence"].values()),
        "controlled_only": bool(args.controlled_only),
    }
    complete.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--controlled-only",
        action="store_true",
        help="Write geometry and hidden raycast labels without RGB-D; for train/validation policy iteration only.",
    )
    parser.add_argument(
        "--mark-zero-vocabulary-ineligible",
        action="store_true",
        help="Record a deterministic data-eligibility exclusion instead of aborting a locked render batch.",
    )
    args = parser.parse_args()

    if normalize_body_name("mug 8caf1bb3f88e9a00e02dfe9e6518aeb0 1 0") != "mug":
        raise RuntimeError("renderer semantic normalizer failed its startup invariant")

    os.environ["MLSPACES_CACHE_DIR"] = str(args.cache)
    os.environ["MLSPACES_ASSETS_DIR"] = str(args.assets)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import molmo_spaces.molmo_spaces_constants as constants
    from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

    constants.DATA_TYPE_TO_SOURCE_TO_VERSION = {
        "scenes": {
            "procthor-10k-train": VERSIONS["train"],
            "procthor-10k-val": VERSIONS["val"],
            "procthor-10k-test": VERSIONS["test"],
        },
        "objects": {"thor": "20251117"},
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    vocabulary_payload = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    expected_train_ids = [int(row["scene_id"]) for row in manifest["splits"]["train"]]
    vocabulary_train_ids = [int(row["scene_id"]) for row in vocabulary_payload["scenes"]]
    if vocabulary_train_ids != expected_train_ids:
        raise RuntimeError("training vocabulary does not match the manifest's training scenes")
    vocabulary = list(vocabulary_payload["vocabulary"])
    if len(vocabulary) != 24:
        raise RuntimeError(f"expected 24 frozen categories, got {len(vocabulary)}")
    scenes = manifest["splits"][args.split]
    stop = len(scenes) if args.stop_index is None else min(args.stop_index, len(scenes))
    records = []
    for scene in scenes[args.start_index:stop]:
        records.append(
            render_scene(
                args,
                scene,
                vocabulary,
                install_scene_with_objects_and_grasps_from_path,
                mujoco,
            )
        )
        print(json.dumps(records[-1]), flush=True)
    print(json.dumps({"split": args.split, "rendered": len(records), "records": records}, indent=2))


if __name__ == "__main__":
    main()
