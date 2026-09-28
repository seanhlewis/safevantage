#!/usr/bin/env python3
"""Matched closed-loop evaluation on rendered Fresh-MSProc scenes.

The controlled sensor is provisional during train/validation development.  It
is deterministic, and policy code receives only scores from acquired views;
simulator presence and visibility remain confined to the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Stable deterministic-seed namespace used by the archived controlled sensor.
# Keep unchanged when reproducing the published run; changing it changes the
# generated observation stream and therefore the reported metrics.
DETERMINISTIC_SENSOR_NAMESPACE = "fresh-msproc-active-v0.1"


# Related analysis harness, not the canonical SafeVantage implementation.
# It compares the SafeVantage branches with separate baselines and diagnostic
# variants. Consult docs/REPRODUCE.md; the experiment label alone does not make a
# row a paper result.
METHODS = (
    "random",
    "nearest_frontier",
    "coverage_gain",
    "category_room_prior",
    "detector_confidence_gain",
    "safevantage_active",
    "safevantage_fixed_horizon",
    "counterevidence_active",
    "safevantage_dc_full",
    "safevantage_dc_no_value",
    "safevantage_dc_no_observability",
    "safevantage_dc_no_independence",
    "safevantage_dc_no_belief",
    "oracle_visibility",
    "max_observability",
    "decision_resolve",
    "residual_safevantage_tau0",
    "residual_safevantage_tau02",
    "residual_safevantage_tau05",
    "residual_safevantage_tau10",
)


def hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def unit_uniform(text: str) -> float:
    return (hash_int(text) + 0.5) / float(2**64)


def controlled_score(scene_id: int, category: str, view_id: int, visible_fraction: float) -> float:
    """A deterministic, noisy semantic observation for policy development."""
    key = f"{DETERMINISTIC_SENSOR_NAMESPACE}::sensor::{scene_id}::{category}::{view_id}"
    u1 = max(unit_uniform(key + "::u1"), 1e-12)
    u2 = unit_uniform(key + "::u2")
    if visible_fraction >= 1e-4:
        base = 0.45 + 0.50 * (1.0 - math.exp(-80.0 * visible_fraction))
        gaussian = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        score = base + 0.08 * gaussian
        if unit_uniform(key + "::miss") < 0.08:
            score *= 0.30
    elif u1 < 0.015:
        score = 0.55 + 0.30 * u2
    else:
        score = 0.05 + 0.30 * u2
    return float(np.clip(score, 0.0, 1.0))


def hashed_order(values: list[Any], namespace: str) -> list[Any]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{namespace}::{value}".encode()).hexdigest())


def episode_queries(hidden: dict) -> list[tuple[str, bool]]:
    scene_id = int(hidden["scene_id"])
    present = [key for key, value in hidden["vocabulary_presence"].items() if value]
    absent = [key for key, value in hidden["vocabulary_presence"].items() if not value]
    if len(present) < 4 or len(absent) < 4:
        return []
    positives = hashed_order(present, f"{DETERMINISTIC_SENSOR_NAMESPACE}::positive::{scene_id}")[:4]
    negatives = hashed_order(absent, f"{DETERMINISTIC_SENSOR_NAMESPACE}::negative::{scene_id}")[:4]
    return [(category, True) for category in positives] + [(category, False) for category in negatives]


def score_views(hidden: dict, category: str) -> dict[int, float]:
    fractions = {
        int(view["view_id"]): float(view["visible_fraction"].get(category, 0.0))
        for view in hidden["views"]
    }
    return {
        view_id: controlled_score(int(hidden["scene_id"]), category, view_id, fraction)
        for view_id, fraction in fractions.items()
    }


def controlled_target_estimate(public: dict, hidden: dict, category: str) -> np.ndarray | None:
    """Estimate a latent 2-D target from the frozen multi-view ray evidence."""
    public_views = {int(view["view_id"]): view for view in public["views"]}
    constraints = []
    for row in hidden["views"]:
        weight = float(row["visible_fraction"].get(category, 0.0))
        if weight < 1e-4:
            continue
        view = public_views[int(row["view_id"])]
        point = np.asarray(view["position_world"][:2], dtype=float)
        yaw = float(view["yaw_rad"])
        direction = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=float)
        constraints.append((point, direction, max(weight, 1e-4)))
    if not constraints:
        return None
    matrix = np.zeros((2, 2), dtype=float)
    vector = np.zeros(2, dtype=float)
    identity = np.eye(2)
    for point, direction, weight in constraints:
        projector = identity - np.outer(direction, direction)
        matrix += weight * projector
        vector += weight * (projector @ point)
    if abs(float(np.linalg.det(matrix))) > 1e-8:
        return np.linalg.solve(matrix, vector)
    point, direction, _ = max(constraints, key=lambda item: item[2])
    return point + 3.0 * direction


def controlled_observations(public: dict, hidden: dict) -> dict[str, dict[int, dict]]:
    """Simulate confidence plus an acquired-view RGB-D target estimate."""
    scene_id = int(hidden["scene_id"])
    public_views = {int(view["view_id"]): view for view in public["views"]}
    hidden_views = {int(view["view_id"]): view for view in hidden["views"]}
    output = {}
    for category in hidden["vocabulary_presence"]:
        scores = score_views(hidden, category)
        latent = controlled_target_estimate(public, hidden, category)
        category_rows = {}
        for view_id, score in scores.items():
            row = {"score": score}
            if score >= 0.45:
                view = public_views[view_id]
                visible = float(hidden_views[view_id]["visible_fraction"].get(category, 0.0)) >= 1e-4
                key = f"{DETERMINISTIC_SENSOR_NAMESPACE}::target::{scene_id}::{category}::{view_id}"
                if visible and latent is not None:
                    angle = 2.0 * math.pi * unit_uniform(key + "::angle")
                    radius = 0.12 * math.sqrt(unit_uniform(key + "::radius"))
                    target = latent + radius * np.asarray([math.cos(angle), math.sin(angle)])
                else:
                    # A false positive gets a self-consistent spurious target,
                    # never the latent target of a real instance elsewhere.
                    origin = np.asarray(view["position_world"][:2], dtype=float)
                    bearing = float(view["yaw_rad"]) + (unit_uniform(key + "::bearing") - 0.5) * 0.9
                    depth = 1.5 + 3.0 * unit_uniform(key + "::depth")
                    target = origin + depth * np.asarray([math.cos(bearing), math.sin(bearing)])
                row["target_world"] = [float(target[0]), float(target[1]), 1.0]
            category_rows[view_id] = row
        output[category] = category_rows
    return output


def scene_sensor_observations(
    scene_dir: Path,
    public: dict,
    hidden: dict,
    sensor: str,
    detector_root: Path | None = None,
) -> dict[str, dict[int, dict]]:
    if sensor == "controlled":
        return controlled_observations(public, hidden)
    if sensor != "yoloworld":
        raise ValueError(f"unknown sensor: {sensor}")
    detector_path = (
        detector_root / f"{scene_dir.name}.json"
        if detector_root is not None
        else scene_dir / "yoloworld.json"
    )
    if not detector_path.exists():
        raise RuntimeError(f"missing frozen detector outputs: {detector_path}")
    payload = json.loads(detector_path.read_text(encoding="utf-8"))
    rows: dict[str, dict[int, dict]] = {category: {} for category in hidden["vocabulary_presence"]}
    for row in payload["rows"]:
        category = str(row["category"])
        if category not in rows:
            continue
        rows[category][int(row["view_id"])] = {
            "score": float(row["score"]),
            "target_world": row.get("target_world"),
            "box_xyxy": row.get("box_xyxy"),
        }
    incomplete = {category: len(values) for category, values in rows.items() if len(values) != len(hidden["views"])}
    if incomplete:
        raise RuntimeError(f"incomplete detector rows: {incomplete}")
    return rows


def cooccurrence_weights(training_vocabulary: dict) -> dict[str, dict[str, float]]:
    vocabulary = [str(value) for value in training_vocabulary["vocabulary"]]
    scenes = [set(map(str, row["categories"])) for row in training_vocabulary["scenes"]]
    weights = {}
    for target in vocabulary:
        positive = [scene for scene in scenes if target in scene]
        negative = [scene for scene in scenes if target not in scene]
        target_weights = {}
        for context in vocabulary:
            if context == target:
                continue
            p_context = (sum(context in scene for scene in positive) + 1.0) / (len(positive) + 2.0)
            n_context = (sum(context in scene for scene in negative) + 1.0) / (len(negative) + 2.0)
            target_weights[context] = float(math.log(p_context / n_context))
        weights[target] = target_weights
    return weights


def enrich_observation(view: dict, bank: dict[str, dict[int, dict]], category: str) -> dict:
    view_id = int(view["view_id"])
    context = {
        name: float(observations[view_id]["score"])
        for name, observations in bank.items()
        if name != category and float(observations[view_id]["score"]) >= 0.45
    }
    return {**view, **bank[category][view_id], "context_scores": context}


def view_lookup(public: dict) -> dict[int, dict]:
    return {int(view["view_id"]): view for view in public["views"]}


def room_geometry(public: dict) -> dict[int, np.ndarray]:
    geodesic = np.asarray(public["geodesic_position_cost_m"], dtype=float)
    finite = geodesic[np.isfinite(geodesic)]
    scale = float(np.mean(finite[finite > 0])) if np.any(finite > 0) else 1.0
    by_position = {}
    room_positions: dict[int, set[int]] = {}
    for view in public["views"]:
        position = int(view["position_id"])
        room = int(view["room_id"])
        by_position[position] = np.asarray(view["position_world"][:2], dtype=float)
        room_positions.setdefault(room, set()).add(position)
    positions = sorted(by_position)
    xy = np.asarray([by_position[position] for position in positions])
    center = np.mean(xy, axis=0)
    radius_scale = float(np.sqrt(np.mean(np.sum((xy - center) ** 2, axis=1)))) or 1.0
    output = {}
    for room, members_set in room_positions.items():
        members = sorted(members_set)
        internal = geodesic[np.ix_(members, members)]
        central = geodesic[np.ix_(members, positions)]
        centroid = np.mean(np.asarray([by_position[position] for position in members]), axis=0)
        output[room] = np.asarray([
            len(members) / len(positions),
            float(np.mean(internal[np.isfinite(internal)])) / scale,
            float(np.mean(central[np.isfinite(central)])) / scale,
            float(np.linalg.norm(centroid - center)) / radius_scale,
            len(room_positions) / 8.0,
        ])
    return output


def view_geometry(public: dict, room_features: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    geodesic = np.asarray(public["geodesic_position_cost_m"], dtype=float)
    finite = geodesic[np.isfinite(geodesic)]
    distance_scale = float(np.mean(finite[finite > 0])) if np.any(finite > 0) else 1.0
    by_position = {}
    room_positions: dict[int, set[int]] = {}
    for view in public["views"]:
        position = int(view["position_id"])
        room = int(view["room_id"])
        by_position[position] = np.asarray(view["position_world"][:2], dtype=float)
        room_positions.setdefault(room, set()).add(position)
    xy = np.asarray(list(by_position.values()))
    scene_center = np.mean(xy, axis=0)
    scene_scale = float(np.sqrt(np.mean(np.sum((xy - scene_center) ** 2, axis=1)))) or 1.0
    room_centers = {
        room: np.mean(np.asarray([by_position[position] for position in members]), axis=0)
        for room, members in room_positions.items()
    }
    output = {}
    for view in public["views"]:
        view_id = int(view["view_id"])
        position = int(view["position_id"])
        room = int(view["room_id"])
        point = by_position[position]
        yaw = float(view["yaw_rad"])
        heading = np.asarray([math.cos(yaw), math.sin(yaw)])
        to_room = room_centers[room] - point
        to_scene = scene_center - point
        room_norm = float(np.linalg.norm(to_room))
        scene_norm = float(np.linalg.norm(to_scene))
        members = sorted(room_positions[room])
        output[view_id] = np.concatenate([room_features[room], np.asarray([
            room_norm / scene_scale,
            float(np.dot(heading, to_room) / room_norm) if room_norm > 1e-6 else 0.0,
            float(np.dot(heading, to_scene) / scene_norm) if scene_norm > 1e-6 else 0.0,
            float(np.mean(geodesic[position, members])) / distance_scale,
            float((point[0] - scene_center[0]) / scene_scale),
            float((point[1] - scene_center[1]) / scene_scale),
            math.sin(yaw),
            math.cos(yaw),
        ])])
    return output


def coverage_utility(candidate: dict, history: list[dict], geodesic: np.ndarray, travel: float) -> float:
    visited_positions = {int(item["position_id"]) for item in history}
    visited_views = {int(item["view_id"]) for item in history}
    position = int(candidate["position_id"])
    if visited_positions:
        geometric_novelty = min(float(geodesic[position, other]) for other in visited_positions)
    else:
        geometric_novelty = 1.0
    yaw_novelty = 1.0 if int(candidate["view_id"]) not in visited_views else 0.0
    room_novelty = 1.0 if candidate["room_id"] not in {item["room_id"] for item in history} else 0.0
    return (geometric_novelty + 0.60 * yaw_novelty + 0.35 * room_novelty) / (0.50 + travel)


def choose_action(
    method: str,
    scene_id: int,
    category: str,
    step: int,
    public: dict,
    history: list[dict],
    detection_trigger: float,
    spent_m: float,
    geodesic_budget_m: float,
    safe_ray_weight: float,
    safe_target_depth_m: float,
    safe_coverage_weight: float,
    safe_confidence_backbone_weight: float,
    safe_provenance_weight: float,
    safe_yaw_diversity_weight: float,
    safe_ray_alignment_weight: float,
    safe_baseline_geometry_weight: float,
    safe_prior_nearest_weight: float,
    safe_prior_semantic_weight: float,
    safe_prior_new_room_weight: float,
    safe_prior_coverage_weight: float,
    safe_exploration_mode: str,
    safe_travel_power: float,
    safe_max_edge_m: float,
    context_weights: dict[str, float],
    room_prior: dict | None,
    oracle_visibility: dict[int, float],
    outcome_model: dict | None = None,
    value_stop: bool = False,
) -> tuple[int, float] | None:
    views = view_lookup(public)
    geodesic = np.asarray(public["geodesic_position_cost_m"], dtype=float)
    room_geometry_features = public.get("_room_geometry_features")
    if room_geometry_features is None:
        room_geometry_features = room_geometry(public)
    view_geometry_features = public.get("_view_geometry_features")
    if view_geometry_features is None:
        view_geometry_features = view_geometry(public, room_geometry_features)
    current_position = int(history[-1]["position_id"])
    visited = {int(item["view_id"]) for item in history}
    candidates = []
    for view_id, candidate in views.items():
        if view_id in visited:
            continue
        travel = float(geodesic[current_position, int(candidate["position_id"])])
        if spent_m + travel <= geodesic_budget_m + 1e-9:
            candidates.append((view_id, candidate, travel))
    if not candidates:
        return None

    tie = lambda view_id: hashlib.sha256(
        f"{DETERMINISTIC_SENSOR_NAMESPACE}::{method}::{scene_id}::{category}::{step}::{view_id}".encode()
    ).hexdigest()
    if method == "oracle_visibility":
        chosen = max(
            candidates,
            key=lambda item: (
                oracle_visibility.get(item[0], 0.0) / (0.50 + item[2]),
                coverage_utility(item[1], history, geodesic, item[2]),
                tie(item[0]),
            ),
        )
    elif method == "random":
        chosen = min(candidates, key=lambda item: tie(item[0]))
    elif method == "nearest_frontier":
        chosen = min(candidates, key=lambda item: (item[2], tie(item[0])))
    elif method == "coverage_gain":
        chosen = max(candidates, key=lambda item: (coverage_utility(item[1], history, geodesic, item[2]), tie(item[0])))
    else:
        # Enter evidence-acquisition mode at the method's frozen support
        # threshold.  A detector-specific calibration must not be silently
        # overridden by a controlled-sensor constant.
        positives = [
            item for item in history if float(item["score"]) >= detection_trigger
        ]
        if not positives:
            room_evidence: dict[int, float] = {}
            room_context: dict[int, dict[str, float]] = {}
            visited_rooms = {int(item["room_id"]) for item in history}
            for observation in history:
                room_id = int(observation["room_id"])
                aggregate = room_context.setdefault(room_id, {})
                for name, score in observation.get("context_scores", {}).items():
                    aggregate[name] = max(aggregate.get(name, 0.0), float(score))
                evidence = sum(
                    max(0.0, context_weights.get(name, 0.0)) * float(score)
                    for name, score in observation.get("context_scores", {}).items()
                )
                room_evidence[room_id] = max(room_evidence.get(room_id, 0.0), evidence)

            def learned_room_probability(room_id: int) -> float:
                if not room_prior:
                    return 0.0
                coefficients = room_prior.get("geometry_coefficients")
                if coefficients:
                    values = room_geometry_features[room_id]
                    mean = np.asarray(room_prior["geometry_mean"], dtype=float)
                    std = np.asarray(room_prior["geometry_std"], dtype=float)
                    normalized = (values - mean) / std
                    logit = float(coefficients[0] + np.dot(np.asarray(coefficients[1:]), normalized))
                else:
                    logit = float(room_prior.get("intercept", 0.0))
                for name, score in room_context.get(room_id, {}).items():
                    logit += float(room_prior.get("weights", {}).get(name, 0.0)) * score
                return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

            def learned_view_probability(view_id: int, room_id: int) -> float:
                coefficients = room_prior.get("view_geometry_coefficients") if room_prior else None
                if not coefficients:
                    return learned_room_probability(room_id)
                values = view_geometry_features[view_id]
                mean = np.asarray(room_prior["view_geometry_mean"], dtype=float)
                std = np.asarray(room_prior["view_geometry_std"], dtype=float)
                normalized = (values - mean) / std
                logit = float(coefficients[0] + np.dot(np.asarray(coefficients[1:]), normalized))
                for name, score in room_context.get(room_id, {}).items():
                    logit += float(room_prior.get("weights", {}).get(name, 0.0)) * score
                return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

            def prior_exploration(item: tuple[int, dict, float]) -> float:
                view_id, candidate, travel = item
                room_id = int(candidate["room_id"])
                semantic_room = room_evidence.get(room_id, 0.0)
                learned_room = learned_view_probability(view_id, room_id)
                new_room = 0.30 if room_id not in visited_rooms else 0.0
                nearest = safe_prior_nearest_weight / (0.50 + travel)
                coverage = coverage_utility(candidate, history, geodesic, travel)
                return (
                    # The matched coverage baseline is the conservative
                    # backbone.  Learned category/room terms must earn an
                    # improvement over it rather than replace exploration.
                    coverage
                    + nearest
                    + safe_prior_semantic_weight * (semantic_room + learned_room) / (0.50 + travel)
                    + safe_prior_new_room_weight * new_room
                    + safe_prior_coverage_weight * coverage
                )

            if method in ("safevantage_active", "safevantage_fixed_horizon") and safe_exploration_mode == "nearest":
                chosen = min(candidates, key=lambda item: (item[2], tie(item[0])))
            elif method in ("safevantage_active", "safevantage_fixed_horizon") and safe_exploration_mode == "low_cost":
                visited_positions = {int(item["position_id"]) for item in history}
                visited_rooms = {int(item["room_id"]) for item in history}

                def low_cost_exploration(item: tuple[int, dict, float]) -> float:
                    view_id, candidate, travel = item
                    position_novelty = 1.0 if int(candidate["position_id"]) not in visited_positions else 0.0
                    yaw_novelty = 1.0 if view_id not in visited else 0.0
                    room_novelty = 1.0 if int(candidate["room_id"]) not in visited_rooms else 0.0
                    value = position_novelty + 0.25 * yaw_novelty + 0.35 * room_novelty
                    return value / ((0.50 + travel) ** safe_travel_power)

                chosen = max(candidates, key=lambda item: (low_cost_exploration(item), tie(item[0])))
            elif method in ("category_room_prior", "safevantage_active", "safevantage_fixed_horizon", "counterevidence_active", "safevantage_dc_full", "safevantage_dc_no_value", "safevantage_dc_no_observability", "safevantage_dc_no_independence", "safevantage_dc_no_belief", "max_observability", "residual_safevantage_tau0", "residual_safevantage_tau02", "residual_safevantage_tau05", "residual_safevantage_tau10", "decision_resolve"):
                chosen = max(candidates, key=lambda item: (prior_exploration(item), tie(item[0])))
            else:
                chosen = max(candidates, key=lambda item: (coverage_utility(item[1], history, geodesic, item[2]), tie(item[0])))
        else:
            best = max(positives, key=lambda item: float(item["score"]))
            best_position = int(best["position_id"])
            best_yaw = float(best["yaw_rad"])
            used_support_positions = {int(item["position_id"]) for item in positives}

            def confidence_gain(item: tuple[int, dict, float]) -> float:
                _, candidate, travel = item
                position = int(candidate["position_id"])
                proximity = math.exp(-float(geodesic[best_position, position]) / 2.5)
                distinct = 1.0 if position not in used_support_positions else 0.0
                yaw_delta = abs(math.atan2(math.sin(float(candidate["yaw_rad"]) - best_yaw), math.cos(float(candidate["yaw_rad"]) - best_yaw))) / math.pi
                return (proximity + 0.55 * distinct + 0.20 * yaw_delta) / (0.50 + travel)

            if method == "category_room_prior":
                chosen = max(
                    candidates,
                    key=lambda item: (
                        (1.0 if int(item[1]["room_id"]) == int(best["room_id"]) else 0.0) / (0.50 + item[2]),
                        confidence_gain(item),
                        tie(item[0]),
                    ),
                )
            elif method == "detector_confidence_gain":
                chosen = max(candidates, key=lambda item: (confidence_gain(item), tie(item[0])))
            elif method in ("safevantage_active", "safevantage_fixed_horizon"):
                local_candidates = [item for item in candidates if item[2] <= safe_max_edge_m]
                if local_candidates:
                    candidates = local_candidates

                def evidence_acquisition(item: tuple[int, dict, float]) -> float:
                    _, candidate, travel = item
                    position = int(candidate["position_id"])
                    baseline_m = float(geodesic[best_position, position])
                    proximity = math.exp(-baseline_m / 2.5)
                    provenance_diversity = 1.0 if position not in used_support_positions else 0.0
                    coverage = coverage_utility(candidate, history, geodesic, travel)
                    # Convert the strongest observation into a provisional
                    # target ray.  A corroborating action should come from a
                    # distinct camera provenance while facing the same
                    # hypothesized object, not merely repeat a nearby yaw.
                    best_xy = np.asarray(best["position_world"][:2], dtype=float)
                    candidate_xy = np.asarray(candidate["position_world"][:2], dtype=float)
                    grounded = [
                        observation for observation in positives
                        if observation.get("target_world") is not None
                    ]
                    if grounded:
                        points = np.asarray([observation["target_world"][:2] for observation in grounded], dtype=float)
                        weights = np.asarray([max(float(observation["score"]), 1e-4) for observation in grounded])
                        target_xy = np.average(points, axis=0, weights=weights)
                    else:
                        target_xy = best_xy + safe_target_depth_m * np.array([math.cos(best_yaw), math.sin(best_yaw)])
                    target_yaw = math.atan2(target_xy[1] - candidate_xy[1], target_xy[0] - candidate_xy[0])
                    angular_error = abs(math.atan2(
                        math.sin(float(candidate["yaw_rad"]) - target_yaw),
                        math.cos(float(candidate["yaw_rad"]) - target_yaw),
                    ))
                    ray_alignment = math.exp(-angular_error / 0.65)
                    useful_baseline = math.exp(-((baseline_m - 1.5) / 1.5) ** 2)
                    denominator = (0.50 + travel) ** safe_travel_power
                    ray_bonus = (
                        safe_ray_alignment_weight * ray_alignment
                        * safe_provenance_weight * provenance_diversity
                        + safe_baseline_geometry_weight * useful_baseline
                    ) / denominator
                    # Confidence gain is the conservative backbone; the ray
                    # term adds claim-grounded multi-provenance value without
                    # giving up its matched-cost behavior.
                    safe_confidence_gain = (
                        safe_confidence_backbone_weight * proximity
                        + 0.55 * safe_provenance_weight * provenance_diversity
                        + safe_yaw_diversity_weight * abs(math.atan2(
                            math.sin(float(candidate["yaw_rad"]) - best_yaw),
                            math.cos(float(candidate["yaw_rad"]) - best_yaw),
                        )) / math.pi
                    ) / denominator
                    return safe_confidence_gain + safe_ray_weight * ray_bonus + safe_coverage_weight * coverage

                chosen = max(candidates, key=lambda item: (evidence_acquisition(item), tie(item[0])))
            elif method in ("counterevidence_active", "safevantage_dc_full", "safevantage_dc_no_value", "safevantage_dc_no_observability", "safevantage_dc_no_independence", "safevantage_dc_no_belief", "max_observability", "residual_safevantage_tau0", "residual_safevantage_tau02", "residual_safevantage_tau05", "residual_safevantage_tau10", "decision_resolve"):
                # Full branch: use a train-fitted observation-outcome model to
                # estimate the value of a candidate test. The model is only
                # queried with acquired support and public geometry. A missing
                # model falls back to the earlier geometry proxy for smoke
                # diagnostics, never to hidden visibility.
                grounded = [item for item in positives if item.get("target_world") is not None]
                if grounded:
                    points = np.asarray([item["target_world"][:2] for item in grounded], dtype=float)
                    weights = np.asarray([max(float(item["score"]), 1e-4) for item in grounded])
                    target_xy = np.average(points, axis=0, weights=weights)
                else:
                    best_xy = np.asarray(best["position_world"][:2], dtype=float)
                    target_xy = best_xy + safe_target_depth_m * np.asarray([math.cos(best_yaw), math.sin(best_yaw)])
                current_confidence = max(float(item["score"]) for item in positives)
                decision_uncertainty = max(0.05, 1.0 - current_confidence)

                if outcome_model is not None and outcome_model.get("weights"):
                    model_mean = np.asarray(outcome_model.get("feature_mean", []), dtype=float)
                    model_std = np.asarray(outcome_model.get("feature_std", []), dtype=float)
                    model_weights = np.asarray(outcome_model["weights"], dtype=float)
                    model_bias = float(outcome_model.get("bias", 0.0))
                    false_positive_rate = float(outcome_model.get("false_positive_rate", 0.05))
                    belief = outcome_model.get("belief_calibration", {})
                    belief_logit = float(belief.get("bias", -4.0)) + float(belief.get("slope", 8.0)) * current_confidence

                    def sigmoid(value: float) -> float:
                        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))

                    p_yes = 0.5 if method == "safevantage_dc_no_belief" else sigmoid(belief_logit)

                    def bayes_loss(prediction: str, probability_yes: float) -> float:
                        if prediction == "yes":
                            return 5.0 * (1.0 - probability_yes)
                        if prediction == "no":
                            return 2.0 * probability_yes
                        return 0.5

                    def model_visibility(candidate: dict) -> float:
                        point = np.asarray(candidate["position_world"][:2], dtype=float)
                        delta = target_xy - point
                        distance = float(np.linalg.norm(delta))
                        direction = delta / distance if distance > 1e-6 else np.zeros(2)
                        heading = np.asarray([math.cos(float(candidate["yaw_rad"])), math.sin(float(candidate["yaw_rad"]))])
                        cosine = float(np.dot(heading, direction)) if distance > 1e-6 else 0.0
                        independent = 1.0 if int(candidate["position_id"]) not in used_support_positions else 0.0
                        if method == "safevantage_dc_no_independence":
                            independent = 1.0
                        baseline = float(geodesic[best_position, int(candidate["position_id"])])
                        raw = np.asarray([distance, cosine, baseline, independent], dtype=float)
                        if len(model_mean) == len(raw) and len(model_std) == len(raw) and len(model_weights) == len(raw):
                            raw = (raw - model_mean) / np.maximum(model_std, 1e-6)
                        predicted = sigmoid(model_bias + float(np.dot(model_weights, raw)))
                        if method == "safevantage_dc_no_observability":
                            predicted = float(outcome_model.get("visibility_positive_rate", predicted))
                        return predicted

                    def decision_value(item: tuple[int, dict, float]) -> float:
                        _, candidate, travel = item
                        p_visible = model_visibility(candidate)
                        p_high = p_yes * p_visible + (1.0 - p_yes) * false_positive_rate
                        p_high = min(max(p_high, 1e-6), 1.0 - 1e-6)
                        posterior_high = p_yes * p_visible / p_high
                        posterior_low = p_yes * (1.0 - p_visible) / (1.0 - p_high)
                        high = dict(candidate)
                        high.update({"score": 0.75, "target_world": [float(target_xy[0]), float(target_xy[1]), 1.0]})
                        low = dict(candidate)
                        low.update({"score": 0.10, "target_world": None})
                        total_positions = len(public["geodesic_position_cost_m"])
                        current_prediction = decide(history, total_positions, 0.55, 0.65, 0.25,
                                                    require_target_consistency=True, target_consistency_m=0.75)
                        high_prediction = decide(history + [high], total_positions, 0.55, 0.65, 0.25,
                                                 require_target_consistency=True, target_consistency_m=0.75)
                        low_prediction = decide(history + [low], total_positions, 0.55, 0.65, 0.25,
                                                require_target_consistency=True, target_consistency_m=0.75)
                        expected = (
                            p_high * bayes_loss(high_prediction, posterior_high)
                            + (1.0 - p_high) * bayes_loss(low_prediction, posterior_low)
                        )
                        current = bayes_loss(current_prediction, p_yes)
                        if method == "safevantage_dc_no_value":
                            return (p_visible * (0.5 + 0.5 * decision_uncertainty)) / (0.50 + travel)
                        return (current - expected - 0.001 * travel) / (0.50 + travel)

                    def decision_resolution_value(item: tuple[int, dict, float]) -> float:
                        """Expected correct terminal decision from one acquired view.

                        This is the single DecisionResolve policy: it scores a view by
                        the model probability that the view produces a *correct*
                        yes/no decision, rather than by raw visibility or expected
                        scalar loss. A wait outcome contributes zero resolution.
                        """
                        _, candidate, travel = item
                        p_visible = model_visibility(candidate)
                        p_high = p_yes * p_visible + (1.0 - p_yes) * false_positive_rate
                        p_high = min(max(p_high, 1e-6), 1.0 - 1e-6)
                        posterior_high = p_yes * p_visible / p_high
                        posterior_low = p_yes * (1.0 - p_visible) / (1.0 - p_high)
                        high = dict(candidate)
                        high.update({"score": 0.75, "target_world": [float(target_xy[0]), float(target_xy[1]), 1.0]})
                        low = dict(candidate)
                        low.update({"score": 0.10, "target_world": None})
                        total_positions = len(public["geodesic_position_cost_m"])
                        current_prediction = decide(history, total_positions, 0.55, 0.65, 0.25)
                        high_prediction = decide(history + [high], total_positions, 0.55, 0.65, 0.25)
                        low_prediction = decide(history + [low], total_positions, 0.55, 0.65, 0.25)

                        def correctness(prediction: str, probability_yes: float) -> float:
                            if prediction == "yes":
                                return probability_yes
                            if prediction == "no":
                                return 1.0 - probability_yes
                            return 0.0

                        current_resolved = 1.0 if current_prediction != "wait" else 0.0
                        expected_correct = (
                            p_high * correctness(high_prediction, posterior_high)
                            + (1.0 - p_high) * correctness(low_prediction, posterior_low)
                        )
                        # The current episode enters this branch only while waiting;
                        # retaining the gain form makes the policy definition explicit
                        # and prevents already-terminal states from earning value.
                        gain = max(0.0, expected_correct - current_resolved)
                        return (gain - 0.001 * travel) / (0.50 + travel)

                    if method == "decision_resolve":
                        scored = [(decision_resolution_value(item), item) for item in candidates]
                    elif method.startswith("residual_safevantage_tau") or method == "max_observability":
                        vis_scores = [(model_visibility(item[1]) / (0.50 + item[2]), item) for item in candidates]
                        if method == "max_observability":
                            scored = vis_scores
                        else:
                            tau = {"residual_safevantage_tau0": 0.0, "residual_safevantage_tau02": 0.02,
                                   "residual_safevantage_tau05": 0.05, "residual_safevantage_tau10": 0.10}[method]
                            dec_scores = [(decision_value(item), item) for item in candidates]
                            best_vis = max(vis_scores, key=lambda pair: (pair[0], tie(pair[1][0])))
                            best_dec = max(dec_scores, key=lambda pair: (pair[0], tie(pair[1][0])))
                            scored = dec_scores if best_dec[0] - best_vis[0] > tau else vis_scores
                    else:
                        scored = [(decision_value(item), item) for item in candidates]
                    best_value, best_item = max(scored, key=lambda pair: (pair[0], tie(pair[1][0])))
                    if value_stop and best_value <= 0.0:
                        return None
                    chosen = best_item
                    return int(chosen[0]), float(chosen[2])

                def counterevidence_value(item: tuple[int, dict, float]) -> float:
                    _, candidate, travel = item
                    position = int(candidate["position_id"])
                    point = np.asarray(candidate["position_world"][:2], dtype=float)
                    target_yaw = math.atan2(target_xy[1] - point[1], target_xy[0] - point[0])
                    angular_error = abs(math.atan2(
                        math.sin(float(candidate["yaw_rad"]) - target_yaw),
                        math.cos(float(candidate["yaw_rad"]) - target_yaw),
                    ))
                    observability = math.exp(-angular_error / 0.55)
                    independent = 1.0 if position not in used_support_positions else 0.0
                    baseline_m = float(geodesic[best_position, position])
                    useful_baseline = math.exp(-((baseline_m - 1.5) / 1.5) ** 2)
                    coverage = coverage_utility(candidate, history, geodesic, travel)
                    expected_test_value = observability * (0.5 + 0.5 * decision_uncertainty) * (0.25 + 0.75 * independent)
                    denominator = (0.50 + travel) ** safe_travel_power
                    return (1.25 * expected_test_value + 0.35 * useful_baseline + 0.20 * coverage) / denominator

                chosen = max(candidates, key=lambda item: (counterevidence_value(item), tie(item[0])))
            else:
                raise ValueError(f"unknown method: {method}")
    return int(chosen[0]), float(chosen[2])


def decide(
    history: list[dict],
    total_positions: int,
    yes_threshold: float = 0.55,
    yes_maximum: float = 0.65,
    no_coverage: float = 0.75,
    require_target_consistency: bool = False,
    target_consistency_m: float = 0.75,
    no_score_ceiling: float | None = None,
    no_min_views: int = 0,
) -> str:
    support = [item for item in history if float(item["score"]) >= yes_threshold]
    support_positions = {int(item["position_id"]) for item in support}
    maximum = max(float(item["score"]) for item in history)
    corroborated = len(support_positions) >= 2
    if corroborated and require_target_consistency:
        grounded = [item for item in support if item.get("target_world") is not None]
        corroborated = any(
            int(first["position_id"]) != int(second["position_id"])
            and float(np.linalg.norm(
                np.asarray(first["target_world"][:2], dtype=float)
                - np.asarray(second["target_world"][:2], dtype=float)
            )) <= target_consistency_m
            for index, first in enumerate(grounded)
            for second in grounded[index + 1:]
        )
    if corroborated and maximum >= yes_maximum:
        return "yes"
    position_coverage = len({int(item["position_id"]) for item in history}) / float(total_positions)
    negative_ceiling = yes_threshold if no_score_ceiling is None else no_score_ceiling
    if position_coverage >= no_coverage and len(history) >= no_min_views and maximum < negative_ceiling:
        return "no"
    return "wait"


def run_episode(
    scene_dir: Path,
    public: dict,
    hidden: dict,
    category: str,
    truth: bool,
    start: int,
    method: str,
    actions: int,
    geodesic_budget_m: float,
    yes_threshold: float,
    yes_maximum: float,
    no_coverage: float,
    safe_yes_threshold: float,
    safe_yes_maximum: float,
    safe_no_coverage: float,
    safe_target_consistency_m: float,
    safe_ray_weight: float,
    safe_target_depth_m: float,
    safe_coverage_weight: float,
    safe_confidence_backbone_weight: float,
    safe_provenance_weight: float,
    safe_yaw_diversity_weight: float,
    safe_ray_alignment_weight: float,
    safe_baseline_geometry_weight: float,
    safe_prior_nearest_weight: float,
    safe_prior_semantic_weight: float,
    safe_prior_new_room_weight: float,
    safe_prior_coverage_weight: float,
    safe_exploration_mode: str,
    safe_detection_trigger: float,
    safe_travel_power: float,
    safe_max_edge_m: float,
    safe_no_score_ceiling: float,
    safe_no_min_views: int,
    force_fixed_horizon: bool,
    sensor: str,
    observation_bank: dict[str, dict[int, dict]],
    context_weights: dict[str, float],
    room_prior: dict | None,
    outcome_model: dict | None,
    common_decoder: bool,
    value_stop: bool,
) -> dict:
    views = view_lookup(public)
    initial_yaw = hash_int(f"{DETERMINISTIC_SENSOR_NAMESPACE}::initial-yaw::{hidden['scene_id']}::{category}::{start}") % 4
    initial_view = start * 4 + initial_yaw
    first = enrich_observation(views[initial_view], observation_bank, category)
    history = [first]
    spent_m = 0.0
    safe_decision = method in ("safevantage_active", "safevantage_fixed_horizon", "counterevidence_active", "safevantage_dc_full", "safevantage_dc_no_value", "safevantage_dc_no_observability", "safevantage_dc_no_independence", "safevantage_dc_no_belief", "decision_resolve")
    decoder_safe = safe_decision and not common_decoder
    method_yes_threshold = safe_yes_threshold if decoder_safe else yes_threshold
    method_yes_maximum = safe_yes_maximum if decoder_safe else yes_maximum
    method_no_coverage = safe_no_coverage if decoder_safe else no_coverage
    for step in range(actions):
        action = choose_action(
            method,
            int(hidden["scene_id"]),
            category,
            step,
            public,
            history,
            safe_detection_trigger if safe_decision else method_yes_threshold,
            spent_m,
            geodesic_budget_m,
            safe_ray_weight,
            safe_target_depth_m,
            safe_coverage_weight,
            safe_confidence_backbone_weight,
            safe_provenance_weight,
            safe_yaw_diversity_weight,
            safe_ray_alignment_weight,
            safe_baseline_geometry_weight,
            safe_prior_nearest_weight,
            safe_prior_semantic_weight,
            safe_prior_new_room_weight,
            safe_prior_coverage_weight,
            safe_exploration_mode,
            safe_travel_power,
            safe_max_edge_m,
            context_weights,
            room_prior,
            {
                int(view["view_id"]): float(view["visible_fraction"].get(category, 0.0))
                for view in hidden["views"]
            },
            outcome_model,
            value_stop,
        )
        if action is None:
            break
        view_id, travel = action
        spent_m += travel
        history.append(enrich_observation(views[view_id], observation_bank, category))
        if (
            method != "safevantage_fixed_horizon"
            and not force_fixed_horizon
            and decide(
                history,
                len(public["geodesic_position_cost_m"]),
                method_yes_threshold,
                method_yes_maximum,
                method_no_coverage,
                require_target_consistency=decoder_safe,
                target_consistency_m=safe_target_consistency_m,
                no_score_ceiling=safe_no_score_ceiling if decoder_safe else None,
                no_min_views=safe_no_min_views if decoder_safe else 0,
            ) != "wait"
        ):
            break
    prediction = decide(
        history,
        len(public["geodesic_position_cost_m"]),
        method_yes_threshold,
        method_yes_maximum,
        method_no_coverage,
        require_target_consistency=decoder_safe,
        target_consistency_m=safe_target_consistency_m,
        no_score_ceiling=safe_no_score_ceiling if decoder_safe else None,
        no_min_views=safe_no_min_views if decoder_safe else 0,
    )
    fp = int(prediction == "yes" and not truth)
    fn = int(prediction == "no" and truth)
    abstain = int(prediction == "wait")
    risk = 5.0 * fp + 2.0 * fn + 0.5 * abstain + 0.001 * spent_m
    return {
        "scene_id": int(hidden["scene_id"]),
        "category": category,
        "truth": bool(truth),
        "start_position_id": int(start),
        "method": method,
        "sensor": sensor,
        "action_budget": int(actions),
        "total_positions": len(public["geodesic_position_cost_m"]),
        "actions_taken": len(history) - 1,
        "geodesic_m": spent_m,
        "prediction": prediction,
        "false_positive": fp,
        "false_negative": fn,
        "abstain": abstain,
        "risk": risk,
        "history": [
            {
                key: item[key] for key in
                ("view_id", "position_id", "room_id", "yaw_rad", "score", "target_world", "box_xyxy", "context_scores")
                if key in item
            }
            for item in history
        ],
        "selector_family": (
            "decision_resolution"
            if method == "decision_resolve"
            else ("counterevidence_observability_proxy"
                  if method in ("counterevidence_active", "safevantage_dc_full")
                  else ("counterevidence_ablation" if method.startswith("safevantage_dc_no_")
                        else ("safevantage_provenance_ray" if safe_decision else "conventional")))
        ),
    }


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def summarize(rows: list[dict]) -> dict:
    output = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        negatives = [row for row in subset if not row["truth"]]
        answered = [row for row in subset if row["prediction"] != "wait"]
        yes_tp = sum(row["prediction"] == "yes" and row["truth"] for row in subset)
        yes_fp = sum(row["prediction"] == "yes" and not row["truth"] for row in subset)
        yes_fn = sum(row["prediction"] != "yes" and row["truth"] for row in subset)
        no_tp = sum(row["prediction"] == "no" and not row["truth"] for row in subset)
        no_fp = sum(row["prediction"] == "no" and row["truth"] for row in subset)
        no_fn = sum(row["prediction"] != "no" and not row["truth"] for row in subset)
        output[method] = {
            "episodes": len(subset),
            "scenes": len({row["scene_id"] for row in subset}),
            "mean_risk": float(np.mean([row["risk"] for row in subset])) if subset else math.nan,
            "answer_rate": len(answered) / len(subset) if subset else math.nan,
            "answered_accuracy": sum((row["prediction"] == "yes") == row["truth"] for row in answered) / len(answered) if answered else math.nan,
            "false_positive_rate": sum(row["false_positive"] for row in negatives) / len(negatives) if negatives else math.nan,
            "macro_f1": 0.5 * (f1(yes_tp, yes_fp, yes_fn) + f1(no_tp, no_fp, no_fn)),
            "mean_geodesic_m": float(np.mean([row["geodesic_m"] for row in subset])) if subset else math.nan,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--sensor", choices=("controlled", "yoloworld"), default="controlled")
    parser.add_argument(
        "--detector-root",
        type=Path,
        help="Read immutable per-scene detector JSON files from this separate result tree.",
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        help="Evaluate the ordered, frozen scene directories listed one per line.",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--training-vocabulary", type=Path, required=True)
    parser.add_argument("--room-prior-model", type=Path)
    parser.add_argument("--outcome-model", type=Path)
    parser.add_argument("--actions", type=int, default=12)
    parser.add_argument("--geodesic-budget-m", type=float, default=20.0)
    parser.add_argument("--yes-threshold", type=float, default=0.55)
    parser.add_argument("--yes-maximum", type=float, default=0.65)
    parser.add_argument("--no-coverage", type=float, default=0.25)
    parser.add_argument("--safe-yes-threshold", type=float)
    parser.add_argument("--safe-yes-maximum", type=float)
    parser.add_argument("--safe-no-coverage", type=float)
    parser.add_argument("--safe-target-consistency-m", type=float, default=0.75)
    parser.add_argument("--safe-ray-weight", type=float, default=0.75)
    parser.add_argument("--safe-target-depth-m", type=float, default=2.0)
    parser.add_argument("--safe-coverage-weight", type=float, default=0.10)
    parser.add_argument("--safe-confidence-backbone-weight", type=float, default=1.0)
    parser.add_argument("--safe-provenance-weight", type=float, default=1.0)
    parser.add_argument("--safe-yaw-diversity-weight", type=float, default=0.20)
    parser.add_argument("--safe-ray-alignment-weight", type=float, default=1.40)
    parser.add_argument("--safe-baseline-geometry-weight", type=float, default=0.35)
    parser.add_argument("--safe-prior-nearest-weight", type=float, default=0.35)
    parser.add_argument("--safe-prior-semantic-weight", type=float, default=0.50)
    parser.add_argument("--safe-prior-new-room-weight", type=float, default=0.25)
    parser.add_argument("--safe-prior-coverage-weight", type=float, default=0.20)
    parser.add_argument("--safe-exploration-mode", choices=("coverage", "nearest", "low_cost"), default="coverage")
    parser.add_argument("--safe-detection-trigger", type=float)
    parser.add_argument("--safe-travel-power", type=float, default=1.0)
    parser.add_argument("--safe-max-edge-m", type=float, default=1.0e9)
    parser.add_argument("--safe-no-score-ceiling", type=float)
    parser.add_argument("--safe-no-min-views", type=int, default=0)
    parser.add_argument("--force-fixed-horizon", action="store_true")
    parser.add_argument("--common-decoder", action="store_true", help="Use the conventional frozen decoder for every acquisition policy.")
    parser.add_argument("--value-stop", action="store_true", help="Allow the calibrated counterevidence controller to stop when all candidate tests have non-positive expected value.")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    safe_yes_threshold = args.yes_threshold if args.safe_yes_threshold is None else args.safe_yes_threshold
    safe_yes_maximum = args.yes_maximum if args.safe_yes_maximum is None else args.safe_yes_maximum
    safe_no_coverage = args.no_coverage if args.safe_no_coverage is None else args.safe_no_coverage
    safe_detection_trigger = safe_yes_threshold if args.safe_detection_trigger is None else args.safe_detection_trigger
    safe_no_score_ceiling = safe_yes_threshold if args.safe_no_score_ceiling is None else args.safe_no_score_ceiling
    selected_methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown_methods = sorted(set(selected_methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    training_vocabulary = json.loads(args.training_vocabulary.read_text(encoding="utf-8"))
    semantic_priors = cooccurrence_weights(training_vocabulary)
    room_priors = None
    if args.room_prior_model is not None:
        payload = json.loads(args.room_prior_model.read_text(encoding="utf-8"))
        if payload.get("split") != "train":
            raise RuntimeError("room prior model must be fitted on the training split")
        room_priors = payload["models"]
    outcome_model = None
    if args.outcome_model is not None:
        outcome_model = json.loads(args.outcome_model.read_text(encoding="utf-8"))
    # Limit the number of *eligible completed* scenes, not transient shard
    # directories created by concurrently running render jobs.
    if args.scene_manifest is not None:
        scene_dirs = [
            Path(line)
            for line in args.scene_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        wrong_split = [path for path in scene_dirs if path.parent.name != args.split]
        if wrong_split:
            raise RuntimeError(
                f"scene manifest contains paths outside split {args.split}: {wrong_split[:3]}"
            )
        incomplete = [path for path in scene_dirs if not (path / "COMPLETE.json").exists()]
        if incomplete:
            raise RuntimeError(f"scene manifest contains incomplete scenes: {incomplete[:3]}")
    else:
        scene_dirs = [
            scene_dir
            for scene_dir in sorted((args.scene_root / args.split).glob("scene_*"))
            if (scene_dir / "COMPLETE.json").exists()
        ]
    scene_dirs = scene_dirs[args.scene_offset:]
    rows = []
    eligible_scenes = 0
    for scene_dir in scene_dirs:
        public = json.loads((scene_dir / "public.json").read_text(encoding="utf-8"))
        public["_room_geometry_features"] = room_geometry(public)
        public["_view_geometry_features"] = view_geometry(public, public["_room_geometry_features"])
        hidden = json.loads((scene_dir / "hidden.json").read_text(encoding="utf-8"))
        observation_bank = scene_sensor_observations(
            scene_dir, public, hidden, args.sensor, args.detector_root
        )
        queries = episode_queries(hidden)
        if not queries:
            continue
        if args.max_scenes is not None and eligible_scenes >= args.max_scenes:
            break
        eligible_scenes += 1
        for category, truth in queries:
            for start in public["start_position_ids"]:
                for method in selected_methods:
                    rows.append(
                        run_episode(
                            scene_dir,
                            public,
                            hidden,
                            category,
                            truth,
                            int(start),
                            method,
                            args.actions,
                            args.geodesic_budget_m,
                            args.yes_threshold,
                            args.yes_maximum,
                            args.no_coverage,
                            safe_yes_threshold,
                            safe_yes_maximum,
                            safe_no_coverage,
                            args.safe_target_consistency_m,
                            args.safe_ray_weight,
                            args.safe_target_depth_m,
                            args.safe_coverage_weight,
                            args.safe_confidence_backbone_weight,
                            args.safe_provenance_weight,
                            args.safe_yaw_diversity_weight,
                            args.safe_ray_alignment_weight,
                            args.safe_baseline_geometry_weight,
                            args.safe_prior_nearest_weight,
                            args.safe_prior_semantic_weight,
                            args.safe_prior_new_room_weight,
                            args.safe_prior_coverage_weight,
                            args.safe_exploration_mode,
                            safe_detection_trigger,
                            args.safe_travel_power,
                            args.safe_max_edge_m,
                            safe_no_score_ceiling,
                            args.safe_no_min_views,
                            args.force_fixed_horizon,
                            args.sensor,
                            observation_bank,
                            semantic_priors[category],
                            room_priors.get(category) if room_priors else None,
                            outcome_model,
                            args.common_decoder,
                            args.value_stop,
                        )
                    )
    if not rows:
        raise RuntimeError("no complete eligible scenes found")
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "split": args.split,
        "sensor": args.sensor,
        "actions": args.actions,
        "force_fixed_horizon": args.force_fixed_horizon,
        "common_decoder": args.common_decoder,
        "value_stop": args.value_stop,
        "decision": {
            "yes_threshold": args.yes_threshold,
            "yes_maximum": args.yes_maximum,
            "no_coverage": args.no_coverage,
            "safe_yes_threshold": safe_yes_threshold,
            "safe_yes_maximum": safe_yes_maximum,
            "safe_no_coverage": safe_no_coverage,
            "safe_target_consistency_m": args.safe_target_consistency_m,
            "safe_no_score_ceiling": safe_no_score_ceiling,
            "safe_no_min_views": args.safe_no_min_views,
        },
        "policy": {
            "safe_ray_weight": args.safe_ray_weight,
            "safe_target_depth_m": args.safe_target_depth_m,
            "safe_coverage_weight": args.safe_coverage_weight,
            "safe_confidence_backbone_weight": args.safe_confidence_backbone_weight,
            "safe_provenance_weight": args.safe_provenance_weight,
            "safe_yaw_diversity_weight": args.safe_yaw_diversity_weight,
            "safe_ray_alignment_weight": args.safe_ray_alignment_weight,
            "safe_baseline_geometry_weight": args.safe_baseline_geometry_weight,
            "safe_prior_nearest_weight": args.safe_prior_nearest_weight,
            "safe_prior_semantic_weight": args.safe_prior_semantic_weight,
            "safe_prior_new_room_weight": args.safe_prior_new_room_weight,
            "safe_prior_coverage_weight": args.safe_prior_coverage_weight,
            "safe_exploration_mode": args.safe_exploration_mode,
            "safe_detection_trigger": safe_detection_trigger,
            "safe_travel_power": args.safe_travel_power,
            "safe_max_edge_m": args.safe_max_edge_m,
            "room_prior_model": str(args.room_prior_model) if args.room_prior_model else None,
        },
        "methods": summarize(rows),
    }
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

