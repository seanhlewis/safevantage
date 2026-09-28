import json

import pytest

from src.analysis.hm3d.evaluate_multisplit import read_jsonl as read_hm3d_jsonl
from src.analysis.scannet.evaluate_negative_closure import read_jsonl as read_scannet_jsonl


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_hm3d_canonical_input_accepts_method_score_fields(tmp_path):
    path = tmp_path / "canonical.jsonl"
    write_jsonl(path, [
        {"case_id": "c1", "scene_id": "s1", "gt_present": 1, "method": "safevantage_full", "support_count": 2},
        {"case_id": "c1", "scene_id": "s1", "gt_present": 1, "method": "conceptgraphs_tau1", "confidence": 0.7},
        {"case_id": "c1", "scene_id": "s1", "gt_present": 1, "method": "vlmaps_tau1", "confidence": 0.6},
    ])
    assert len(read_hm3d_jsonl(path, "canonical")) == 3


def test_hm3d_canonical_input_reports_missing_method_score(tmp_path):
    path = tmp_path / "canonical.jsonl"
    write_jsonl(path, [{"case_id": "c1", "scene_id": "s1", "gt_present": 1, "method": "safevantage_full"}])
    with pytest.raises(ValueError, match="requires support_count"):
        read_hm3d_jsonl(path, "canonical")


def test_scannet_input_enforces_frozen_disjoint_scene_splits(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rows = []
    for split, count in (("dev", 6), ("test", 12)):
        for index in range(count):
            rows.append({
                "split": split,
                "scene_id": f"{split}-{index}",
                "epistemic_state": "observed_absent",
                "max_p_clear": 0.1,
                "coverage_fraction": 0.8,
                "source_coverage_fraction": 0.75,
            })
    write_jsonl(path, rows)
    assert len(read_scannet_jsonl(path)) == 18


def test_scannet_input_rejects_out_of_range_fraction(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rows = []
    for split, count in (("dev", 6), ("test", 12)):
        for index in range(count):
            rows.append({
                "split": split,
                "scene_id": f"{split}-{index}",
                "epistemic_state": "observed_absent",
                "max_p_clear": 0.1,
                "coverage_fraction": 1.2,
                "source_coverage_fraction": 0.75,
            })
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="between 0 and 1"):
        read_scannet_jsonl(path)
