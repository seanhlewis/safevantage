import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "analysis" / "ablations" / "analyze_grouped_vantage.py"


def load_script():
    spec = importlib.util.spec_from_file_location("grouped_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mask_scope_is_explicit():
    module = load_script()
    assert "position_score_1" not in module.MASKED_NAMES
    assert "position_score_2" in module.MASKED_NAMES
    assert "position_score_3" in module.MASKED_NAMES
    assert "grounded_fraction" in module.MASKED_NAMES
    assert "minimum_grounded_pair_distance_clipped_5m" in module.MASKED_NAMES
    assert module.MASKED_PREFIXES == ("support_positions_ge_",)


def test_row_key_excludes_method_for_pairing():
    module = load_script()
    first = {
        "scene_id": 1,
        "category": "chair",
        "start_position_id": 2,
        "action_budget": 8,
        "method": "full",
    }
    second = {**first, "method": "ablated"}
    assert module.row_key(first) == module.row_key(second)


