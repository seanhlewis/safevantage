import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_rows(path, budget):
    rows = []
    for scene_id in (13, 91):
        for method, prediction, truth, risk, travel in (
            ("safevantage_active", "yes", True, 0.01, 1.5),
            ("safevantage_dc_full", "wait", True, 0.51, 2.5),
            ("safevantage_active", "no", False, 0.02, 1.0),
            ("safevantage_dc_full", "no", False, 0.02, 2.0),
        ):
            rows.append({
                "scene_id": scene_id,
                "method": method,
                "action_budget": budget,
                "sensor": "controlled",
                "prediction": prediction,
                "truth": truth,
                "risk": risk,
                "geodesic_m": travel,
                "actions_taken": budget,
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scene_summary_and_bootstrap_retain_paired_metrics(tmp_path, monkeypatch):
    rows8 = tmp_path / "test8.jsonl"
    rows12 = tmp_path / "test12.jsonl"
    write_rows(rows8, 8)
    write_rows(rows12, 12)

    summarize = load("summarize_paired_scenes", ROOT / "scripts" / "summarize_paired_scenes.py")
    table = tmp_path / "paired.csv"
    monkeypatch.setattr(sys, "argv", [
        "summarize_paired_scenes.py",
        "--input", f"8={rows8}",
        "--input", f"12={rows12}",
        "--output", str(table),
    ])
    summarize.main()

    with table.open(newline="", encoding="utf-8") as stream:
        table_rows = list(csv.DictReader(stream))
    assert len(table_rows) == 8
    assert "category" not in table_rows[0]
    assert len({row["scene_group_id"] for row in table_rows}) == 2
    assert all(int(row["episodes"]) == 2 for row in table_rows)
    assert next(row for row in table_rows if row["method"] == "safevantage_dc_full")["pred_wait_truth_yes"] == "1"

    bootstrap = load("bootstrap_paired_scenes", ROOT / "scripts" / "bootstrap_paired_scenes.py")
    output = tmp_path / "bootstrap.json"
    monkeypatch.setattr(sys, "argv", [
        "bootstrap_paired_scenes.py", "--input", str(table), "--output", str(output),
        "--resamples", "20", "--seed", "0",
    ])
    bootstrap.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["budgets"]["8"]["scenes"] == 2
    delta = result["budgets"]["8"]["paired_deltas"]
    assert delta["comparison"] == "safevantage_active"
    assert delta["risk"]["mean"] > 0
    assert delta["macro_f1"]["mean"] < 0


def test_training_vocabulary_reads_training_split_only(tmp_path, monkeypatch):
    scene_root = tmp_path / "scenes"
    for scene_name, presence in (
        ("scene_1", {"chair": True, "lamp": False}),
        ("scene_2", {"chair": False, "table": True}),
    ):
        scene = scene_root / "train" / scene_name
        scene.mkdir(parents=True)
        (scene / "COMPLETE.json").write_text("{}", encoding="utf-8")
        (scene / "hidden.json").write_text(
            json.dumps({"vocabulary_presence": presence}), encoding="utf-8"
        )
    test_scene = scene_root / "test" / "scene_3"
    test_scene.mkdir(parents=True)
    (test_scene / "COMPLETE.json").write_text("{}", encoding="utf-8")
    (test_scene / "hidden.json").write_text(
        json.dumps({"vocabulary_presence": {"test_only_category": True}}), encoding="utf-8"
    )

    builder = load("build_training_vocabulary", ROOT / "scripts" / "build_training_vocabulary.py")
    output = tmp_path / "training_vocab.json"
    monkeypatch.setattr(sys, "argv", [
        "build_training_vocabulary.py", "--scene-root", str(scene_root), "--output", str(output)
    ])
    builder.main()
    vocabulary = json.loads(output.read_text(encoding="utf-8"))
    assert vocabulary["fit_split"] == "train"
    assert vocabulary["scene_count"] == 2
    assert vocabulary["vocabulary"] == ["chair", "table"]
    assert all("test_only_category" not in scene["categories"] for scene in vocabulary["scenes"])
