#!/usr/bin/env python3
"""Compatibility entry point for the ScanNet negative-closure evaluator."""
from pathlib import Path
import runpy

script = Path(__file__).resolve().parents[1] / "scannet" / "evaluate_negative_closure.py"
runpy.run_path(str(script), run_name="__main__")
