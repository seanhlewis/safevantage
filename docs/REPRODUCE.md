# Reproduce SafeVantage

This guide maps the implementation to its archived results and reproduction commands.

## Implementation map

The native ProcTHOR policy is implemented in `src/safevantage/procthor/evaluate_procthor.py`. In the shared harness, `safevantage_dc_full` is the revised SafeVantage policy used in the 232-house experiment. It uses the training-fitted observation-outcome model in `src/safevantage/procthor/fit_observation_outcome.py` to value additional evidence. The fitter uses training data only; evaluation receives the frozen model.

`safevantage_active` is the legacy SafeVantage reference, and `safevantage_fixed_horizon` is its fixed-horizon form. The `safevantage_dc_no_*` variants are mechanism ablations of the revised policy. Comparison methods include `random`, `nearest_frontier`, `coverage_gain`, `category_room_prior`, and `detector_confidence_gain`. `counterevidence_active` is a separately named controller variant; `oracle_visibility` is a diagnostic upper bound.

`src/analysis/procthor/evaluate_decision_resolution.py` contains supporting comparison and decision-resolution analyses. HM3D evaluations and the grouped-vantage ablation are under `src/analysis/`.

## Results

The JSON summaries under `results/` are the archived outputs used in the paper analyses. Experiment definitions and method roles are specified in `configs/` and `protocols/`.

### Native ProcTHOR comparison

`results/procthor/native_232/primary/` contains the primary 8- and 12-action summaries and fitted selective-head parameters for the revised SafeVantage policy, its legacy reference, and comparison policies. `paired_scene_metrics.csv` contains 1,856 rows: 232 held-out scene groups × four methods × two action budgets. Rows retain confusion counts and sums needed to recalculate pooled metrics. Opaque scene-group IDs pair the same held-out scene across methods and budgets.

`paired_scene_bootstrap.json` contains intervals from 5,000 paired scene resamples using NumPy seed 0. The scripts below recompute the paired table and intervals from locally available episode outputs.

### Additional experiments

- `results/procthor/native_232/ablation/` contains native policy ablation summaries and fitted outcome-model parameters.
- `results/procthor/locked_baselines/` contains a separate baseline-selection analysis with its own estimator and operating-point selection.
- `results/hm3d/` contains HM3D multi-split and negative-closure summaries.
- `results/ablations/grouped_vantage/` contains grouped-vantage ablation summaries and model parameters.

## Environment and inputs

Use Python 3.11 or newer and install the direct analysis dependencies:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
```

Install the simulator and renderer dependencies required by each experiment from their official sources. Dataset links and access terms are in [`../data/README.md`](../data/README.md).

## Native ProcTHOR evaluation

The evaluator reads prepared scene records with this layout:

```text
SCENE_ROOT/
  train/scene_*/{COMPLETE.json,public.json,hidden.json}
  validation/scene_*/{COMPLETE.json,public.json,hidden.json}
  test/scene_*/{COMPLETE.json,public.json,hidden.json}
```

The renderer builds this tree from the pinned MolmoSpaces scene resources. Obtain ProcTHOR assets under the upstream terms listed in [`../data/README.md`](../data/README.md). Clone MolmoSpaces at the revision in `configs/procthor_preparation.json`, install the dependencies below, and set `MLSPACES_CACHE_DIR` and `MLSPACES_ASSETS_DIR` to writable directories.

```bash
git clone https://github.com/allenai/molmospaces.git ../molmospaces
git -C ../molmospaces checkout 1320b266d2b47aaa81c5f7a419cb9d3474e6994d
python -m pip install 'molmospaces-resources==0.0.2' 'compress-json' 'packaging~=26.2' 'tqdm>=4.65.0' 'pydantic>=2.0.0,<3'
export PYTHONPATH="../molmospaces${PYTHONPATH:+:$PYTHONPATH}"
```

Create the fixed scene manifest and fetch its one-scene smoke bundle:

```bash
python scripts/prepare_procthor/build_scene_manifest.py \
  --repo ../molmospaces \
  --config configs/procthor_preparation.json \
  --output data/local_inputs/procthor/manifest.json \
  --cache "$MLSPACES_CACHE_DIR" \
  --assets "$MLSPACES_ASSETS_DIR"
```

Build the 24-category vocabulary from training scenes only:

```bash
python scripts/prepare_procthor/build_training_inventory.py \
  --manifest data/local_inputs/procthor/manifest.json \
  --output data/local_inputs/procthor/training_vocabulary.json \
  --cache "$MLSPACES_CACHE_DIR" \
  --assets "$MLSPACES_ASSETS_DIR" \
  --vocabulary-size 24
```

Render and check each split. These jobs are independent once the training vocabulary is frozen. Each scene should be assigned to one renderer process at a time.

```bash
for SPLIT in train validation test; do
  python scripts/prepare_procthor/render_prepared_scenes.py \
    --manifest data/local_inputs/procthor/manifest.json \
    --vocabulary data/local_inputs/procthor/training_vocabulary.json \
    --split "$SPLIT" \
    --output-dir data/local_inputs/procthor \
    --cache "$MLSPACES_CACHE_DIR" \
    --assets "$MLSPACES_ASSETS_DIR" \
    --width 320 --height 240 \
    --mark-zero-vocabulary-ineligible
  python scripts/prepare_procthor/validate_prepared_scenes.py \
    --manifest data/local_inputs/procthor/manifest.json \
    --scene-root data/local_inputs/procthor \
    --split "$SPLIT"
done
```

Full rendering processes 640 scenes and writes RGB-D views. For a scene tree prepared elsewhere, stage it at another path:

```bash
python scripts/prepare_local_inputs.py \
  --dataset procthor \
  --source /path/to/prepared/scenes \
  --destination data/local_inputs/procthor_external
```

Add `--mode copy` to copy rather than link. Set `SCENE_ROOT` to the prepared tree. Before policy evaluation, the runner lists the categories represented in `train/`; the preparation step separately freezes the 24-category inventory.

Run the main evaluation:

```bash
bash scripts/run_native_232.sh
```

It fits the observation-outcome model on training scenes, evaluates the methods at 8 and 12 actions, fits selective decision heads on training rows, applies them to test rows, and reduces test outputs to paired scene metrics. Generated outputs are written under `results/generated/native_232/`.

Recompute the checked-in paired intervals with:

```bash
python scripts/bootstrap_paired_scenes.py \
  --input results/procthor/native_232/primary/paired_scene_metrics.csv \
  --output results/generated/paired_scene_bootstrap.json
```

For fresh episode outputs, build the paired table with:

```bash
python scripts/summarize_paired_scenes.py \
  --input 8=results/generated/native_232/test_8_head_applied.jsonl \
  --input 12=results/generated/native_232/test_12_head_applied.jsonl \
  --output results/generated/native_232/paired_scene_metrics.csv \
  --split test
```

The paired table retains scene-level confusion counts and sums of risk, travel, and actions, sufficient to recalculate pooled metrics and paired percentile intervals.

## Additional evaluations

- HM3D multi-split: `python src/analysis/hm3d/evaluate_multisplit.py --help`; follow [`../protocols/hm3d/multisplit_preregistration.md`](../protocols/hm3d/multisplit_preregistration.md).
- ScanNet negative closure: `python src/analysis/hm3d/evaluate_negative_closure.py --help`; the input format is documented in [`../data/README.md`](../data/README.md).
- ProcTHOR policy ablation: `SCENE_ROOT=/path/to/prepared/scenes bash scripts/run_native_232_ablation.sh`. It evaluates the counterevidence controller, four SafeVantage mechanism ablations, and two comparison policies at 8 and 12 actions using the common decoder.
- Decision-resolution analyses: `python src/analysis/procthor/evaluate_decision_resolution.py --help`.

