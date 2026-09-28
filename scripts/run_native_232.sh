#!/usr/bin/env bash
set -euo pipefail

: "${SCENE_ROOT:?Set SCENE_ROOT to an authorized ProcTHOR scene directory}"
OUTPUT_DIR="${OUTPUT_DIR:-results/generated/native_232}"
mkdir -p "$OUTPUT_DIR"
OUTCOME_MODEL="$OUTPUT_DIR/observation_outcome.json"
TRAINING_VOCABULARY="${TRAINING_VOCABULARY:-$OUTPUT_DIR/training_vocabulary.json}"

python scripts/build_training_vocabulary.py \
  --scene-root "$SCENE_ROOT" \
  --output "$TRAINING_VOCABULARY"

python src/safevantage/procthor/fit_observation_outcome.py \
  --scene-root "$SCENE_ROOT" \
  --output "$OUTCOME_MODEL"

METHODS="coverage_gain,detector_confidence_gain,safevantage_active,safevantage_dc_full"
for ACTIONS in 8 12; do
  for SPLIT in train test; do
    python src/safevantage/procthor/evaluate_procthor.py \
      --scene-root "$SCENE_ROOT" \
      --split "$SPLIT" \
      --sensor controlled \
      --methods "$METHODS" \
      --training-vocabulary "$TRAINING_VOCABULARY" \
      --outcome-model "$OUTCOME_MODEL" \
      --actions "$ACTIONS" \
      --geodesic-budget-m 20.0 \
      --common-decoder \
      --force-fixed-horizon \
      --rows "$OUTPUT_DIR/${SPLIT}_${ACTIONS}_head_input.jsonl" \
      --summary "$OUTPUT_DIR/${SPLIT}_${ACTIONS}_raw_summary.json"
  done

  # Fit on the training split only, then freeze and apply the decision head
  # to the test split. This mirrors the paper's end-to-end evaluation path.
  DECISION_MODEL="$OUTPUT_DIR/decision_head_${ACTIONS}.json"
  python src/analysis/procthor/evaluate_head_f1_cap.py train \
    --rows "$OUTPUT_DIR/train_${ACTIONS}_head_input.jsonl" \
    --safe-method safevantage_active \
    --output "$DECISION_MODEL"
  python src/analysis/procthor/evaluate_head_f1_cap.py apply \
    --model "$DECISION_MODEL" \
    --rows "$OUTPUT_DIR/test_${ACTIONS}_head_input.jsonl" \
    --output-rows "$OUTPUT_DIR/test_${ACTIONS}_head_applied.jsonl" \
    --summary "$OUTPUT_DIR/test_${ACTIONS}_head_applied_summary.json"
done

# Retain only compact per-scene sufficient statistics for paired uncertainty
# analysis; episode-level rows remain local run outputs.
python scripts/summarize_paired_scenes.py \
  --input "8=$OUTPUT_DIR/test_8_head_applied.jsonl" \
  --input "12=$OUTPUT_DIR/test_12_head_applied.jsonl" \
  --output "$OUTPUT_DIR/paired_scene_metrics.csv" \
  --split test
python scripts/bootstrap_paired_scenes.py \
  --input "$OUTPUT_DIR/paired_scene_metrics.csv" \
  --output "$OUTPUT_DIR/paired_scene_bootstrap.json"
