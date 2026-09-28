#!/usr/bin/env bash
set -euo pipefail

: "${SCENE_ROOT:?Set SCENE_ROOT to the prepared ProcTHOR scene directory}"
RUN_DIR="${OUTPUT_DIR:-results/generated/native_232}"
OUTPUT_DIR="$RUN_DIR/ablation"
mkdir -p "$OUTPUT_DIR"
VOCABULARY="${TRAINING_VOCABULARY:-$RUN_DIR/training_vocabulary.json}"
OUTCOME_MODEL="$RUN_DIR/observation_outcome.json"
METHODS="coverage_gain,detector_confidence_gain,counterevidence_active,safevantage_dc_no_value,safevantage_dc_no_observability,safevantage_dc_no_independence,safevantage_dc_no_belief"

if [[ ! -f "$VOCABULARY" ]]; then
  python scripts/build_training_vocabulary.py \
    --scene-root "$SCENE_ROOT" \
    --output "$VOCABULARY"
fi
if [[ ! -f "$OUTCOME_MODEL" ]]; then
  python src/safevantage/procthor/fit_observation_outcome.py \
    --scene-root "$SCENE_ROOT" \
    --output "$OUTCOME_MODEL"
fi

for ACTIONS in 8 12; do
  python src/safevantage/procthor/evaluate_procthor.py \
    --scene-root "$SCENE_ROOT" \
    --split test \
    --sensor controlled \
    --methods "$METHODS" \
    --training-vocabulary "$VOCABULARY" \
    --outcome-model "$OUTCOME_MODEL" \
    --actions "$ACTIONS" \
    --geodesic-budget-m 20.0 \
    --common-decoder \
    --force-fixed-horizon \
    --rows "$OUTPUT_DIR/test_${ACTIONS}_rows.jsonl" \
    --summary "$OUTPUT_DIR/summary_actions_${ACTIONS}.json"
done
