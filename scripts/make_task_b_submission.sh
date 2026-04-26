#!/bin/bash
# ============================================================
# Task B submission script
# Usage: bash scripts/make_task_b_submission.sh
#
# Modify TEST_INPUT when test data arrives.
# ============================================================

set -e
cd "$(dirname "$0")/.."

DATA_ROOT="data/student_release"
VAL_INPUT="${DATA_ROOT}/task_B_tte/val_input.pkl"
VAL_GT="${DATA_ROOT}/task_B_tte/val_gt.pkl"
TEST_INPUT="${DATA_ROOT}/task_B_tte/test_input.pkl"

MODEL_PATH="outputs/task_b/best_model.pkl"
CONFIG="configs/task_b_advanced.yaml"
OUT_DIR="outputs/submissions"

mkdir -p "$OUT_DIR"

echo "=== Task B: Validating on val set ==="
python -m src.task_b.run_task_b \
  --input "$VAL_INPUT" \
  --output "${OUT_DIR}/task_b_val.pkl" \
  --gt "$VAL_GT" \
  --model-path "$MODEL_PATH" \
  --config "$CONFIG" \
  --mode val

if [ -f "$TEST_INPUT" ]; then
  echo "=== Task B: Generating test submission ==="
  python -m src.task_b.run_task_b \
    --input "$TEST_INPUT" \
    --output "${OUT_DIR}/task_b_test.pkl" \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --mode predict

  echo "Test submission saved to ${OUT_DIR}/task_b_test.pkl"
else
  echo "Test input not found at $TEST_INPUT — skipping test prediction."
fi

echo "=== Done ==="
