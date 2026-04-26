#!/bin/bash
# ============================================================
# Task A submission script
# Usage: bash scripts/make_task_a_submission.sh
#
# Modify TEST_INPUT_8 and TEST_INPUT_16 when test data arrives.
# ============================================================

set -e
cd "$(dirname "$0")/.."

# --- Paths (modify when test data arrives) ---
DATA_ROOT="data/student_release"
TEST_INPUT_8="${DATA_ROOT}/task_A_recovery/test_input_8.pkl"
TEST_INPUT_16="${DATA_ROOT}/task_A_recovery/test_input_16.pkl"

# For validation runs (always available)
VAL_INPUT_8="${DATA_ROOT}/task_A_recovery/val_input_8.pkl"
VAL_INPUT_16="${DATA_ROOT}/task_A_recovery/val_input_16.pkl"
VAL_GT="${DATA_ROOT}/task_A_recovery/val_gt.pkl"

METHOD="linear_with_speed_smoothing"
CONFIG="configs/task_a_advanced.yaml"
OUT_DIR="outputs/submissions"

mkdir -p "$OUT_DIR"

echo "=== Task A: Validating on val sets ==="
python -m src.task_a.run_task_a \
  --input "$VAL_INPUT_8" \
  --output "${OUT_DIR}/task_a_val_8.pkl" \
  --gt "$VAL_GT" \
  --method "$METHOD" \
  --config "$CONFIG" \
  --mode val

python -m src.task_a.run_task_a \
  --input "$VAL_INPUT_16" \
  --output "${OUT_DIR}/task_a_val_16.pkl" \
  --gt "$VAL_GT" \
  --method "$METHOD" \
  --config "$CONFIG" \
  --mode val

if [ -f "$TEST_INPUT_8" ]; then
  echo "=== Task A: Generating test submissions ==="
  python -m src.task_a.run_task_a \
    --input "$TEST_INPUT_8" \
    --output "${OUT_DIR}/task_a_test_8.pkl" \
    --method "$METHOD" \
    --config "$CONFIG" \
    --mode predict

  python -m src.task_a.run_task_a \
    --input "$TEST_INPUT_16" \
    --output "${OUT_DIR}/task_a_test_16.pkl" \
    --method "$METHOD" \
    --config "$CONFIG" \
    --mode predict

  echo "Test submissions saved to $OUT_DIR"
else
  echo "Test input not found at $TEST_INPUT_8 — skipping test prediction."
fi

echo "=== Done ==="
