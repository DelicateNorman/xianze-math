#!/bin/bash
# ============================================================
# Task B submission script
# Usage: bash scripts/make_task_b_submission.sh
#
# Modify TEST_INPUT when test data arrives.
# ============================================================

set -e
cd "$(dirname "$0")/.."

find_data_root() {
  for root in "${XIANZE_DATA_ROOT:-}" "${DATA_ROOT:-}" "data/student_release" "data"; do
    if [ -n "$root" ] && [ -d "${root}/task_B_tte" ]; then
      printf '%s\n' "$root"
      return
    fi
  done
  printf '%s\n' "data/student_release"
}

find_file() {
  local dir="$1"
  shift
  local result
  for pattern in "$@"; do
    result=$(find "$dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | sort | head -n 1)
    if [ -n "$result" ]; then
      printf '%s\n' "$result"
      return
    fi
  done
}

require_file() {
  local label="$1"
  local path="$2"
  if [ -z "$path" ] || [ ! -f "$path" ]; then
    echo "Missing ${label}. Set the corresponding environment variable or check DATA_ROOT." >&2
    exit 1
  fi
}

DATA_ROOT="$(find_data_root)"
TASK_B_DIR="${DATA_ROOT}/task_B_tte"

# Paths can be overridden without editing the script:
#   DATA_ROOT=/path/to/student_release bash scripts/make_task_b_submission.sh
#   TASK_B_TEST_INPUT=/path/to/file.pkl bash scripts/make_task_b_submission.sh
VAL_INPUT="${TASK_B_VAL_INPUT:-$(find_file "$TASK_B_DIR" "val_input.pkl" "*val*input*.pkl")}"
VAL_GT="${TASK_B_VAL_GT:-$(find_file "$TASK_B_DIR" "val_gt.pkl" "*gt*.pkl")}"
TEST_INPUT="${TASK_B_TEST_INPUT:-$(find_file "$TASK_B_DIR" "test_input.pkl" "*test*input*.pkl")}"

MODEL_PATH="outputs/task_b/best_model.pkl"
CONFIG="configs/task_b_advanced.yaml"
OUT_DIR="outputs/submissions"

mkdir -p "$OUT_DIR"

echo "Using DATA_ROOT=${DATA_ROOT}"
require_file "Task B val input" "$VAL_INPUT"
require_file "Task B val ground truth" "$VAL_GT"

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
