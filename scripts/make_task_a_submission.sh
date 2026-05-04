#!/bin/bash
# ============================================================
# Task A submission script
# Usage: bash scripts/make_task_a_submission.sh
#
# Modify TEST_INPUT_8 and TEST_INPUT_16 when test data arrives.
# ============================================================

set -e
cd "$(dirname "$0")/.."

find_data_root() {
  for root in "${XIANZE_DATA_ROOT:-}" "${DATA_ROOT:-}" "data/student_release" "data"; do
    if [ -n "$root" ] && [ -d "${root}/task_A_recovery" ]; then
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
TASK_A_DIR="${DATA_ROOT}/task_A_recovery"

# Paths can be overridden without editing the script:
#   DATA_ROOT=/path/to/student_release bash scripts/make_task_a_submission.sh
#   TASK_A_TEST_INPUT_8=/path/to/file.pkl bash scripts/make_task_a_submission.sh
TEST_INPUT_8="${TASK_A_TEST_INPUT_8:-$(find_file "$TASK_A_DIR" "test_input_8.pkl" "*test*8*.pkl")}"
TEST_INPUT_16="${TASK_A_TEST_INPUT_16:-$(find_file "$TASK_A_DIR" "test_input_16.pkl" "*test*16*.pkl")}"

VAL_INPUT_8="${TASK_A_VAL_INPUT_8:-$(find_file "$TASK_A_DIR" "val_input_8.pkl" "*val*8*.pkl")}"
VAL_INPUT_16="${TASK_A_VAL_INPUT_16:-$(find_file "$TASK_A_DIR" "val_input_16.pkl" "*val*16*.pkl")}"
VAL_GT="${TASK_A_VAL_GT:-$(find_file "$TASK_A_DIR" "val_gt.pkl" "*gt*.pkl")}"

METHOD="local_segment_template_interpolation"
CONFIG="configs/task_a_advanced.yaml"
OUT_DIR="outputs/submissions"

mkdir -p "$OUT_DIR"

echo "Using DATA_ROOT=${DATA_ROOT}"
require_file "Task A val input 8" "$VAL_INPUT_8"
require_file "Task A val input 16" "$VAL_INPUT_16"
require_file "Task A val ground truth" "$VAL_GT"

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

echo "=== Task A: Generating test submissions when files exist ==="
if [ -n "$TEST_INPUT_8" ] && [ -f "$TEST_INPUT_8" ]; then
  python -m src.task_a.run_task_a \
    --input "$TEST_INPUT_8" \
    --output "${OUT_DIR}/task_a_test_8.pkl" \
    --method "$METHOD" \
    --config "$CONFIG" \
    --mode predict
else
  echo "Test input 8 not found — skipping 1/8 test prediction."
fi

if [ -n "$TEST_INPUT_16" ] && [ -f "$TEST_INPUT_16" ]; then
  python -m src.task_a.run_task_a \
    --input "$TEST_INPUT_16" \
    --output "${OUT_DIR}/task_a_test_16.pkl" \
    --method "$METHOD" \
    --config "$CONFIG" \
    --mode predict

  echo "Task A test submissions saved to $OUT_DIR"
else
  echo "Test input 16 not found — skipping 1/16 test prediction."
fi

echo "=== Done ==="
