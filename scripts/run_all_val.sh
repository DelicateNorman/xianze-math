#!/bin/bash
# Run all validation experiments and log results.
set -e
cd "$(dirname "$0")/.."

bash scripts/make_task_a_submission.sh
bash scripts/make_task_b_submission.sh
echo "All validations complete. See experiments/ for results."
