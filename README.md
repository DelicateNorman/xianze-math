# Xi'an Taxi Trajectory Modeling

## Tasks
- **Task A**: Trajectory Recovery — restore missing GPS points from sparse trajectories
- **Task B**: Travel Time Estimation — predict trip duration from full path + departure time

## Environment
```bash
pip install -r requirements.txt
```

## Data
The code supports either released-data layout:
```
data/
├── data_ds15/
├── data_org/
├── task_A_recovery/
└── task_B_tte/
```
or:
```
data/student_release/
├── data_ds15/
├── data_org/
├── task_A_recovery/
└── task_B_tte/
```

You can also point scripts to another location without editing files:
```bash
DATA_ROOT=/path/to/student_release bash scripts/make_task_a_submission.sh
DATA_ROOT=/path/to/student_release bash scripts/make_task_b_submission.sh
```

## Run Task A Validation
```bash
python -m src.task_a.run_task_a \
  --input data/task_A_recovery/val_input_8.pkl \
  --gt    data/task_A_recovery/val_gt.pkl \
  --output outputs/submissions/task_a_val_8.pkl \
  --method local_segment_template_interpolation \
  --config configs/task_a_advanced.yaml \
  --mode val
```

## Run Task A Prediction
```bash
python -m src.task_a.run_task_a \
  --input data/task_A_recovery/test_input_8.pkl \
  --output outputs/submissions/task_a_test_8.pkl \
  --method local_segment_template_interpolation \
  --config configs/task_a_advanced.yaml \
  --mode predict
```

## Run Task B Validation
```bash
python -m src.task_b.run_task_b \
  --input  data/task_B_tte/val_input.pkl \
  --gt     data/task_B_tte/val_gt.pkl \
  --output outputs/submissions/task_b_val.pkl \
  --config configs/task_b_advanced.yaml \
  --mode val
```

## Run Task B Prediction
```bash
python -m src.task_b.run_task_b \
  --input      data/task_B_tte/test_input.pkl \
  --output     outputs/submissions/task_b_test.pkl \
  --model-path outputs/task_b/best_model.pkl \
  --config     configs/task_b_advanced.yaml \
  --mode predict
```

## Results
See `experiments/task_a_results.csv` and `experiments/task_b_results.csv`.

Latest Task A validation:
- `local_segment_template_interpolation`: 62.03 m MAE / 89.07 m RMSE on `val_input_8`
- `local_segment_template_interpolation`: 116.50 m MAE / 163.59 m RMSE on `val_input_16`
