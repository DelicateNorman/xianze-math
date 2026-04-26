# Xi'an Taxi Trajectory Modeling

## Tasks
- **Task A**: Trajectory Recovery — restore missing GPS points from sparse trajectories
- **Task B**: Travel Time Estimation — predict trip duration from full path + departure time

## Environment
```bash
pip install -r requirements.txt
```

## Data
Place released data under:
```
data/student_release/
```
Or symlink:
```bash
ln -s /path/to/student_release data/student_release
```

## Run Task A Validation
```bash
python -m src.task_a.run_task_a \
  --input data/student_release/task_A_recovery/val_input_8.pkl \
  --gt    data/student_release/task_A_recovery/val_gt.pkl \
  --output outputs/submissions/task_a_val_8.pkl \
  --method linear_with_speed_smoothing \
  --config configs/task_a_advanced.yaml \
  --mode val
```

## Run Task A Prediction
```bash
python -m src.task_a.run_task_a \
  --input data/student_release/task_A_recovery/test_input_8.pkl \
  --output outputs/submissions/task_a_test_8.pkl \
  --method linear_with_speed_smoothing \
  --config configs/task_a_advanced.yaml \
  --mode predict
```

## Run Task B Validation
```bash
python -m src.task_b.run_task_b \
  --input  data/student_release/task_B_tte/val_input.pkl \
  --gt     data/student_release/task_B_tte/val_gt.pkl \
  --output outputs/submissions/task_b_val.pkl \
  --config configs/task_b_advanced.yaml \
  --mode val
```

## Run Task B Prediction
```bash
python -m src.task_b.run_task_b \
  --input      data/student_release/task_B_tte/test_input.pkl \
  --output     outputs/submissions/task_b_test.pkl \
  --model-path outputs/task_b/best_model.pkl \
  --config     configs/task_b_advanced.yaml \
  --mode predict
```

## Results
See `experiments/task_a_results.csv` and `experiments/task_b_results.csv`.
