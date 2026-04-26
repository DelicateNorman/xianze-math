# Ablation Notes

## Task A Ablation Results

| Method | 1/8 MAE (m) | 1/8 RMSE (m) | 1/16 MAE (m) | 1/16 RMSE (m) |
|--------|-------------|--------------|--------------|----------------|
| linear_time_interpolation | 92.04 | 121.82 | 170.39 | 224.55 |
| linear_with_speed_smoothing | 92.04 | 121.82 | 170.39 | 224.55 |
| knn_template_refinement | TBD | TBD | TBD | TBD |

注：speed_smoothing与baseline相同，说明线性插值后无超速异常段。

## Task B Ablation Results

| Method | MAE (s) | RMSE (s) | MAPE (%) |
|--------|---------|----------|----------|
| global_speed_baseline | 273.92 | 362.72 | 23.88 |
| time_bucket_speed_model | 238.53 | 316.43 | 20.65 |
| **ensemble (GB + residual)** | **22.77** | **33.63** | **1.97** |
