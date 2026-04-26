# Ablation Notes

## Task A Ablation Results

| Method | 1/8 MAE (m) | 1/8 RMSE (m) | 1/16 MAE (m) | 1/16 RMSE (m) | Notes |
|--------|-------------|--------------|--------------|----------------|-------|
| linear_time_interpolation | 92.04 | 121.82 | 170.39 | 224.55 | Baseline |
| linear_with_speed_smoothing | 92.04 | 121.82 | 170.39 | 224.55 | 与baseline相同（无超速段）|
| knn_template_refinement | 216.09 | 338.54 | — | — | **2.35x worse** — 失败方法 |
| **catmull_rom_interpolation** | **89.16** | **116.18** | **166.22** | **215.65** | **最终方法** (-3.1% MAE) |

**Key Insights:**
- 速度平滑对本数据集无效：15s采样下线性插值后基本无超速段
- KNN模板方法严重劣化：全轨迹级别的相似度无法对齐局部间隙形状
- Catmull-Rom给出3-5%改善：利用已知点切线方向更好跟随路网曲线
- 残差学习无效（Mean deviation≈0）：剩余误差为不可预测的道路曲率噪声
- **结论：无路网数据下，Catmull-Rom已近最优**

## Task B Ablation Results

| Method | MAE (s) | RMSE (s) | MAPE (%) | Train Time | Notes |
|--------|---------|----------|----------|------------|-------|
| global_speed_baseline | 273.92 | 362.72 | 23.88 | <1s | 全局平均速度 |
| time_bucket_speed_model | 238.53 | 316.43 | 20.65 | <1s | 按小时分桶速度 |
| ensemble (GB + residual) | 22.77 | 33.63 | 1.97 | ~3.5min | GradientBoosting 200树 |
| **ensemble (HistGBM + residual)** | **19.31** | **28.61** | **1.67** | **~34s** | **最终方法** |

**Key Insights:**
- num_points × 15.64 ≈ travel_time（采样间隔约15.64s）是最强单特征
- 残差学习（TB baseline + ML残差）将MAPE从20.65%→1.67%（提升12x）
- HistGBM比GradientBoosting快6x且效果更好（直方图分箱对连续特征更鲁棒）
- 新增分位数特征（p25/p50/p75/p90_segment_distance）和slow_segment_ratio有效捕捉速度分布
- **结论：物理规则baseline + ML残差集成是本项目最强框架**
