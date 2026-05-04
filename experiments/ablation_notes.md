# Ablation Notes

## Task A Ablation Results

| Method | 1/8 MAE (m) | 1/8 RMSE (m) | 1/16 MAE (m) | 1/16 RMSE (m) | Notes |
|--------|-------------|--------------|--------------|----------------|-------|
| linear_time_interpolation | 92.04 | 121.82 | 170.39 | 224.55 | Baseline |
| linear_with_speed_smoothing | 92.04 | 121.82 | 170.39 | 224.55 | 与baseline相同（无超速段）|
| knn_template_refinement | 216.09 | 338.54 | — | — | **2.35x worse** — 失败方法 |
| catmull_rom_interpolation | 89.16 | 116.18 | 166.22 | 215.65 | 曲线插值，优于linear |
| pchip_time_interpolation | 87.59 | 116.09 | 163.53 | 216.03 | 保形三次Hermite插值，MAE优于Catmull |
| **local_segment_template_interpolation** | **64.73** | **92.02** | **120.73** | **168.34** | **最终方法**：训练集局部缺口模板残差 |
| **local_segment_template_interpolation (wide index)** | **62.03** | **89.07** | **116.50** | **163.59** | **2026-05-04改进**：扩大训练片段索引覆盖 |

**Key Insights:**
- 速度平滑对本数据集无效：15s采样下线性插值后基本无超速段
- KNN模板方法严重劣化：全轨迹级别的相似度无法对齐局部间隙形状
- Catmull-Rom给出3-5%改善：利用已知点切线方向更好跟随路网曲线
- PCHIP在MAE上进一步优于Catmull-Rom，但RMSE改善有限
- 局部缺口模板显著优于全轨迹KNN：关键是只匹配相同span的局部片段，并学习相对直线插值的弯曲残差
- 该方法只使用data_ds15/train.pkl，不使用data_org/val_gt查表，避免验证集泄漏
- 扩大局部片段索引覆盖（max_segments_per_span 25万→50万，samples_per_traj_span 3→8）后，最近邻检索更容易找到同区域/同方向的局部道路形状，1/8 MAE 64.73→62.03，1/16 MAE 120.73→116.50
- **结论：Task A最终采用PCHIP fallback + wide local segment template residual correction**

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
