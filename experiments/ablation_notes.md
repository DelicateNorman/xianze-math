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
| **local_segment_template_interpolation (wide index, top_k=12)** | **61.66** | **89.21** | **115.73** | **163.61** | **当前默认**：更低MAE，RMSE基本持平 |
| **local_segment_template_interpolation (adaptive confidence blend)** | **61.51** | **88.59** | **115.37** | **162.00** | **当前最佳**：按近邻距离自适应融合PCHIP fallback |
| **local_segment_template_interpolation (dense 2M templates)** | **57.40** | **83.73** | **106.48** | **152.18** | **当前最佳**：大规模局部模板库 + 置信融合 |

**Key Insights:**
- 速度平滑对本数据集无效：15s采样下线性插值后基本无超速段
- KNN模板方法严重劣化：全轨迹级别的相似度无法对齐局部间隙形状
- Catmull-Rom给出3-5%改善：利用已知点切线方向更好跟随路网曲线
- PCHIP在MAE上进一步优于Catmull-Rom，但RMSE改善有限
- 局部缺口模板显著优于全轨迹KNN：关键是只匹配相同span的局部片段，并学习相对直线插值的弯曲残差
- 该方法只使用data_ds15/train.pkl，不使用data_org/val_gt查表，避免验证集泄漏
- 扩大局部片段索引覆盖（max_segments_per_span 25万→50万，samples_per_traj_span 3→8）后，最近邻检索更容易找到同区域/同方向的局部道路形状，1/8 MAE 64.73→62.03，1/16 MAE 120.73→116.50
- 细搜索显示 top_k=12 比 top_k=20 的 MAE 更低（1/8: 62.03→61.66，1/16: 116.50→115.73），RMSE只小幅波动；alpha>1 会明显放大噪声，不采纳
- 自适应置信融合按最近邻特征距离决定历史残差权重，近邻不够相似时回退更多 PCHIP，可同时降低 MAE 和 RMSE
- 将局部模板索引进一步扩展到 200 万片段/span 后，1/16 MAE 从 115.37 降到 106.48，说明该方法仍处于“历史局部道路形状覆盖不足”阶段，计算换覆盖能带来实质收益
- seed 消融在验证子集上显示 seed=7 略优于 seed=42（sum MAE 163.111 vs 163.360），但当前默认保留已全量验证的 seed=42 配置；seed=7 可作为下一轮全量复核候选
- **结论：Task A最终采用PCHIP fallback + dense local segment template residual correction + adaptive confidence blend**

## Task B Ablation Results

| Method | MAE (s) | RMSE (s) | MAPE (%) | Train Time | Notes |
|--------|---------|----------|----------|------------|-------|
| global_speed_baseline | 273.92 | 362.72 | 23.88 | <1s | 全局平均速度 |
| time_bucket_speed_model | 238.53 | 316.43 | 20.65 | <1s | 按小时分桶速度 |
| ensemble (GB + residual) | 22.77 | 33.63 | 1.97 | ~3.5min | GradientBoosting 200树 |
| ensemble (HistGBM + residual) | 19.31 | 28.61 | 1.67 | ~34s | 原最终方法 |
| n-count median lookup | 21.19 | 33.30 | 1.82 | <1s | 只利用点数结构，作为新残差底座 |
| HGB + n-count residual + phase features | 16.34 | 25.36 | 1.41 | ~30s | 单模型已超过16.37门槛 |
| XGB + n-count residual + phase features | 16.32 | 25.31 | 1.40 | ~7s/model | XGBoost残差模型 |
| **sampling_residual_ensemble** | **16.27** | **25.25** | **1.40** | **~2.5min** | **当前最佳**：点数中位数baseline + 采样相位增强 + HGB/XGB/LGBM残差集成 |

**Key Insights:**
- num_points × 15.64 ≈ travel_time（采样间隔约15.64s）是最强单特征
- 单纯点数查表可到 21.19s MAE，但需要几何/时间相位特征解释剩余残差
- 残差学习从“分时段速度baseline”改为“点数中位数baseline”后，学习目标更接近采样端点误差与交通扰动
- HistGBM比GradientBoosting快6x且效果更好（直方图分箱对连续特征更鲁棒）
- 新增分位数特征（p25/p50/p75/p90_segment_distance）和slow_segment_ratio有效捕捉速度分布
- LightGBM/XGBoost 需要 macOS OpenMP 运行库（conda-forge: llvm-openmp），装好后可参与集成
- **结论：采样结构baseline + phase增强特征 + 多模型残差集成是当前最强框架**
