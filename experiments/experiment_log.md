# Experiment Log

---

## Experiment 2026-04-26 — Project Initialization

### Task
Both

### Method
Project setup, data exploration

### Config
- No experiments run yet

### Observations
- data_ds15/train.pkl: 132,657 trajectories, 50–240 points per trajectory
- data_ds15/val.pkl: 16,582 trajectories
- Task A val_input_8.pkl: 16,582 items (1/8 keep rate)
- Task A val_input_16.pkl: 16,582 items (1/16 keep rate)
- Trajectory coords format: [lon, lat] WGS-84, float32 in stored pkl
- Timestamps: Unix seconds integer
- Task A mask: True=known, False=missing (to predict)
- Task B input: full coords + departure_timestamp, predict travel_time in seconds

### Analysis for Report/PPT
- Data format confirmed, ready for implementation

### Next Step
Run Task A baseline (linear interpolation) and Task B baseline (global speed).

---

## Experiment 2026-04-26 18:56 — Task A Baseline

### Task
Task A

### Method
linear_time_interpolation

### Config
- config file: configs/task_a_baseline.yaml
- random seed: 42

### Result
- val_input_8:  MAE=92.04 m, RMSE=121.82 m (N=1,092,547 points)
- val_input_16: MAE=170.39 m, RMSE=224.55 m (N=1,170,232 points)

### Observations
- 1/16保留率比1/8误差约大85%，符合预期（缺口更长，直线假设偏差更大）
- 速度平滑方法（linear_with_speed_smoothing）在本数据集上与线性插值结果完全相同，说明线性插值后极少出现超速段（出租车在15秒采样间隔下速度通常合理）

### Analysis for Report/PPT
- linear interpolation 是稳定 baseline，适合展示"物理可解释基准线"
- 1/8 vs 1/16 的结果对比（85%误差增大）可以说明任务难度与保留率的关系

### Next Step
Task B 实验；考虑 KNN 模板增强 Task A

---

## Experiment 2026-04-26 18:57–19:02 — Task B 全量实验

### Task
Task B

### Method
global_speed_baseline → time_bucket_speed_model → ensemble_with_speed_constraints (残差GradientBoosting)

### Config
- configs/task_b_baseline.yaml / task_b_advanced.yaml
- random seed: 42
- GB: n_estimators=200, max_depth=5, lr=0.05
- residual_learning: enabled

### Result
| Method | MAE (s) | RMSE (s) | MAPE (%) |
|--------|---------|----------|----------|
| global_speed_baseline | 273.92 | 362.72 | 23.88 |
| time_bucket_speed_model | 238.53 | 316.43 | 20.65 |
| ensemble (GB + residual) | **22.77** | **33.63** | **1.97** |

### Observations
- 残差回归集成效果极为显著，MAE从238s降至22.77s（降低90%）
- 残差学习让模型专注于学习规则模型无法捕捉的复杂偏差
- 轨迹几何特征（total_distance, tortuosity, bearing_change等）贡献了主要预测能力

### Analysis for Report/PPT
- 三个方法的对比表非常适合报告：展示逐步改进的技术路线
- 残差学习是本项目核心创新点之一，MAPE从24%→2%是强有力的数据支撑
- "规则基础+ML残差"框架可以作为方法设计的核心叙事

### Next Step
生成可视化图表（EDA、误差分布、TTE散点图）；更新消融表

---
