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

## Experiment 2026-04-26 — Task A KNN Template Refinement

### Task
Task A

### Method
knn_template_refinement (alpha=0.3/0.2, top_k=5, start/end thresh=1.0km)

### Config
- config file: configs/task_a_advanced.yaml
- KDTree index on start+end points for fast candidate lookup
- alpha_8=0.3 (1/8 keep rate), alpha_16=0.2 (1/16 keep rate)

### Result
- val_input_8: MAE=216.09 m, RMSE=338.54 m — 2.35x WORSE than linear

### Observations
- KNN模板方法严重劣化（MAE从92m→216m）
- 根本原因：以整条轨迹的起终点搜索"相似轨迹"，但插补的是轨迹中的局部段落
- 历史轨迹形状无法对齐当前轨迹的局部间隙，导致模板坐标引入大量偏差
- alpha越大误差越大，alpha=0完全退化为线性插值
- **结论：轨迹级别KNN对该任务无效，应放弃**

### Analysis for Report/PPT
- 这是一次"有益的失败实验"：说明直觉方法不一定有效
- 证明了在无路网数据情况下线性插值已是近优方案
- 可用于报告的"失败案例分析"部分，体现方法探索深度

### Next Step
尝试 Catmull-Rom 样条插值（考虑已知点处的切线方向）

---

## Experiment 2026-04-26 — Task A Catmull-Rom Spline Interpolation

### Task
Task A

### Method
catmull_rom_interpolation

### Config
- config file: configs/task_a_advanced.yaml
- 边界用镜像反射填充（reflect last interval）
- 少于4个已知点时退化为线性插值

### Result
- val_input_8:  MAE=89.16 m, RMSE=116.18 m (vs linear: 92.04/121.82)
- val_input_16: MAE=166.22 m, RMSE=215.65 m (vs linear: 170.39/224.55)
- 1/8改善: MAE -3.1%, RMSE -4.6%
- 1/16改善: MAE -2.4%, RMSE -3.9%

### Observations
- Catmull-Rom在已知点处使用相邻点方向作为切线，能更好地跟随路网曲线
- 在简单直线段上与线性插值相同，在弯道段上改善明显
- 改善幅度有限（3-5%）：主要瓶颈是无路网信息，无法判断具体走哪条路
- 尝试了基于特征的残差学习（岭回归）：提升几乎为零（74.55m vs 74.58m）
  - 说明剩余误差是因道路曲率无法预测，非系统性偏差

### Analysis for Report/PPT
- Catmull-Rom是本项目Task A最终方法，相比linear有实质改善
- "无路网数据下线性插值已近最优"这一结论具有方法论意义
- 残差学习失败的分析（误差均值≈0，无可学信号）可写进局限性分析

### Next Step
切换默认方法为catmull_rom；进行Task B HistGBM优化

---

## Experiment 2026-04-26 19:58–20:01 — Task A PCHIP and Local Segment Template

### Task
Task A

### Method
1. pchip_time_interpolation
2. local_segment_template_interpolation

### Config
- config file: configs/task_a_advanced.yaml
- PCHIP: shape-preserving cubic Hermite interpolation over known timestamps
- Local segment template:
  - train data: data_ds15/train.pkl only
  - spans: [8, 16]
  - max_segments_per_span: 250000
  - samples_per_traj_span: 3
  - top_k: 20
  - alpha: 1.0
  - fallback: PCHIP for unsupported/boundary spans

### Result
- PCHIP:
  - val_input_8: MAE=87.59 m, RMSE=116.09 m
  - val_input_16: MAE=163.53 m, RMSE=216.03 m
- Local segment template:
  - val_input_8: MAE=64.73 m, RMSE=92.02 m
  - val_input_16: MAE=120.73 m, RMSE=168.34 m

### Observations
- PCHIP 比 Catmull-Rom 的 MAE 更低：1/8 从 89.16m 降到 87.59m，1/16 从 166.22m 降到 163.53m。
- 整条轨迹 KNN 失败不代表历史轨迹无效，失败点在于相似性粒度错误。
- 新方法改为“局部缺口模板”：只匹配相邻两个已知点之间的局部片段，按相同 span、局部位置和端点位移检索训练片段。
- 对候选训练片段计算相对于直线插值的弯曲残差，再加到当前缺口上，因此能学习常见道路弯曲形态。
- 该方法没有使用 data_org 或 val_gt 查表，只使用 data_ds15/train.pkl，属于可泛化训练集统计方法。
- 相比 Catmull-Rom，局部模板在 1/8 上 MAE 降低 27.4%，在 1/16 上 MAE 降低 27.4%。

### Analysis for Report/PPT
- 这是 Task A 的主要创新点：从“轨迹级相似”改为“缺口级相似”，解决了 KNN 模板无法局部对齐的问题。
- 方法可以解释为“物理插值 baseline + 历史局部道路形状残差修正”。
- 可在报告中对比三层路线：linear → PCHIP/Catmull 曲线插值 → local segment template 历史残差。

### Next Step
将默认 Task A 方法切换为 local_segment_template_interpolation；课堂测试时确保 data_ds15/train.pkl 路径可用。

---

## Experiment 2026-04-26 19:33 — Task B HistGradientBoosting v2

### Task
Task B

### Method
ensemble_with_speed_constraints (HistGradientBoostingRegressor + residual learning)

### Config
- config file: configs/task_b_advanced.yaml
- model: hist_gradient_boosting
- n_estimators: 500, max_depth: 6, learning_rate: 0.05
- residual_learning: enabled (train on residuals from time_bucket baseline)
- 新增特征: est_tt_from_npts (= (n-1)*15.64s), p25/p50/p75/p90_segment_distance, slow_segment_ratio

### Result
- MAE=19.31 s, RMSE=28.61 s, MAPE=1.67%
- vs GradientBoosting v1: MAE 22.77s→19.31s (-15%), MAPE 1.97%→1.67% (-15%)
- 训练速度: ~34s (vs GradientBoosting ~3.5min, 快6x)

### Observations
- **关键洞察**: num_points * 15.64 ≈ travel_time（采样间隔约15.64秒）
  - 该特征直接提供行程时间的近优估计
  - 结合mean_segment_distance可近似恢复平均速度
- HistGBM不仅更快（6x），效果也更好（直方图分箱对连续特征更鲁棒）
- 新增的分位数特征（p25/p50/p75/p90_segment_distance）帮助捕捉轨迹速度分布
- slow_segment_ratio捕捉走走停停的行为（交通拥堵指标）
- MAPE=1.67%意味着平均预测误差仅为真实时间的1.67%，已达到实用水平

### Analysis for Report/PPT
- MAPE从24%（全局速度baseline）→1.67%（HistGBM集成）是本项目最强的改进证据
- "采样间隔×点数"的洞察是一个物理可解释的强特征，适合在报告中重点介绍
- 三阶段方法链（全局速度→分时段速度→ML残差集成）展示了逐步改进的设计逻辑

### Next Step
整理最终提交脚本，更新消融表，生成可视化，完整commit

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

## Experiment 2026-05-04 14:16–14:17 — Task A Wide Local Segment Index

### Task
Task A

### Method
local_segment_template_interpolation (wide index)

### Config
- config file: configs/task_a_advanced.yaml
- train data: data/data_ds15/train.pkl
- spans: [8, 16]
- max_segments_per_span: 500000
- samples_per_traj_span: 8
- top_k: 20
- alpha: 1.0
- fallback: PCHIP for unsupported/boundary spans

### Result
- val_input_8: MAE=62.03 m, RMSE=89.07 m
- val_input_16: MAE=116.50 m, RMSE=163.59 m
- Compared with previous local segment template:
  - 1/8 MAE 64.73 -> 62.03 (-4.2%), RMSE 92.02 -> 89.07 (-3.2%)
  - 1/16 MAE 120.73 -> 116.50 (-3.5%), RMSE 168.34 -> 163.59 (-2.8%)

### Observations
- 子集参数搜索显示，主要收益来自扩大训练片段覆盖，而不是简单增大 top_k。
- top_k=20, alpha=1.0 在验证子集上 MAE 最低；top_k=40 会引入更多不够相似的片段，误差上升。
- 该改进仍只使用 data_ds15/train.pkl，不使用 data_org 或 val_gt 查表。
- 运行时间仍然适合课堂现场：单个验证文件约十几秒级，代价主要在 KDTree 索引构建。

### Analysis for Report/PPT
- 可以把该实验解释为“历史先验的覆盖度消融”：局部模板方法有效，但检索库覆盖不足会限制收益。
- 宽索引提升说明，同一 span 下的局部道路形状存在稀疏性，需要更充分采样训练片段。

### Next Step
继续尝试更强的局部特征表示，例如方向归一化残差、近邻距离自适应 alpha、以及按空间网格限制候选片段。

---

## Experiment 2026-05-04 14:21 — Task A Fine Search on Template Neighbors

### Task
Task A

### Method
local_segment_template_interpolation (wide index, top_k=12)

### Config
- config file: configs/task_a_advanced.yaml
- max_segments_per_span: 500000
- samples_per_traj_span: 8
- top_k: 12
- alpha: 1.0
- max_feature_distance: 2.5

### Result
- val_input_8: MAE=61.66 m, RMSE=89.21 m
- val_input_16: MAE=115.73 m, RMSE=163.61 m
- Compared with top_k=20 wide index:
  - 1/8 MAE 62.03 -> 61.66 (-0.6%), RMSE 89.07 -> 89.21 (+0.2%)
  - 1/16 MAE 116.50 -> 115.73 (-0.7%), RMSE 163.59 -> 163.61 (+0.0%)

### Observations
- 子集细搜索比较了 top_k in {8,12,15,20,25,30} 和 alpha in {0.9,1.0,1.1,1.2}。
- top_k=12, alpha=1.0 的 MAE 最优；top_k 继续增大容易混入局部形状不相似的片段。
- alpha>1 会放大历史残差噪声，MAE/RMSE 都明显变差。
- top_k=12 的 RMSE 相比 top_k=20 仅有 0.1m 量级波动，MAE 改善更稳定，因此采纳为默认配置。

### Analysis for Report/PPT
- 这个实验可以支撑“近邻数量不是越大越好”：局部模板依赖高质量相似片段，过多邻居会稀释局部道路形状先验。
- 可作为 Task A 参数消融补充，不需要占主要篇幅。

### Next Step
探索是否可以通过方向归一化、空间网格约束或近邻距离自适应权重进一步降低 RMSE。

---

## Experiment 2026-05-04 14:26–14:27 — Task A Adaptive Confidence Blend

### Task
Task A

### Method
local_segment_template_interpolation + adaptive confidence blend

### Config
- config file: configs/task_a_advanced.yaml
- max_segments_per_span: 500000
- samples_per_traj_span: 8
- top_k: 12
- alpha: 1.0
- confidence_blend: linear
- confidence_threshold: 1.2
- fallback: PCHIP

### Result
- val_input_8: MAE=61.51 m, RMSE=88.59 m
- val_input_16: MAE=115.37 m, RMSE=162.00 m
- Compared with top_k=12 wide index without confidence blend:
  - 1/8 MAE 61.66 -> 61.51 (-0.2%), RMSE 89.21 -> 88.59 (-0.7%)
  - 1/16 MAE 115.73 -> 115.37 (-0.3%), RMSE 163.61 -> 162.00 (-1.0%)

### Observations
- 近邻距离可以作为模板可信度：最近邻越远，说明训练集中缺少高度相似的局部道路形状。
- 对低置信缺口直接套用历史残差会增加异常误差；线性置信融合让这些缺口更多回退到 PCHIP。
- 子集搜索中 hard threshold 不稳定，linear confidence threshold=1.0~1.5 都有效，1.2 在 MAE/RMSE之间最稳。
- 该方法仍不使用 val_gt 或 data_org 查表，只改变训练片段残差与无监督插值 fallback 的融合权重。

### Analysis for Report/PPT
- 这是 Task A 可以重点讲的“可靠性控制”：历史先验不是无条件使用，而是用近邻相似度估计置信度。
- 该设计解释了为什么 RMSE 改善明显：低置信模板容易制造大误差，自适应回退能抑制尾部误差。

### Next Step
下一步可尝试方向归一化 residual 或局部空间网格约束，但需要警惕过拟合验证集。

---
