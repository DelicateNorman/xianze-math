# 课堂汇报结构

## 1. 问题背景（1 min）
- 西安2016年10月出租车GPS轨迹数据，132,657条训练轨迹
- 任务A：轨迹修复 —— 给定稀疏保留点，恢复缺失GPS坐标
- 任务B：行程时间估计 —— 给定完整路径和出发时间，预测总行程时间
- 指标：A用Haversine MAE/RMSE（米），B用MAE/RMSE/MAPE（秒/%）

## 2. 数据观察（1.5 min）
- 图：轨迹空间分布（热力图）
- 图：轨迹长度分布（50~240点）
- 图：行程时间分布（10~68 min）
- 图：不同时段平均速度（早晚高峰明显下降）
- 结论：数据具有明显时间周期性和空间集中性，需要时空建模

## 3. 整体方法框架（0.5 min）
```
原始轨迹
→ 时空特征提取（距离/速度/方向/时段）
→ Task A：稀疏插值 + 速度约束 + 历史模板
→ Task B：特征工程 + 分时段速度基线 + 残差回归
→ 评估与误差分析
```

## 4. Task A 方法（2 min）
流程图：
```
稀疏输入
→ PCHIP保形插值（稳定fallback）
→ 按相邻已知点切分局部缺口
→ 从训练集检索相同span的相似局部片段
→ 学习历史片段相对直线插值的弯曲残差
→ residual correction
→ 输出恢复轨迹
```
- 重点讲清楚：整条轨迹KNN失败，因为全局OD相似无法保证局部缺口对齐
- 新方法是“缺口级相似”，只使用 data_ds15/train.pkl，不用 data_org/val_gt 查表
- 展示1/8和1/16两种难度下的轨迹案例图（真值vs预测vs输入）

## 5. Task A 结果（1 min）
方法对比表（见 experiments/ablation_notes.md）
- linear: 92.04m / 170.39m MAE
- Catmull-Rom: 89.16m / 166.22m MAE
- PCHIP: 87.59m / 163.53m MAE
- local segment template: 64.73m / 120.73m MAE
- wide local segment template: 62.03m / 116.50m MAE
- wide local segment template (top_k=12): 61.66m / 115.73m MAE
- adaptive confidence blend: 61.51m / 115.37m MAE，RMSE 88.59m / 162.00m
- dense 2M template index: 57.40m / 106.48m MAE，RMSE 83.73m / 152.18m
- 可讲消融：扩大训练片段覆盖后，检索到同区域/同方向局部道路形状的概率更高；盲目增大 top-k 或 alpha 反而会混入/放大不相似片段
- 可讲机制：最近邻距离越大，历史模板越不可信，因此更多回退到 PCHIP，主要减少尾部大误差
- 可讲突破：不使用外部路网，用更密的训练片段库近似“局部道路形状先验”，1/16 MAE 相对原局部模板下降 14.25m
误差分布图（见 outputs/figures/）

## 6. Task B 方法（2 min）
流程图：
```
完整轨迹 + 出发时间
→ 距离/形状/时间/空间特征提取
→ 分时段速度模型（基础估计）
→ GradientBoosting 学习残差
→ 叠加 → 最终预测
```
- 特征重要性图

## 7. Task B 结果（1 min）
- 方法对比表
- 真实时间vs预测时间散点图

## 8. 误差分析与总结（1 min）
- Task A：1/16下两已知点间隔长，直线插值误差大，为何弯道路段更难
- Task B：极短行程MAPE高（小分母问题），高峰期误差略大
- 创新点回顾
- 未来工作：路网增强、序列模型
