# CLAUDE.md

本文件用于约束 Claude Code / Claude CLI 在本项目中的开发行为、工程规范、实验记录方式与最终交付要求。请将本文件放在项目根目录，并在后续开发中持续遵守。

---

## 0. 项目背景

本项目为数学建模课程小组作业，基于 **2016 年 10 月中国西安市出租车 GPS 轨迹数据** 完成两个独立评分任务：

* **任务 A：轨迹修复（Trajectory Recovery）**
* **任务 B：行程时间估计（Travel Time Estimation, TTE）**

本项目不仅追求验证集和课堂测试集上的指标表现，也需要为最终课程展示和大作业报告积累完整材料，包括：

* 问题分析
* 数据探索与可视化
* 方法设计
* 创新点总结
* 实验结果记录
* 消融实验与方法对比
* 失败尝试分析
* 小组分工说明
* 可复现代码与运行说明

Claude 在编写代码的同时，必须同步维护实验记录和方法文档，不能只写代码不写说明。

---

## 1. 项目目标

### 1.1 总体目标

构建一个结构清晰、可复现、便于课堂现场运行的数学建模工程项目，完成以下目标：

1. 能够读取课程提供的 `.pkl` 数据文件。
2. 能够完成任务 A 的轨迹修复，输出符合要求的 `.pkl` 预测文件。
3. 能够完成任务 B 的行程时间估计，输出符合要求的 `.pkl` 预测文件。
4. 能够在验证集上自动评估 MAE、RMSE、MAPE 等指标。
5. 能够记录每次实验的配置、方法、指标和结果文件。
6. 能够为最终报告和 PPT 自动积累图表、结果表格和方法说明。
7. 能够在课堂测试时快速切换到 test input，并稳定生成提交文件。

### 1.2 工程目标

项目必须满足以下工程要求：

* 代码结构模块化，不允许所有逻辑堆在一个脚本中。
* 每个任务必须有独立入口脚本。
* 每个任务必须支持验证集评估和测试集预测两种模式。
* 所有路径、超参数、方法选择应尽量通过配置文件控制。
* 所有实验结果必须自动保存。
* 每完成一个重要功能或模块，必须进行一次 Git commit。
* 任何可能影响实验结果的修改，必须在提交信息和实验日志中说明。

---

## 2. 数据与任务理解

### 2.1 数据目录

课程释放数据目录大致如下：

```text
student_release/
├── 作业说明.txt
├── data_org/
│   ├── train.pkl
│   └── val.pkl
├── data_ds15/
│   ├── train.pkl
│   └── val.pkl
├── task_A_recovery/
│   ├── val_input_8.pkl
│   ├── val_input_16.pkl
│   └── val_gt.pkl
└── task_B_tte/
    ├── val_input.pkl
    └── val_gt.pkl
```

其中：

* `data_org`：原始采样率数据，约 3 秒一个点。
* `data_ds15`：降采样数据，约 15 秒一个点。
* 两个文件夹中的轨迹索引一一对应，也可以通过 `order_id` 关联。
* 两个任务的测试数据均基于 `data_ds15` 生成。

### 2.2 轨迹格式

训练数据中，每条轨迹是一个 dict，包含：

```python
{
    "vehicle_id": ...,      # 车辆 ID
    "order_id": ...,        # 订单 ID
    "coords": ...,          # 经纬度坐标，WGS-84，通常为 [lon, lat]
    "timestamps": ...       # Unix 秒时间戳
}
```

### 2.3 任务 A：轨迹修复

任务 A 的目标是：给定一条轨迹中少量保留点，恢复中间被置为 NaN 的 GPS 坐标。

输入包含：

```python
{
    "traj_id": ...,
    "timestamps": ...,
    "coords": ...,      # 已知点为真实坐标，未知点为 NaN
    "mask": ...         # True 表示已知点，False 表示待预测点
}
```

需要分别处理：

* `val_input_8.pkl`：约 1/8 保留率
* `val_input_16.pkl`：约 1/16 保留率

提交格式：

```python
{
    "traj_id": ...,
    "coords": ...       # 完整坐标，NaN 已被替换，已知点保持不变
}
```

评测指标：

* MAE：所有待预测点的 Haversine 距离平均值，单位为米。
* RMSE：所有待预测点的 Haversine 距离均方根，单位为米。

### 2.4 任务 B：行程时间估计

任务 B 的目标是：给定完整路径坐标序列和出发时间，预测整个行程耗时。

输入包含：

```python
{
    "traj_id": ...,
    "coords": ...,
    "departure_timestamp": ...
}
```

提交格式：

```python
{
    "traj_id": ...,
    "travel_time": ...   # 单位：秒
}
```

评测指标：

* MAE：平均绝对误差，单位为秒。
* RMSE：均方根误差，单位为秒。
* MAPE：平均绝对百分比误差，单位为百分比。

---

## 3. 推荐项目结构

Claude 应优先按照以下结构组织项目：

```text
xian-taxi-modeling/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── task_a_baseline.yaml
│   ├── task_a_advanced.yaml
│   ├── task_b_baseline.yaml
│   └── task_b_advanced.yaml
├── data/
│   └── student_release/              # 本地数据，不提交 Git
├── src/
│   ├── common/
│   │   ├── io.py                      # pkl 读写
│   │   ├── geo.py                     # Haversine、距离、速度、方位角
│   │   ├── time_features.py           # 小时、星期、节假日/时段特征
│   │   ├── logging_utils.py           # 日志工具
│   │   └── seed.py                    # 随机种子
│   ├── task_a/
│   │   ├── dataset.py
│   │   ├── methods.py                 # 插值、速度约束、模板匹配等方法
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── run_task_a.py
│   ├── task_b/
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── models.py                  # 规则模型、ML 模型、集成模型
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── run_task_b.py
│   └── visualization/
│       ├── plot_trajectories.py
│       ├── plot_error_distribution.py
│       └── plot_feature_analysis.py
├── scripts/
│   ├── eda.py
│   ├── run_all_val.sh
│   ├── make_task_a_submission.sh
│   └── make_task_b_submission.sh
├── outputs/
│   ├── task_a/
│   ├── task_b/
│   ├── figures/
│   └── submissions/
├── experiments/
│   ├── experiment_log.md
│   ├── task_a_results.csv
│   ├── task_b_results.csv
│   └── ablation_notes.md
├── report_materials/
│   ├── method_notes.md
│   ├── innovation_points.md
│   ├── presentation_notes.md
│   └── figures_to_use.md
└── tests/
    ├── test_io.py
    ├── test_geo.py
    └── test_submission_format.py
```

注意：

* `data/`、`outputs/` 中的大文件不应提交到 Git。
* `experiments/` 和 `report_materials/` 需要提交到 Git，因为它们是报告和 PPT 的素材来源。
* 所有预测文件应保存在 `outputs/submissions/` 下。

---

## 4. Git 版本控制规范

### 4.1 基本原则

本项目必须使用 Git 控制版本。Claude 每完成一个重要功能、模块、实验或文档更新，都必须提醒用户进行 commit，或者在用户允许的情况下直接执行 commit。

不得在大量修改堆积后才提交。

### 4.2 Commit 频率

以下情况必须 commit：

1. 完成项目初始化。
2. 完成数据读取模块。
3. 完成 Haversine 评估函数。
4. 完成任务 A baseline。
5. 完成任务 A 验证集评估脚本。
6. 完成任务 A 改进方法。
7. 完成任务 A 提交文件生成脚本。
8. 完成任务 B baseline。
9. 完成任务 B 特征工程。
10. 完成任务 B 模型训练与验证。
11. 完成任务 B 提交文件生成脚本。
12. 完成 EDA 可视化。
13. 完成实验结果汇总。
14. 完成报告素材文档。
15. 完成课堂展示素材整理。

### 4.3 Commit Message 格式

统一使用如下格式：

```text
[type] short description
```

推荐 type：

* `[init]`：项目初始化
* `[data]`：数据读取、数据处理
* `[metric]`：评测指标
* `[task-a]`：任务 A 相关功能
* `[task-b]`：任务 B 相关功能
* `[model]`：模型或算法更新
* `[exp]`：实验结果更新
* `[viz]`：可视化更新
* `[doc]`：文档更新
* `[fix]`：bug 修复
* `[refactor]`：代码重构

示例：

```text
[init] create project structure and basic config files
[data] add pickle loading and trajectory validation utilities
[metric] implement haversine MAE and RMSE for trajectory recovery
[task-a] add linear interpolation baseline for recovery task
[task-b] add distance-speed baseline for travel time estimation
[exp] record validation results for task A interpolation methods
[doc] update method notes and innovation summary for presentation
```

### 4.4 分支建议

推荐使用以下分支：

```text
main              # 稳定版本，可用于课堂测试
feature/task-a    # 任务 A 开发
feature/task-b    # 任务 B 开发
feature/eda       # 数据探索与可视化
feature/report    # 报告与 PPT 素材整理
```

如果时间紧张，可以只使用 `main`，但必须保持小步 commit。

### 4.5 .gitignore 要求

`.gitignore` 至少包含：

```text
__pycache__/
*.pyc
.env
.venv/
venv/
.DS_Store

# data
data/
*.pkl

# outputs
outputs/

# notebooks checkpoints
.ipynb_checkpoints/

# logs
*.log
```

注意：如果需要提交小型示例文件，应放在 `examples/` 中，不要提交完整课程数据。

---

## 5. 任务 A 实现路径：轨迹修复

### 5.1 任务 A 的核心理解

任务 A 本质上是一个时空轨迹插补问题。输入点是等间隔时间序列，部分坐标已知，部分坐标缺失。需要恢复缺失点的经纬度。

最直接的 baseline 是在已知点之间做线性插值。但出租车轨迹不是严格直线运动，因此可以进一步考虑：

* 经纬度空间插值
* 基于时间比例的插值
* 基于速度平滑的插值
* 基于历史轨迹模板的修正
* 基于道路方向或城市路网结构的修正
* 基于异常速度过滤的后处理

### 5.2 任务 A 第一阶段：可靠 baseline

必须先实现一个可靠 baseline，不要一开始就写复杂模型。

Baseline 方法：

1. 对每条轨迹读取 `coords`、`timestamps`、`mask`。
2. 找到所有已知点。
3. 对经度和纬度分别按照时间戳做一维线性插值。
4. 保证已知点坐标不被修改。
5. 输出完整坐标序列。
6. 在验证集上只对 `mask == False` 的点计算 MAE 和 RMSE。

该方法应命名为：

```text
linear_time_interpolation
```

需要记录：

* `val_input_8.pkl` 上的 MAE / RMSE
* `val_input_16.pkl` 上的 MAE / RMSE
* 两个难度下误差分布图
* 若干条轨迹的可视化对比图

### 5.3 任务 A 第二阶段：速度与方向约束修正

在线性插值基础上加入简单物理约束：

* 计算相邻预测点之间的速度。
* 若速度超过合理阈值，进行平滑处理。
* 对急剧跳变点进行局部修正。
* 可加入移动平均或 Savitzky-Golay 平滑，但要避免把已知点也平滑掉。

建议方法名：

```text
linear_with_speed_smoothing
```

需要比较：

* 纯线性插值
* 速度平滑后处理
* 不同速度阈值的效果

### 5.4 任务 A 第三阶段：历史轨迹 / KNN 模板修正

可以利用训练集中的完整轨迹作为历史模板。思路：

1. 对每条待修复轨迹，提取其已知点作为查询片段。
2. 在训练集中寻找起终点相近、方向相近、路径形状相似的历史轨迹。
3. 使用相似轨迹对线性插值结果进行修正。
4. 可将最终结果设为：

```text
final_prediction = alpha * linear_prediction + (1 - alpha) * template_prediction
```

相似度特征可以包括：

* 起点距离
* 终点距离
* 总位移方向
* 轨迹包围盒
* 已知点对齐误差
* 轨迹总长度近似

建议方法名：

```text
knn_template_refinement
```

注意：

* KNN 方法可能计算量较大，必须先做候选过滤。
* 可先用起点和终点附近距离过滤候选轨迹。
* 必须保证课堂测试时运行时间可接受。
* 如果性能不稳定，保留为 advanced 方法，不影响 baseline 提交。

### 5.5 任务 A 第四阶段：可选路网增强

如果时间允许，可以使用 OSM 路网信息进行 map matching 或道路方向约束。但由于课程现场测试需要稳定运行，不应让路网方法成为唯一方案。

可选创新描述：

* 利用城市道路结构约束轨迹补全方向。
* 将线性插值结果投影到可能道路走廊中。
* 结合历史轨迹密度进行非直线修复。

工程要求：

* 路网增强必须可开关。
* 没有路网文件时，代码仍然能运行 baseline。

---

## 6. 任务 B 实现路径：行程时间估计

### 6.1 任务 B 的核心理解

任务 B 本质上是一个轨迹级回归问题。输入是完整路径坐标和出发时间，需要预测总行程时间。

由于输入包含完整路径坐标，因此可以构造较强的手工特征：

* 路径总距离
* 起终点直线距离
* 曲折度
* 点数
* 平均点间距
* 起点/终点空间区域
* 出发小时
* 是否高峰期
* 星期几
* 车辆或城市区域历史速度统计

### 6.2 任务 B 第一阶段：规则 baseline

先实现一个基于距离和平均速度的 baseline：

1. 计算轨迹 Haversine 总距离。
2. 按训练集估计全局平均速度。
3. 预测：

```text
travel_time = total_distance / global_average_speed
```

建议方法名：

```text
global_speed_baseline
```

需要记录：

* MAE
* RMSE
* MAPE
* 误差分布
* 不同时段误差分析

### 6.3 任务 B 第二阶段：分时段速度模型

考虑早晚高峰和不同时间段速度差异。

特征：

* `hour`
* `day_of_week`
* `is_peak_hour`
* `is_night`

方法：

* 按小时统计训练集平均速度。
* 按工作日/周末统计平均速度。
* 预测时使用对应时段速度。

建议方法名：

```text
time_bucket_speed_model
```

### 6.4 任务 B 第三阶段：机器学习回归模型

在规则 baseline 的基础上构造特征，训练回归模型。

推荐模型：

* Ridge Regression
* RandomForestRegressor
* GradientBoostingRegressor
* XGBoost / LightGBM（如果安装方便）

如果环境不稳定，优先使用 scikit-learn 内置模型。

核心特征建议：

```text
spatial features:
- total_distance
- straight_distance
- tortuosity = total_distance / straight_distance
- num_points
- mean_segment_distance
- max_segment_distance
- std_segment_distance
- start_lon, start_lat
- end_lon, end_lat
- delta_lon, delta_lat
- bounding_box_width
- bounding_box_height

time features:
- departure_hour
- day_of_week
- is_peak_hour
- is_weekend
- minute_of_day

trajectory shape features:
- mean_bearing_change
- std_bearing_change
- stop_like_ratio if inferable
```

建议方法名：

```text
feature_regression_model
```

### 6.5 任务 B 第四阶段：集成与后处理

可以将多个模型结果集成：

```text
prediction = w1 * time_bucket_speed + w2 * random_forest + w3 * gradient_boosting
```

也可以对预测值进行合理约束：

* 不小于某个最短时间。
* 不大于某个极端上限。
* 根据路径距离限制速度范围。

建议方法名：

```text
ensemble_with_speed_constraints
```

---

## 7. 实验记录要求

Claude 每完成一个方法或实验，必须更新：

```text
experiments/experiment_log.md
experiments/task_a_results.csv
experiments/task_b_results.csv
```

### 7.1 experiment_log.md 格式

每次实验记录如下：

```markdown
## Experiment YYYY-MM-DD HH:MM

### Task
Task A / Task B

### Method
方法名称

### Config
- config file:
- important parameters:
- random seed:

### Result
- MAE:
- RMSE:
- MAPE:  # only for Task B

### Observations
- 哪些样本表现好？
- 哪些样本误差大？
- 是否存在异常？

### Analysis for Report/PPT
- 这个实验可以支撑什么结论？
- 是否体现创新点？
- 是否适合放入最终报告或 PPT？

### Next Step
下一步要做什么。
```

### 7.2 task_a_results.csv 字段

```text
experiment_id,datetime,method,input_file,mae_meter,rmse_meter,config,notes
```

### 7.3 task_b_results.csv 字段

```text
experiment_id,datetime,method,mae_second,rmse_second,mape_percent,config,notes
```

---

## 8. 报告与 PPT 素材沉淀要求

本课程最终需要提交大作业报告 PDF 和课堂汇报 PPT。因此 Claude 在开发过程中必须同步维护以下文档。

### 8.1 method_notes.md

记录每个方法的原理，要求能直接改写进报告。

每个方法包含：

```markdown
## 方法名称

### 核心思想

### 输入与输出

### 算法流程

### 为什么这样设计

### 优点

### 局限性
```

### 8.2 innovation_points.md

记录项目创新点。建议从以下角度组织：

1. **数据驱动的轨迹修复**
   不仅使用简单插值，还结合历史轨迹模式、速度约束和轨迹形状相似性。

2. **物理可解释的时空约束**
   通过速度、方向、时间间隔等约束减少不合理预测。

3. **面向行程时间估计的多层特征建模**
   结合距离、轨迹形状、出发时段和城市空间位置构造特征。

4. **可复现实验管理**
   每个方法都有配置、指标、输出文件和实验日志，方便报告分析和课堂展示。

5. **课堂测试友好的稳定工程设计**
   baseline、advanced、submission 生成脚本相互独立，保证现场数据到来后能快速运行。

### 8.3 presentation_notes.md

记录课堂汇报可用内容，结构建议：

```markdown
# 课堂汇报结构

## 1. 问题背景
- 西安出租车轨迹数据
- 两个任务：轨迹修复与行程时间估计

## 2. 数据观察
- 轨迹长度分布
- 时间分布
- 空间分布
- 速度分布

## 3. 任务 A 方法
- baseline
- 改进方法
- 创新点
- 验证集结果

## 4. 任务 B 方法
- baseline
- 特征工程
- 回归模型
- 验证集结果

## 5. 实验分析
- 方法对比
- 消融实验
- 误差案例

## 6. 总结
- 最终方法
- 主要创新
- 局限与未来改进
```

### 8.4 figures_to_use.md

记录哪些图可以用于报告和 PPT。

图表建议包括：

* 轨迹空间分布热力图
* 单条轨迹修复前后对比图
* Task A 不同方法 MAE/RMSE 对比柱状图
* Task A 误差分布直方图
* Task B 真实时间 vs 预测时间散点图
* Task B 不同时段平均速度图
* Task B 不同方法指标对比表
* 特征重要性图

---

## 9. 评估函数要求

### 9.1 Haversine 距离

必须实现 Haversine 距离，单位为米。

要求：

* 输入经纬度格式为 `[lon, lat]`。
* 支持 numpy array 批量计算。
* 对 NaN 做必要处理。
* 不允许用欧氏距离直接代替地理距离作为最终评测指标。

### 9.2 Task A 评估

Task A 只评估被 mask 掉的位置，即：

```python
mask == False
```

已知点不参与 MAE/RMSE 计算。

同时必须检查：

* 输出长度是否与输入一致。
* `traj_id` 是否一致。
* 已知点是否保持不变。
* 输出中是否还有 NaN。

### 9.3 Task B 评估

Task B 需要计算：

* MAE
* RMSE
* MAPE

同时必须检查：

* `traj_id` 是否一致。
* `travel_time` 是否为正数。
* 是否存在 NaN 或 inf。
* 单位是否为秒。

---

## 10. 课堂测试运行要求

课堂测试时会发放：

* Task A: `test_input_8.pkl`, `test_input_16.pkl`
* Task B: `test_input.pkl`

代码必须支持通过命令行指定输入路径和输出路径。

### 10.1 Task A 测试运行示例

```bash
python -m src.task_a.run_task_a \
  --input data/student_release/task_A_recovery/test_input_8.pkl \
  --output outputs/submissions/task_a_test_8.pkl \
  --method linear_with_speed_smoothing \
  --config configs/task_a_advanced.yaml \
  --mode predict

python -m src.task_a.run_task_a \
  --input data/student_release/task_A_recovery/test_input_16.pkl \
  --output outputs/submissions/task_a_test_16.pkl \
  --method linear_with_speed_smoothing \
  --config configs/task_a_advanced.yaml \
  --mode predict
```

### 10.2 Task B 测试运行示例

```bash
python -m src.task_b.run_task_b \
  --input data/student_release/task_B_tte/test_input.pkl \
  --output outputs/submissions/task_b_test.pkl \
  --model-path outputs/task_b/best_model.pkl \
  --config configs/task_b_advanced.yaml \
  --mode predict
```

### 10.3 一键运行脚本

必须准备课堂测试用脚本：

```bash
bash scripts/make_task_a_submission.sh
bash scripts/make_task_b_submission.sh
```

脚本中需要有清晰注释，方便现场快速修改测试文件路径。

---

## 11. 代码质量要求

### 11.1 基本要求

* Python 代码应尽量使用类型标注。
* 关键函数必须有 docstring。
* 不要写过长函数。
* 不要在函数内部硬编码数据路径。
* 不要依赖 notebook 才能运行。
* 不要默认覆盖已有实验结果。
* 输出文件名应包含任务、方法、时间戳或实验 ID。

### 11.2 随机性控制

如果使用机器学习模型，必须设置随机种子。

建议统一 seed：

```python
SEED = 42
```

### 11.3 异常处理

代码应能处理：

* 文件路径不存在
* pkl 格式不符合预期
* coords 中存在 NaN
* 轨迹点数异常
* travel_time 非正数
* 模型文件不存在

### 11.4 性能要求

课堂测试需要现场运行，因此：

* baseline 方法必须非常快。
* advanced 方法如果较慢，必须提供 fallback。
* KNN 或模板匹配方法必须有候选过滤和最大候选数量限制。
* 不能在预测阶段进行过重训练。
* Task B 模型应提前训练并保存，测试时只加载模型预测。

---

## 12. 推荐开发顺序

Claude 应按以下顺序开发，不要跳步。

### 阶段 1：工程初始化

1. 创建项目结构。
2. 创建 `.gitignore`。
3. 创建 `requirements.txt`。
4. 创建 `README.md` 初版。
5. 创建基础配置文件。
6. Git commit：

```text
[init] create project structure and basic files
```

### 阶段 2：通用工具

1. 实现 pkl 读写。
2. 实现 Haversine 距离。
3. 实现轨迹基础统计函数。
4. 实现日志工具。
5. 写基础测试。
6. Git commit：

```text
[data] add common io and geo utilities
```

### 阶段 3：EDA

1. 统计轨迹数量、长度、时间范围。
2. 绘制轨迹长度分布。
3. 绘制空间分布。
4. 绘制速度分布。
5. 更新 `report_materials/figures_to_use.md`。
6. Git commit：

```text
[viz] add exploratory data analysis and trajectory figures
```

### 阶段 4：任务 A baseline

1. 实现线性插值。
2. 实现 Task A 评估。
3. 在 val_input_8 和 val_input_16 上测试。
4. 更新实验记录。
5. Git commit：

```text
[task-a] implement linear interpolation baseline and evaluation
```

### 阶段 5：任务 A advanced

1. 实现速度平滑。
2. 可选实现 KNN 模板修正。
3. 做方法对比和消融。
4. 更新实验记录与方法文档。
5. Git commit：

```text
[task-a] add speed smoothing and template refinement methods
```

### 阶段 6：任务 B baseline

1. 实现距离计算。
2. 估计全局平均速度。
3. 实现规则 baseline。
4. 评估 MAE/RMSE/MAPE。
5. Git commit：

```text
[task-b] implement global speed baseline for travel time estimation
```

### 阶段 7：任务 B advanced

1. 构造时空特征。
2. 训练 ML 回归模型。
3. 比较多个模型。
4. 保存最佳模型。
5. 更新实验记录。
6. Git commit：

```text
[task-b] add feature regression models and validation experiments
```

### 阶段 8：提交脚本

1. 写 Task A 提交文件生成脚本。
2. 写 Task B 提交文件生成脚本。
3. 测试输出格式。
4. Git commit：

```text
[task-a][task-b] add submission generation scripts
```

### 阶段 9：报告和 PPT 素材整理

1. 汇总实验结果表格。
2. 汇总创新点。
3. 整理关键图。
4. 写展示逻辑。
5. Git commit：

```text
[doc] organize report and presentation materials
```

---

## 13. 重点创新方向建议

本项目不应只停留在“调包跑模型”，应至少形成 2–3 个可以在报告和课堂展示中讲清楚的创新点。

### 13.1 任务 A 可讲创新点

可选创新点：

1. **时间一致的轨迹插值**
   不是按点序号简单插值，而是按真实时间戳比例恢复坐标。

2. **速度物理约束修正**
   利用出租车运动速度上限和局部平滑约束，减少不合理跳变。

3. **历史轨迹模板增强**
   利用训练集中相似起终点、相似方向的历史轨迹，对直线插值结果进行形状修正。

4. **难度自适应策略**
   对 1/8 和 1/16 两种缺失率使用不同平滑强度或模板权重。

### 13.2 任务 B 可讲创新点

可选创新点：

1. **轨迹几何特征建模**
   不只看起终点距离，还建模总路径长度、曲折度、方向变化等轨迹形状特征。

2. **时段感知速度估计**
   将出发时间映射到城市交通状态，区分高峰、夜间、普通时段。

3. **规则模型与机器学习模型融合**
   使用物理可解释的速度 baseline 提供稳定下界，再用回归模型学习复杂残差。

4. **误差可解释分析**
   分析哪些轨迹预测困难，例如长距离、强曲折、高峰时段、异常绕路轨迹。

---

## 14. 最低可交付版本

如果时间紧张，必须至少完成以下内容：

### Task A

* 线性时间插值
* Haversine MAE/RMSE 评估
* 8 和 16 两种输入的提交文件生成
* 轨迹可视化若干张

### Task B

* 路径总距离特征
* 全局速度 baseline
* 至少一个机器学习回归模型
* MAE/RMSE/MAPE 评估
* 提交文件生成

### 文档

* README.md
* experiment_log.md
* method_notes.md
* innovation_points.md

---

## 15. Claude 工作准则

Claude 在本项目中必须遵守以下准则：

1. 先理解任务和数据格式，再写代码。
2. 每写一个模块，先说明目的，再实现。
3. 不要为了复杂而复杂，必须保证课堂测试可运行。
4. 所有方法都要有 baseline 对照。
5. 所有实验结果都要记录。
6. 所有图表都要考虑能否用于报告和 PPT。
7. 每完成重要模块，提醒用户 Git commit。
8. 如果修改了输出格式，必须同步修改格式检查代码。
9. 如果发现数据异常，必须记录在 `experiments/experiment_log.md`。
10. 如果某个高级方法效果不好，也要记录失败原因，因为这可以写进实验分析。

---

## 16. 当前优先级

Claude 当前应优先完成：

1. 创建项目结构。
2. 实现通用 pkl 读取和保存。
3. 实现 Haversine 距离。
4. 实现 Task A 线性插值 baseline。
5. 实现 Task A 验证集评估。
6. 实现 Task B 距离速度 baseline。
7. 实现 Task B 验证集评估。
8. 再考虑高级方法和可视化。

不要一开始就尝试复杂模型或路网增强。

---

## 17. 最终交付检查清单

提交前必须检查：

### 代码

* [ ] Task A 能读取 val/test input。
* [ ] Task A 能输出两个 `.pkl` 文件。
* [ ] Task A 输出格式与要求一致。
* [ ] Task A 输出无 NaN。
* [ ] Task B 能读取 val/test input。
* [ ] Task B 能输出一个 `.pkl` 文件。
* [ ] Task B 输出格式与要求一致。
* [ ] Task B 预测时间均为正数。
* [ ] 所有路径可通过命令行参数指定。
* [ ] 课堂测试脚本可运行。

### 实验

* [ ] Task A 至少有两个方法对比。
* [ ] Task B 至少有两个方法对比。
* [ ] 结果记录在 CSV 中。
* [ ] 实验日志有解释。
* [ ] 有误差分析。

### 报告/PPT

* [ ] 有问题分析。
* [ ] 有数据可视化。
* [ ] 有方法流程图素材。
* [ ] 有结果对比表。
* [ ] 有创新点总结。
* [ ] 有失败案例或局限性分析。
* [ ] 有小组分工说明。

### Git

* [ ] 已完成多次小步 commit。
* [ ] main 分支可运行。
* [ ] 大文件没有提交到仓库。
* [ ] README 中有运行说明。

---

## 18. 建议 README 最小内容

README.md 至少应包含：

```markdown
# Xi'an Taxi Trajectory Modeling

## Tasks
- Task A: Trajectory Recovery
- Task B: Travel Time Estimation

## Environment
pip install -r requirements.txt

## Data
Place released data under:

data/student_release/

## Run Task A Validation
python -m src.task_a.run_task_a --mode val ...

## Run Task A Prediction
python -m src.task_a.run_task_a --mode predict ...

## Run Task B Validation
python -m src.task_b.run_task_b --mode val ...

## Run Task B Prediction
python -m src.task_b.run_task_b --mode predict ...

## Results
See experiments/task_a_results.csv and experiments/task_b_results.csv.
```

---

## 19. 给 Claude 的最后提醒

这个项目的评分并不只看测试排名。方法创新和课堂汇报占比很高。因此实现过程中必须形成“能讲清楚”的方法体系：

* 为什么这个任务难？
* baseline 为什么合理？
* 我们做了什么改进？
* 改进是否真的有效？
* 哪些样本仍然失败？
* 这些失败说明了什么？
* 如果继续优化，可以怎么做？

请始终把代码、实验和展示三件事一起推进。
