# Method Notes

## Task A Methods

### linear_time_interpolation

**核心思想**
假设出租车在相邻已知采样点之间以近似匀速运动，根据真实时间戳对经纬度分别进行线性插值。

**输入与输出**
- 输入：稀疏轨迹（coords含NaN）、timestamps、mask
- 输出：完整坐标序列，NaN已替换，已知点坐标保持不变

**算法流程**
1. 提取 mask=True 的已知点坐标和时间戳
2. 分别对经度、纬度按时间戳建立一维线性插值函数
3. 对所有时间戳插值得到完整坐标
4. 将已知点坐标原值回写（保证精确不变）
5. 对边界缺失点用最近已知点填充

**为什么这样设计**
出租车在短时间内速度变化不大，时间线性插值对采样密度较高时误差小，且实现简单稳定。

**优点**
- 极快，O(N log N)
- 无需训练数据
- 适合课堂测试现场运行

**局限性**
- 默认两点间走直线，无法恢复道路弯曲
- 1/16 保留率下两已知点距离长，误差增大

---

### linear_with_speed_smoothing

**核心思想**
在线性插值基础上，对导致速度超阈值的未知点进行移动平均平滑，减少不合理跳变。

**输入与输出**
同上，额外参数：max_speed_kmh, smoothing_window, iterations

**算法流程**
1. 执行 linear_time_interpolation 得到初始坐标
2. 计算相邻点速度（Haversine距离/时间差）
3. 识别超速段，对两端的未知点进行局部移动平均
4. 重复若干次迭代
5. 对全部未知点做一次轻度平滑

**为什么这样设计**
真实出租车速度有物理上限，超速说明插值产生了不合理跳变。平滑只作用于未知点，不改变已知点。

**优点**
- 比纯线性插值更符合车辆运动规律
- 速度阈值可调

**局限性**
- 平滑后仍是直线趋势，不能恢复道路弯曲
- 速度阈值设置影响效果

---

### pchip_time_interpolation

**核心思想**
在时间线性插值基础上使用保形三次 Hermite 插值（PCHIP），在保持已知点精确通过的同时，用局部斜率估计生成更平滑的弯道轨迹。

**算法流程**
1. 提取 mask=True 的已知点时间戳和经纬度
2. 对经度、纬度分别按时间戳建立 PCHIP 插值函数
3. 对所有时间戳生成完整坐标
4. 边界外缺失点回退到最近已知点填充
5. 将已知点坐标原值回写

**为什么这样设计**
普通三次样条容易在稀疏点之间过冲，Catmull-Rom 对切线估计敏感。PCHIP 是局部保形插值，能改善弯道段 MAE，同时比自然三次样条更稳。

**验证结果**
- 1/8：MAE 87.59m，RMSE 116.09m
- 1/16：MAE 163.53m，RMSE 216.03m

---

### knn_template_refinement

**核心思想**
从训练集中找起终点相似的历史轨迹，将历史路径形状作为模板，与线性插值结果加权融合。

**输入与输出**
同上，额外参数：train_trajectories, alpha, start/end阈值, top_k

**算法流程**
1. 执行 linear_with_speed_smoothing 得到初始坐标
2. 提取查询轨迹起终点
3. 遍历训练集，筛选起终点距离在阈值内的候选轨迹
4. 按起终点距离之和排序，取 top-k
5. 将每条历史轨迹重采样到与查询轨迹等长
6. 取均值为模板坐标
7. final = (1-alpha)*linear + alpha*template
8. 恢复已知点

**为什么这样设计**
城市出租车轨迹具有重复性，相似起终点的行程往往走相近路线。

**优点**
- 能部分恢复非直线路径
- 利用训练集历史先验

**局限性**
- 训练集相似轨迹不足时效果退化
- 计算量较大，需要候选过滤
- 验证结果显示该方法明显劣化：全轨迹相似无法对齐当前缺口的局部形状

---

### local_segment_template_interpolation

**核心思想**
从“整条轨迹相似”改为“局部缺口相似”。对每一对相邻已知点之间的缺口，在训练集中检索相同 span、局部位置和端点位移相似的短片段，学习这些片段相对直线插值的弯曲残差，再叠加到当前缺口。

**输入与输出**
- 输入：Task A 稀疏轨迹、data_ds15/train.pkl 训练轨迹
- 输出：完整坐标序列
- fallback：缺少训练数据或 unsupported span 时使用 PCHIP

**算法流程**
1. 从训练集中采样局部片段，按 span 建立索引（当前配置 spans=[8,16]）
2. 对每个训练片段计算端点直线插值
3. 保存真实片段相对直线插值的 residual
4. 对待恢复轨迹的每个缺口，构造特征：局部 midpoint + endpoint displacement
5. 用 KDTree 检索 top-k 相似训练片段
6. 按特征距离加权平均 residual
7. final = linear_gap + alpha * mean_residual
8. 已知点坐标原值回写

**为什么这样设计**
之前的 KNN 模板方法以整条轨迹起终点检索历史轨迹，但 Task A 实际要恢复的是中间局部缺口，整条轨迹相似不等于局部缺口形状相似。局部模板把匹配粒度降到缺口级别，解决了对齐问题。

**合规性说明**
该方法只使用 data_ds15/train.pkl 学习历史局部形状，不使用 data_org 或 val_gt 查表，不依赖验证集答案。

**验证结果**
- 原始索引（25万片段/span，3 samples/traj/span）：
  - 1/8：MAE 64.73m，RMSE 92.02m
  - 1/16：MAE 120.73m，RMSE 168.34m
- 宽索引（50万片段/span，8 samples/traj/span）：
  - 1/8：MAE 62.03m，RMSE 89.07m
  - 1/16：MAE 116.50m，RMSE 163.59m

**参数消融结论**
扩大训练片段覆盖比盲目增大 top-k 更有效。top-k=20、alpha=1.0 在验证集上最优；top-k=40 会混入更多局部形状不够相似的片段，导致误差回升。

**优点**
- 比 Catmull-Rom 和 PCHIP 明显更低误差
- 方法可解释：线性物理初值 + 历史道路弯曲残差
- 只需训练集，无需外部路网

**局限性**
- 需要训练集路径可用
- 当前配置重点优化主缺口 span=8/16，边界短缺口交给 PCHIP
- 宽索引会增加内存和索引构建时间，但仍在课堂现场可接受范围内
- 如果测试集空间分布和训练集差异很大，模板收益会下降

---

## Task B Methods

### global_speed_baseline

**核心思想**
用训练集全局中位数速度，通过 total_distance / speed 估计行程时间。

**公式**
travel_time = total_distance / global_median_speed

**优点**
- 极简，可解释
- 作为所有方法的基准

**局限性**
- 不区分时段、区域、路径形状

---

### time_bucket_speed_model

**核心思想**
按出发小时统计不同时段中位数速度，用时段速度替代全局速度。

**优点**
- 能捕捉早晚高峰拥堵
- 样本不足时回退全局速度

**局限性**
- 忽略空间区域差异
- 未建模路径形状复杂度

---

### ensemble_with_speed_constraints (残差回归集成)

**核心思想**
先用 time_bucket_speed_model 得到可解释基础估计，再用 GradientBoosting 学习残差，最后叠加。

**特征体系**
- 距离类：total_distance, straight_distance, tortuosity, 分段统计
- 形状类：num_points, bearing_change, bounding_box
- 时间类：departure_hour, day_of_week, is_peak_hour, is_weekend, is_night
- 空间类：start/end lon/lat, grid坐标

**优点**
- 规则部分可解释
- ML部分学习拥堵、绕行等复杂模式
- 残差目标比直接预测更集中

**局限性**
- 需要提前训练保存模型
- 特征构建需要较完整轨迹
