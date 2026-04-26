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
