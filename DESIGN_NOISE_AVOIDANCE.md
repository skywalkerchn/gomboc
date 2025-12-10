# Gömböc 噪声规避设计方案

## 一、问题分析

### 当前状态
- **观测空间**: 原始 CartPole 状态 `[x, x_dot, theta, theta_dot]` (4维)
- **噪声块**: 仅存在于渲染层，通过 `NoiseField` 管理
- **问题**: Agent 无法感知噪声块位置，无法学习规避行为

### 目标
让 agent 能够：
1. 感知噪声块的存在和位置
2. 学习规避噪声块的策略
3. 在完成主任务(CartPole平衡)的同时避开噪声

---

## 二、方案对比与建议

### 推荐实现顺序

**建议采用渐进式实现：B → A → (可选)混合方案**

理由：
1. **方案B作为baseline**: 快速验证"噪声规避"在信息充分时是否可学
2. **方案A作为目标**: 更符合真实场景，视觉端到端
3. **代码复用**: 两个方案共享 `NoiseField` 和 reward 设计

---

## 三、方案 B：低维特征方案（推荐先实现）

### 3.1 设计概述

**核心思想**: 将噪声块信息投影到 CartPole 状态空间
- 由于 CartPole 是1D运动（cart只能左右移动），噪声块的 y 坐标对碰撞不重要
- 重点关注：噪声块的 x 坐标与 cart 的相对位置

### 3.2 观测空间设计

```python
# 方案 B1: 简单拼接（推荐baseline）
obs = [
    # CartPole 原始状态 (4维)
    cart_position,      # x
    cart_velocity,      # x_dot
    pole_angle,         # theta
    pole_angular_vel,   # theta_dot

    # 噪声特征 (K个最近块，每个块3维特征)
    # 对每个块 i (i=1..K):
    rel_x_i,           # 相对 cart 的水平距离
    rel_vx_i,          # 相对水平速度
    block_size_i,      # 块的大小（影响碰撞半径）
]
# 总维度: 4 + K*3

# 如果实际块数 < K，用特殊值填充（如 999.0 表示"无块"）
```

**参数建议**:
- `K = 3~5`: 只关注最近的几个块，减少观测维度
- 距离计算：使用 CartPole 的 cart 位置与噪声块的 x 坐标
- 坐标系转换：需要将像素坐标转换为 CartPole 的物理坐标系

### 3.3 实现接口

```python
class NoiseFeaturesWrapper(gym.ObservationWrapper):
    """
    将噪声块信息转换为低维特征并附加到观测中
    """
    def __init__(
        self,
        env: gym.Env,
        noise_field: NoiseField,
        k_nearest: int = 5,
        pixel_to_cart_scale: float = None,  # 像素到CartPole坐标的比例
    ):
        super().__init__(env)
        self.noise_field = noise_field
        self.k_nearest = k_nearest

        # 扩展观测空间
        low_orig = env.observation_space.low
        high_orig = env.observation_space.high

        # 噪声特征: [rel_x, rel_vx, size] * k_nearest
        noise_feat_dim = k_nearest * 3
        noise_low = np.full(noise_feat_dim, -10.0)  # 相对距离/速度范围
        noise_high = np.full(noise_feat_dim, 10.0)

        self.observation_space = spaces.Box(
            low=np.concatenate([low_orig, noise_low]),
            high=np.concatenate([high_orig, noise_high]),
            dtype=np.float32
        )

    def observation(self, obs):
        """提取噪声特征并拼接到观测"""
        cart_x = obs[0]  # CartPole的cart位置

        # 获取当前所有噪声块
        blocks = self.noise_field.blocks

        # 计算每个块到cart的距离（只考虑x方向）
        if len(blocks) > 0:
            distances = []
            for block in blocks:
                # 将像素坐标转换为CartPole坐标
                block_x_cart = self.pixel_to_cart_x(block.x)
                rel_x = block_x_cart - cart_x
                distances.append((abs(rel_x), block, rel_x))

            # 按距离排序，取最近的k个
            distances.sort(key=lambda t: t[0])
            nearest_blocks = distances[:self.k_nearest]

            # 提取特征
            noise_feat = []
            for _, block, rel_x in nearest_blocks:
                block_vx_cart = self.pixel_to_cart_v(block.vx)
                rel_vx = block_vx_cart - obs[1]  # cart的速度
                block_size_cart = self.pixel_to_cart_size(block.size)

                noise_feat.extend([rel_x, rel_vx, block_size_cart])

            # 如果块数不足k，填充
            while len(noise_feat) < self.k_nearest * 3:
                noise_feat.extend([999.0, 0.0, 0.0])  # 特殊标记：无块
        else:
            # 没有噪声块，全部填充
            noise_feat = [999.0, 0.0, 0.0] * self.k_nearest

        return np.concatenate([obs, np.array(noise_feat, dtype=np.float32)])
```

### 3.4 坐标系转换

**关键问题**: 像素空间 vs CartPole物理空间

```python
class CoordinateConverter:
    """
    转换像素坐标系和CartPole物理坐标系

    CartPole:
        - x ∈ [-2.4, 2.4] (cart position)
        - screen center = x=0

    Pixel:
        - x ∈ [0, width]
        - screen center = width/2
    """
    def __init__(self, frame_width: int, frame_height: int,
                 cart_range: Tuple[float, float] = (-2.4, 2.4)):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cart_min, self.cart_max = cart_range
        self.cart_range_total = cart_max - cart_min

    def pixel_to_cart_x(self, pixel_x: float) -> float:
        """像素 x 坐标 → CartPole cart 坐标"""
        # 归一化到 [0, 1]
        normalized = pixel_x / self.frame_width
        # 映射到 [-2.4, 2.4]
        return self.cart_min + normalized * self.cart_range_total

    def pixel_to_cart_v(self, pixel_vx: float) -> float:
        """像素速度 → CartPole 速度"""
        # 速度比例与位置相同
        return pixel_vx * (self.cart_range_total / self.frame_width)

    def pixel_to_cart_size(self, pixel_size: float) -> float:
        """像素大小 → CartPole 长度单位"""
        return pixel_size * (self.cart_range_total / self.frame_width)
```

### 3.5 优缺点

**优点**:
1. ✅ 实现简单，训练快速（MLP即可）
2. ✅ 便于调试和分析（特征可解释）
3. ✅ 作为baseline验证reward设计是否合理
4. ✅ 计算开销小

**缺点**:
1. ❌ 带有"特权信息"（直接给出相对位置）
2. ❌ 不符合"纯视觉"的设定
3. ❌ 特征设计需要领域知识

---

## 四、方案 A：像素观测方案

### 4.1 设计概述

**核心思想**: Agent 直接从像素学习，真正的端到端视觉RL

### 4.2 观测空间设计

```python
# 观测空间: RGB图像
obs_shape = (H, W, 3)  # 例如 (400, 600, 3)

# 可选优化:
# 1. 灰度化: (H, W, 1)
# 2. 下采样: (H//2, W//2, 3)
# 3. Frame stacking: (H, W, 3*n_frames) 用于捕捉运动信息
```

**建议配置**:
- **分辨率**: 84x84 (Atari标准) 或 128x128
- **预处理**: 灰度化可选，但彩色更容易区分噪声块
- **Frame stack**: 4帧，用于隐式估计速度

### 4.3 实现接口

```python
class NoiseVisionWrapper(gym.ObservationWrapper):
    """
    将观测改为包含噪声块的渲染图像
    """
    def __init__(
        self,
        env: gym.Env,
        noise_field: NoiseField,
        frame_size: Tuple[int, int] = (84, 84),
        grayscale: bool = False,
    ):
        super().__init__(env)
        self.noise_field = noise_field
        self.frame_size = frame_size
        self.grayscale = grayscale

        # 设置观测空间
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(*frame_size, channels),
            dtype=np.uint8
        )

    def observation(self, obs):
        """渲染带噪声的图像作为观测"""
        # 获取原始渲染帧
        frame = self.env.render()

        # 在帧上绘制噪声块
        frame_with_noise = self.noise_field.update_and_draw(frame)

        # 预处理
        frame_processed = self._preprocess(frame_with_noise)

        return frame_processed

    def _preprocess(self, frame):
        """图像预处理：缩放、灰度化等"""
        import cv2

        # 下采样
        frame_resized = cv2.resize(frame, self.frame_size,
                                   interpolation=cv2.INTER_AREA)

        # 灰度化（可选）
        if self.grayscale:
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
            frame_gray = np.expand_dims(frame_gray, axis=-1)
            return frame_gray

        return frame_resized
```

### 4.4 CNN Policy 设计

**轻量级CNN（推荐baseline）**:

```python
class CNNPolicy(nn.Module):
    """
    Nature DQN-style CNN for CartPole with noise
    """
    def __init__(self, action_dim: int, frame_channels: int = 3):
        super().__init__()

        # Feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(frame_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算展平后的维度（需要根据输入尺寸计算）
        # 对于 84x84: ~3136
        conv_out_dim = self._get_conv_out_dim(frame_channels)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out_dim(self, channels):
        """计算卷积输出维度"""
        dummy_input = torch.zeros(1, channels, 84, 84)
        return self.conv(dummy_input).numel()

    def forward(self, obs):
        # obs: (batch, H, W, C) → (batch, C, H, W)
        obs = obs.permute(0, 3, 1, 2).float() / 255.0

        features = self.conv(obs)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value
```

**计算成本估计**:
- 单次前向传播: ~5-10ms (GPU)
- 训练时间: 约为MLP的10-20倍
- 建议使用GPU训练

### 4.5 优缺点

**优点**:
1. ✅ 真正的视觉端到端学习
2. ✅ 符合"从视觉规避噪声"的目标
3. ✅ 易于扩展到复杂场景
4. ✅ 不需要手工设计特征

**缺点**:
1. ❌ 训练成本高（样本效率低）
2. ❌ 需要GPU和更多调参
3. ❌ 初期可能难以收敛

---

## 五、混合方案（可选，未来扩展）

### 5.1 设计思路

结合两种方案的优势：
- 底层状态（CartPole 4维）→ 快速控制
- 视觉信息（噪声块）→ 感知障碍

```python
observation_space = spaces.Dict({
    "state": Box(...),      # CartPole 状态 (4,)
    "vision": Box(...),     # 渲染图像 (H, W, 3)
})
```

### 5.2 Policy 结构

```python
class HybridPolicy(nn.Module):
    def __init__(self):
        # 状态编码器（MLP）
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU()
        )

        # 视觉编码器（CNN）
        self.vision_encoder = ConvNet(...)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + conv_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
```

### 5.3 何时使用

- 如果方案B和A都成功，想进一步提升性能
- 研究"显式状态 vs 隐式视觉"的信息融合
- 不建议作为第一步实现

---

## 六、Reward 设计建议

### 6.1 原始任务 Reward

```python
# CartPole 默认 reward
r_balance = 1.0  # 每步保持平衡
```

### 6.2 噪声规避 Reward

**方案1: 碰撞惩罚**
```python
def check_collision(cart_x: float, blocks: List[NoiseBlock],
                   converter: CoordinateConverter) -> bool:
    """检测cart是否与任何噪声块碰撞"""
    cart_half_width = 0.25  # CartPole cart的半宽（估计值）

    for block in blocks:
        block_x_cart = converter.pixel_to_cart_x(block.x)
        block_half_width = converter.pixel_to_cart_size(block.size / 2)

        # 简单的1D碰撞检测
        if abs(cart_x - block_x_cart) < (cart_half_width + block_half_width):
            # 还需要检查y方向（cart在底部）
            if block.y + block.size / 2 >= cart_y_pixel:
                return True
    return False

# Reward
if check_collision(...):
    r_collision = -1.0  # 碰撞惩罚
else:
    r_collision = 0.0
```

**方案2: 距离奖励（稠密reward）**
```python
def get_min_distance(cart_x: float, blocks: List[NoiseBlock]) -> float:
    """返回cart到最近噪声块的距离"""
    if not blocks:
        return float('inf')

    distances = []
    for block in blocks:
        block_x_cart = converter.pixel_to_cart_x(block.x)
        distances.append(abs(cart_x - block_x_cart))

    return min(distances)

# Reward
min_dist = get_min_distance(...)
r_distance = 0.1 * np.clip(min_dist, 0, 2.0)  # 距离越远奖励越高
```

**推荐组合**:
```python
r_total = r_balance + w_collision * r_collision + w_distance * r_distance
# 例如: w_collision = 5.0, w_distance = 0.5
```

### 6.3 Reward Wrapper

```python
class NoiseAvoidanceRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, noise_field, converter,
                 collision_penalty=-5.0, distance_weight=0.1):
        super().__init__(env)
        self.noise_field = noise_field
        self.converter = converter
        self.collision_penalty = collision_penalty
        self.distance_weight = distance_weight

    def reward(self, reward):
        # 获取当前状态
        cart_x = self.env.unwrapped.state[0]

        # 碰撞检测
        if check_collision(cart_x, self.noise_field.blocks, self.converter):
            r_noise = self.collision_penalty
            # 可选：结束episode
            # self.env.unwrapped.done = True
        else:
            # 距离奖励
            min_dist = get_min_distance(cart_x, self.noise_field.blocks,
                                       self.converter)
            r_noise = self.distance_weight * np.clip(min_dist, 0, 2.0)

        return reward + r_noise
```

---

## 七、代码结构建议

### 7.1 目录结构

```
gomboc/
├── cartpole_with_noise.py          # 现有主脚本
├── noise_avoidance/                # 新模块
│   ├── __init__.py
│   ├── wrappers.py                 # 观测/reward wrappers
│   │   ├── NoiseFeaturesWrapper    # 方案B
│   │   ├── NoiseVisionWrapper      # 方案A
│   │   ├── NoiseAvoidanceRewardWrapper
│   │   └── CoordinateConverter
│   ├── policies.py                 # CNN policies
│   │   ├── CNNPolicy
│   │   └── HybridPolicy
│   └── utils.py                    # 辅助函数
│       ├── check_collision
│       ├── get_min_distance
│       └── preprocess_frame
├── policies/                       # 现有MLP policies
│   ├── dqn.py
│   ├── ppo.py
│   └── ...
└── train_noise_avoidance.py       # 新训练脚本
```

### 7.2 接口设计原则

1. **NoiseField 保持独立**: 不依赖具体环境，只负责管理噪声块
2. **Wrapper 解耦**: 观测wrapper和reward wrapper分离
3. **Policy 可插拔**: 支持MLP/CNN/Hybrid
4. **坐标转换抽象**: `CoordinateConverter` 作为独立组件

---

## 八、实现路线图

### Phase 1: 方案B Baseline (1-2天)

1. ✅ 实现 `CoordinateConverter`
2. ✅ 实现 `NoiseFeaturesWrapper`
3. ✅ 实现 `NoiseAvoidanceRewardWrapper`
4. ✅ 修改现有MLP policy支持新观测维度
5. ✅ 训练并验证是否能学到规避行为

**成功标准**:
- Agent的 collision rate 明显低于随机策略
- 在保持平衡的同时，平均距离噪声块更远

### Phase 2: 方案A 视觉RL (3-5天)

1. ✅ 实现 `NoiseVisionWrapper`
2. ✅ 实现 CNN policy
3. ✅ 集成 frame stacking（可选）
4. ✅ 训练并调参（可能需要GPU）
5. ✅ 对比B和A的性能

**成功标准**:
- CNN policy能达到接近B的性能
- 证明纯视觉方案的可行性

### Phase 3: 优化与扩展 (可选)

1. 实现混合方案
2. 尝试不同的CNN架构（ResNet, attention等）
3. 迁移到其他环境（如Acrobot）
4. 研究噪声块的不同模式对学习的影响

---

## 九、潜在问题与解决方案

### 9.1 坐标系对齐问题

**问题**: 像素坐标与CartPole物理坐标不一致

**解决方案**:
- 通过 `render()` 获取cart的实际像素位置
- 使用CartPole的 `x_threshold = 2.4` 校准比例
- 添加单元测试验证转换正确性

### 9.2 训练不稳定

**问题**: Reward shaping不当导致策略退化

**解决方案**:
- 先用稀疏reward（只有碰撞惩罚）
- 如果学不到，再加入稠密的距离reward
- 使用curriculum learning：先少量噪声块，再逐渐增加

### 9.3 计算效率

**问题**: 方案A训练太慢

**解决方案**:
- 使用向量化环境（SubprocVecEnv）
- 降低图像分辨率（64x64）
- 考虑使用更高效的算法（SAC, Rainbow等）

---

## 十、总结与建议

### 关键决策点

| 维度 | 方案B | 方案A |
|-----|------|------|
| **实现难度** | ⭐⭐ | ⭐⭐⭐⭐ |
| **训练效率** | 快 | 慢 |
| **样本效率** | 高 | 低 |
| **可解释性** | 强 | 弱 |
| **目标一致性** | 中（特权信息） | 强（端到端） |
| **扩展性** | 需要重新设计特征 | 通用 |

### 最终建议

1. **第一步**: 实现方案B
   - 验证问题设定的可行性
   - 快速迭代reward设计
   - 为方案A提供性能upper bound

2. **第二步**: 实现方案A
   - 证明视觉端到端的可行性
   - 可以和B做消融实验

3. **未来工作**:
   - 如果两者都成功，可尝试混合方案
   - 研究从B到A的迁移学习
   - 扩展到更复杂的环境

### 代码复用策略

- `NoiseField`: 完全复用
- `Reward wrapper`: B和A共用
- `Policy`: 需要分别实现MLP和CNN
- `Training loop`: 尽量统一接口

---

## 附录：关键代码示例

见后续实现文件：
- `noise_avoidance/wrappers.py`
- `noise_avoidance/policies.py`
- `train_noise_avoidance.py`
