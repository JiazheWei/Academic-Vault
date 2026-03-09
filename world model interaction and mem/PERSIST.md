# Paper Info

- **Title**: Beyond Pixel Histories: World Models with Persistent 3D State
- **Authors**: Samuel Garcin*, Thomas Walker*, Steven McDonagh, Tim Pearce, Hakan Bilen, Tianyu He, Kaixin Wang, Jiang Bian
- **Venue/Year**: Preprint (arXiv:2603.03482), March 2026
- **Paper Link**: https://arxiv.org/abs/2603.03482
- **Code Link**: francelico.github.io/persist.github.io


```ad-tldr
这篇论文提出了 PERSIST，一种全新范式的 world model。它不再像以往方法那样依赖像素历史帧作为记忆，而是维护一个**持久的、动态演化的 latent 3D 场景状态**（world-frame），通过相机模型查询该 3D 状态并投影到屏幕空间来引导像素帧的生成。实验表明 PERSIST 在空间记忆、3D 一致性和长时序稳定性方面显著优于 Oasis 和 WorldMem 等 baseline。
```

---

# Introduction

## Task and Applications

本文研究的是**交互式世界模型（Interactive World Models）**任务：模型持续响应用户动作，以自回归方式生成视频帧，模拟一个可交互的虚拟环境。

应用场景包括：
- 照片级真实感的沉浸式交互体验（类似神经游戏引擎）
- 在学习到的模拟器中安全训练 embodied agents
- 视频游戏、数字孪生仿真、具身 AI 导航

## Technical Challenges

现有的自回归（AR）视频生成方法（如 Oasis、WorldMem 等）面临以下核心难题：

1. **有限的上下文窗口**：高维像素观测的计算代价极高，AR 模型实际上只能看到过去几秒的帧。即使用 learned autoencoder 压缩，context window 依然很短。
2. **像素观测的信息冗余与不完备**：每一帧像素只提供了世界的**局部、视角相关、高度冗余**的信息。随着 memory bank 增长，检索相关历史帧变得越来越困难。
3. **缺乏 3D 表示**：3D 一致性只能隐式从数据中学习，没有显式的几何约束，导致长时序生成中空间记忆退化、几何不一致。

## 与之前工作的区别/定位

| 维度      | 之前的主流方法              | PERSIST                   |
| ------- | -------------------- | ------------------------- |
| 记忆载体    | 像素帧（pixel histories） | 动态演化的 latent 3D 场景        |
| 信息检索    | 从不断增长的像素历史中检索关键帧     | 通过相机参数从 3D 状态中几何投影获取      |
| 3D 一致性  | 隐式学习                 | 通过构造强制保证（by construction） |
| 记忆成本    | 随 episode 长度增长       | 固定成本（fixed-cost memory）   |
| 3D 环境表示 | 无（或静态不变）             | 动态演化，能捕捉屏幕外的环境变化          |

核心切入角度：**从"检索像素历史"转向"主动生成引导帧"**。受传统 3D 游戏引擎（维护持久 3D 状态并从中渲染像素帧）的启发，PERSIST 将 world model 分解为三个耦合组件：世界帧模型、相机模型、世界到像素的生成模块。

## 解决 Challenge 的 Pipeline

### Contribution 1: Persistent 3D State（持久的 3D 状态表示）

**解决什么问题？** 解决像素历史作为记忆载体带来的信息冗余、视角依赖、和检索困难的问题。

**Key insight**: 不把像素当作记忆的主要载体，而是在 latent 3D 空间中建模环境的结构和动态。这个 3D 表示（world-frame）以 agent 为中心，捕捉场景如何随时间演化，相机参数则作为"查询键"从中提取当前帧所需的 3D 信息。

**具体做法**: 定义代理状态 $\tilde{s} = \langle \mathbf{w}, \mathbf{c} \rangle$，其中 $\mathbf{w}$ 是以 agent 为中心的 world-frame（3D voxel grid 的 latent 表示），$\mathbf{c}$ 是相机状态。通过 3D VAE 编码 voxel grid，然后用 DiT 作为 denoiser 预测其演化。

### Contribution 2: 解耦式 Pipeline 设计（World → Camera → Projection → Pixel）

**解决什么问题？** 将复杂的世界模拟目标分解为更容易学习的子任务。

**Key insight**: 模仿传统游戏引擎的渲染管线，将世界模拟分解为：3D 场景演化预测、相机位姿预测、几何投影、像素渲染四个步骤。每个组件可以独立训练，推理时组合使用无需 fine-tuning。

### Contribution 3: World-to-Pixel Generation（可微投影 + 学习的 Deferred Shading）

**解决什么问题？** 如何将 3D latent state 有效地转化为高质量的像素观测。

**Key insight**: 借鉴 Neural Deferred Shading 的思路——先将 3D 特征光栅化到屏幕空间（通过深度排序的 depth peeling），再用学习的 neural shader（pixel denoiser）渲染最终像素帧。这种方式既保留了 3D 几何引导，又允许模型学习任意渲染函数（如纹理、光照、粒子效果等）。

### Contribution 4: 新能力的涌现

**解决什么问题？** 传统像素级 world model 无法实现的 3D 感知控制。

**具体能力**: 从单张 RGB 图像生成多样的 3D 环境；显式 3D 初始化以获得更好的控制；generation 中途进行 3D 编辑（如修改地形、放置物体）；学习到屏幕外持续演化的环境动态（如视野外的洞穴被水填满）。

---

# Method

## Overview

- **输入**：初始像素帧 $\mathbf{o}_0$、初始相机 $\mathbf{c}_0$、每步的动作 $\mathbf{a}_t$（可选：初始 world-frame $\mathbf{w}_0$）
    
- **输出**：持续生成的像素帧序列 $\mathbf{o}_1, \mathbf{o}_2, \ldots$（及对应的 3D world-frame 序列）
    
- **Pipeline 整体流程**：每个 timestep 依次执行四步——预测 3D world-frame → 预测相机参数 → 将 world-frame 投影到屏幕空间 → 生成像素帧
    
- **模块连接关系**：
    

```
Actions + Past Context
        ↓
[World Denoiser (Wθ)] → world-frame w̄_t（latent 3D voxel）
        ↓
    3D-VAE decode → w_t
        ↓
[Camera Model (Cθ)] → camera params c_t
        ↓
[World Projection (R)] → w_2D（depth-ordered screen-space features）
        ↓
[Pixel Denoiser (Pθ)] → pixel latent ō_t
        ↓
    2D-VAE decode → pixel frame o_t
```

## 数据准备与预处理

- 数据集来自 Luanti（开源 Minecraft-like 体素游戏引擎），通过 Craftium 平台收集
- 每条轨迹包含：像素观测 $O$、动作 $A$（23维 multi-hot 编码：按键 + 离散化鼠标移动）、world-frame $W$（$48^3$ voxel grid）、相机视角 $C$
- 训练 2D-VAE 将像素帧编码为 $\bar{\mathbf{o}} \in \mathbb{R}^{36 \times 64 \times 16}$（patch size 10×10 像素）
- 训练 3D-VAE 将 world-frame 编码为 $\bar{\mathbf{w}} \in \mathbb{R}^{12 \times 12 \times 12 \times 48}$（patch size $4^3$ voxels）
- 数据规模：约 40M 交互，约 100K 轨迹，460 小时游戏数据（24Hz）

多模块训练方式预览：

```
真实环境 E (Luanti)
    ↓ 收集轨迹
原始数据: [O, A, W, C]  
    ↓ VAE 编码
Latent 数据: [Ō, A, W̄, C]
    ↓ 分别送入三个模块训练
    ├── Wθ: 学习 W̄ 的时序演化
    ├── Cθ: 学习 C 的时序演化  
    └── Pθ: 学习从 W̄ 投影结果生成 Ō
```

## Module 1: World-Frame Prediction（世界帧预测，$\mathcal{W}_\theta$）

- **这个模块做什么？** 在每个 timestep 预测当前的 3D world-frame latent $\bar{\mathbf{w}}_t$
    
- **Motivation**: world-frame 是整个系统的"空间记忆"核心。它以 agent 为中心，表示周围固定区域的 3D 场景，需要根据历史状态和动作预测其演化。
    
- **具体怎么做？**
    

$$\bar{\mathbf{w}}_t \sim \mathcal{W}_\theta(\bar{\mathbf{w}}_t | \bar{W}_{t-K}^{t-1}, A_{t-K}^t, C_{t-K-1}^{t-1}, \bar{O}_{t-K-1}^{t-1})$$

架构：Rectified Flow 模型 + Causal DiT backbone，包含交错的 spatial（3D）、temporal 和 cross-attention 模块。

关键设计：

- **3D 空间注意力**：将原始 2D spatial attention 改为处理三维空间，用 voxel token 质心的 XYZ 坐标的绝对位置编码替换 RoPE
    
- **动作和相机条件注入**：通过 MLP 联合嵌入后，加到 denoising timestep embedding，再通过 AdaLN 注入每个模块
    
- **像素信息引入**：用 Plücker embeddings 编码像素 patch 到 3D 空间的投影关系，拼接到像素 patch 后通过 cross-attention 注入
    
- **支持无 3D 条件初始化**：$\mathcal{W}_\theta$ 支持 $\bar{W} = \varnothing$ 的情况，即仅从 $\langle \mathbf{o}_0, \mathbf{c}_0 \rangle$ 生成初始 world-frame $\mathbf{w}_0$
    
- **为什么能 work？** 在 latent 3D 空间而非像素空间建模动态，使得模型能捕捉屏幕外的环境变化（如水流、物体碰撞），并且记忆检索成本不随 episode 长度增长。
    

## Module 2: Camera Model（相机模型，$\mathcal{C}_\theta$）

- **这个模块做什么？** 预测每个 timestep 的相机参数 $\mathbf{c}_t$
    
- **Motivation**: 相机参数充当"空间查询键"，决定从 3D world-frame 中提取哪部分信息来渲染当前帧。
    
- **具体怎么做？**
    

$$\mathbf{c}_t = \mathcal{C}_\theta(C_{t-1-K}^{t-1}, W_{t-K}^t, A_{t-K}^t)$$

相机用 10 维向量表示：$\mathbf{c} = \langle \text{pos}, \text{rot}, \text{fov} \rangle$，其中 $\text{pos} \in \mathbb{R}^3$（位置），$\text{rot} \in \mathbb{R}^6$（6D 连续旋转表示），$\text{fov} \in \mathbb{R}$（视场角）。

架构：1D causal transformer + RoPE temporal embeddings。

关键设计：

- **输出重参数化**：由于相机只沿 pitch 和 yaw 旋转，输出重参数化为 $\bar{\mathbf{c}} = \langle \text{pos}, \Delta\text{pitch}, \Delta\text{yaw}, \Delta\text{fov} \rangle$，预测残差而非绝对值
- 训练用 MSE loss
- World-frame 裁剪为内部 $4^3$ voxels 后与 action 一起通过 MLP + AdaLN 注入

## Module 3: World Projection（世界投影，$\mathcal{R}$）

- **这个模块做什么？** 将 3D world-frame 投影到屏幕空间，形成像素对齐的 3D 特征图
    
- **Motivation**: 建立 world-frame 与像素帧之间的几何对应关系，为 pixel denoiser 提供 3D 引导
    
- **具体怎么做？**
    

$$\mathcal{R}(\mathbf{c}, \mathbf{w}) = (\tilde{\mathbf{w}}_{2D}, \mathbf{d})$$

其中 $\tilde{\mathbf{w}}_{2D} \in \mathbb{R}^{h \times w \times l \times m}$ 是每像素的**深度排序的** voxel 特征列表（$l$ 层，$m$ 通道），$\mathbf{d} \in \mathbb{R}^{h \times w \times l}$ 是对应的线性深度。

实现：使用 GPU 原生的三角形光栅化 + **depth peeling** 技术。将 voxel 特征赋给静态 voxel grid mesh 的面，然后光栅化生成深度排序的特征栈。投影的屏幕尺寸与像素 latent 相同，保证像素级对齐。

最终将 $\tilde{\mathbf{w}}_{2D}$ 和 $\mathbf{d}$ 通过 channel-wise 拼接合并为 $\mathbf{w}_{2D} \in \mathbb{R}^{h \times w \times z}$，其中 $z = l \times (m + 1)$。

## Module 4: Pixel Frame Prediction（像素帧预测 / Learned Deferred Shader，$\mathcal{P}_\theta$）

- **这个模块做什么？** 根据投影的 3D 特征和历史像素帧，生成当前时间步的像素帧
    
- **Motivation**: $\mathbf{w}_{2D}$ 提供了几何结构信息，但缺少纹理、光照、粒子效果、UI 覆盖等细节。需要一个 learned renderer 来补全这些信息。
    
- **具体怎么做？**
    

$$\bar{\mathbf{o}}_t \sim \mathcal{P}_\theta(\bar{\mathbf{o}}_t | W_{2D,t-K}^t, A_{t-K}^t, \bar{O}_{t-K}^{t-1})$$

架构：Rectified Flow + Causal DiT，包含交错的 spatial 和 temporal 模块。

关键设计：

- $\mathbf{w}_{2D}$ 通过 1D 卷积投影到 latent space，然后 **channel-wise 拼接**到 $\bar{\mathbf{o}}$
    
- **关键：分配给 $\mathbf{w}_{2D}$ 的 latent 通道数 > $\bar{\mathbf{o}}$ 的通道数**（752 vs 16），强烈偏置模型以 3D 信息作为主要信息源
    
- Action 通过 MLP + AdaLN 注入
    
- **为什么能 work？** 这本质上是一个 **learned deferred shader**：几何信息已经由投影步骤提供，pixel denoiser 只需学习"如何着色"，大大降低了任务难度。同时由于 $\mathbf{w}_{2D}$ 是像素对齐的，空间一致性通过构造得到保证。
    

## 核心亮点深度解析

### 亮点 1：从"像素检索"到"3D 状态投影"的范式转换

**Intuition**: 传统方法（如 WorldMem）把"记忆"存储在不断增长的像素帧 buffer 中，然后根据当前相机状态检索最相关的历史帧作为 context。这存在根本性缺陷——每个像素帧只包含世界的一个视角切片，信息高度冗余且不完整，随着 buffer 增长检索变得越来越困难。

PERSIST 的 key insight 是：**不要从像素历史中检索，而是维护一个 3D 世界状态，通过几何投影直接生成引导帧**。这和传统游戏引擎的思路一致——引擎不会保存每一帧截图作为记忆，而是维护 3D 场景图，需要时渲染。

**与之前方法的关键区别**：

- 记忆成本**固定**，不随 episode 长度增长
- 信息检索是**几何投影**而非**相似度搜索**
- 3D 一致性**by construction**而非隐式学习

**为什么更好？** 在 600 步长 episode 的评估中，PERSIST 的 FVD 从 Oasis 的 706 降到 181（PERSIST-XL），用户评分在所有维度上均显著领先。

### 亮点 2：独立训练 + 噪声增强的组合策略

**Intuition**: PERSIST 的各模块独立训练，推理时才组合。这面临严重的 exposure bias 问题——训练时 Wθ 看到 ground truth 的像素 latent，但推理时看到的是 Pθ 的预测；反之亦然。

**设计**:

1. 每个 diffuser 使用 **Diffusion Forcing** 训练，使其对自身预测的 exposure bias 更鲁棒
2. 关键创新：训练 $\mathcal{W}_\theta$ 时对 $\bar{O}$ 施加 10% 随机噪声增强，训练 $\mathcal{P}_\theta$ 时对 $\bar{W}$ 施加 10% 噪声增强。这模拟了推理时另一个模块预测不完美的情况。

**为什么更好？** 这种简单但有效的策略让各组件无需联合训练或 fine-tuning 即可组合，大大简化了训练流程，同时保持了推理时的稳定性。

## Training

- **Loss Function**:
    
    - World Denoiser ($\mathcal{W}_\theta$) 和 Pixel Denoiser ($\mathcal{P}_\theta$)：Rectified Flow 的 Conditional Flow Matching (CFM) loss，即 $\mathcal{L}(\theta) = |\mathcal{V}_\theta(\mathbf{x}^\tau, \tau) - (\mathbf{x}^0 - \mathbf{x}^1)|^2$
    - Camera Model ($\mathcal{C}_\theta$)：MSE loss 应用于 $\bar{\mathbf{c}}$ 的各分量
    - 2D-VAE：MSE 重建 + KL divergence（系数 1e-6）
    - 3D-VAE：Cross-entropy（预测 voxel 类别标签）+ KL divergence（系数 1e-6）
- **训练策略**:
    
    - 各模块**完全独立训练**，推理时组合无需 fine-tuning
    - 先训练 VAE 并 pre-encode latent，再训练动态模型
    - 使用 Diffusion Forcing（每帧独立采样噪声级别）
    - 跨模块 10% 噪声增强缓解 exposure bias
    - Optimizer: AdamW, lr = 1e-4
    - Batch size: 64（动态模型），256（Camera model）
- **关键超参数**:
    
    - 推理时 denoising steps = 20
    - Context noise: $\tau_{ctx} = 0.02$（Wθ）/ $0.1$（Pθ）
    - Denoising step scheduling: $\tau^k = \frac{\eta k}{1+(\eta-1)k}$, $\eta = 3$
    - Context window: $K_W = 8$（World）, $K_P = 16$（Pixel）, $K_C = 8$（Camera）

---

# Experiment

## 资源消耗

| 组件               | GPU     | 训练时间  | 参数量                     |
| ---------------- | ------- | ----- | ----------------------- |
| 2D-VAE           | 8× A100 | 4 天   | 227M                    |
| 3D-VAE           | 8× A100 | 12 天  | 138M                    |
| 3D Denoiser (S)  | 8× H100 | 3 天   | 686M                    |
| 3D Denoiser (XL) | 8× H100 | 10 天  | 686M（同参数量，patch size=1） |
| Pixel Denoiser   | 8× H100 | 10 天  | 460M                    |
| Camera Model     | 1× A100 | 16 小时 | 234M                    |

推理时每帧 20 denoising steps。论文未报告具体推理速度（FPS）。

## 数据集 / Benchmark

- **训练数据**: 来自 Luanti（开源体素游戏引擎）+ Craftium 平台收集的程序化生成世界，约 40M 交互，100K 轨迹，460 小时@24Hz
- **评估集**: 148 条评估轨迹（Craftium），来自训练集中未见过的游戏世界；baselines 使用 MineDojo 评估集
- **评估策略**: 四种移动模式——Free Play、Move Forward、Move Backward、Orbit（专门设计来测试空间/时间一致性）
- **评估指标**: FVD（Fréchet Video Distance）+ 28 人用户研究（800+ rollout 评分），评分维度包括 Per-Frame Visual Fidelity、3D Spatial Consistency、Temporal Consistency、Overall Score（1-5 分）

## 定量结果

| Method         | FVD↓    | Visual Fidelity↑ | 3D Consistency↑ | Temporal Consistency↑ | Overall↑     |
| -------------- | ------- | ---------------- | --------------- | --------------------- | ------------ |
| Oasis          | 706     | 2.1±0.1          | 1.9±0.1         | 1.8±0.1               | 1.9±0.1      |
| WorldMem       | 596     | 1.7±0.09         | 1.7±0.09        | 1.5±0.08              | 1.5±0.07     |
| **PERSIST-S**  | **209** | **2.8±0.1**      | **2.7±0.1**     | **2.5±0.1**           | **2.6±0.09** |
| **PERSIST-XL** | **181** | **2.8±0.09**     | 2.5±0.09        | 2.5±0.09              | 2.6±0.08     |
| PERSIST-XL+w₀  | 116     | 3.2±0.1          | 2.8±0.1         | 2.8±0.1               | 3.0±0.1      |

核心发现：

- PERSIST 所有配置在所有指标上**显著优于** baselines
- FVD 提升巨大：PERSIST-XL (181) vs Oasis (706)，降低 74%
- PERSIST-S 和 PERSIST-XL 用户评分接近，说明 3D 表示对空间分辨率降低具有鲁棒性
- 提供 ground truth $\mathbf{w}_0$（PERSIST-XL+w₀）进一步提升效果，验证模型能有效利用 3D 信息

## 定性结果

- 600 步 episode 对比（Figure 5）：Oasis 在 t=100 时场景已严重退化（纹理漂移、结构崩溃）；WorldMem 稍好但也出现明显不一致；PERSIST 维持了连贯的环境
- 3D 编辑能力（Figure 6）：可以在生成中途编辑地形、生物群落、放置树木等
- 屏幕外动态（Figure 7）：洞穴在视野外被水填满，水后来流到 agent 所在位置——这种 off-screen dynamics 是纯像素方法无法实现的
- 从单张 RGB 生成多样初始 world-frame（Figure 13）：同一初始图像可生成不同的 3D 环境
- 2000 步 episode（Figure 14）：虽然有 glitches 出现，但 3D 表示全局仍保持连贯，有自我恢复能力

## Ablation Study

论文没有设计独立的 ablation study 章节，但通过以下实验对比可以视为 ablation：

1. **Oasis 可视为去掉 $\mathbf{w}_{2D}$ 引导的 ablation**：Oasis 使用与 $\mathcal{P}_\theta$ 相同的 DiT backbone，但只靠最近 K 帧像素。对比 PERSIST-S vs Oasis 可见 3D 引导的巨大作用（FVD: 209 vs 706）。
2. **PERSIST-S vs PERSIST-XL**：空间 token 数量差 8 倍（216 vs 1728），但用户评分几乎相同，说明 3D 表示有效性对空间分辨率鲁棒。
3. **PERSIST-XL vs PERSIST-XL+w₀**：提供 GT 初始 world-frame 进一步提升所有指标，验证模型确实在利用 3D 信息而非忽略它。

---

# Limitations & Future Work

- **作者提到的局限**:
    
    1. **依赖 GT 3D 监督**：训练时需要 ground truth 的 3D voxel grid（world-frame），限制了在没有 3D 标注的真实数据上的应用。未来方向：用 2D-to-3D foundation models（如 SAM 3D、VGGT）生成合成 3D 标注。
    2. **Exposure bias 导致的累积误差**：虽然有噪声增强和 diffusion forcing 缓解，但长 episode（>2000 步）仍会出现越来越频繁的 glitches。未来方向：端到端 post-training on generated rollouts（类似 Self-Forcing）。
    3. **有限的空间范围**：world-frame 只覆盖 agent 周围固定区域（$48^3$ voxels），远离的区域信息会丢失。未来方向：3D memory bank，通过空间 chunk 加载实现无限空间记忆。
    4. **Ground truth 仍然显著优于生成结果**：用户研究中 ground truth 视频在所有指标上仍明显领先，说明还有很大提升空间。
- **我观察到的局限/疑问**:
    
    1. 目前仅在体素环境（Luanti/Minecraft）中验证，该环境天然适合 voxel grid 表示。对于连续表面的真实世界场景，如何定义和学习 world-frame 是一个开放问题。
    2. 没有报告推理延迟/FPS，而交互式体验对实时性要求很高。每帧需要运行 world denoiser（20步）+ camera model + 投影 + pixel denoiser（20步），实际速度可能较慢。
    3. 各模块独立训练虽然简化了流程，但也可能限制了整体性能上界。端到端训练或至少端到端 fine-tuning 可能带来进一步提升。
    4. 数据收集策略（随机动作序列）可能不够多样，对更复杂的交互行为（如建造、破坏方块）的建模能力有待验证。

# Personal Notes

- **核心启发**：将世界模型从"像素历史检索"转向"3D 状态维护+投影"是一个非常优雅的范式转换。这种思路可以推广到任何需要长期空间记忆的生成任务中。
- **可借鉴的设计**：
    - 跨模块噪声增强（10% random noise augmentation）是一个简单而有效的缓解模块间 distribution shift 的策略，值得在多模块 pipeline 中广泛应用。
    - Depth peeling 实现 3D-to-2D 投影是一个巧妙的工程选择，兼顾了效率和信息完整性。
    - 通过分配更多通道给 3D 引导（752 vs 16）来偏置模型关注 3D 信息，是一种简单有效的 inductive bias 注入方式。
- **值得深入探索的方向**：
    - 将 PERSIST 的思路扩展到非体素环境（如 NeRF/3DGS 表示）
    - 探索端到端 post-training 对长 episode 稳定性的提升效果
    - 3D memory bank 的设计——如何高效存储和检索空间 chunks
    - 将该框架应用于 embodied agent 训练，验证 3D world model 是否能产生更好的策略