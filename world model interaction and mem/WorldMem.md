# Paper Info

- **Title**: WORLDMEM: Long-term Consistent World Simulation with Memory
- **Authors**: Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, Xingang Pan
- **Venue/Year**: NeurIPS 2025 / arXiv: 2504.12369v3 (2026.01.01)
- **Affiliations**: S-Lab (NTU), Peking University, Shanghai AI Laboratory
- **Paper Link**: https://arxiv.org/abs/2504.12369
- **Code/Project Page**: https://xizaoqu.github.io/worldmem

## TL;DR

WORLDMEM 提出了一种基于 **记忆库（Memory Bank）** 的世界模拟框架，通过存储历史生成帧及其状态信息（位姿、时间戳），并利用 **状态感知的记忆注意力（State-aware Memory Attention）** 从记忆中检索相关信息，解决了视频扩散模型因有限上下文窗口导致的 **长期3D空间一致性丧失** 问题。在 Minecraft 和 RealEstate10K 上均取得显著提升。

---

# Introduction

## Task and Applications

这篇论文研究的是 **交互式世界模拟（Interactive World Simulation）** 任务。世界模拟的目标是：给定当前状态和动作，预测环境的下一个状态，从而构建一个可交互的虚拟世界。

实际应用场景包括：自动驾驶中的导航仿真（Navigation Simulation）、作为传统游戏引擎的替代方案（如用扩散模型实时生成游戏画面），以及 Agent 学习的环境模型。

## Technical Challenges

现有方法面临的核心挑战是 **有限的时间上下文窗口（Limited Temporal Context Window）**：

1. **遗忘问题**：视频生成模型由于计算和显存限制，只能在固定长度的上下文窗口内工作。超出窗口的历史内容会被丢弃，当相机转回之前看过的场景时，模型无法记住之前的内容，导致重新生成的画面与之前不一致。
2. **3D 一致性崩塌**：最直观的例子是——向左转再向右转回来，看到的场景和之前完全不同，这在真实世界中是不可能的。
3. **显式 3D 重建的局限**：一种自然的解决思路是做 3D 重建来保持一致性，但 3D 表示在动态、可演化的环境中不够灵活——一旦重建完成，修改和交互就很困难，且对大规模无界场景容易丢失细节。
4. **隐式记忆方法的缺陷**：SlowFastGen 通过 LoRA 存储抽象特征，紧凑但丢失视觉保真度和空间特异性；固定 token 数的场景表示（如 SRT）难以捕捉多样和演化环境的复杂性。

## 与之前工作的区别/定位

WORLDMEM 的核心区别在于采用了 **geometry-free（无几何）的记忆方案**，同时又不牺牲细节保真度：

- **不做显式 3D 重建**：避免了 3D 表示不灵活、动态场景处理困难的问题
- **不用抽象隐式表示**：不像 LoRA 那样压缩到高度抽象的特征空间，而是 **直接存储 token 级别的 latent 特征**，保留足够的视觉细节
- **核心创新**：引入 **状态感知（State-aware）** 的记忆机制——每个记忆单元不仅存帧的 latent token，还存其位姿和时间戳，通过注入这些状态线索到注意力机制中，让模型能进行 **跨视角、跨时间的时空推理**

## 解决 Challenge 的 Pipeline

### Contribution 1: Token 级别的 Memory Bank

**解决什么问题？** 解决上下文窗口有限导致的历史信息丢失。

**Key insight**：对于生成当前帧，通常只需要历史中的一小部分内容是相关的。因此不需要把所有历史都放进上下文，只需存储所有历史 latent token，然后按需检索。

**具体做法**：维护一个记忆库，存储所有历史生成帧的 latent token 以及对应的位姿 $\mathbf{p} \in \mathbb{R}^5$ (x, y, z, pitch, yaw) 和时间戳 $t$。通过基于 FOV 重叠度和时间距离的 confidence 评分进行贪心检索。

### Contribution 2: State-aware Memory Attention

**解决什么问题？** 解决如何从检索到的记忆中准确提取信息——尤其是在大视角差异和时间间隔下重建之前看到的场景。

**Key insight**：仅靠视觉 token 做 cross-attention 是不够的（因为同一场景在不同视角下的 token 差异很大），需要给 attention 注入显式的空间-时间状态线索，让模型知道"记忆帧是从哪个位置、什么时候看到的"。

**具体做法**：将位姿通过 Plücker Embedding 编码、时间戳通过 Sinusoidal Embedding 编码，加到 Q 和 K 上后再做 cross-attention。并采用 **相对状态编码**（query 设为零参考，key 归一化为相对值），简化学习目标。

### Contribution 3: 时间戳建模动态世界演化

**解决什么问题？** 现实世界是动态演化的（如植物生长、雪融化），同一地点在不同时间可能外观不同。

**Key insight**：通过在状态信息中加入 timestamp，模型不仅能记住"在哪里看到了什么"，还能推理"那是什么时候看到的"，从而在时间演化的世界中保持一致性。

---

# Method

## Overview

- **输入**：初始帧 + 动作序列（Minecraft 中为 25 维，包含移动、视角调整、事件触发等）
    
- **输出**：逐帧自回归生成的第一人称视角视频序列
    
- **Pipeline 整体流程**：
    
    1. 基于 Diffusion Forcing 的条件 DiT 自回归生成帧序列
    2. 每次生成新帧时，已生成的帧（latent token + 状态）存入 Memory Bank
    3. 从 Memory Bank 中根据 confidence 评分检索最相关的记忆帧
    4. 在 DiT block 中通过 Memory Block（state-aware cross-attention）将记忆信息注入当前生成过程
- **模块连接关系**：
    

```
Action + Timestep
      ↓
[Memory Bank] → Memory Retrieval → Retrieved Memory Units (tokens + states)
      ↓                                        ↓
Input Noisy Frames → [Spatial Block] → [Temporal Block] → [Memory Block (Cross-Attn with State Embedding)] → Output
      ↑                                                              ↑
      └──── × N denoising steps ──────────────────────────────────────┘
```

## Module 1: Interactive World Simulator (Baseline)

- **这个模块做什么？** 基于条件 Diffusion Transformer (CDiT) + Diffusion Forcing (DF) 的自回归视频生成器。
    
- **Motivation**：需要一个能根据动作信号逐帧生成第一人称视频的基础架构。
    
- **具体怎么做的？**
    
    标准视频扩散模型对所有帧使用同一噪声水平 $k$： $$p_\theta(\mathbf{x}_t^{k-1}|\mathbf{x}_t^k) = \mathcal{N}(\mathbf{x}_t^{k-1}; \mu_\theta(\mathbf{x}_t^k, k), \sigma_k^2 \mathbf{I})$$
    
    Diffusion Forcing 引入 **逐帧噪声水平** $k_t$，允许不同帧有不同噪声： $$p_\theta(\mathbf{x}_t^{k_t-1}|\mathbf{x}_t^{k_t}) = \mathcal{N}(\mathbf{x}_t^{k_t-1}; \mu_\theta(\mathbf{x}_t^{k_t}, k_t), \sigma_{k_t}^2 \mathbf{I})$$
    
    这使得自回归生成成为一个特例（只有最后一帧或几帧有噪声），可以灵活地超越训练长度生成视频。
    
    架构由多个 DiT Block 组成，每个 block 包含 Spatial Block（空间注意力）和 Temporal Block（因果时间注意力）。动作信号通过 MLP 投影后加到 timestep embedding 上，通过 AdaLN 注入 Temporal Block。
    

## Module 2: Memory Bank 与 Memory Retrieval

- **这个模块做什么？** 存储所有历史生成帧的信息，并在需要时检索最相关的子集。
    
- **Motivation**：上下文窗口有限（训练时只有 8 帧），超出窗口的内容会被遗忘，导致长期不一致。
    
- **Technical Challenge**：记忆帧可能非常多（如 600 帧），但条件输入的 memory window 有限（如 8 帧），如何选出最相关的？
    
- **数据结构**：Memory Bank = ${(\mathbf{x}_i^m, \mathbf{p}_i, t_i)}_{i=1}^N$，其中 $\mathbf{x}_i^m$ 是 token 级别的 latent 特征，$\mathbf{p}_i \in \mathbb{R}^5$ 是位姿，$t_i$ 是时间戳。
    
- **检索算法（Algorithm 1）**：
    
    1. **计算 Confidence 评分**：
        - FOV 重叠度 $o$：通过 Monte Carlo 采样（默认采 10000 个 3D 点）计算每个记忆帧与当前帧的视野重叠比例
        - 时间差 $d = |t_i - t_c|$
        - 综合评分 $\alpha = o \cdot w_o - d \cdot w_t$（$w_o=1$, $w_t=0.2/t_c$）
    2. **贪心选择 + 相似性去重**：
        - 选 confidence 最高的帧加入结果集
        - 移除与已选帧相似度 > 阈值 $t_r=0.9$ 的候选帧
        - 重复直到选满 $L_M$ 帧
- **为什么能 work？** 虽然简单，但 FOV 重叠度直接反映了"这个记忆帧与当前视角共享了多少空间内容"，是非常直觉且有效的相关性度量。相似性去重则避免选入大量冗余的相邻帧。
    

## Module 3: State Embedding

- **这个模块做什么？** 将位姿和时间戳编码为稠密特征向量，作为记忆注意力中的状态线索。
    
- **Motivation**：纯视觉 token 在视角差异大时无法提供足够的空间对应信息，需要显式的空间-时间坐标作为 "anchor"。
    
- **具体编码方式**：
    
    **位姿编码**：采用 Plücker Embedding。给定 5D 位姿 $\mathbf{p} = (x, y, z, \text{pitch}, \text{yaw})$，首先构造外参矩阵 $\mathbf{T}$，然后对每个像素 $(u,v)$ 计算： $$\pi_{uv} = K^{-1}[u, v, 1]^T, \quad d_{uv} = R_c \pi_{uv} + c$$ $$l_{uv} = (c \times d_{uv}, d_{uv}) \in \mathbb{R}^6$$ 得到 $L_i \in \mathbb{R}^{H \times W \times 6}$ 的稠密位姿特征。
    
    **时间戳编码**：Sinusoidal Embedding → MLP。
    
    最终状态嵌入： $$\mathbf{E} = G_p(\text{PE}(\mathbf{p})) + G_t(\text{SE}(t))$$ 其中 $G_p$ 和 $G_t$ 是 MLP。
    
- **Key insight**：Plücker Embedding 是一种 **逐像素** 的稠密位姿编码（Dense），而不是像稀疏位姿那样只给整帧一个向量。这对于像素级的空间对应推理至关重要。消融实验中 Dense vs Sparse 的 rFID 从 39.23 降到 29.34，证明了这一点。
    

## Module 4: State-aware Memory Attention (Memory Block)

- **这个模块做什么？** 在每个 DiT Block 中，利用 state-aware cross-attention 从记忆帧中提取与当前帧相关的信息。
    
- **这是整篇论文最核心的设计。**
    
- **具体流程**：
    
    设 $\mathbf{X}_q \in \mathbb{R}^{l_q \times d}$ 为当前帧的 feature（query），$\mathbf{X}_k \in \mathbb{R}^{l_k \times d}$ 为记忆帧的 feature（key/value）。
    
    **Step 1**：注入状态嵌入到 Q 和 K： $$\tilde{\mathbf{X}}_q = \mathbf{X}_q + \mathbf{E}_q, \quad \tilde{\mathbf{X}}_k = \mathbf{X}_k + \mathbf{E}_k$$
    
    **Step 2**：做 cross-attention（注意 V 不加状态嵌入）： $$\mathbf{X}' = \text{CrossAttn}(Q = p_q(\tilde{\mathbf{X}}_q), ; K = p_k(\tilde{\mathbf{X}}_k), ; V = p_v(\mathbf{X}_k))$$
    
    **关键设计——相对状态编码**：
    
    - Query 帧的状态统一设为零参考（位姿设为 identity，时间戳设为 0）
    - Key 帧的状态归一化为相对于 query 的相对值
    - 这样模型只需学习"记忆帧相对于我在哪里、相对于我是什么时间"的关系，而不需要学习绝对坐标下的推理
- **Memory 注入到 Pipeline 的方式**：
    
    - 训练时：记忆帧赋予最低噪声 $k_{\min}=15$（近似干净），上下文帧随机采样 $[k_{\min}, k_{\max}]$
    - 推理时：记忆帧和上下文帧都用 $k_{\min}$，当前生成帧用 $k_{\max}=1000$
    - 通过特殊的 temporal attention mask $A_{\text{mask}}$ 确保记忆帧只在 Memory Block 中被访问，记忆帧之间不互相影响

## 核心亮点深度解析

### 亮点 1：状态感知注意力中"V 不加状态嵌入"的设计

这是一个精妙的设计：Q 和 K 都加了状态嵌入，但 V 保持原始视觉特征。

**Intuition**：状态嵌入的作用是帮助 attention 计算"哪些记忆 token 与当前 token 空间位置对应"（即 routing），但最终传递给当前帧的信息应该是纯粹的视觉内容，不应混入坐标信息。这类似于 NeRF 中位置编码只参与查询但不直接成为输出的设计哲学。

**与之前方法的区别**：之前的记忆方法（如 StreamingT2V）主要靠视觉相似度做 attention，在视角差异大时完全失效。WORLDMEM 通过在 Q-K 中注入精确的空间坐标，让模型能做类似"变换矩阵"的推理——即使当前帧和记忆帧的视角完全不同，只要坐标对得上，就能正确检索到对应像素的信息。

### 亮点 2：渐进式训练策略（Progressive Sampling）

训练时先从小范围（2m）的记忆帧开始训练，然后逐步扩大到大范围（8m）。

**Intuition**：让模型先学会在简单情况（小视角差、短时间差）下做记忆对齐，建立基础能力后，再逐步挑战更困难的大时空跨度推理。类似 curriculum learning 的思想。

**效果**：相比直接用大范围训练，rFID 从 42.96 大幅降至 15.37，说明这种渐进策略对学习空间推理能力至关重要。

## Training

- **Loss Function**：标准的扩散模型去噪 Loss，遵循 Diffusion Forcing 的逐帧噪声水平范式。论文未详细说明 loss 的特殊设计（使用标准 MSE 去噪 loss）。
- **训练策略**：
    - Minecraft 实验基于 Oasis (Decart et al., 2024) 作为 base model
    - RealEstate10K 实验基于 DFoT (Song et al., 2025) 作为 base model
    - 使用 **渐进式采样策略**：先小范围 → 再大范围
    - 训练时上下文窗口 8 帧 + 记忆窗口 8 帧
- **关键超参数**：
    - Optimizer: Adam, lr = $2 \times 10^{-5}$
    - 分辨率：Minecraft 640×360（latent 32×18，patch 16×9）；RealEstate 256×256
    - 训练步数：Minecraft ~500K steps, 4 GPUs, batch size 4/GPU；RealEstate ~50K steps
    - 噪声水平：$k_{\min}=15$, $k_{\max}=1000$
    - 检索超参：$t_r=0.9$, $w_o=1$, $w_t=0.2/t_c$

---

# Experiment

## 资源消耗

- **训练**：4 GPUs，Minecraft 约 500K steps；RealEstate 约 50K steps
- **显存**：训练时 without memory 33GB → with memory 51GB；推理时 9GB → 11GB
- **推理速度**：without memory 1.03 it/s → with memory 0.89 it/s（单卡 H200）；检索延迟在 600 帧 memory bank 时约 0.1s（生成一帧约 0.9s，检索开销 ~10-20%）
- **Memory Bank 占用**：600 帧 visual token [600, 16, 18, 32] float32 约 21MB，非常轻量
- 使用加速技术（timestep distillation、early exit、sparse attention）后可达约 10 FPS

## 数据集 / Benchmark

**Minecraft (MineDojo)**：

- 训练集约 12K 个长视频，每个 1500 帧
- 多种地形（平原、热带草原、冰原、沙漠）、多种动作模态（移动、视角控制、事件触发）
- 测试 300 个视频

**RealEstate10K**：

- 训练集约 65K 短视频片段，带相机位姿标注
- 设计了 5 种评估轨迹（起止位姿相同的闭环），覆盖 100 个场景
- 轨迹长度 37-60 帧，超过所有 baseline 的训练长度上限（25 帧）

**评估指标**：PSNR（像素保真度）、LPIPS（感知相似度）、rFID（重建 FID，整体真实感）

## 定量结果

**Minecraft - Within Context Window**（上下文窗口内一致性）：

|Method|PSNR ↑|LPIPS ↓|rFID ↓|
|---|---|---|---|
|Full Seq.|20.14|0.0691|13.87|
|Diffusion Forcing|24.11|0.0094|13.88|
|**WORLDMEM**|**25.98**|**0.0072**|**13.73**|

**Minecraft - Beyond Context Window**（超出上下文窗口的长期一致性，600 帧记忆 + 生成 100 帧）：

|Method|PSNR ↑|LPIPS ↓|rFID ↓|
|---|---|---|---|
|Diffusion Forcing|17.32|0.4376|51.28|
|**WORLDMEM**|**23.98**|**0.1429**|**15.37**|

提升非常显著：PSNR 从 17.32→23.98（+6.66），rFID 从 51.28→15.37（降低 70%）。

**RealEstate10K**：

|Method|PSNR ↑|LPIPS ↓|rFID ↓|
|---|---|---|---|
|CameraCtrl|13.19|0.3328|133.81|
|TrajAttn|14.22|0.3698|128.36|
|Viewcrafter|21.72|0.1729|58.43|
|DFoT|16.42|0.2933|110.34|
|**WORLDMEM**|**23.34**|**0.1672**|**43.14**|

在真实场景上同样全面领先，超越了使用显式 3D 重建的 Viewcrafter。

## 定性结果

- **Minecraft**：在 600 帧记忆库基础上生成 100 帧，能准确保持多种环境（草原、沙漠等）中的一致性，还能正确建模动态事件（如下雨、植物生长、南瓜灯融化周围积雪）
- **RealEstate10K**：360° 闭环旋转后首尾帧高度一致，而其他方法的首尾帧差异明显
- **交互能力**：放置干草、种植种子后四处游走再回头看，放置的物体仍然存在且植物已生长

## Ablation Study

**1. Embedding 设计**（Table 2）：

|Pose type|Embed. type|PSNR|LPIPS|rFID|
|---|---|---|---|---|
|Sparse|Absolute|20.67|0.2887|39.23|
|Dense|Absolute|23.63|0.1830|29.34|
|**Dense**|**Relative**|**23.98**|**0.1429**|**15.37**|

→ Dense (Plücker) 比 Sparse 大幅提升；Relative 比 Absolute 在 rFID 上再降一半。

**2. 记忆检索策略**（Table 3）：Random → +Confidence → +Similarity，rFID 从 47.35→24.33→15.37，每一步都有显著提升。

**3. 训练采样策略**（Table 5）：Progressive 远优于 Small-range 和 Large-range，rFID 15.37 vs 46.55/42.96。

**4. 时间条件**（Table 6）：加入 timestamp 后 PSNR 从 23.17→25.12，rFID 从 23.89→16.53，证明时间信息对建模动态世界演化很重要。

**5. 记忆上下文长度**（Table 7）：长度 8 最优（PSNR 25.32），16 帧反而下降（过多记忆引入噪声）。

**6. 长期生成对比**（Figure 7）：300 帧序列中，无 Memory Block / Random 检索几乎立刻偏离 GT；无 Relative Embedding 100 帧后开始退化；完整方法 300 帧后仍保持高 PSNR。

---

# Limitations & Future Work

- **作者提到的局限**：
    1. 无法保证总能从记忆库中检索到所有必要信息（如被障碍物遮挡的视角，仅靠 FOV 重叠不够）
    2. 当前的环境交互缺乏多样性和真实感
    3. 记忆库的存储线性增长，处理超长序列时可能受限
- **我观察到的局限/疑问**：
    1. **位姿依赖**：方法依赖已知的位姿信息。虽然作者提供了位姿预测模块，但性能有一定下降（PSNR 25.32→23.13），在位姿累积误差大的场景可能更严重
    2. **记忆帧数量固定**：memory window 固定为 8 帧，对于极度复杂的场景（如 360° 全景）可能不够，但增大到 16 帧反而变差，说明当前的 attention 机制在处理更多记忆帧时还不够鲁棒
    3. **FOV 重叠作为唯一检索信号**：遮挡关系、语义相关性等更高层的信息未被考虑
    4. **评估主要集中在 Minecraft**：Minecraft 的纹理相对简单且规则，真实场景的验证（RealEstate10K）只有短片段非交互式数据，真正复杂的真实世界长期交互场景尚未验证

# Personal Notes

- **State-aware Attention 的范式**可以迁移到其他需要跨视角/跨时间对应的任务，如长视频编辑、多视角一致性生成、视频 inpainting 等
- **Relative State Embedding** 的设计思想（将 query 设为零参考）非常优雅，避免了绝对坐标的累积误差，值得在其他涉及坐标条件的生成任务中借鉴
- **渐进式训练策略**（从简单到困难）的有效性再次印证了 curriculum learning 在 spatial reasoning 任务上的价值
- Memory Bank 的 token 级存储 + 基于物理意义的检索（FOV overlap）+ state-aware attention 的组合，形成了一个清晰、可扩展的框架，有望成为长期世界模拟的标准范式
- 未来可以探索：非均匀记忆压缩（远处/旧的记忆逐步压缩）、语义级记忆检索、与 3D 表示的混合方案