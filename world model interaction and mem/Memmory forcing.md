---
name: paper-reader
description: 深度阅读学术论文并按结构化模板输出详细清晰的中文笔记
triggers:
  - 读论文
  - read paper
  - paper reading
  - 论文笔记
  - 讲解论文
---
# Paper Info

- **Title**: Memory Forcing: Spatio-Temporal Memory for Consistent Scene Generation on Minecraft
- **Authors**: Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, Li Jiang
- **Venue/Year**: arXiv preprint, 2025 (arXiv:2510.03198v1)
- **Paper Link**: https://arxiv.org/abs/2510.03198
- **Code Link**: 未公开代码，项目页面：https://junchao-cs.github.io/MemoryForcing-demo/

## TL;DR

这篇论文提出了 Memory Forcing，一个面向 Minecraft 交互式场景生成的训练框架。它通过 Hybrid Training 和 Chained Forward Training 两种训练策略，配合基于 3D 几何的 Geometry-indexed Spatial Memory，让自回归视频扩散模型在**探索新场景时依赖 temporal memory**，在**重访旧场景时利用 spatial memory**，从而同时实现高质量新场景生成和长期空间一致性，且保持高效的检索与存储。

---

# Introduction

## Task and Applications

本文研究的是**交互式视频生成**任务，具体场景是 Minecraft 游戏的世界建模。玩家通过动作（移动、转向等）实时操控视角，模型需要以自回归方式根据历史帧和玩家动作预测未来帧。这个任务的核心挑战在于：模型既需要在**探索新区域**时生成高质量的、合理的新内容，又需要在**重新访问已探索区域**时保持空间一致性（即之前看过的建筑/地形不应该改变）。

应用场景包括交互式游戏引擎模拟、虚拟世界生成、以及基于世界模型的 embodied agent 训练等。

## Technical Challenges

之前的方法面临一个**根本性的 trade-off**，论文通过 Figure 1 非常直观地展示了两种范式的失败模式：

**Challenge 1: 纯 Temporal Memory 方法缺乏长期空间一致性。** 如 Oasis、NFD 等模型只依赖滑动窗口内的最近若干帧作为上下文。当玩家转一圈回到原地时，之前的视觉信息已经滑出窗口，模型无法"记住"之前的场景，导致重访时生成出与之前不一致的内容。

**Challenge 2: 加入 Spatial Memory 后，新场景生成质量下降。** 如 WorldMem 加入了基于 pose 的长期记忆检索，虽然在重访时有一定一致性，但在探索全新区域时，由于没有足够相关的 spatial memory，模型过度依赖不充分的检索结果反而导致生成质量退化。

**Challenge 3: Teacher Forcing 训练与推理时的 gap。** 传统 teacher-forced 训练使用 ground truth 帧作为条件，推理时却使用模型自身预测的帧（可能有误差/漂移），导致推理时模型过度依赖短时 temporal cues 而忽略 spatial memory。

**Challenge 4: 现有检索效率问题。** 基于外观的帧级检索（WorldMem）对视角和光照变化敏感，且随序列增长检索延迟线性增加；State-space 方法（LSVM）将历史压缩成隐状态，虽然高效但缺乏显式空间索引。

## 与之前工作的区别/定位

核心区别在于：之前的方法要么只做 temporal memory 要么无差别地加入 spatial memory，而本文的 Memory Forcing 让模型**学会在不同场景下动态切换**记忆使用策略——探索时主要用 temporal memory，重访时依靠 spatial memory。这不仅是一种新的架构设计，更是一个**训练框架**层面的创新。

与 WorldMem 对比：WorldMem 用 pose-based 检索，存储所有历史帧，复杂度 O(n)；本文用 3D 点云的 point-to-frame 检索，复杂度 O(1)，存储量减少 98.2%，速度快 7.3 倍。

## 解决 Challenge 的 Pipeline

### Contribution 1: Hybrid Training（混合训练策略）

**解决什么问题？** 解决 temporal memory 和 spatial memory 的 trade-off，即模型需要"知道"什么时候该用哪种 memory。

**Key insight：** 不同数据集天然代表不同的游玩模式——VPT 数据集是人类玩家的探索性游玩（很少重访），MineDojo 数据集是合成轨迹有频繁的位置重访。用不同的 memory 策略训练不同数据集，就能让模型学会根据场景切换策略。

**具体做法：** 固定上下文窗口 L 帧，其中 L/2 帧固定为 temporal context，另外 L/2 帧灵活分配：VPT 数据用 extended temporal memory（更长的时间上下文），MineDojo 数据用 spatial memory（检索出的历史帧）。

### Contribution 2: Chained Forward Training (CFT)

**解决什么问题？** 解决 teacher forcing 导致的 train-test gap，让模型学会在自身预测有误差的情况下依然能利用 spatial memory 维持一致性。

**Key insight：** 训练时用模型自身的预测帧逐步替换 ground truth 帧，这样会在窗口间产生更大的 pose 变化（因为预测不完美），**迫使模型主动依赖 spatial memory 来纠正漂移**。

**具体做法：** 顺序处理时间窗口，前一个窗口的预测帧被塞进后一个窗口的输入中（不传梯度），形成级联依赖。

### Contribution 3: Geometry-indexed Spatial Memory

**解决什么问题？** 解决检索效率和检索质量问题。

**Key insight：** 维护一个显式 3D 点云，每个 3D 点记录它来源于哪一帧。检索时将当前视角可见的 3D 点投影出来，统计可见点最多来自哪些历史帧，即可快速找到空间上最相关的帧。这种方法天然对视角变化鲁棒，且复杂度与序列长度无关。

**具体做法：** Point-to-Frame Retrieval + Incremental 3D Reconstruction（基于 VGGT 的流式重建 + 选择性关键帧 + 体素下采样）。

---

# Method

## Overview

- **输入**：历史视频帧序列 $X_{1:T}$、玩家动作序列 $A_{1:T}$、相机 pose 信息
    
- **输出**：自回归生成的未来视频帧
    
- **Pipeline 整体流程**：
    
    1. 维护一个流式 3D 点云（Incremental 3D Reconstruction）
    2. 根据当前 pose 做 Point-to-Frame Retrieval，找到空间上相关的历史帧作为 spatial memory
    3. 结合 temporal memory（最近的若干帧）和 spatial memory 构建上下文窗口
    4. 送入 DiT backbone 进行扩散去噪生成下一帧
    5. 训练时使用 Hybrid Training + Chained Forward Training
- **模块连接关系**：
    

```
Camera Trajectory → Incremental 3D Reconstruction → 3D Point Cloud
                                                        ↓
Current Pose → Point-to-Frame Retrieval (查询点云) → Spatial Memory Frames
                                                        ↓
Temporal Memory Frames + Spatial Memory Frames + Action → Context Window
                                                        ↓
                            DiT Backbone (Self-Attention + Memory Cross-Attention) → Next Frame
```

## Module 1: Memory-Augmented Architecture（记忆增强架构）

- **这个模块做什么？** 在 DiT backbone 中加入 Memory Cross-Attention，将长期 spatial memory 注入到生成过程中。
    
- **Motivation：** 标准的 DiT 只有 self-attention 处理窗口内帧之间的关系，无法利用窗口外的历史帧信息。
    
- **具体怎么做的？**
    
    - 基础架构沿用 NFD 的 DiT，包含 Spatio-Temporal Self-Attention、adaLN-zero 动作条件注入、3D 位置编码
    - 在每个 DiT Block 中加入一个 **Memory Cross-Attention** 层：

$$\text{Attention}(\tilde{Q}, \tilde{K}_{\text{spatial}}, V_{\text{spatial}}) = \text{Softmax}\left(\frac{\tilde{Q}\tilde{K}_{\text{spatial}}^T}{\sqrt{d}}\right)V_{\text{spatial}}$$

- 其中 $\tilde{Q}$ 和 $\tilde{K}_{\text{spatial}}$ 在标准 Q/K 基础上加入了 Plücker 坐标，编码当前视角和历史视角之间的相对 pose 信息
    
- 检索到的历史帧做 key 和 value，当前帧 token 做 query
    
- **为什么能 work？** Plücker 坐标提供了几何上的相对位置关系，让 cross-attention 可以感知"这个历史帧是从哪个角度看的"，从而更好地融合空间信息。
    

## Module 2: Hybrid Training（混合训练策略）

- **这个模块做什么？** 通过不同数据分布上使用不同的 memory 策略来训练，让模型学会动态选择依赖哪种 memory。
    
- **Motivation：** 如果统一用 spatial memory 训练，模型在新场景（没有有效 spatial memory 可用）时会退化；如果不用 spatial memory 训练，模型不会利用它。需要让模型**在两种模式之间灵活切换**。
    
- **Technical Challenge：** 如何在一个模型中同时学会两种截然不同的策略？
    
- **具体怎么做的？**
    

上下文窗口 $W$ 的构建公式：

$$W = [T_{\text{fixed}}, M_{\text{context}}] = \begin{cases} [T_{\text{fixed}}, M_{\text{spatial}}] & \text{如果正在重访已观察过的区域} \ [T_{\text{fixed}}, T_{\text{extended}}] & \text{如果正在探索新场景} \end{cases}$$

- $T_{\text{fixed}}$：固定的 $L/2$ 帧最近 temporal context
- $M_{\text{spatial}}$：通过 Geometry-indexed Spatial Memory 检索的历史帧
- $T_{\text{extended}}$：更早时间步的额外 temporal 帧

实际操作：

- 在 **MineDojo 合成数据集**（频繁重访）上使用 spatial memory 条件
    
- 在 **VPT 数据集**（人类探索游玩）上使用 extended temporal context 条件
    
- 两种数据混合训练
    
- **为什么能 work？** 这本质上是利用数据分布的差异作为隐式的"场景类型标签"。MineDojo 轨迹有大量重访行为，用 spatial memory 训练让模型学会"看到相关 spatial memory 就用它"；VPT 数据没有重访，用纯 temporal 训练让模型学会"没有有效 spatial memory 时就专注于 temporal 线索"。推理时模型自然根据检索到的 spatial memory 的质量/相关性来决定依赖程度。
    

## Module 3: Chained Forward Training (CFT)

- **这个模块做什么？** 在训练时引入模型自身的预测帧替换 ground truth，弥合 train-test gap。
    
- **Motivation：** 标准 teacher forcing 训练时用 GT 帧，推理时用自己的预测帧，两者之间存在分布偏移（exposure bias）。这导致推理时误差逐步累积，pose 漂移变大，模型来不及适应就已经偏离了。
    
- **具体怎么做的？** 详见 Algorithm 1：
    
    1. 对于一段视频，按窗口大小 $W$ 滑动处理
    2. 在第 $j$ 个窗口内，如果某个位置的帧已经在之前的窗口中被预测过，就使用**预测帧**替换 ground truth
    3. 计算该窗口的 diffusion loss
    4. 生成当前窗口最后一帧的预测（**不传梯度**，用更少的去噪步骤），存入预测帧缓存供下一个窗口使用

训练目标：

$$\mathcal{L}_{\text{chain}} = \frac{1}{T}\sum_{j=0}^{T-1} \mathbb{E}_{t,\epsilon}\left[|\epsilon - \epsilon_\theta(W_j(x, \hat{x}), C_j, t)|^2\right]$$

其中 $C_j = {A_j, P_j, M_{\text{spatial}}}$ 包含动作、pose 和 spatial memory。

- **为什么能 work？**
    - 预测帧不完美 → 窗口间 pose 漂移更大 → 模型被迫更多依赖 spatial memory 来校正
    - 训练时就见过了"自己预测的噪声帧"，推理时面对类似情况就不会 panic
    - 这本质上是 Scheduled Sampling 思想在扩散模型+空间记忆场景下的巧妙应用

## Module 4: Geometry-indexed Spatial Memory（几何索引空间记忆）

### 4.1 Point-to-Frame Retrieval

- **做什么？** 根据当前视角从 3D 点云中找到最相关的 top-8 历史帧。
    
- **具体做法：**
    
    1. 将全局点云投影到当前相机 pose
    2. 找出当前视角可见的所有 3D 点 $P_{\text{visible}}^t$
    3. 统计这些可见点各自来源于哪一帧（每个 3D 点存储了它的 source frame index）
    4. 选取出现次数最多的 top-8 帧：

$$H_t = \arg\max_{k=1,...,8} \text{Count}(\text{source}(p_i) : p_i \in P_{\text{visible}}^t)$$

- **为什么好？**
    - 复杂度 O(1)，与序列长度无关（因为点云通过体素下采样保持有界大小）
    - 对视角变化天然鲁棒（同一个 3D 点从不同角度看都能被检索到）
    - 不需要存储所有历史帧，只存关键帧

### 4.2 Incremental 3D Reconstruction

- **做什么？** 流式地构建和更新全局 3D 点云。
    
- **关键帧选择策略：** 当一帧满足以下条件之一时成为关键帧：
    
    - 它看到了之前未观察过的区域（NovelCoverage）
    - 当前历史帧数量不足阈值 $\tau_{\text{hist}} = L/2$

$$\text{IsKeyframe}(t) = \text{NovelCoverage}(I_t, G_{\text{global}}) \text{ or } (|H_t| < \tau_{\text{hist}})$$

- **3D 重建过程：**
    1. 使用 VGGT 网络对关键帧窗口生成相对深度图和置信度
    2. Cross-Window Scale Alignment 模块对齐不同窗口间的深度尺度（通过重叠帧的深度对应关系做最小二乘）
    3. 使用 pose 导出的外参矩阵做深度反投影生成 3D 点

$$E = \begin{bmatrix} R(\text{pitch, yaw}) & -RC \ 0^T & 1 \end{bmatrix}$$

4. 体素下采样整合到全局表示中

- **为什么高效？**
    - 选择性重建：只处理有新空间覆盖的帧，重访区域不重复计算
    - 体素下采样：任何 pose 区域的点密度有上界，保证检索复杂度恒定
    - 存储量随**空间覆盖范围**而非时间长度增长

## 核心亮点深度解析

### 亮点 1: Hybrid Training 的设计哲学

**Intuition：** 这个设计的核心 insight 是——不同的游戏行为模式（探索 vs 重访）对 memory 的需求是截然不同的，而且这两种需求可以通过**数据集的天然差异**来解耦学习。

**与之前方法的关键区别：** WorldMem 对所有场景统一使用 spatial memory，这就导致在新场景中没有有效 spatial memory 时模型不知所措。而 Memory Forcing 通过混合训练让模型学会了一种**隐式的切换机制**：当检索到的 spatial memory 与当前帧不太相关时（探索模式），模型自动更依赖 temporal context；反之则利用 spatial memory。

**为什么更好？** 这避免了显式设计一个"何时用何种 memory"的规则（这种规则很难设计好），而是让模型从数据中自己学会这种判断能力。

### 亮点 2: Chained Forward Training 的巧妙之处

**Intuition：** CFT 的核心洞察是——训练时的 pose 漂移太小（因为用了 GT 帧，pose 几乎完美），推理时 pose 漂移很大（因为用了预测帧，累积误差）。通过在训练时刻意引入"不完美预测帧"制造更大的 pose 漂移，模型被**迫使**更加重视 spatial memory 作为"锚点"来维持一致性。

**与 Self-Forcing (Huang et al., 2025) 的关系：** Self-Forcing 也是试图弥合 train-test gap，但 CFT 特别巧妙之处在于它将 bridge train-test gap 和 encourage spatial memory usage 两个目标统一到了一个训练策略中。

## Training

- **数据集：**
    
    - **VPT 数据集** (Baker et al., 2022)：约 4000+ 小时的人类 Minecraft 游玩视频，配有 25 维动作向量。排除了没有动作或 GUI 可见的帧。用于探索模式的训练（extended temporal memory）。
    - **MineDojo 合成数据集** (Fan et al., 2022)：按 WorldMem 的配置生成，11k 个视频，每个 1500 帧，包含频繁的位置重访和相邻视角。用于重访模式的训练（spatial memory）。
- **Loss Function：**
    
    - 标准 Diffusion Forcing loss：$\mathcal{L} = \mathbb{E}[|\epsilon_{1:T} - \epsilon_\theta(\tilde{X}_{1:T}, k_{1:T})|^2]$
    - Chained Forward Training loss：$\mathcal{L}_{\text{chain}} = \frac{1}{T}\sum_{j=0}^{T-1}\mathbb{E}_{t,\epsilon}[|\epsilon - \epsilon_\theta(W_j(x,\hat{x}), C_j, t)|^2]$
    - 每个帧有独立的噪声等级 $k_t \in [0,1]$
- **训练策略：**
    
    - 在 VPT 数据上先做预训练（temporal-only），然后加入 MineDojo 数据做 hybrid training
    - CFT 在 hybrid training 基础上进行，逐步引入预测帧替换
    - CFT 中的预测帧生成用更少的去噪步骤，且不传梯度（detach）
- **关键超参数：**
    
    - 训练步数：约 400k steps 收敛
    - GPU：24 块 NVIDIA H20/H100
    - Batch size：4
    - 优化器：Adam，学习率 4e-5
    - 上下文窗口大小：16 帧
    - 帧分辨率：384×224
    - 帧 tokenization：2D VAE（沿用 NFD），16× 空间压缩，每帧 24×14 个连续 token
    - Point-to-Frame 检索 top-k：8 帧
    - 关键帧历史数阈值 $\tau_{\text{hist}} = L/2$

---

# Experiment

## 资源消耗

- **训练**：24 块 H20/H100 GPU，约 400k steps 收敛（batch size 4）
- **推理速度**：本文方法在 0-3999 帧范围内总平均速度 31.21 FPS（包含完整 3D 重建+检索 pipeline），而 WorldMem 仅 4.27 FPS
- **模型参数量**：论文未明确提及具体参数量
- **存储效率**：对于 4000 帧视频，WorldMem 需存储全部 4000 帧，本文方法仅需约 72.65 帧（减少 98.2%）

## 数据集 / Benchmark

评估使用了三个基于 MineDojo 构建的数据集：

1. **Long-term Memory**：150 个长视频序列（1500 帧），测试空间一致性
2. **Generalization Performance**：150 个视频（800 帧），来自 9 种未见过的 Minecraft 地形
3. **Generation Performance**：300 个视频（800 帧），评估新环境中的生成质量

评估指标：FVD（↓）、PSNR（↑）、SSIM（↑）、LPIPS（↓），所有方法统一在第 600-800 帧（200帧）上评估。

## 定量结果

|维度|方法|FVD↓|PSNR↑|SSIM↑|LPIPS↓|
|---|---|---|---|---|---|
|Long-term Memory|Oasis|196.8|16.83|0.5654|0.3791|
||NFD|220.8|16.35|0.5819|0.3891|
||WorldMem|122.2|19.32|0.5983|0.2769|
||**Ours**|**84.9**|**21.41**|**0.6692**|**0.2156**|
|Generation Perf.|Oasis|285.7|14.51|0.5063|0.4704|
||NFD|349.6|14.64|0.5417|0.4343|
||WorldMem|290.8|14.71|0.4906|0.4531|
||**Ours**|**185.9**|**17.99**|**0.6155**|**0.3031**|

**核心发现：** 在所有三个评估维度上全面超越所有 baseline。特别值得注意的是，在 Generation Performance（新场景生成）上，本文方法比 WorldMem 提升巨大（FVD 从 290.8 降到 185.9），验证了 Hybrid Training 成功避免了加入 spatial memory 后新场景质量退化的问题。

## 定性结果

- **Long-term Memory（Fig. 3）**：回到之前访问过的位置时，本文方法生成的场景与之前高度一致；WorldMem 虽有一定记忆但产生不稳定视角和伪影；Oasis/NFD 完全无法保持空间一致性
- **Generalization（Fig. 4 上半部分）**：在未见地形上，本文方法生成稳定一致；WorldMem 和 NFD 出现伪影；Oasis 场景不一致
- **Generation（Fig. 4 下半部分）**：本文方法展示了自然的运动动态——远处场景随接近逐渐清晰；WorldMem 质量严重退化；NFD 远景几乎无变化；Oasis 远景过于简化
- **Generalization on frozen ocean（Fig. 5）**：WorldMem 在冰冻海洋地形上会生成类似训练集平原的场景（泛化失败），而本文方法正确保持了冰冻海洋地形

## Ablation Study

**训练策略消融（Table 3）：**

|训练策略|检索策略|FVD↓|PSNR↑|
|---|---|---|---|
|FT（全参数微调）|Pose-based|366.1|15.09|
|HT-w/o-CFT|Pose-based|230.4|16.24|
|HT-w/o-CFT|3D-based|225.9|16.24|
|**MF (HT+CFT)**|**3D-based**|**165.9**|**18.17**|

**关键发现：**

- **Hybrid Training vs 全参数微调**：FVD 从 366.1 降到 230.4，说明混合训练有效教会模型平衡两种 memory
- **加入 CFT 的效果**：FVD 从 225.9 进一步降到 165.9，CFT 贡献显著，证实了在训练中引入 model rollout 能有效促进 spatial memory 的利用
- **3D-based vs Pose-based 检索**：在同样的训练策略（MF）下，3D-based 检索大幅优于 pose-based（165.9 vs 隐含对比），且效率更高

**检索效率消融（Table 2）：**

- WorldMem 的检索速度随序列增长从 10.11 FPS 降到 1.47 FPS（线性衰减）
- 本文方法反而随序列增长更快（18.57 → 37.84 FPS），因为重访区域不增加新关键帧
- 在 3000-3999 帧范围内，本文方法比 WorldMem 快 25.7 倍

---

# Limitations & Future Work

- **作者提到的局限**：
    
    1. 目前仅在 Minecraft 环境验证，可能无法直接泛化到其他环境
    2. 固定分辨率 384×224，可能限制高保真度应用
- **我观察到的局限/疑问**：
    
    1. **3D 重建的准确性依赖**：整个 spatial memory 系统建立在 VGGT 的深度估计和 pose 信息的准确性上。在 Minecraft 这种方块世界中 3D 重建相对容易，但在更复杂的真实世界场景中，深度估计误差会直接影响点云质量和检索精度
    2. **CFT 中预测帧质量的影响**：CFT 使用少量去噪步骤生成预测帧，这些帧的质量如何影响训练稳定性？如果预测帧质量太差，是否会导致训练不稳定？论文未对此做充分讨论
    3. **两种数据集的混合比例**：VPT 和 MineDojo 数据的混合比例如何确定？是否对此做了敏感性分析？
    4. **动态物体处理**：Minecraft 中有移动的生物等动态物体，3D 点云无法处理动态物体的位置变化，这是否会造成检索错误？

# Personal Notes

- **Hybrid Training 的思路非常值得借鉴**：用数据分布差异来隐式教会模型做 mode switching，而不是显式设计切换规则。这个思路可以推广到其他需要"情境依赖策略选择"的任务中
- **CFT 将 exposure bias 缓解与 spatial memory 利用统一到一个训练策略中**，这种"一石二鸟"的设计思路值得学习
- **Point-to-Frame Retrieval 的设计很优雅**：利用 3D 点的来源信息做帧检索，将 3D 空间和 2D 帧之间的对应关系显式建模，比纯基于 appearance 或 pose 的检索都更 robust
- **对于需要长期记忆的序列生成任务**，显式的几何表示（3D 点云）+ 高效的索引机制是一种很有前途的方向，可能在自动驾驶场景重建、具身导航等领域也有应用
- **Memory Forcing 名字的含义**：类似 Teacher Forcing，但这里是"强制模型学会使用 memory"——通过训练策略设计让模型不得不依赖 spatial memory，这种"forcing"的训练哲学很有启发性