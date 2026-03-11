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

- **Title**: RELIC: Interactive Video World Model with Long-Horizon Memory
- **Authors**: Yicong Hong*, Yiqun Mei*, Chongjian Ge*, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, Hao Tan （*共同一作，Yicong Hong 为 Project Lead）
- **Venue/Year**: arXiv preprint, December 2025（Adobe Research）
- **Paper Link**: https://arxiv.org/abs/2512.04040
- **Code Link**: 项目主页 https://relic-worldmodel.github.io/ （截至论文发布未提供代码）

## TL;DR

RELIC 是一个 14B 参数的交互式视频世界模型，给定一张图片和文字描述，用户可以通过键盘/鼠标实时（16 FPS）探索任意场景长达 20 秒。其核心创新在于：(1) 用高度压缩的历史 latent token + 绝对相机位姿编码实现高效的长时空间记忆；(2) 提出 replayed back-propagation 技术，使得在 20 秒长视频上的 Self-Forcing 蒸馏变得 memory-efficient。在 action following 精度、长视频稳定性和空间记忆检索方面均优于已有方法。

---

# Introduction

## Task and Applications

本文研究的是**交互式视频世界模型（Interactive Video World Model）**：给定一张起始图像，用户通过键盘/鼠标等输入控制相机运动，模型实时生成对应的视频流，模拟"走进一张图片并自由探索"的体验。

应用场景非常广泛：自动驾驶模拟、具身智能（Embodied AI）、机器人训练、空间智能、沉浸式虚拟内容创作等。

## Technical Challenges

构建一个实用的视频世界模型需要**同时**满足三个关键需求，而现有方法往往只能解决其中一个：

1. **实时长时流式生成（Real-time Long-horizon Streaming）**：视频必须以实时延迟响应用户连续的控制输入。这要求 few-step 甚至 one-step 去噪，同时 autoregressive rollout 不能 drift（质量退化、颜色过饱和、输出静止等）。
    
2. **一致的空间记忆（Consistent Spatial Memory）**：当用户回头看之前探索过的区域时，模型不能"遗忘"之前生成的内容，必须保持 3D 空间一致性。这需要存储和检索长历史信息。
    
3. **精确的用户控制（Precise User Control）**：模型需要准确地响应用户的动作指令（如前进、左转、抬头等）。
    

**核心矛盾**在于：长时空间记忆需要额外的计算和 GPU 显存来存储、传输和推理过去的 token，这会带来严重的 FLOPs 和显存带宽瓶颈，与实时性需求直接冲突。

## 与之前工作的区别/定位

之前的方法大致可以分为三类，各有局限：

- **循环模型更新（Recurrent model updates）**类方法（如 Zhang et al. 2025b, Po et al. 2025）：空间记忆受限于模型内部状态容量，且通常只适用于特定视觉域。
- **外部记忆库 + 手工检索启发式**（如 Yu et al. 2025b, Xiao et al. 2025）：需要手工设计的检索规则。
- **显式 3D 场景表示**（如 Ma et al. 2025, Ren et al. 2025）：引入强归纳偏置，受限于重建精度和运行时开销。

RELIC 的核心定位是：**不使用任何显式 3D 表示或手工记忆启发式**，而是通过在 KV cache 中存储高度压缩的、带有相机位姿编码的历史 latent token，让模型隐式地学会 3D 一致的内容检索。同时，RELIC 是第一个使用 20 秒长上下文教师模型进行蒸馏的方法（而非之前工作中常用的 5 秒短上下文教师），从而提供更强的长时一致性监督。

## 解决 Challenge 的 Pipeline

### Contribution 1: Camera-Aware Compressed Memory（相机感知的压缩记忆机制）

**解决什么问题？** 长时空间记忆与实时推理的矛盾——保留完整的 KV cache 会导致显存和计算量线性增长。

**Key insight**：VAE 编码的 latent 空间存在大量的空间冗余，对 latent token 进行适度的空间下采样后，大部分原始信息仍可恢复。同时，通过将绝对相机位姿编码到 Q/K 投影中，模型可以通过视角感知的上下文对齐来隐式地进行 3D 内容检索。

**具体做法**：KV cache 分为两个分支——近期的 rolling window cache（不压缩）和远期的 compressed memory cache（空间下采样 1×/2×/4× 交替），平均将 token 数量压缩约 4 倍（如从 120K 降到 30K）。

### Contribution 2: Long-Horizon Teacher Fine-tuning（20 秒长上下文教师模型）

**解决什么问题？** 5 秒的短上下文教师无法提供长时一致性的监督，也无法处理剧烈的视角变化。

**Key insight**：要让学生模型学会长时记忆检索，教师模型本身必须先具备长时生成能力。

**具体做法**：对 Wan-2.1 进行课程学习式的微调（5s → 10s → 20s），并使用 YaRN 技术扩展 RoPE 位置编码。

### Contribution 3: Replayed Back-Propagation（重放式反向传播）

**解决什么问题？** 在 20 秒长视频上进行 Self-Forcing 蒸馏时，完整的计算图会导致 GPU 显存不可承受。

**Key insight**：可以将反向传播从全序列微分拆解为逐块微分——先无梯度地生成完整序列并缓存 DMD score difference map，然后逐块重放前向传播并反传梯度，每处理完一个块就释放其计算图。

**具体做法**：三步走——(a) 无梯度全序列 rollout → (b) 缓存 score difference → (c) 逐块重放+反传。峰值 GPU 显存从整个 rollout 降低到单个 block。

### Contribution 4: 高质量 UE 数据集（精心策划的 Unreal Engine 数据）

**解决什么问题？** 现实世界数据存在动作分布不均衡、动作耦合严重、缺乏视点重访等问题。

**Key insight**：少量但精心策划的、控制精确的合成数据，从强预训练 backbone 出发就足以提升可控性同时保持泛化能力。

**具体做法**：350 个 UE 场景，1400+ 条人类控制的相机轨迹，1600+ 分钟视频，并专门设计了动作平衡采样和 time-reverse augmentation。

---

# Method

## Overview

- **输入**：一张 RGB 图像 $f_0$、一段文字描述 $c_{\text{text}}$、一系列用户动作控制 $(A_1, A_2, \ldots, A_T)$
- **输出**：实时生成的视频流 $(\hat{f}_1, \hat{f}_2, \ldots, \hat{f}_T)$，480×832 分辨率，16 FPS，最长 20 秒
- **Pipeline 整体流程**：采用两阶段 pipeline——先训练一个双向（bidirectional）视频扩散教师模型（支持 20 秒生成），再将其蒸馏为一个 few-step 的自回归（causal）学生模型。
- **模块连接关系**：

```
Input Image + Text + Actions
    │
    ▼
[Stage 1] Action-Conditioned Bidirectional Teacher (20s, Wan-2.1 14B fine-tuned)
    │  (课程学习：5s → 10s → 20s)
    │
    ▼
[Stage 2] Causal AR Student (Wan-2.1 14B, causal attention)
    │  ├── ODE Initialization (Hybrid Forcing)
    │  ├── Self-Forcing Distillation (Replayed Back-Propagation)
    │  └── Memory Mechanism (Compressed Memory Cache + Rolling Window Cache)
    │
    ▼
[Runtime] Real-time Inference (4× H100, torch.compile, FP8, FlashAttention v3)
    │
    ▼
Generated Video Stream (16 FPS, 480×832)
```

## Module 1: Action-Conditioned Teacher Model

- **这个模块做什么？** 作为蒸馏的"老师"，生成高质量的 20 秒动作控制视频，提供长时一致性的监督信号。
    
- **Motivation / 为什么需要这个模块？** 之前的方法（如 APT-2, LongLive）使用 5 秒的短上下文教师来蒸馏长视频学生，但 5 秒窗口不足以学习长时记忆检索或应对剧烈视角变化。教师必须先"看过"长时序列，才能教会学生如何处理长时依赖。
    
- **Technical Challenge**：将 Wan-2.1 从 5 秒扩展到 20 秒（81 帧 → 317 帧），token 数量激增至 120K+，给训练带来巨大的显存压力。
    
- **具体怎么做的？**
    
    **Base Architecture**：基于 Wan-2.1 14B text-to-video DiT 模型，包含 ST-VAE（8× 空间压缩、4× 时间压缩，时间因果架构）和 DiT transformer blocks。文本通过 umT5 编码后经 cross-attention 注入，去噪时间步通过共享 MLP 注入。
    
    **Action Space**：设计了 13 维动作空间 $A \in \mathbb{R}^{13}$，覆盖 6 个平移动作（前/后/左/右/上/下）、6 个旋转动作（上仰/下俯/左转/右转/顺时针滚/逆时针滚）+ 1 个静止动作。每个动作是非负标量值（非二值），可编码相对速度。
    
    **Action Conditioning**：同时注入两种信号——
    
    - **Relative action embeddings**：通过专用 encoder 编码后，加到 self-attention 层之后的 latent 上。指导模型生成帧间场景转换。
    - **Absolute camera pose embeddings**：通过对 relative motion 积分得到绝对位姿 $P_t = \sum_{i=1}^{t} (R_i)^T \Delta P_i^c$, $R_t = \prod_{i=1}^{t} \Delta R_i^c$，编码后加到 Q 和 K 投影上（V 不变）。作为跨视角和跨时间的空间内容检索的代理。
    
    **Long-Horizon Training**：课程学习策略——5 秒训练 5000 iter → 10 秒训练 1000 iter → 20 秒训练 4000 iter。使用 YaRN 技术扩展 RoPE 位置编码以适应更长序列。
    
- **为什么能 work？** 将 relative action 和 absolute pose 分别注入不同位置的设计非常巧妙：relative action 指导"下一步该怎么动"，absolute pose 指导"当前在哪里，该检索哪些历史内容"。两者的计算角色完全不同，分开注入避免了信息混淆。
    

## Module 2: Memory Mechanism（记忆压缩机制）

- **这个模块做什么？** 在 AR 学生模型推理时，高效地存储和检索长时历史信息，使模型能在回看之前探索过的区域时保持空间一致性。
    
- **Motivation / 为什么需要这个模块？** 最简单的方法是保留所有历史 token 在 KV cache 中，但 KV cache 显存和 attention 计算量都随序列长度线性增长，无法实时。而只用短滑动窗口又会丢失长程信息。
    
- **Technical Challenge**：在大幅压缩 token 数量的同时保留足够的信息以恢复高保真内容。
    
- **具体怎么做的？**
    
    KV cache 由两个分支组成：
    
    1. **Rolling Window Cache**（滚动窗口缓存）：存储最近 $w$ 个 latent 的未压缩 KV token。保持小窗口以防止模型仅依赖短期模式。
        
    2. **Compressed Long-Horizon Memory Cache**（压缩长时记忆缓存）：存储从序列开头到 $i-w$ 的 latent 的空间下采样 KV token。采用交替压缩比的配置，具体为：
        
    
    $$S = [1, 4, 2, 4, 4, 4, 2, 4, 4, 2, 4, 4, 4, 2, 4, 4, 2, 4]$$
    
    其中 1× 表示不压缩，2× 和 4× 表示空间下采样。该配置循环应用，latent frame $i$ 的压缩比为 $s_i = S[i \mod \text{len}(S)]$。
    
    平均下来总 token 数压缩约 4 倍（如 120K → 30K），KV cache 显存和 attention FLOPs 也同比例降低。
    
- **为什么能 work？** VAE 编码的 latent 空间具有大量空间冗余。直觉上，一张 60×104 的 latent map 下采样到 30×52 或 15×26 后，主要的语义信息和空间结构仍然保留。再配合绝对相机位姿编码到 Q/K 上，模型可以通过视角对齐精准地从压缩记忆中检索对应的空间内容。
    

## Module 3: Distillation Framework（蒸馏框架）

- **这个模块做什么？** 将 20 秒双向教师模型蒸馏为 few-step 的自回归学生模型，使其能实时流式生成。
    
- **Motivation / 为什么需要这个模块？** 双向教师虽然能生成高质量长视频，但推理速度远达不到实时。需要通过蒸馏得到一个 few-step 的学生模型。
    
- **Technical Challenge**：在 20 秒长视频上进行 Self-Forcing 蒸馏时，计算图随视频长度线性增长，峰值 GPU 显存不可承受。
    
- **具体怎么做的？** 分两步：
    
    **Step 1: ODE Initialization with Hybrid Forcing**
    
    将双向教师转换为因果模型。不同于之前单独使用 teacher forcing 或 diffusion forcing，RELIC 提出混合策略：给定 $B$ 个 latent block 的训练序列，前 $B-K$ 个 block 是 clean compressed latent（teacher forcing），后 $K$ 个 block 加不同尺度噪声，因果地同时条件化于前面的压缩 clean latent 和后面的 noisy latent（diffusion forcing）。这为长时记忆检索提供了更强的初始化。
    
    **Step 2: Long-Video Distillation with Replayed Back-Propagation**
    
    基于 Self-Forcing 框架，使用 Distribution Matching Distillation (DMD) loss：
    
    $$\nabla_\theta \mathcal{L}_{KL} \approx -\mathbb{E}_u \left[ \int \left( s^{\text{data}}(\Psi(G_\theta(\epsilon, c_{\text{text}}), u)) - s^{\text{gen}}(\Psi(G_\theta(\epsilon, c_{\text{text}}), u)) \right) \frac{dG_\theta(\epsilon, c_{\text{text}})}{d\theta} , d\epsilon \right]$$
    
    其中 $\Psi$ 是前向扩散过程，$G_\theta$ 是学生生成器。
    
    **Replayed back-propagation 的三步流程**：
    
    (a) **无梯度全序列 rollout**：$\hat{x}_{0:L} = \text{stop-grad}(G_\theta(\epsilon_{0:L}))$
    
    (b) **缓存 score difference**：$\Delta\hat{s}_{0:L} = s^{\text{data}}(\hat{x}_{0:L}) - s^{\text{gen}}(\hat{x}_{0:L})$
    
    (c) **逐块重放 + 反传**：
    
    $$\nabla_\theta \mathcal{L}_{KL} \approx \sum_{l=1}^{L} -\Delta\hat{s}_l \frac{\partial G_\theta}{\partial \theta}$$
    
    每处理完一个 block 就立即释放其计算图，然后处理下一个。参数在全部 replay 完成后统一更新一次。
    
- **为什么能 work？** 关键洞察是：DMD loss 的梯度可以分解为两个独立的部分——score difference（不依赖学生梯度）和学生的 Jacobian $\partial G_\theta / \partial \theta$。前者可以预计算并缓存，后者可以逐块计算。这样就把全序列微分转化为了逐块微分，峰值显存从整个 rollout 降到单个 block，但仍能捕捉反映教师完整长视频分布的梯度。
    

## Module 4: Data Curation Pipeline（数据策划）

- **这个模块做什么？** 构建高质量的合成视频-动作-文本三元组数据集。
    
- **Motivation / 为什么需要这个模块？** 现实世界数据有三大缺陷：(1) 动作分布严重不均衡（主要是前进）；(2) 动作过度耦合（转弯+前进同时发生）；(3) 缺乏视点重访（很少回头看）。
    
- **具体怎么做的？**
    
    1. **数据采集**：350 个 UE 场景（室内+室外），人类操作者使用碰撞约束的相机控制器导航，记录 6-DoF 轨迹并渲染为 720p 视频。
        
    2. **数据过滤**：从相机运动（过快/抖动）、视点稳定性（微抖/震荡）、曝光光照（过曝/欠曝）、渲染质量（缺失纹理/几何弹出）四个维度过滤。
        
    3. **数据标注**：
        
        - Camera Pose：直接从 UE 获取完整 6-DoF 位姿
        - Action：通过相邻帧相对位姿计算 13 维动作标签，并按平均位移量归一化以消除场景尺度差异
        - Caption：将长视频切成 5 秒段，用 GPT-5 生成静态场景描述（刻意避免描述相机运动）
    4. **数据增强**：
        
        - **Balanced Action Sampling**：确保各动作分布均衡
        - **Time-Reverse Augmentation**：在采样的训练片段中随机选取一个枢轴点 $t^* \sim U(T/2, T)$，构造回文式序列 $f_{1:t^_}$ + $f_{t^_:(2t^*-T)}$，强制模型学习空间回溯和长时记忆

## 核心亮点深度解析

### 亮点 1: Camera-Aware Memory in KV Cache

这是 RELIC 最核心的创新。

**Intuition**：在一个 3D 场景中，当用户从位置 A 走到位置 B 再走回位置 A 时，应该看到与之前一致的内容。传统方法要么用显式 3D 表示（如 NeRF/3DGS），要么用外部记忆库+检索。RELIC 的洞察是：**如果模型知道每个历史 token 对应的绝对相机位姿，那么当当前 query 的位姿与某个历史 token 的位姿接近时，attention 机制本身就可以自动完成"检索"操作。** 这本质上是把 3D 空间检索隐式地编码在了 transformer 的 attention 中。

**和之前方法的关键区别**：

- vs 显式 3D 表示（3DGS/NeRF）：不需要显式重建，避免了重建误差和运行时开销
- vs 外部记忆库：不需要手工设计检索规则，让模型端到端学习检索
- vs 循环模型：不受内部状态容量限制，理论上可以存储任意长的历史

**为什么比之前的方案更好？** 通过空间压缩（平均 4×），120K token 被压缩到 30K，使得在 20 秒上下文中保持实时推理成为可能。同时，压缩是在空间维度上进行的，保留了时间维度的完整性，这对于时序一致的内容检索至关重要。

### 亮点 2: Replayed Back-Propagation

**Intuition**：在长视频蒸馏中，DMD loss 需要对整个 self-rollout 做反向传播，计算图太大。但注意到 DMD 梯度的特殊结构——score difference 部分和学生 Jacobian 部分可以解耦。score difference 只需要前向计算（frozen models），可以预计算；学生的 Jacobian 可以逐块计算因为每个块的前向传播只依赖于之前的 context（已经在无梯度 rollout 中生成好了）。

**和之前方法的关键区别**：之前的长视频蒸馏方法（如 LongLive, APT-2）使用短上下文教师（5 秒），把长视频拆成松散耦合的短片段。这限制了学生的长程记忆检索能力。RELIC 可以直接在 20 秒的完整上下文上蒸馏，实现了学生和教师在长视频分布上的直接对齐。

## Training

- **数据集**：自建的 Unreal Engine 渲染数据集。350 个授权 3D 场景，1400+ 条人类控制的相机轨迹，1600+ 分钟的 720p 视频。视频平均时长约 75 秒，最长可达 9 分钟。13 维动作标签均衡采样。
    
- **Loss Function**：
    
    - **ODE Initialization 阶段**：MSE loss，让学生在 4 个去噪时间步上回归教师预计算的 ODE 轨迹。
    - **Self-Forcing Distillation 阶段**：DMD loss（Distribution Matching Distillation），最小化学生分布与教师数据分布之间的 reverse KL 散度。通过 real-data score $s^{\text{data}}$ 和 generated-data score $s^{\text{gen}}$ 的差来近似梯度。
- **训练策略**：
    
    - **Teacher 训练**：课程学习——5s (5000 iter) → 10s (1000 iter) → 20s (4000 iter)。使用 YaRN 扩展 RoPE。
    - **Student 蒸馏**：同样渐进式增加 rollout 长度——5s (250 iter) → 10s (150 iter) → 20s (150 iter)。
    - **并行策略**：FSDP + Sequence Parallelism + Tensor Parallelism 的组合，32 张 H100 GPU (80GB)。
- **关键超参数**：
    
    - 模型大小：14B 参数（基于 Wan-2.1）
    - 分辨率：480×832
    - 帧率：16 FPS
    - 去噪步数：4 步（推理时）
    - Rolling window 大小：$w$（论文未给出具体值）
    - 压缩配置：$S = [1, 4, 2, 4, 4, 4, 2, 4, 4, 2, 4, 4, 4, 2, 4, 4, 2, 4]$，循环应用
    - VAE：使用 Tiny VAE（来自 MotionStream）替代原始 VAE 以加速解码

---

# Experiment

## 资源消耗

- **训练**：32 张 H100 GPU（80GB），具体训练总时间论文未明确给出
- **推理**：4 张 H100 GPU，实现 16 FPS 实时生成
- **模型参数量**：14B
- **推理优化**：torch.compile + FP8 E4M3 KV cache + FlashAttention v3 + 手动算子融合 + 序列并行/张量并行

## 数据集 / Benchmark

**测试集**：从 Adobe Stock 收集 220 张图像，涵盖真实场景（风景、城市、室内）和非真实场景（卡通、矢量艺术、油画）。随机分成 11 组，每组使用预定义的动作脚本评估所有 baseline，生成 220 个视频/baseline。输出时长固定为 20 秒。

**评估指标**：

- **Visual Quality**：VBench 的多个维度——Subject Consistency, Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality，以及这些指标的平均分
- **Action Accuracy**：用 ViPE 从生成视频重建相机轨迹，通过 Sim(3) Umeyama 对齐后计算 Relative Pose Error (RPE-trans, RPE-rot)

## 定量结果

与两个 SOTA baseline 对比（20 秒视频）：

|Model|Average Score↑|Image Quality↑|Aesthetic↑|RPE-Trans↓|RPE-Rot↓|
|---|---|---|---|---|---|
|Matrix-Game-2.0|0.7447|0.6551|0.4931|0.1122|1.48|
|Hunyuan-GameCraft|0.7885|0.6737|0.5874|0.1149|1.23|
|**RELIC**|**0.8015**|**0.6665**|**0.5967**|**0.0906**|**1.00**|

关键发现：

- RELIC 在整体视觉质量（Average Score）上最优
- 虽然训练分辨率只有 480p，但图像质量与 720p 训练的 GameCraft 接近
- 在 Aesthetics 和 Action Accuracy（特别是旋转 RPE）上显著领先

## 定性结果

**Action Accuracy**（Figure 8）：

- Tilt Up 命令下：Matrix-Game-2.0 在顶部产生黑色空洞；GameCraft 几乎不动
- Truck Left 命令下：GameCraft 错误地执行 Pan Left（旋转而非平移）；Matrix-Game-2.0 原地不动

**Memory**（Figure 9）：

- 在"先向上看，再向下，再左转，最后右转"的场景中，GameCraft 会遗忘长椅等物体，Matrix-Game-2.0 很快丢失输入图像的上下文，而 RELIC 能准确恢复之前生成的内容

**vs Marble (World Labs)**（Figure 10）：

- Marble 基于 Gaussian Splatting，会产生 floater 等重建伪影。RELIC 输出更干净稳定。

**Diversity**：RELIC 能泛化到油画、漫画、矢量艺术、低多边形渲染等多种风格域，展现出正确的距离感知（远处物体移动慢）和 3D 形状理解。

## Ablation Study

论文未提供详细的 ablation study 表格。但从方法描述中可以推断，以下设计经过了实验验证：

- Hybrid Forcing vs 单独的 Teacher Forcing / Diffusion Forcing（Hybrid 收敛更快）
- 压缩配置 $S$ 的选择是经验性确定的
- Time-reverse augmentation 的效果在记忆检索中被验证

---

# Limitations & Future Work

- **作者提到的局限**：
    
    1. 生成视频的多样性和场景动态性有限，主要因为训练数据以静态 UE 场景为主
    2. 难以生成超长（分钟级别）视频
    3. 大模型 + KV cache + 多步去噪的组合在资源受限环境下延迟较高
- **我观察到的局限/疑问**：
    
    1. **缺少 Ablation Study**：论文没有提供系统性的消融实验，如压缩比的影响、rolling window 大小的影响、relative action vs absolute pose 分别的贡献等
    2. **仅限相机运动**：当前只支持相机 6-DoF 控制，不支持场景中物体的交互（如开门、拿起物品等）
    3. **训练数据域偏**：350 个 UE 场景虽然多样，但与真实世界仍有 domain gap。泛化到真实场景的能力主要依赖 Wan-2.1 的预训练
    4. **Evaluation 局限**：只与 2 个 baseline 比较，且测试集较小（220 张图）。缺少与 Genie-3 等强 baseline 的定量对比
    5. **20 秒限制**：对于真正的"世界探索"场景，20 秒仍然较短，且作者承认扩展到分钟级别是困难的
    6. **4 张 H100 的推理需求**：这对于消费级应用来说成本过高

# Personal Notes

- **Camera-aware memory 的思路**非常值得借鉴：将 3D 空间检索隐式地编码在 attention 中（通过位姿编码到 Q/K），避免了显式 3D 表示的各种问题。这个思路可以推广到其他需要空间一致性的生成任务。
    
- **Replayed back-propagation** 是一个通用的技巧，可以用于任何需要长序列反向传播但显存不足的场景。核心思想是利用 loss 梯度的可分解性，将预计算和反传解耦。
    
- **数据策划的重要性**：论文花了大量篇幅介绍数据，包括动作平衡、去耦合、视点重访设计。这说明在 world model 领域，高质量的、有针对性的训练数据可能比模型架构创新更重要。
    
- **Time-reverse augmentation** 是一个简单但巧妙的增强方法，值得在其他需要"记忆回溯"能力的任务中尝试。
    
- **Relative action + Absolute pose 的分离注入策略**（relative → add to latent after self-attention; absolute → add to Q/K）体现了对两种信号不同计算角色的深刻理解，可以启发其他多条件生成任务的条件注入设计。