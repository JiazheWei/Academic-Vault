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

- **Title**: Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory
- **Authors**: Ruiqi Wu*, Xuanhua He*, Meng Cheng*, Tianyu Yang, Yong Zhang‡, Zhuoliang Kang, Xunliang Cai, Wei Xiaoming, Chunle Guo†, Chongyi Li, Ming-Ming Cheng (*Equal Contribution, †Corresponding Author, ‡Project Leader)
- **Venue/Year**: arXiv 2026.02（Nankai University & Meituan & HKUST）
- **Paper Link**: https://arxiv.org/abs/2602.02393
- **Code Link**: https://github.com/MeiGen-AI/Infinite-World

## TL;DR

Infinite-World 提出了一个鲁棒的交互式世界模型，通过 **Hierarchical Pose-free Memory Compressor (HPMC)** 和 **Uncertainty-aware Action Labeling (UAL)** 两个核心设计，首次在**真实世界数据**上实现了 **1000+ 帧**的连贯视觉记忆和动作可控生成，在 ELO 评分上以 1719 分大幅领先第二名 HY-World-1.5（1542 分）。

---

# Introduction

## Task and Applications

这篇论文研究的是 **Interactive World Model（交互式世界模型）** 任务。具体来说，就是给定一个初始图像/场景描述和用户的动作指令（如"向左转"、"前进"），模型需要自回归地生成后续的视频帧，构建一个用户可以实时交互探索的虚拟世界。

实际应用场景非常广泛：自动驾驶模拟、游戏世界生成、虚拟现实/增强现实中的场景漫游、机器人导航训练环境等。核心诉求是：模型不仅要生成高质量的视觉内容，还要能**长时间保持场景的一致性**（比如走了一圈回到原地，场景应该和之前一样），并且能**准确响应用户的动作指令**。

## Technical Challenges

现有方法主要面临三大挑战：

**1. 长时记忆的计算瓶颈**：交互式世界模型需要自回归生成视频帧，随着生成帧数增加，历史信息不断累积。直接存储所有历史 latent 会导致**显存线性增长**，在 80GB H800 上很快就会 OOM。之前的方法如 SlowFastGen 通过推理时 LoRA 微调来记忆场景，但计算开销巨大；PFP 虽然预训练了 memory compressor，但其压缩比固定，显存占用仍然线性增长。

**2. 真实世界数据的噪声 pose**：现有表现较好的世界模型（如 Matrix-Game 2.0）大多在**合成数据**上训练，因为合成数据有完美的 ground-truth camera pose。而真实世界视频的 pose 估计不可避免地存在噪声——微小的抖动、漂移等。如果直接用这些 noisy pose 作为 action label 来训练，模型会学到错误的"动作-响应"映射，导致动作可控性很差。

**3. 真实世界视频缺乏"回访"（revisit）数据**：要让模型学会长程空间记忆（loop-closure，即回到之前去过的地方时能生成一致的画面），训练数据中需要包含大量"回到同一地点"的视频。但普通的互联网视频中这种回访极为稀缺。

## 与之前工作的区别/定位

与之前的主流方法相比，Infinite-World 的核心区别在于：

- **不依赖合成数据**：Matrix-Game 2.0 等方法依赖完美 GT pose 的合成数据训练，Infinite-World 直接在真实世界视频上训练
- **不需要显式的几何先验**：不需要 camera pose 作为输入条件，而是通过端到端联合优化让模型自己学会从压缩的历史信息中锚定空间位置
- **固定计算预算的长程记忆**：HPMC 将历史信息递归压缩到固定大小的表示中，计算成本恒定，突破了之前线性增长的瓶颈
- **对噪声的鲁棒性**：通过 Uncertainty-aware Action Labeling 显式处理噪声 pose，不是简单丢弃不确定的帧，而是用三态逻辑保留时序连续性

## 解决 Challenge 的 Pipeline

### Contribution 1: Hierarchical Pose-free Memory Compressor (HPMC)

**解决什么问题？** 长时记忆的计算瓶颈——随着生成帧数增加，如何以固定的计算/显存成本保持对遥远历史的记忆。

**Key insight**：采用两级递归压缩策略（Local Compression + Global Compression），将历史 latent 递归蒸馏为固定大小的表示。关键在于**compressor 和 DiT backbone 端到端联合训练**，让压缩器自主学习哪些历史信息对未来帧的生成最重要，从而在不需要显式 pose 的情况下实现 pose-free 的空间锚定。

### Contribution 2: Uncertainty-aware Action Labeling (UAL)

**解决什么问题？** 真实世界 pose 估计噪声导致的动作可控性差。

**Key insight**：将连续的运动 pose 解耦为平移和旋转原语，然后用三态逻辑（No-operation / Discrete Action / Uncertain）进行标注。对于不确定的运动帧，**不丢弃而是标注为 "Uncertain"**，从而在训练时既保留了时序连续性（对视频模型至关重要），又屏蔽了噪声对 action space 的污染。

### Contribution 3: Revisit-Dense Finetuning Strategy

**解决什么问题？** 真实世界数据中"回访"场景稀缺，模型难以学习 loop-closure 能力。

**Key insight**：通过一个 pilot toy study 发现，loop-closure 能力可以用**极少量的数据"激活"**。因此，仅用一个约 30 分钟的 revisit-dense 数据集进行微调，就能有效激活模型的千帧级空间记忆能力。这是一个非常实用且高效的策略。

---

# Method

## Overview

- **输入**：初始帧（文本描述生成或用户提供）+ 每一步的动作指令（6 个离散方向：前/后/左/右/上看/下看 + 无操作）
    
- **输出**：自回归生成的视频帧序列（可达 1000+ 帧）
    
- **Pipeline 整体流程**：
    
    1. 文本描述经 T5 编码器得到条件特征
    2. 每次生成一个 chunk 的帧（由 DiT backbone 进行 diffusion 生成）
    3. 生成完的 chunk 被 VAE encode 成 latent，送入 HPMC 进行压缩并更新记忆
    4. 压缩后的固定大小记忆和当前动作 embedding 一起送入 DiT，指导下一个 chunk 的生成
    5. 循环往复，实现无限长度的交互式生成
- **模块连接关系**：
    

```
Text Prompt → T5 Encoder → Text Embedding ─────────────────────────────────────────────┐
                                                                                        ↓
Initial Frame → VAE Encoder → Latent ──→ [Historical Latents Buffer] ──→ HPMC ──→ Compressed Memory ──→ DiT Backbone ──→ Denoised Latent → VAE Decoder → Generated Frame
                                                ↑                                       ↑
                                           (recursive update)                     Action Embedding
                                                                                        ↑
                                                                              Action Encoder (tri-state)
```

## Module 1: Hierarchical Pose-free Memory Compressor (HPMC)

- **这个模块做什么？** 将不断增长的历史帧 latent 序列压缩为**固定大小的记忆表示**，供 DiT backbone 在生成新帧时参考历史信息。
    
- **Motivation / 为什么需要这个模块？** 交互式世界模型需要自回归生成上千帧，如果直接存储所有历史 latent，显存会线性增长并迅速 OOM。之前的方法要么不压缩（PFP 等显存还是线性增长），要么压缩但不和生成模型联合优化（丢失关键信息）。
    
- **Technical Challenge**：如何在有限的固定预算下，保留最关键的历史信息？压缩比越高、历史越远，信息损失越大，但 loop-closure 恰恰需要远处的记忆。
    
- **具体怎么做的？** HPMC 采用**两级递归压缩**：
    
    **第一级：Local Compression（局部压缩）**
    
    - 对最近的若干帧 latent 进行压缩，捕获**细粒度的短程动态**
    - 将相邻 chunk 的 raw latent 压缩成更紧凑的表示
    - 当历史帧数量未超过处理阈值 $T_{max}$ 时，只使用 Local Compression 即可
    
    **第二级：Global Compression（全局压缩）**
    
    - 当历史序列长度超过 $T_{max}$ 时启动
    - 采用**自适应滑动窗口采样**机制，从已经 Local Compressed 的历史中进一步蒸馏出全局记忆
    - 将远距离历史信息压缩到固定预算的表示中
    
    **Context Injection to DiT**：压缩后的历史记忆通过时间维度的拼接（temporal concatenation）注入到 DiT 的输入中，直接指导 diffusion 生成过程。
    
    **Joint Optimization（联合优化）**：这是 HPMC 最关键的设计——compressor $f_\phi$ 与 DiT backbone **端到端联合训练**。通过让压缩器的梯度来自"未来帧的生成 loss"，模型自主学习在压缩过程中保留哪些信息对 loop-closure 最重要。
    
- **为什么能 work？Key insight**：不是人为设计什么信息该保留，而是让生成 loss 自动驱动压缩器学习"什么历史信息对生成未来帧最有用"。这样，模型可以自动发现和保留空间锚点信息，实现 pose-free 的长程一致性。
    

## Module 2: Uncertainty-aware Action Labeling (UAL)

- **这个模块做什么？** 将真实世界视频中估计出的连续 camera pose 变化，转换为鲁棒的**离散动作标签**，供模型学习"动作-视觉响应"的映射。
    
- **Motivation / 为什么需要这个模块？** 真实世界视频的 pose estimation 不可避免存在噪声。如果直接将 noisy continuous pose 作为 action conditioning，模型会学到混乱的动作-响应关系，导致交互时用户发出"向左转"的指令，模型可能不转或乱转。
    
- **Technical Challenge**：如何从 noisy pose 中提取可靠的动作信号？直接丢弃不确定的帧会破坏时序连续性（视频模型训练非常依赖连续帧的学习），而强行标注又会引入噪声。
    
- **具体怎么做的？**
    
    **Step 1: Motion Decoupling（运动解耦）**
    
    - 给定视频序列，首先估计相邻帧的相对 camera pose 变化
    - 将连续 pose 解耦为**平移（translation）**和**旋转（rotation）**两个独立的运动原语
    - 分别在各自的轴上进行离散化
    
    **Step 2: Tri-state Logic Labeling（三态逻辑标注）** 每个运动原语被标注为三种状态之一：
    
    |状态|含义|处理方式|
    |---|---|---|
    |**No-operation**|运动幅度极小，几乎静止|标注为"无动作"，模型学习在此条件下保持画面稳定|
    |**Discrete Action**|运动方向明确、幅度显著|标注为具体的离散方向（前/后/左/右/上看/下看）|
    |**Uncertain**|运动幅度处于模糊区间，无法可靠判断|标注为"不确定"，训练时**保留该帧用于视觉学习**，但**不对 action 施加监督**|
    
    **Step 3: Action Encoding**
    
    - 离散动作通过 learnable embedding 编码
    - 编码后的 action embedding 注入 DiT backbone 中，指导生成方向
- **为什么能 work？Key insight**：Uncertain 状态的引入是核心创新。传统做法要么丢弃不确定帧（损失训练数据和时序连续性），要么强行标注（引入噪声）。三态逻辑同时解决了这两个问题——Uncertain 帧参与视觉 loss 的训练但不参与 action loss 的训练，最大化利用数据同时保护 action space 的纯净性。
    

## Module 3: Revisit-Dense Finetuning Strategy

- **这个模块做什么？** 通过在一个精心构造的、包含大量"回访"片段的小数据集上微调，激活模型的长程 loop-closure 能力。
    
- **Motivation / 为什么需要这个模块？** 互联网视频中很少有"走了一圈回到同一地点"的片段，模型在大规模预训练中很难学会这种空间记忆。
    
- **具体怎么做的？**
    
    - 作者先进行了一个 **Pilot Toy Study**：在一个合成环境中系统性地研究什么因素对 loop-closure 能力最关键
    - 关键发现：(1) Loop-closure 能力可以用**极少量数据"激活"**；(2) 超出训练时见过的时间窗口反而会导致性能下降
    - 基于这些发现，构建了一个**仅约 30 分钟**的 Revisit-Dense Dataset (RDD)，其中包含密集的回访场景
    - 在大规模预训练后，用这个小数据集进行微调
- **为什么能 work？** 预训练阶段已经让模型具备了强大的视觉生成能力，loop-closure 更像是一种"技能"而非"知识"——少量示范就能激活。这类似于大语言模型中，instruction tuning 只需少量数据就能激活模型的指令遵循能力。
    

## 核心亮点深度解析

### 亮点 1：HPMC 的端到端联合优化

这是本文最有创新性的设计。之前的 memory compression 方法（如 PFP）采用**两阶段策略**——先用重建 loss 预训练 compressor，再将压缩后的表示给到生成模型。这种方式的问题是：compressor 优化的目标（重建保真度）和最终使用目标（帮助生成未来帧）之间存在 gap。重建保真度高不代表保留了对生成最有用的信息。

Infinite-World 的做法是让 compressor 的 loss 直接来自 DiT 的生成 loss。这意味着：

- 如果 compressor 丢掉了对 loop-closure 重要的信息 → 生成 loss 增大 → 反向传播迫使 compressor 保留这些信息
- 模型自动学会在固定预算下做最优的信息取舍

这和之前方法的关键区别在于：**从"重建导向"转变为"生成导向"的压缩**，信息的保留与丢弃完全由下游任务（生成）驱动。

### 亮点 2：Tri-state Logic 对噪声的优雅处理

传统的二值化动作标注（有动作 vs 无动作）在面对噪声时非常脆弱——一个微小的 pose 抖动可能被错误标注为"前进"，模型因此学到虚假的动作-响应关系。

Uncertain 状态的引入本质上是一种**"我不确定就不教"**的策略，但又比简单丢弃数据更聪明——Uncertain 帧的视觉信息仍然被利用（对视频的视觉质量学习有贡献），只是不对动作施加错误的监督信号。这是一个非常实用的工程设计，对于任何涉及 noisy label 的场景都有借鉴价值。

## Training

- **数据集：**
    
    - **预训练阶段**：使用**开放域视频数据集**（open-domain video）进行预训练，让模型学习通用的视频生成能力和初步的视觉理解
    - **微调阶段**：使用自建的 **Revisit-Dense Dataset (RDD)**，约 30 分钟时长的视频数据，包含密集的回访场景。该数据集是专门为激活 loop-closure 能力而构建的
- **Loss Function**：
    
    - 主要的 loss 是标准的 **Diffusion Loss**（DiT backbone 的 denoising loss）
    - HPMC 通过端到端联合训练，其优化信号来自 DiT 的 diffusion loss——不需要额外的 loss
    - Action conditioning 部分：只对标注为 "No-operation" 和 "Discrete Action" 的帧计算 action-related loss，"Uncertain" 帧不参与 action loss
- **训练策略**：
    
    - **两阶段训练**：
        1. **Stage 1: 大规模预训练**：在 open-domain 视频上训练，学习视觉生成和短程动态
        2. **Stage 2: Revisit-Dense Finetuning**：在 RDD 上微调，激活千帧级别的 loop-closure 能力
    - 基础生成模型基于 **Wan2.1-T2V-1.3B** 的 DiT backbone，使用其 VAE 和 T5 text encoder
    - HPMC compressor 与 DiT backbone 联合训练
- **关键超参数**：论文未详细列出所有超参数（如 learning rate 等）。推理时使用 H800 GPU。
    

---

# Experiment

## 资源消耗

- 推理在 **80GB H800 GPU** 上进行
- HPMC 的核心优势之一就是**近乎恒定的显存占用**：在 initial growth phase 之后，随着上下文视频长度增加，显存几乎不再增长。相比之下，不压缩的 baseline 显存会迅速耗尽
- 模型参数量基于 Wan2.1-T2V-1.3B，具体增加了 HPMC compressor 和 action encoder 的参数
- 论文未提供具体的训练 GPU 数量和训练时间

## 数据集 / Benchmark

- **VBench**：用于客观评估，包含 Motion Smoothness（运动流畅度）、Dynamic Degree（动态程度）、Aesthetic Quality（美学质量）、Image Quality（图像质量）四个指标
- **User Study**：人工主观评估，评估 Memory Consistency（记忆一致性）、Visual Fidelity（视觉保真度）、Action Responsiveness（动作响应性），并计算 ELO Rating

## 定量结果

### VBench 客观指标

|Model|Mot. Smo.↑|Dyn. Deg.↑|Aes. Qual.↑|Img. Qual.↑|Avg. Score↑|
|---|---|---|---|---|---|
|Hunyuan-GameCraft|0.9855|0.9896|0.5380|0.6010|0.7785|
|Matrix-Game 2.0|0.9788|**1.0000**|0.5267|**0.7215**|0.8068|
|Yume 1.5|0.9861|0.9896|**0.5840**|0.6969|0.8141|
|HY-World-1.5|**0.9905**|**1.0000**|0.5280|0.6611|0.7949|
|**Infinite-World**|0.9876|**1.0000**|0.5440|0.7159|0.8119|

在 VBench 上，Infinite-World 的 Avg. Score 排名第二（仅略低于 Yume 1.5），但在 Dynamic Degree 上达到满分，Image Quality 接近最好水平。

### User Study 主观评估

|Model|Memory↓|Fidelity↓|Action↓|ELO Rating↑|
|---|---|---|---|---|
|Hunyuan-GameCraft|2.67|2.49|2.56|1311|
|Matrix-Game 2.0|2.98|2.91|1.78|1432|
|Yume 1.5|2.43|1.91|2.47|1495|
|HY-World-1.5|2.59|2.78|**1.50**|1542|
|**Infinite-World**|**1.92**|**1.67**|1.54|**1719**|

（Memory/Fidelity/Action 是排名，越低越好）

**Infinite-World 以 1719 的 ELO Rating 大幅领先**，比第二名 HY-World-1.5 高出 177 分。在 Memory Consistency（1.92）和 Visual Fidelity（1.67）上均排名第一，Action Responsiveness（1.54）仅略逊于 HY-World-1.5（1.50）。

值得注意的是，作者指出 Yume 1.5 虽然在某些客观指标上表现不错，但实际交互时经常退化为简单的"向前走"轨迹，回避了复杂视角转换的挑战。

## 定性结果

论文展示了多种场景下的生成结果：

- 奇幻城市街景（建筑雕刻在巨树中，发光的树液）
- 室内禅意泳池
- 盐湖天镜日落
- 阁楼工作室
- 冰川湖
- 珊瑚礁浅滩

这些场景展示了模型在视觉质量、风格多样性和空间一致性方面的能力。

## Ablation Study

### HPMC 的有效性（显存消耗对比）

三种配置对比：无压缩 / 直接压缩（单级）/ 层级压缩（HPMC）

- **无压缩**：显存随上下文长度迅速线性增长，很快 OOM
- **直接压缩**：显存增长变慢但仍然线性
- **HPMC**：在初始增长阶段后，**显存几乎恒定**，显著优于其他方案

### UAL 的有效性

|配置|Action Rank|
|---|---|
|Baseline（无 UAL）|2.95|
|+ UAL|2.17|
|Full model（+ UAL + RDD FT）|最佳|

UAL 在不同训练阶段都能**一致地提升 Action Responsiveness**，证明三态逻辑确实有效地屏蔽了 pose 噪声。

### RDD Finetuning 的有效性

- RDD 微调能显著提升 Memory Consistency 排名
- 仅需约 30 分钟的 revisit-dense 数据即可激活千帧级 loop-closure 能力

---

# Limitations & Future Work

- **作者提到的局限**：
    
    - 论文未显式讨论详细的 limitations 章节，但可以推断：
    - 模型基于 1.3B 参数的 DiT，生成分辨率和视觉质量可能还有提升空间
    - 30 分钟的 RDD 数据集是特定构建的，泛化性有待验证
- **我观察到的局限/疑问**：
    
    1. **推理速度**：论文未明确报告每帧的生成延迟，对于"交互式"应用来说，实时性是关键因素。基于 diffusion model 的方法通常较慢，是否能达到实时交互的要求？
    2. **动作空间的局限性**：目前只支持 6 个离散方向 + 无操作，对于更精细的连续控制（如精确的旋转角度、变速移动）支持有限
    3. **3D 一致性**：虽然通过记忆实现了"看起来一致"，但本质上不是真正的 3D 表示，在极端视角变化或精确几何要求的场景下可能会暴露不一致
    4. **Revisit-Dense Data 的获取**：30 分钟看似不多，但如何系统性地大规模获取这类数据仍是一个实际挑战
    5. **与显式 3D 方法的对比缺失**：论文主要和其他 video generation-based 世界模型对比，缺乏与 NeRF/3DGS 等显式 3D 方法在场景一致性上的深入比较

# Personal Notes

- **HPMC 的"生成导向压缩"思想**非常有启发性。在任何需要压缩长序列历史信息的场景（如长视频理解、长对话记忆）中，"以下游任务 loss 驱动压缩"都可能优于"以重建 loss 驱动压缩"
- **Tri-state Logic** 对 noisy label 的处理范式很通用，可以迁移到其他存在标注噪声的任务中——"不确定的标签不用于该维度的监督，但样本仍用于其他维度的学习"
- **少量数据激活特定能力**的发现（Revisit-Dense Finetuning）与 LLM 中 instruction tuning 的范式非常相似，暗示视频生成模型可能也存在类似的 "capability awakening" 机制，值得深入探索
- 世界模型从"合成数据"走向"真实世界数据"是重要趋势，本文的训练范式为这个方向提供了实用的解决方案