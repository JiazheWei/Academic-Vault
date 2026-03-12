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

- **Title**: Video World Models with Long-term Spatial Memory
- **Authors**: Tong Wu*, Shuai Yang*, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, Gordon Wetzstein （* 共同一作）
- **Venue/Year**: NeurIPS 2025 Poster
- **Paper Link**: https://arxiv.org/abs/2506.05284
- **Code Link**: https://spmem.github.io/ （项目主页，代码未明确开源）

## TL;DR

这篇论文受人类记忆机制启发，提出了一个**三种记忆协同**的视频世界模型框架——**Spatial Memory（长期空间记忆）+ Working Memory（短期工作记忆）+ Episodic Memory（情景记忆）**——来解决视频世界模型在长时间自回归生成时"遗忘"之前场景的问题。核心是用一个基于几何的 3D 静态点云作为长期空间记忆，在重新访问同一场景时保持一致性。实验在 view recall consistency 和 VBench 上均显著优于 baseline。

---

# Introduction

## Task and Applications

这篇论文研究的是 **Video World Model（视频世界模型）** 任务：给定动作信号（如相机轨迹、文本 prompt），自回归地生成视频帧来模拟一个交互式的虚拟世界。

应用场景包括：

- 游戏/虚拟世界的实时内容生成
- 机器人训练的仿真环境
- 电影/动画的虚拟场景创作
- 交互式 3D 内容探索

## Technical Challenges

现有 Video World Model 面临的核心问题是**长时间一致性缺失（Long-term Forgetting）**：

1. **有限的上下文窗口**：Diffusion Transformer 中 attention 的计算复杂度随帧数二次增长，因此模型只能用很少的"上下文帧"来生成新帧（working memory 太小）。
2. **场景重访时遗忘**：当自回归生成了很多帧后，如果相机回头看之前的场景，模型已经"忘了"那里长什么样，导致前后不一致。
3. **缺乏持久的 3D 理解**：现有方法只靠图像帧来回忆过去，没有显式的 3D 空间理解，导致空间一致性差。

## 与之前工作的区别/定位

- **与扩展上下文窗口的方法（如 frame packing、SSM）不同**：这些方法仍然依赖"图像级别"的过去表示，没有显式的 3D 空间记忆。
- **与 point cloud conditioning 方法（如 ViewCrafter、DiffusionAsShader）不同**：这些方法主要用点云做相机控制，但不处理动态场景的时序推进，也没有"存储-检索"式的记忆机制。
- **本文的独特切入角度**：借鉴认知科学中人类记忆的三种类型（空间记忆、工作记忆、情景记忆），设计了一个结构化的多类型记忆系统，其中最核心的创新是**基于几何的长期静态空间记忆**。

## 解决 Challenge 的 Pipeline

### Contribution 1: 三种记忆机制的统一框架

**解决什么问题？** 视频世界模型缺乏长期记忆，生成长视频时场景不一致。

**Key insight**：人类依靠不同类型的记忆来维持对世界的一致认知——空间记忆记住环境布局，工作记忆处理当前信息，情景记忆回忆重要事件。视频世界模型也需要类似的分层记忆结构。

**具体做法**：

- **Spatial Memory（空间记忆）**：用 TSDF-Fusion 维护一个全局静态 3D 点云，记住世界的几何结构
- **Working Memory（工作记忆）**：用最近 k+1 帧作为上下文，提供短期动态信息
- **Episodic Memory（情景记忆）**：存储稀疏的历史关键帧，保留长期视觉细节

### Contribution 2: Geometry-grounded 数据集构建

**解决什么问题？** 没有现成的数据集包含"3D 空间记忆 + 视频生成"的配对训练数据。

**Key insight**：可以从真实视频中提取 3D 信息（相机位姿、深度图），通过 TSDF-Fusion 分离静态/动态元素，构建"静态点云引导 + 动态视频监督"的配对数据。

**具体做法**：从 MiraData 数据集中提取 97 帧视频片段，前 49 帧为 source，后 48 帧为 target，用 Mega-SaM 提取 4D 重建信息，TSDF-Fusion 获取静态点云，共构建 90K 训练样本。

### Contribution 3: 记忆引导的视频生成架构

**解决什么问题？** 如何有效地将三种记忆信号注入到视频 diffusion model 中。

**Key insight**：不同类型的记忆信息需要不同的注入方式——点云渲染适合用 ControlNet 式的条件注入，上下文帧适合直接拼接，历史帧适合用 cross attention 查询。

---

# Method

## Overview

- **输入**：相机轨迹（camera poses）、文本描述（action text）、历史帧序列
    
- **输出**：未来 N-k 帧视频
    
- **Pipeline 整体流程**：
    
    1. 从已有生成结果中维护三种记忆：静态点云（Spatial）、最近帧（Working）、关键帧（Episodic）
    2. 给定新的目标相机轨迹，从静态点云渲染出条件图像
    3. 将三种记忆信号分别注入 DiT 模型
    4. DiT 去噪生成新视频帧
    5. 从新帧中提取点图并更新全局静态点云
- **模块连接关系**：
    

```
Input Camera Trajectory
       │
       ├──→ Spatial Memory (Static Point Cloud) ──→ 渲染条件图 ──→ Condition DiT ──→ 逐层注入 Main DiT
       │
       ├──→ Working Memory (Recent 5 frames) ──→ VAE Encode ──→ 拼接到 Main DiT 输入
       │
       ├──→ Episodic Memory (Sparse Keyframes) ──→ VAE Encode ──→ Historical Cross Attention
       │
       └──→ Action Text ──→ T5 Encoder ──→ Text Conditioning
                                                    │
                                              Main DiT (Denoising)
                                                    │
                                              VAE Decode ──→ Output Video Frames
                                                    │
                                              CUT3R + TSDF ──→ 更新 Spatial Memory
```

## Module 1: Spatial Memory（空间记忆）—— 基于 TSDF-Fusion 的静态点云

- **这个模块做什么？** 维护一个全局的 3D 静态点云，表征已生成世界中不会动的部分（如建筑、道路、地形）。
    
- **Motivation / 为什么需要？** 视频世界模型需要在重新访问某个区域时保持场景一致。纯图像级的记忆无法有效编码 3D 空间关系，而 3D 点云天然具有空间位置信息，可以通过相机位姿精确查询。
    
- **Technical Challenge**：如何区分场景中的静态元素（建筑）和动态元素（人物、车辆）？动态物体的深度在不同帧间不一致，直接融合会产生噪声。
    
- **具体怎么做？** 采用 TSDF（Truncated Signed Distance Function）Fusion：
    
    $$D'(v) = \frac{W(v) \cdot D(v) + w_i \cdot d_i(v)}{W(v) + w_i}, \quad W'(v) = W(v) + w_i$$
    
    其中 $D(v)$ 和 $W(v)$ 分别是体素 $v$ 的 TSDF 值和权重，$d_i(v)$ 是第 $i$ 帧观测到的截断符号距离，$w_i$ 是帧级置信权重。
    
    这个融合过程**天然过滤动态元素**：动态物体在不同帧的深度观测不一致，它们对应的体素会积累低置信度、高噪声的 TSDF 值，在最终融合体积中被自然抑制。
    
- **为什么能 work？** TSDF-Fusion 是经典的 3D 重建方法，其多帧加权平均的机制使得只有在多帧中位置一致的结构（即静态物体）才能获得高置信度，动态物体因为帧间位置不一致而被"投票"掉。
    

**推理阶段的更新策略**：使用 CUT3R（一个在线递归 3D 感知模型）替代 Mega-SAM 进行实时 4D 重建。CUT3R 具有有状态的递归模型，每次新观测时增量更新内部状态，输出统一世界坐标系下的逐像素 3D 点和相机参数。每步推理后保存 CUT3R 的 state dict，确保坐标系对齐。

## Module 2: Working Memory（工作记忆）—— 最近帧上下文

- **这个模块做什么？** 将最近生成的 k+1 帧（论文中 k=4，即 5 帧）作为条件输入，提供短期动态连续性。
    
- **Motivation / 为什么需要？** 静态点云只包含不动的场景结构，无法表达动态物体（人物走路、车辆行驶）的运动连续性。最近帧包含了动态元素的最新状态和运动趋势。
    
- **Technical Challenge**：如何在有限帧数内既保持运动连续又不引入太多计算开销。
    
- **具体怎么做？** 采用简单的自回归策略：将最近 5 帧的 source video tokens 与 target video tokens 沿帧维度拼接，直接输入到 DiT 模型中。同时，target 的 condition tokens 也与 recent context tokens 拼接以确保帧级对应。
    
- **为什么能 work？** 这是视频生成模型的常规做法，近帧包含了最直接的运动信息和动态上下文，是保持时序连贯的基础。
    

## Module 3: Episodic Memory（情景记忆）—— 稀疏历史关键帧

- **这个模块做什么？** 存储一组有代表性的历史关键帧，作为长期视觉细节的参考。
    
- **Motivation / 为什么需要？** 融合后的静态点云往往**太稀疏**，无法保留过去看到的视觉细节（如特定人物的外貌、物体的纹理）。需要额外的"照片级"参考来补充。
    
- **Technical Challenge**：如何选择哪些帧值得存储？存太多会增加计算负担，存太少又可能丢失关键信息。
    
- **具体怎么做？**
    
    1. 在生成过程中，监测每帧新暴露的未知区域大小（通过 mask-based visibility check）
    2. 当新暴露的区域超过预设阈值时，将当前帧选为关键帧并加入记忆集合
    3. 这些关键帧通过 3DVAE 编码并 patchify 为 reference tokens
    4. 在 DiT 中添加 **Historical Cross Attention** 层：当前生成帧的 tokens 作为 query，历史关键帧的 tokens 作为 key 和 value
- **为什么能 work？** 通过"新区域面积"作为选择标准，可以自然地在场景变化较大时（如转弯、进入新区域）存储关键帧。Cross attention 允许模型按需查询历史视觉细节，比如回到之前区域时可以"回忆"该区域的视觉细节。
    

## Module 4: Memory-guided Video Generation（记忆引导的生成架构）

- **这个模块做什么？** 将三种记忆信号统一注入到视频 diffusion model 中进行条件生成。
    
- **具体架构设计**：
    
    1. **静态点云条件注入**：
        - 从当前空间记忆沿目标轨迹渲染条件图像（无点云覆盖区域设为黑色）
        - 用预训练的 3DVAE 编码为 condition latents
        - 采用类似 ControlNet 的设计：复制 CogVideoX 的前 18 个 DiT blocks 作为 Condition DiT
        - Condition DiT 每个 block 的输出经过**零初始化线性层**后加到主 DiT 对应的 feature map 上
    2. **工作记忆注入**：
        - 最近 5 帧经 VAE 编码后，在帧维度上与目标视频 tokens 拼接
        - 直接输入到主 DiT
    3. **情景记忆注入**：
        - 历史关键帧经 3DVAE 编码和 patchify 后成为 reference tokens
        - 在主 DiT 的每个 block 中增加 Historical Cross Attention 层
        - 当前生成帧 tokens 作为 query，reference tokens 作为 key/value
    4. **文本条件**：
        - Action text 通过 T5 编码器编码后注入

## 核心亮点深度解析

### 亮点 1：TSDF-Fusion 实现动静分离的空间记忆

**Intuition**：在视频世界模型中，"世界"可以分为两部分——不变的静态场景（建筑、地形）和变化的动态元素（人物、车辆）。静态场景是维持长期一致性的锚点，而动态元素需要模型自由生成。如果把所有信息都混在一起存储，模型很难分辨哪些是需要严格保持的、哪些是可以自由变化的。

**与之前方法的关键区别**：之前的 point cloud conditioning 方法（如 ViewCrafter、DiffusionAsShader）直接用完整的点云做相机控制，不区分动态和静态。这导致两个问题：1）动态物体的点云在不同时刻不一致会引入噪声；2）模型可能被静态的点云"锁死"，无法自由生成合理的动态内容。本文通过 TSDF-Fusion 只保留静态部分，让模型在保持场景一致的同时有自由度去生成动态内容。

**为什么比之前方案更好？** 这种设计实现了"静态靠记忆约束，动态靠模型生成"的优雅分工：

- 静态部分由 3D 点云精确约束（Spatial Memory）
- 动态部分由近帧提供运动连续性（Working Memory）+ 文本引导新动作
- 视觉细节由关键帧补充（Episodic Memory）

### 亮点 2：类认知科学的三层记忆架构

**Intuition**：人类记忆不是单一系统，而是多系统协作。空间记忆帮我们记住"空间在哪"，工作记忆帮我们处理"当前在做什么"，情景记忆帮我们"回忆过去的细节"。这三者缺一不可。

**和之前方法的区别**：之前的长上下文方法（如 frame packing、SSM）试图用一种统一的方式压缩所有历史信息，不加区分地处理不同类型的记忆需求。本文的分层设计让每种记忆用最合适的表示和注入方式。

## Training

- **数据集**：自建数据集，基于 MiraData（大规模长视频数据集）。从 raw video 中切出 97 帧的 clip，前 49 帧为 source，后 48 帧为 target。用 Mega-SaM 提取相机位姿和深度，TSDF-Fusion 分离静态点云。用 Qwen 为 target 帧生成 action 文本描述。共 **90K 样本**。
    
- **Loss Function**：标准的 diffusion denoising loss： $$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon} | \epsilon_\theta(x_t, t) - \epsilon |_2^2$$ 论文未提到额外的特殊 loss（如点云一致性 loss 等），仅用标准去噪目标。
    
- **训练策略**：
    
    - 基座模型：CogVideoX-5B-I2V，从 DiffusionAsShader（DaS）预训练权重初始化
    - Condition DiT：复制主 DiT 的前 18 个 block，新增的零初始化线性层和 Historical Cross Attention 是新参数
    - 训练分辨率：480 × 720，视频长度 49 帧
- **关键超参数**：
    
    - 迭代次数：6,000 iterations
    - Learning rate：2 × 10⁻⁵
    - Mini-batch size：8
    - GPU：8 × NVIDIA A100
    - 推理时使用最近 5 帧作为 working memory

---

# Experiment

## 资源消耗

- **训练**：8 × NVIDIA A100，6000 iterations（具体训练时长论文未明确给出，但 6000 iterations 对于这个规模的模型来说相对较短，说明是在预训练模型基础上 finetune）
- **模型参数量**：基于 CogVideoX-5B，约 5B 参数（加上 Condition DiT 和 Cross Attention 的额外参数）
- **推理**：每步自回归生成需要 CUT3R 进行 4D 重建 + TSDF-Fusion 更新点云 + DiT 去噪，推理速度论文未具体报告

## 数据集 / Benchmark

- **训练集**：自建，90K 样本，来源于 MiraData
- **测试集**：从 MiraData 随机选取 500 个未见序列
- **评估指标**：
    - **View Recall Consistency**：PSNR、SSIM、LPIPS（前向+反向轨迹在同一相机位姿的配对帧对比）
    - **视频质量**：VBench 6 个指标（Aesthetic Quality、Imaging Quality、Temporal Flickering、Motion Smoothness、Subject Consistency、Background Consistency）
    - **用户研究**：Camera Accuracy、Static Consistency、Dynamic Plausibility 三个维度

## 定量结果

### View Recall Consistency（核心结果）

|Method|PSNR ↑|SSIM ↑|LPIPS ↓|
|---|---|---|---|
|TrajectoryCrafter|11.71|0.4380|0.5996|
|DaS|12.01|0.4512|0.5874|
|Wan2.1-Inpainting|12.16|0.4506|0.5875|
|**Ours**|**19.10**|**0.6471**|**0.3069**|

PSNR 从 ~12 提升到 **19.10**，提升幅度巨大（约 7dB），说明空间记忆机制对场景重访一致性有显著效果。

### VBench 视频质量

本文方法在 Aesthetic Quality（0.5835）、Temporal Flickering（0.7580）、Motion Smoothness（0.9886）上均为最优。Wan2.1 在 Imaging Quality 上略高，但其 inpainting 模型容易生成相对静态的场景。

### 用户研究

|Method|Cam-Acc ↑|Stat-Cons ↑|Dyn-Plaus ↑|
|---|---|---|---|
|TrajectoryCrafter|1.632|1.780|1.626|
|DaS|2.566|2.440|2.703|
|Wan2.1-Inpainting|2.176|2.396|2.270|
|**Ours**|**3.626**|**3.385**|**3.401**|

20 位有经验的评估者参与，1-4 分打分，本文方法在所有三个维度上大幅领先。

## 定性结果

论文 Figure 4 展示了三个方面的对比：

1. **Camera Accuracy**：显著相机运动变化时，baseline 无法准确跟随，本文方法可以
2. **View Recall**：重访相同相机位姿时，baseline 忘记了之前的细节或无法补全稀疏点云，本文方法保持一致
3. **Action Accuracy**：给定动作指令生成动态内容时，baseline 容易出现角色消失、动作漂移，本文方法表现良好

## Ablation Study

|变体|Aesthetic ↑|Imaging ↑|Flickering ↑|Smoothness ↑|Subject ↑|Background ↑|
|---|---|---|---|---|---|---|
|w/o Episodic Memory|0.5603|0.6485|0.7260|0.9870|0.9326|0.9489|
|w/o Working Memory|0.5551|0.6384|0.6740|0.9862|0.9331|0.9453|
|**Full Model**|**0.5835**|**0.6701**|**0.7580**|**0.9886**|**0.9359**|**0.9506**|

关键发现：

- **去掉 Working Memory 影响最大**：Motion Smoothness 和 Temporal Flickering 下降最多，说明近帧对短期动态连贯至关重要
- **去掉 Episodic Memory**：Subject Consistency 下降，说明关键帧对保留角色/物体细节很重要
- **三者协同效果最佳**：验证了三种记忆机制的互补性

---

# Limitations & Future Work

- **作者提到的局限**：
    
    1. **TSDF-Fusion 的不完美**：当相机位姿变化剧烈（如大角度转弯、快速移动）时，4D 重建可能失败，导致 TSDF-Fusion 错误地过滤掉大量静态区域的点云，造成空间记忆极度稀疏。论文 Figure 6 用蜘蛛侠在高楼间快速摆动的例子说明了这个问题。
    2. **只解决了"遗忘"问题，未解决"漂移"问题**：长时间自回归生成的误差累积导致的图像质量退化（drift）是另一个未解决的挑战。
    3. **主要关注空间一致性**：Frame packing 等方法主要关注角色一致性，未来可以结合两者。
- **我观察到的局限/疑问**：
    
    1. **推理开销较大**：每步需要 CUT3R 重建 + TSDF 更新 + 点云渲染 + DiT 推理，推理速度可能是瓶颈，论文未报告具体推理时间。
    2. **训练数据规模和多样性**：90K 样本基于 MiraData，场景多样性可能有限，尤其是室内场景、复杂动态交互等场景覆盖不足。
    3. **PSNR 绝对值仍然不高**：虽然相比 baseline 提升巨大（19.10 vs 12），但 19.10 的 PSNR 说明精确记忆每个视觉细节仍然很难，重访时的像素级一致性还有很大提升空间。
    4. **Episodic Memory 的选择策略较简单**：基于"新区域面积阈值"的策略可能不够 robust，没有考虑帧的信息量、多样性、重要性等因素。
    5. **没有验证真正的交互式生成**：测试都是预定义轨迹，没有展示用户实时交互控制下的效果。

# Personal Notes

- **核心启发**：将认知科学中的记忆分类思想引入到视频生成模型中，是一个非常优雅的框架设计。这种"不同类型的信息用不同的表示和机制来处理"的思路，在很多生成任务中都有借鉴价值。
- **可借鉴的 idea**：
    1. TSDF-Fusion 做动静分离是一个工程上非常实用的 trick，可以用在其他需要区分场景中静态/动态元素的任务中。
    2. "零初始化线性层"注入条件信号的方式（类似 ControlNet）已经成为条件注入的标准范式。
    3. Historical Cross Attention 的设计可以推广到其他需要长期记忆的序列生成任务。
- **值得深入探索的方向**：
    1. 如何在推理时更高效地维护和查询空间记忆？比如用 hash grid 或 neural implicit 替代显式点云。
    2. 如何让记忆机制也能处理动态物体的长期一致性？比如为每个动态实体维护独立的记忆。
    3. 能否让模型自己学会何时存储、何时检索，而不是用手工规则？