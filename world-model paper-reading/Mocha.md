[Arxiv](https://arxiv.org/pdf/2601.08587)
# Introduction

## task and applications
*   **任务 (Task)**: 视频角色替换 (Video Character Replacement, VCR)。即在视频中无缝替换角色，同时精确保留原始背景、场景动态和角色动作。
*   **应用 (Applications)**: 影视后期制作、个性化广告、虚拟试穿 (Virtual Try-on)、数字替身/虚拟形象 (Digital Avatars) 的创建。

## Technical challenges for previous problems
*   **依赖显式结构引导**: 现有的主流方法（如VACE, Wan-Animate）多采用基于重建（Reconstruction-based）的范式。它们需要密集的**逐帧分割掩码 (Per-frame segmentation masks)** 以及显式的结构引导（如**骨骼/Skeleton**、**深度图/Depth**）。
*   **复杂场景泛化能力差**: 在涉及遮挡、角色与物体交互、不寻常的姿态（如杂技）或复杂光照的场景中，提取准确的结构信息非常困难。
*   **结果缺陷**: 依赖显式引导导致在复杂场景下容易出现视觉伪影、动作不连贯和时间上的不一致性。此外，逐帧mask的获取成本高，计算开销大。

## 解决challenge 的pipeline是什么
MoCha 提出了一个**非重建基础 (Non-reconstruction-based)** 的端到端框架。它利用视频扩散模型固有的时间感知和跟踪能力，通过**上下文学习 (In-Context Learning)** 的方式，将源视频的动作和表情解耦并迁移到参考角色上。它**仅需要单帧掩码 (Single-frame mask)**，无需骨骼或深度图等结构引导。

### contribution 1: 方法创新
提出了MoCha框架，这是首个仅需任意单帧掩码且无需结构引导的端到端视频角色替换框架。引入了**条件感知 RoPE (Condition-aware RoPE)** 机制来适应多模态输入，并采用基于强化学习 (RL) 的后训练阶段来增强面部身份的一致性。

### contribution 2: 数据集构建
提出了一个综合的数据构建流程，解决了高质量成对训练数据稀缺的问题。包含三个来源：
1.  利用虚幻引擎5 (UE5) 构建的高保真渲染数据集；
2.  基于Flux和LivePortrait合成的表情驱动数据集；
3.  从现有视频-掩码对中筛选并增强的增强型视频掩码数据集。

---

# Method

## overview
*   **输入**:
    *   源视频 (Source Video, $V_s$)
    *   单帧指定掩码 (Designation Frame Mask, $M_s$, 仅需一帧)
    *   参考角色图像集 (Reference Images, $\{I_i\}$)
*   **输出**:
    *   目标视频 (Target Video, $V_t$)
*   **Pipeline组成**:
    *   基于 **DiT (Diffusion Transformer)** 架构，使用 **Wan-2.1-T2V-14B** 作为基础模型。
    *   **VAE Encoder**: 将视频、掩码、参考图压缩为Latents。
    *   **In-Context Learning**: 将所有条件Token（源视频、目标视频噪声、掩码、参考图）在帧维度拼接。
    *   **Condition-aware RoPE**: 特殊设计的位置编码以处理多模态输入。
    *   **Decoder**: 生成最终视频。
    *   **Post-Training**: 基于RL的微调阶段。

## module 1：In-Context Learning with Condition-aware RoPE
*   **核心逻辑**: 将视频生成视为序列预测问题。将源视频latents ($x_s$)、目标视频latents ($x_t$)、掩码latents ($x_m$) 和参考图像latents ($x_I$) 拼接成一个长序列 $x = [x_t, x_s, x_m, x_{I_1}, ...]$ 输入到DiT中。
*   **Technical Challenge**: 朴素的3D RoPE（旋转位置编码）会给不同条件分配不同的时间索引，导致生成长度受限且难以对齐（例如参考图没有时间概念）。
*   **Motivation & Solution (Condition-aware RoPE)**:
    *   **帧对齐**: 强制源视频 ($x_s$) 和目标视频 ($x_t$) 共享相同的帧索引 (0 到 $f-1$)，因为它们在时间上是一一对应的。
    *   **参考图处理**: 给参考图像 ($I_i$) 分配固定的帧索引 **-1**，并通过高度/宽度维度的偏移来区分不同的参考图。
    *   **灵活掩码**: 掩码 ($x_m$) 的帧索引 $f_M$ 根据用户指定的帧号动态计算，使其支持任意帧作为掩码输入。
*   **为什么能work**: 这种编码方式让模型能够理解源视频和目标视频之间的空间对应关系，同时通过Attention机制从参考图中提取身份信息，利用源视频的Motion作为Prompt引导生成。

## module 2：Identity-Enhancing Post-Training
*   **核心逻辑**: 在基础训练之后，引入强化学习 (RL) 策略进一步微调模型。
*   **Technical Challenge**: 仅靠MSE损失训练，模型可能无法完美保持参考角色的面部ID，或者出现直接“复制粘贴”参考图导致融合不自然的现象。
*   **Motivation & Solution**:
    *   **Reward Function**: 计算生成视频帧与参考图像的 **Arcface** 余弦相似度作为奖励 ($R_{face}$)。
    *   **Loss Function**: 结合像素级MSE损失（保证视频结构）和身份奖励损失：$\mathcal{L}_{RL} = (1 - R_{face}) + ||V_t - \hat{V}_t||_2$。
    *   **LoRA**: 仅微调LoRA参数以节省显存并防止破坏原有生成能力。
*   **为什么能work**: 直接优化身份相似度指标，迫使模型在生成的细节纹理上更接近参考人脸，同时利用MSE约束防止背景和动作崩坏。

---

# Experiment

## 资源消耗
*   **硬件**: 8张 NVIDIA H20 GPUs。
*   **训练步数**: 上下文学习阶段训练 30K 步；后训练阶段训练 500 步。
*   **基础模型**: Wan-2.1-T2V-14B。
*   **微调方式**: 使用 LoRA (Rank 64)。

## 数据集/bench是什么
*   **训练数据集 (100K samples)**:
    1.  **UE5 Rendered Data**: 60K。利用虚幻引擎随机组合场景、角色、动作渲染的完美配对视频。
    2.  **Expression-Driven Data**: 20K。用Flux重绘电影画面人物，再用LivePortrait驱动生成的面部动画数据。
    3.  **Augmented Video-Mask Data**: 20K。来自VIVID-10M和VPData，经过YOLOv12筛选和增强的真实视频数据。
*   **Benchmark (测试集)**:
    *   **Synthetic Benchmark**: 引擎渲染的完美配对数据（未参与训练）。
    *   **Real-world Benchmark**: 收集的包含复杂场景（多人、快速运动、复杂光照）的真实视频。
*   **评价指标**:
    *   **SSIM**: 结构相似性（Structural Similarity）。
    *   **LPIPS**: 感知相似性（Perceptual Similarity）。
    *   **PSNR**: 峰值信噪比，衡量重建质量。
    *   **VBench**: 视频生成综合评测，包括 Subject Consistency（主体一致性）、Background Consistency（背景一致性）、Aesthetic Quality（美学质量）、Temporal Flickering（时域闪烁）、Motion Smoothness（运动平滑度）。

## 结果如何
*   **定量对比**:
    *   在 **SSIM, LPIPS, PSNR** 上均优于 SOTA 方法（VACE, HunyuanCustom, Wan-Animate）。例如 SSIM 达到 0.746 (vs 次优 0.692)。
    *   在 **VBench** 指标上，特别是 **Subject Consistency (92.25)** 和 **Background Consistency (94.40)** 上表现最佳，说明身份保留和背景维持能力最强。
*   **定性分析**:
    *   能处理**复杂光照**、**物体交互**和**多角色遮挡**，而对比方法（如VACE）常因重建范式丢失光照细节。
    *   具有强大的跟踪能力 (Tracking ability)，Attention Map 显示模型能自动在整个视频中跟踪单帧Mask标记的对象。
*   **消融实验**:
    *   **Real-Human Data**: 显著提升了面部真实感和表情保真度，减少了合成数据的“塑料感”。
    *   **RL Post-Training**: 显著提升了面部身份一致性 (Facial Identity Preservation)。