# Introduction

## task and applications
**任务**：视频生成中的参考基准相机控制（Reference-based Camera Control）。
**应用**：
1.  **Image-to-Video (I2V)**：让静态图像按照参考视频的运镜方式动起来（如推拉摇移）。
2.  **Video-to-Video (V2V)**：对已有视频进行重拍（Re-shooting），保持原内容但改变运镜，或者保持运镜改变内容。
3.  **电影制作与内容创作**：帮助创作者通过简单的参考视频（如泰坦尼克号中的经典镜头）来控制生成视频的相机运动，营造特定的氛围和叙事效果。

## Technical challenges for previous problems
1.  **显式相机参数难以获取与交互**：
    *   现有的方法大多依赖显式的相机参数序列（如外参矩阵），用户难以手写或构建这些复杂的参数。
    *   从动态视频中提取精准的相机参数（Camera Pose Estimation）非常困难，容易受到场景中物体运动的干扰，导致估计不准，进而影响生成控制。
2.  **现有无参数方法的局限性**：
    *   之前的无参数方法（如MotionClone）依赖于反转过程（Inversion）提取时间注意力权重作为运动表征，这增加了推理的计算开销。
    *   这种隐式的引导先验往往不够鲁棒，难以处理复杂的相机运动。
3.  **缺乏成对的高质量训练数据**：
    *   真实世界中很难获取“同一场景、不同运镜”或“不同场景、同一运镜”的完美成对数据用于训练模型学习解耦相机运动。

## 解决challenge 的pipeline是什么
提出了 **CamCloneMaster**，这是一个无需训练显式相机参数、也无需测试时微调（Test-time fine-tuning）的框架。它通过直接将参考视频的特征作为条件输入到扩散模型中，通过**Token Concatenation（Token拼接）**的方式，让模型直接学习并复制参考视频的相机运动模式。

### contribution 1: 提出了CamCloneMaster框架
**怎么做的？key insight是什么？**
*   **Key Insight**：相机运动信息内嵌在视频像素中，不需要中间步骤提取参数。通过将参考视频的Latent直接作为条件注入模型，可以让模型隐式地理解并复制运动。
*   **做法**：设计了一种基于Token Concatenation的注入机制。将“相机运动参考视频”和“内容参考视频”（可选）通过3D VAE编码后，与噪声Latent在**帧维度（Frame Dimension）**上进行拼接，作为一个统一的序列输入到DiT（Diffusion Transformer）中。这避免了额外的控制模块（如ControlNet）或复杂的参数估计。

### contribution 2: 构建了Camera Clone Dataset
**为了解决什么问题？具体怎么做的？**
*   **解决问题**：解决缺乏用于学习“相机克隆”的高质量成对视频数据的问题。
*   **做法**：利用 **Unreal Engine 5 (UE5)** 构建了一个大规模合成数据集。
    *   包含40个场景、66个角色、97.75K种不同的相机轨迹，共391K个视频。
    *   构建了“三元组”数据：(1) 相机运动参考视频 $V_{cam}$，(2) 内容参考视频 $V_{cont}$，(3) 目标视频 $V$（具有$V_{cont}$的内容和$V_{cam}$的运镜）。
    *   通过多机位同步拍摄和成对轨迹设计，确保数据的精确对应。

### contribution 3: 统一了I2V和V2V任务
**怎么做的？**
*   利用Token拼接的灵活性，在一个模型中同时支持I2V和V2V。
*   对于I2V，只输入相机参考视频和首帧图像；对于V2V，额外输入内容参考视频。
*   采用了混合训练策略（50% I2V任务，50% V2V重拍任务），仅微调DiT中的3D时空注意力层，保持了基础模型的生成能力。

# Method

## overview
**输入**：
1.  噪声视频 Latent ($z_t$)
2.  相机运动参考视频 ($V_{cam}$)
3.  内容参考视频 ($V_{cont}$, 可选，用于V2V)
4.  文本提示 ($c_{text}$)

**输出**：
*   生成的视频 ($x$)，其运镜跟随$V_{cam}$，内容符合文本或$V_{cont}$。

**Pipeline组成**：
1.  **3D VAE Encoder**：将所有视频输入编码为Latent Space的特征。
2.  **Patchify & Token Concatenation**：将条件Latent和噪声Latent分块化（Patchify），并在**帧维度**上拼接成一个长序列。
3.  **DiT Blocks (Diffusion Transformer)**：基于Transformer的主干网络，处理拼接后的序列。其中只有**3D Spatial-Temporal Attention**层参与微调，其他层冻结。
4.  **3D VAE Decoder**：将去噪后的Latent解码为最终视频。

## module 1: Reference Videos Injection via Token Concatenation (通过Token拼接注入参考视频)
**为什么能work？Motivation是什么？**
*   **Motivation**：相比于Channel拼接或使用额外的ControlNet，在**帧维度**拼接Token能让DiT的自注意力机制（Self-Attention）直接建模参考视频帧与生成视频帧之间的时空关系，从而更有效地转移运动模式。
*   **具体操作**：
    $$x_{input} = \text{Frame\_Concat}(x_t, x_{cam}, x_{cont})$$
    将噪声Token $x_t$ 与相机参考Token $x_{cam}$ (以及内容参考 $x_{cont}$) 串联。
*   **优势**：参数高效，不需要引入新的层，且天然支持多参考输入。

## module 2: Training Strategy (训练策略)
**具体怎么做的？**
*   **冻结参数**：为了保留预训练模型的生成质量和先验知识，冻结了除3D注意力层以外的所有参数。
*   **微调对象**：仅微调DiT Block中的 **3D Spatial-Temporal Attention Layers**。这是模型处理时序动态和运动的核心区域。
*   **混合训练**：为了让模型同时具备I2V和V2V能力，训练数据中50%执行相机控制的I2V生成，50%执行V2V的重拍任务（Re-generation）。

# Experiment

## 资源消耗
*   **硬件**：64张 NVIDIA H800 GPUs。
*   **Batch Size**：64。
*   **训练步数**：12,000 steps。
*   **分辨率**：训练时调整为 $384 \times 672$。

## 数据集/bench是什么
**评估数据集**：
1.  **RealEstate10K**：随机抽取1,000个视频作为**相机运动参考**（提供真实的复杂运镜）。
2.  **Koala-36M**：随机抽取1,000个视频作为**内容参考**。

**评估指标**：
1.  **Visual Quality (视觉质量)**：
    *   **Imaging Quality, CLIP Score**：图像质量与文本相关性。
    *   **FVD (Fréchet Video Distance)**, **FID**：视频分布距离，衡量生成真实感。
2.  **Camera Accuracy (相机控制准确性)**：
    *   使用SOTA姿态估计模型 **MegaSaM** 提取生成视频的轨迹，与Reference进行对比。
    *   **RotErr (Rotation Error)**：旋转误差。
    *   **TransErr (Translation Error)**：平移误差。
    *   **CamMC (Camera Motion Consistency)**：相机运动一致性。
3.  **Dynamic Quality (动态质量)**：
    *   使用 **VBench** 指标，包括 Dynamic Degree, Motion Smoothness, Subject/Background Consistency。

## 结果如何
1.  **定量结果**：
    *   在I2V和V2V任务中，CamCloneMaster在**相机控制准确性**（RotErr, TransErr）上显著优于现有方法（如CameraCtrl, MotionClone, CamI2V）。例如，RotErr仅为1.49，而竞争对手普遍在2.8以上。
    *   在**视觉质量**（FVD, FID）上也达到了SOTA水平，说明控制信号的注入没有破坏生成质量。
2.  **定性结果**：
    *   生成的视频能精准复刻复杂的运镜（如推拉+旋转），且保持了主体的一致性。
    *   相比MotionClone，不需要Inversion，推理速度更快且鲁棒性更强。
3.  **用户调研**：
    *   在47名参与者的盲测中，CamCloneMaster在相机准确性、视频文本一致性、时间一致性上均获得了约85%的偏好率，大幅领先基线模型。