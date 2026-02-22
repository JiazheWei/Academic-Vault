![[image-1.png|875]]

# Introduction

## task and applications
**Task:**
本文的任务是**单视频的相机控制生成式重渲染（Camera-Controlled Generative Rendering from A Single Video）**。简单来说，就是给定一段视频，在保持原视频动态场景和内容不变的情况下，按照新的相机轨迹“重拍”这段视频。

**Applications:**
1.  **后期运镜修改：** 允许创作者在后期制作中改变原始素材的相机轨迹（如推拉摇移），增强视觉表现力。
2.  **视频稳像（Video Stabilization）：** 通过生成平滑的相机轨迹来修复抖动的视频。
3.  **视频超分辨率（Video Super-Resolution）：** 通过“推近（zoom-in）”的相机轨迹实现局部高清化。
4.  **视频外绘（Video Outpainting）：** 通过“拉远（zoom-out）”的相机轨迹生成原视频视场外的内容。

## Technical challenges for previous problems
1.  **缺乏配对训练数据：** 现实世界中很难获得同一动态场景下、精确同步且具有不同相机轨迹的多视角视频数据。
2.  **泛化能力差：** 现有方法（如GCD）通常在特定领域的合成数据上训练，在真实世界（in-the-wild）视频上的泛化效果不佳。
3.  **需要逐视频优化（Per-video optimization）：** 一些方法（如ReCapture）需要对每个视频进行长时间的优化，无法实现即时推断。
4.  **条件注入机制效果不佳：** 现有的视频生成模型在注入参考视频（Source Video）作为条件时，往往采用简单的通道拼接（Channel Concatenation），导致生成结果在外观保持和动态同步上效果不理想。
5.  **4D重建的局限性：** 基于单目视频的4D重建技术目前仍面临挑战，限制了基于重建-渲染管线的效果。

## 解决challenge 的pipeline是什么
作者提出了 **ReCamMaster**，这是一个基于预训练文本到视频（T2V）模型的生成式重渲染框架。
**核心思路：** 利用生成式模型强大的先验知识，通过一种新颖的**Frame Dimension Conditioning（帧维度条件化）**机制注入源视频信息，并利用大规模合成数据进行训练。

### contribution 1: 构建了高质量多相机同步视频数据集 (Multi-Cam Video Dataset)
**怎么做的？**
*   使用 **Unreal Engine 5 (UE5)** 渲染引擎。
*   构建了包含40个高质量3D环境、136K个写实视频、122K条不同相机轨迹的数据集。
*   **Key Insight:** 由于现实中无法获取大规模同步多视角数据，作者发现精心策划的合成数据（模拟真实世界的拍摄特征、多样的场景和运镜）能够帮助模型很好地泛化到真实世界的视频中。

### contribution 2: 提出了 Frame Dimension Conditioning (帧维度条件化) 机制
**为了解决什么问题？**
*   解决现有方法（如通道拼接）在保持源视频外观和时间同步性方面的不足。
**具体怎么做的？**
*   摒弃了在通道维度拼接或增加额外Attention层的做法。
*   直接将源视频（Source Video）的Token和目标视频（Target Video）的Token在**帧维度（Frame Dimension）**进行拼接，作为Transformer的输入。这使得模型利用自身的3D Self-Attention机制，在所有层级上都能让源视频和目标视频进行充分的时空交互。

### contribution 3: 精心设计的训练策略
**具体怎么做的？**
*   只微调相机的Encoder和3D-Attention层，冻结其他参数以保留T2V模型的生成能力。
*   在训练时对条件视频Latent添加噪声，以缩小合成数据与真实数据的域差距。
*   混合训练策略：以一定概率随机mask掉源视频的帧，统一相机控制的T2V、I2V和V2V任务，增强生成原视频视野外内容的能力。

---

# Method

## overview

**输入是什么？**
1.  **源视频 ($V_s$)：** 原始的单目视频。
2.  **目标相机轨迹 ($cam_t$)：** 用户指定的新的相机姿态序列（旋转和平移矩阵）。
3.  **文本提示词 ($p_t$)：** 描述视频内容的文本。

**输出是什么？**
*   **目标视频 ($V_t$)：** 内容、动态与源视频一致，但视角符合目标相机轨迹的新视频。

**大概pipeline的组成是什么？**
*   基于 **Latent Diffusion Model (LDM)**，具体是一个基于Transformer的DiT架构。
*   使用3D-VAE将视频编码为Latent。
*   通过 **Frame Dimension Conditioning** 注入源视频特征。
*   通过 **Camera Encoder** 注入目标相机参数。
*   模型预测噪声并去噪生成目标视频Latent，最后解码为像素视频。

## module 1: Conditional Video Injection Mechanism (Frame Dimension Conditioning)
**Motivation / Challenge:**
以前的方法（Channel Concatenation）虽然简单，但实验表明在保持视频动态一致性和减少伪影方面表现不佳。另一种View-Attention方法则需要额外的模块。

**Mechanics / 为什么能work？**
*   **做法：** 将源视频的Latent $z_s$ 和目标视频的噪声Latent $z_t$ 分别patchify得到Token。然后直接在帧的维度上将它们拼接：$x_{input} = [x_s, x_t]_{\text{frame-dim}}$。
*   **原理：** 输入Token数量翻倍。由于DiT模型内部使用的是3D (Spatial-Temporal) Attention，这种拼接方式使得源视频和目标视频的每一帧在每一层Transformer Block中都能直接进行Self-Attention交互。
*   **优势：** 相比于通道拼接，这种方式提供了更灵活、更深层的时空特征交互，极大地提升了同步性和内容一致性。

## module 2: Camera Pose Conditioning
**Technical Challenge:**
推断时，很难从野外（in-the-wild）视频中获得极其精确的**源视频**相机参数。

**Mechanics / 做法：**
*   **策略：** **只以目标相机 ($cam_t$) 为条件**，不输入源相机参数。让模型自己去理解源视频的相机运动。
*   **具体实现：** 将目标相机参数（外参矩阵）通过一个可学习的 MLP (Camera Encoder $\mathcal{E}_c$) 映射为特征，并在每个Transformer Block中加到视觉特征上：$F_i = F_o + \mathcal{E}_c(cam_t)$。
*   **参数选择：** 使用相机外参（旋转和平移），不使用内参（为了适应不知道内参的真实视频）。

## module 3: Training Strategy (Enhancing Robustness)
**为了解决什么问题？**
合成数据与真实数据的Domain Gap，以及模型对视野外内容（Out-painting）的生成能力。

**具体怎么做的？**
1.  **加噪训练：** 在训练时对作为条件的源视频Latent加入适量的噪声（200-500步），模拟真实数据的分布，减少对合成数据纹理的过拟合。
2.  **统一生成任务：** 以20%的概率将源视频全帧替换为高斯噪声（变身为T2V任务），或替换部分帧（变身为I2V任务）。这迫使模型学习生成源视频中未出现的连贯物体，从而支持大幅度运镜下的内容补全。

---

# Experiment

## 资源消耗
*   **基础模型：** 内部预训练的DiT文本到视频模型。
*   **训练设置：** 训练10,000步，分辨率 $384 \times 672$。
*   **Batch Size：** 40。
*   **学习率：** $1 \times 10^{-4}$。

## 数据集/bench是什么
**训练数据集：**
*   自建的 **UE5 Dataset**：136K个视频，覆盖40个3D环境，122K种不同的相机轨迹。

**测试数据集 (Evaluation Set)：**
*   从 **WebVid** 中随机选取的1000个真实视频。
*   配合10种不同的标准相机轨迹（平移、缩放、摇摄等）。

**评价指标：**
1.  **视觉质量 (Visual Quality):**
    *   **FID / FVD:** 衡量生成图像/视频的分布与真实的距离（越低越好）。
    *   **CLIP-T:** 文本-视频一致性。
    *   **VBench:** 综合视频生成质量评估。
2.  **相机控制精度 (Camera Accuracy):**
    *   **RotErr / TransErr:** 使用GLOMAP提取生成视频的轨迹，计算与目标轨迹的旋转和平移误差。
3.  **视图同步性 (View Synchronization):**
    *   **Mat. Pix. (K):** 使用GIM (Image Matching) 计算源视频和生成视频的匹配像素数。
    *   **CLIP-V:** 源视频和目标视频对应帧的CLIP相似度。
    *   **FVD-V:** 在SV4D中提出的用于衡量多视角一致性的指标。

## 结果如何
1.  **定量比较 (Quantitative Results):**
    *   **ReCamMaster 在几乎所有指标上都显著优于 SOTA 方法**（如 GCD, Trajectory-Attention, DaS）。
    *   特别是在 **CLIP-V** (90.36 vs DaS的87.33) 和 **FVD** (122.74 vs DaS的159.60) 上提升明显，说明生成视频既清晰又与源视频高度同步。
    *   相机控制误差极低 (RotErr 1.22)。

2.  **消融实验 (Ablation Studies):**
    *   证明了 **Frame Dimension Conditioning** 远优于 Channel Dimension Conditioning（基线方法）和 View Dimension Conditioning。
    *   证明了高质量合成数据集对于模型泛化至关重要（相比于玩具数据集Toy Data，各项指标大幅提升）。
    *   证明了加噪训练和混合任务训练策略有效提升了图像质量和美学评分。

3.  **应用展示:**
    *   展示了在视频稳像、超分和外绘方面的成功应用，且支持非重叠视角的生成。


# 框架源码阅读


![[Pasted image 20260129165218.png]]


