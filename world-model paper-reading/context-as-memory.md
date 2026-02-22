[Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval](https://arxiv.org/abs/2506.03141v2)

![[image-4.png]]

# Introduction
## task and applications
**任务**：交互式长视频生成（Interactive Long Video Generation）。即在生成视频的过程中，允许用户通过交互信号（主要是相机轨迹/Camera Pose）来控制视频的生成，且生成的视频需要保持长期的场景一致性。
**应用**：游戏引擎（如生成式游戏）、模拟器（Simulation）、虚拟世界探索等。

## Technical challenges for previous problems
1.  **缺乏长期记忆能力（Scene Consistency）**：现有的长视频生成方法（如无限延展生成的Oasis或Diffusion Forcing）通常只能利用有限的最近几帧作为上下文。当相机视角移动并返回之前的位置时，模型往往会生成完全不同的场景，无法保持场景内容和空间关系的一致性。
2.  **资源与计算效率矛盾**：如果将所有历史生成的帧都作为上下文输入，计算开销巨大且不切实际。
3.  **信息干扰**：过多的无关历史帧会引入噪声，反而干扰当前帧的生成。
4.  **3D重建的局限性**：部分现有方法尝试通过3D重建（如Gaussian Splatting）来辅助记忆，但在大规模场景中，3D重建的累积误差会使得后续生成质量下降，且速度受限。

## 解决challenge 的pipeline是什么
文章提出了 **Context-as-Memory** 框架。其核心思想是直接将历史生成的帧存储为“记忆”，并在生成新帧时，通过 **Memory Retrieval（记忆检索）** 模块筛选出与当前视角最相关的历史帧，将它们作为条件输入到视频扩散模型中，从而实现场景一致的长视频生成。

### contribution 1：Context-as-Memory 策略
**怎么做的？key insight是什么？**
*   **Key Insight**：不需要复杂的后处理（如3D重建或特征提取），直接存储生成的 RGB 帧（或其 Latent）就是最直观、无损的记忆方式。
*   **做法**：将筛选出的历史上下文帧（Context Frames）与待预测的帧（Predicted Frames）在帧维度（Frame Dimension）上进行拼接（Concatenation），直接输入到 Transformer 模型中进行联合注意力计算。这种方式简单有效，无需额外的 Adapter 模块。

### contribution 2：Memory Retrieval（记忆检索）模块
**为了解决什么问题？具体怎么做的？**
*   **解决问题**：解决“全量历史帧计算开销过大”以及“无关帧引入噪声”的问题。
*   **做法**：利用相机轨迹信息，设计了一种基于 **FOV（视场角）重叠** 的规则搜索算法。只有当历史帧的相机视野与当前帧的相机视野有显著重叠时，才会被检索出来作为条件。此外，还引入了去冗余策略（剔除相邻的重复帧）。

### contribution 3：长视频数据集构建
**为了解决什么问题？具体怎么做的？**
*   **解决问题**：现有数据集通常只包含短视频片段，缺乏带有精准相机标注的长视频数据，无法有效训练和评估长视频记忆能力。
*   **做法**：基于 Unreal Engine 5 (UE5) 构建了一个包含100个视频、每个视频7601帧的长视频数据集。数据涵盖12种不同风格的场景，并配有精确的相机姿态标注和多模态大模型生成的Caption。

# Method

## overview
**输入是什么？**
1.  **当前待生成的视频片段**（在训练时是加噪的Latent，推理时是噪声）。
2.  **用户控制信号**：目标相机的轨迹/姿态（Camera Poses）。
3.  **记忆上下文**：通过 Memory Retrieval 检索到的 $k$ 个历史 RGB 帧及其对应的相机姿态。
4.  **文本提示**（Text Prompt）。

**输出是什么？**
符合相机轨迹控制且与历史场景保持一致的新视频帧序列。

**Pipeline组成**：
基于 DiT (Diffusion Transformer) 架构。
1.  **检索**：根据当前相机位姿，从历史库中检索相关帧。
2.  **编码**：通过 3D VAE Encoder 将检索到的上下文帧和当前噪声帧都编码为 Latent。
3.  **拼接与注入**：将 Context Latent 和 Noisy Latent 在时间（帧）维度拼接。
4.  **相机条件注入**：将相机参数编码后注入模型。
5.  **去噪**：DiT 模型进行去噪预测，但在 Attention 计算中，Context Latent 保持固定（不更新），仅作为 Key/Value 提供信息，只更新 Noisy Latent。
6.  **解码**：3D VAE Decoder 解码生成视频。

## module 1: Context Frames Learning Mechanism (上下文学习机制)
**为什么能work？Motivation是什么？**
*   **Motivation**：需要一种灵活的方式处理变长的上下文，且不能破坏预训练模型的生成能力。传统的 Adapter 或 Cross-Attention 可能不足以捕捉强时空相关性。
*   **做法**：
    *   **拼接（Concatenation）**：类似于 ReCamMaster，在输入层直接将 Context 和 Target 在帧维度拼接。
    *   **位置编码（Positional Encoding）**：为了适应变长序列，保持待生成帧的位置编码与预训练时一致，给新加入的 Context 帧分配新的位置编码。使用 RoPE (Rotary Positional Embedding) 处理。
    *   **Masking**：在训练和推理时，只对 Target Frames 进行去噪（预测噪声），Context Frames 视为 clean condition 参与 Attention 计算但不计算 Loss。

## module 2: Memory Retrieval (记忆检索)
**Technical challenge是什么？**
如何在成千上万的历史帧中，快速且准确地找到对当前生成有帮助的“那一瞥”，同时排除冗余信息。

**具体做法**：
这是一个基于规则的搜索算法，分为三步：
1.  **FOV Overlap Check（视场角重叠检测）**：
    *   利用相机参数，计算当前相机与历史相机视野的几何重叠。简化为检测两个相机原点发出的四条射线（左右边界）是否在 XY 平面上相交。
    *   同时计算距离，排除距离太远导致的假性重叠。
2.  **冗余剔除（Non-adjacent Filtering）**：
    *   相邻的帧通常包含高度重复的信息。策略是：在连续的候选帧序列中，随机只选一帧（或选最远的一帧）。
3.  **最终选择**：
    *   如果筛选后数量仍超过设定的 Context Size（例如20帧），则优先选择在空间或时间上跨度较大的帧，以最大化信息覆盖。

# Experiment

## 资源消耗
*   **训练硬件**：8块 NVIDIA A100 GPUs。
*   **基座模型**：内部开发的 1B 参数量的 Text-to-Video DiT 模型。
*   **推理速度**：随着 Context Size 增加，速度会下降。例如 Context Size 为 1 时约为 1.60 fps，Context Size 为 20 时降至 0.97 fps（但在可接受范围内）。

## 数据集/bench是什么
*   **数据集**：
    *   **自建 UE5 数据集**：用于训练和定量评估。包含100个长视频，场景多样（城市、自然、室内等），轨迹平滑且受限（XY平面移动，Z轴旋转）。
    *   **Open-domain 数据**：收集自互联网的图片，用于测试泛化能力（图生视频）。
*   **对比方法（Baselines）**：
    1.  **1st Frame**（仅首帧作为 Context）。
    2.  **Random Context**（首帧+随机历史帧）。
    3.  **DFoT (Diffusion Forcing Transformer)**：滑动窗口机制，只看最近的N帧。
    4.  **FramePack**：分层压缩机制（最近的帧保留，远的帧压缩）。
*   **评估指标**：
    *   **图像质量**：FID, FVD。
    *   **记忆能力/一致性**：PSNR, LPIPS。
    *   **评估方式**：
        1.  **Ground Truth Comparison**：用生成的帧与真实数据的帧对比。
        2.  **History Context Comparison**：设计“旋转离开再旋转回来”的轨迹，比较重新看到的场景与之前生成的场景的一致性（这是评估记忆能力的关键）。

## 结果如何
1.  **定量结果**：
    *   **一致性**：Context-as-Memory 在 PSNR 和 LPIPS 指标上显著优于所有 Baseline，证明其记忆能力最强。
    *   **质量**：FID 和 FVD 也达到最佳，说明充分的上下文有助于减少长视频生成中的误差累积。
    *   **检索策略消融**：FOV+去冗余（Non-adj）的策略效果最好，证明剔除无效和重复信息对提升性能至关重要。
2.  **定性结果**：
    *   可视化展示中，当相机旋转一圈回到原位，Baseline 方法（如 DFoT, FramePack）生成的建筑物或物体细节往往变了（出现幻觉），而本文方法能保持与之前生成的内容高度一致。
3.  **泛化能力**：
    *   在 Open-domain 测试中（输入一张网图作为首帧），模型也能在未见过的场景中实现“回眸一致”，证明了方法不依赖于特定的训练场景，具有良好的泛化性。