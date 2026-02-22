[Long-Context Autoregressive Video Modeling  with Next-Frame Prediction](https://arxiv.org/pdf/2503.19325)


![[image-5.png]]

![[image-6.png|610x433]]


# TL;DR
文章想解决的问题和self forcing那几篇不一样，除了常说的train-test gap（推理的时候模型基于自己生成的烂图继续推，而不是基于gt的干净视频），还有一种gap: 先前teacher forcing的方法会在原始干净帧上面加噪，再让模型基于加噪的干净帧去训，但真实推理过程中模型基于自己先前生成的帧推理，自己生成的帧可能有伪影，但是绝对没高斯噪声。这是第二个train test gap。

但是既然self forcing做到了在训练的时候就能看自己的烂帧，还有training过程中做到一帧一帧自回归生成，FAR的意义感觉不大。

# Introduction

## task and applications
**任务**：长上下文视频建模（Long-Context Video Modeling）。
**应用**：旨在让生成式模型充当“世界模拟器（World Simulators）”，这要求模型在延长的跨度内保持时间连贯性（temporal coherence），例如记住观察到的环境布局，而不仅仅是短期的动作连贯。

## Technical challenges for previous problems
1.  **现有模型主要基于短片段**：目前的SOTA模型（如Wan, Cosmos）通常在约5秒的短片段上训练，难以捕捉长距离依赖。
2.  **测试时外推（Test-time extrapolation）效果有限**：虽然有一些training free的方法试图在推理时延长视频，但它们无法有效利用长距离上下文，生成质量随长度增加而下降。
3.  **计算成本过高（Token Explosion）**：直接在长视频上训练极其昂贵，因为视觉Token的数量随时间增长极快（比语言模型快得多），导致计算量难以承受。
4.  **Token-AR的缺陷**：基于离散Token的自回归模型（Token-AR）由于矢量量化（Vector Quantization）会有信息损失，且通常不如Diffusion模型生成质量高。
5. **Test time training:** 在测试的时候进行训练，但这会导致用户推理的时间非常长。
6.  **训练-推理偏差**：混合AR-Diffusion模型在训练时通常基于加噪的上下文，而推理时使用的是干净的上下文，导致分布偏移。

## 解决challenge 的pipeline是什么

### contribution 1: Frame AutoRegressive (FAR) Baseline
**怎么做的？Key insight是什么？**
*   **Insight**：结合自回归的长距离依赖建模能力和Diffusion（Flow Matching）的高质量生成能力，但在**连续潜空间（Continuous Latent Space）**进行，避免离散化的损失。
*   **做法**：建立了一个名为FAR的强基线模型。它是一个基于Transformer的模型，但在帧与帧之间采用**因果注意力（Causal Attention）**，在帧内采用全注意力。它将每一帧视为一个自回归单元，使用**Flow Matching**目标进行训练。

### contribution 2: Stochastic Clean Context (随机干净上下文)
**为了解决什么问题？具体怎么做的？**
*   **问题**：解决混合AR-Diffusion模型中的“训练-推理差距（Training-Inference Gap）”。训练时模型看到的上下文是加噪的（noised latent），推理时看到的是生成的干净帧（clean latent）。
*   **做法**：在训练过程中，**随机将一部分加噪的上下文帧替换为对应的干净潜在帧**，并分配一个特殊的时间步嵌入（如 -1）。这迫使模型在训练阶段就学会利用干净的上下文信号，从而在推理时表现更好。

```ad-note
本文的“干净”指的是是否去噪干净，无论这个帧是否有伪影，模型预测的质量高不高，只要完成了去噪步骤，在这里就被定义为干净的。以前的方法在自回归生成的过程中，后续帧的预测与前续帧的预测没有完全隔开，导致预测后续帧的时候会参考前面还没有完成去噪的帧的上下文，所以是不干净的。
```



### contribution 3: Long Short-Term Context Modeling with Asymmetric Patchify Kernels (基于非对称Patchify核的长短时上下文建模)
**为了解决什么问题？具体怎么做的？**
*   **问题**：解决长视频训练中Vision Token数量爆炸导致的计算成本过高问题。
*   **Insight**：**上下文冗余（Context Redundancy）**。视频中相邻的帧对于保持局部动作连贯性至关重要（需要高分辨率），而遥远的帧主要充当“记忆”（只需宏观信息）。
*   **做法**：
    *   **非对称设计**：将上下文分为“短期”和“长期”。
    *   **短期上下文**：使用标准的Patchify Kernel（保留细节）。
    *   **长期上下文**：使用**大的Patchify Kernel**（例如4x4甚至更大），大幅压缩Token数量。
    *   这使得模型能在训练极长视频时，Token数量保持在可接受范围内，同时不丢失关键的长期语义信息。

# Method

## overview
*   **输入**：一段视频序列的帧（训练时）或已生成的过去帧（推理时）。
*   **输出**：下一帧的预测结果。
*   **Pipeline组成**：
    1.  **VAE压缩**：使用预训练的VAE将视频帧压缩到Latent Space。
    2.  **FAR Transformer**：核心网络，采用因果时空注意力机制。
    3.  **Flow Matching**：基于流匹配的目标函数，预测从噪声到数据的速度场。
    4.  **上下文管理**：应用“随机干净上下文”和“非对称Patchify”策略来处理输入。

## module 1: Frame AutoRegressive (FAR) Framework
*   **Motivation**：为了超越基于离散Token的AR模型（Token-AR）和传统的Video DiT。Token-AR有量化损失，Video DiT通常是非因果的，难以做长视频生成。
*   **做法**：
    *   在Latent Space上操作。
    *   **因果掩码（Causal Mask）**：修改Attention Mask，使得第 $t$ 帧只能看到 $1$ 到 $t$ 帧的信息，确保自回归生成。
    *   **Flow Matching**：使用 $L(\theta) = E[\|v_\theta(x(t), t) - (x_1 - x_0)\|^2]$ 进行训练，模拟从高斯噪声到图像Latent的轨迹。

## module 2: Stochastic Clean Context (SCC)
*   **Technical Challenge**：训练时，上下文帧是根据时间步 $t$ 加噪的；推理时，上下文是之前生成的“干净”帧（虽然可能有瑕疵，但未加高斯噪）。这种分布不匹配导致生成质量下降（如闪烁）。
*   **做法**：
    *   在训练batch中，以一定概率（如0.1）将上下文帧替换为未加噪的Clean Latent。
    *   给这些帧赋予一个特殊的时间步 $t=-1$，告诉网络“这是干净的参考帧”。
    *   这无需额外的推理成本，也不增加计算量，但显著提升了指标。

## module 3: Long Short-Term Context Modeling (Asymmetric Patchify)
*   **Technical Challenge**：视频越长，Token越多，Attention计算是 $O(N^2)$，显存和计算瞬间爆炸。
*   **做法**：
    *   **Short-Term Window**（如最近16帧）：使用常规Kernel（如2x2），保留高频细节，保证动作平滑。
    *   **Long-Term Window**（遥远的过去）：使用大Kernel（如4x4，Token数减少4倍），提取全局语义，作为长期记忆。
    *   通过这种方式，即使Vision Context长度增加，Token Context长度增长也很缓慢，从而实现高效训练。

## module 4: Multi-Level Inference-Time KV Cache
*   **Motivation**：自回归生成每一步都要重新计算以前的特征，非常慢。
*   **做法**：
    *   由于引入了非对称核，KV Cache也需要分级。
    *   **L1 Cache**：存短期窗口的高分辨率特征。
    *   **L2 Cache**：当帧移出短期窗口进入长期窗口时，重新编码（Re-encode）为低分辨率特征并存入L2 Cache。
    *   推理速度因此大幅提升（从1300多秒降至100秒左右）。

# Experiment

## 资源消耗
*   相比于全注意力/全分辨率模型，**训练时间缩短约5倍**，显存占用大幅降低（见论文Fig 6）。
*   推理时结合KV Cache，速度提升超过**10倍**。

## 数据集/bench是什么
*   **短视频生成/预测**：UCF-101 (Generation & Prediction), BAIR (Prediction).
*   **长视频预测（Long-Context）**：Minecraft (200K videos), DMLab (40K videos) —— 这是一个基于动作条件（Action-conditioned）的长视频预测任务。
*   **对比方法**：Latte (Video DiT), Token-AR (如MAGVITv2, CogVideo), TECO, FitVid等。
*   **指标**：FVD (Fréchet Video Distance), SSIM, PSNR, LPIPS。

## 结果如何
1.  **短视频性能**：
    *   FAR在UCF-101上的收敛速度优于Video DiT（如Latte）。
    *   在无条件和有条件生成上均取得了SOTA或极具竞争力的FVD分数（优于Token-AR模型）。
2.  **长视频性能**：
    *   **Direct Training vs Extrapolation**：直接在长视频上训练（FAR-Long）的效果显著优于在短视频上训练并通过测试时外推（Test-Time Extrapolation）的方法。
    *   在Minecraft和DMLab上，FAR利用长上下文（144帧上下文预测后续帧）取得了最低的LPIPS和FVD，证明了其有效利用长期记忆的能力。
3.  **Ablation Study**：
    *   **Stochastic Clean Context**：去除该模块会导致FVD从347恶化到399，证明了消除训练推理偏差的重要性。
    *   **Asymmetric Kernel**：使用[4,4]的大核在保持性能（SSIM/LPIPS几乎不变）的同时，显著降低了训练内存（从38.9G降至15.3G）。