[arxiv](https://arxiv.org/abs/2506.08009)

![[image-2.png]]


![[image-3.png]]



这篇文章介绍了一种名为 **Self Forcing** 的自回归视频扩散模型训练范式，旨在解决训练与测试之间的分布差异（Exposure Bias），实现高质量、低延迟的实时视频生成。
# Introduction

## task and applications
*   **Task:** 自回归（Autoregressive, AR）视频生成。
*   **Applications:** 实时流媒体视频生成（Real-time streaming video generation）。具体应用场景包括：实时交互式内容创作、游戏模拟、机器人学习以及直播流媒体。

## Technical challenges for previous problems
1.  **Exposure Bias（曝光偏差）:** 传统的 Teacher Forcing (TF) 训练在每一步都基于真实的 Ground-truth 上下文进行去噪，但在推理（测试）时，模型必须依赖自己生成的（可能包含错误的）历史帧。这种训练-测试的不匹配导致误差随着生成过程不断累积，视频质量随时间推移而下降。
2.  **Distribution Mismatch in Recent Solutions:** 现有的 Diffusion Forcing (DF) 虽然引入了噪声上下文，但它训练时的输入是基于数据分布加噪的，而推理时的输入是模型之前的输出。CausVid 等方法虽然使用了分布匹配损失，但其优化的分布（基于 DF 输出）与实际推理时的分布（AR 输出）不一致。
3.  **Efficiency vs. Quality:** 双向注意力模型（如 DiT）质量高但延迟大，无法流式输出；传统 AR 模型延迟低但往往依赖有损矢量量化，视觉质量不如扩散模型。

## 解决challenge 的pipeline是什么
文章提出了 **Self Forcing** 训练范式。该 Pipeline 在训练阶段显式地执行自回归 **Self-Rollout**（自展开），即当前帧的生成是基于模型自身之前生成的帧（通过 KV Cache），而不是 Ground-truth。然后，对生成的完整视频序列应用**整体分布匹配损失**（Holistic Distribution Matching Loss），迫使模型从自身的预测误差中学习并修正。

### contribution 1: Self Forcing Training Paradigm
**怎么做的？key insight是什么？**
*   **做法：** 在训练过程中模仿推理过程。模型通过 KV Caching 进行多帧的自回归生成。生成第 $i$ 帧时，条件是模型之前生成的第 $1$ 到 $i-1$ 帧（存储在 KV Cache 中），而不是数据集中的真实帧。
*   **Key Insight：** 只有让模型在训练时“看到”并基于自己生成的（不完美的）历史数据进行预测，才能真正消除 Exposure Bias。这使得训练分布 $p_{\theta}$ 与推理分布对齐。

### contribution 2: Efficient Implementation (Few-step Backbone & Gradient Truncation)
**为了解决什么问题？具体怎么做的？**
*   **问题：** 在训练中进行完整的自回归展开（Unrolling）并反向传播，计算成本过高且显存难以承受。
*   **做法：**
    1.  使用 **Few-step Diffusion**（如4步）作为基座，减少单帧去噪步数。
    2.  提出 **Stochastic Gradient Truncation**（随机梯度截断）：在训练的每一步，只对每帧最后一步去噪过程进行梯度回传，并且切断当前帧对历史帧 KV Cache 的梯度流（Detach gradients）。
    3.  随机采样一个去噪步数 $s$ 作为输出，而不是总是展开所有步数，以兼顾效率和监督信号。这使得训练效率甚至优于并行的 Teacher Forcing/Diffusion Forcing。

### contribution 3: Rolling KV Cache Mechanism
**为了解决什么问题？具体怎么做的？**
*   **问题：** 现有的滑动窗口推理（Sliding Window）在处理长视频时，要么需要重新计算 KV Cache（计算量 $O(TL^2)$），要么因为重叠少导致时序不连贯（Flickering）。
*   **做法：**
    1.  引入 **Rolling KV Cache**：维护一个固定大小的 Cache，生成新帧时逐出最旧的 token，加入新的 token，无需重算，复杂度降为 $O(TL)$。
    2.  **Training Fix：** 为了解决首帧（First Frame）分布不同导致的伪影问题，在训练时限制注意力窗口，让模型在去噪后面的 Chunk 时无法 attend 到第一个 Chunk，从而模拟长视频生成中“丢失首帧信息”的状态。

# Method

## overview
*   **输入：** 文本提示词（Text Prompts）。
*   **输出：** 连续生成的视频帧序列。
*   **Pipeline 组成：**
    1.  一个基于 Transformer 的自回归视频扩散模型（使用 Causal Attention）。
    2.  **训练阶段：** 使用 Self-Rollout 策略生成视频片段 $\rightarrow$ 计算整体分布匹配损失（DMD/SiD/GAN Loss）更新参数。
    3.  **推理阶段：** 使用 Rolling KV Cache 进行无限长视频流式生成。

## module 1: Autoregressive Diffusion Post-Training via Self-Rollout
**为什么能work？motivation是什么？technical challenge是什么？**
*   **Motivation:** 彻底解决训练和推理输入来源不一致的问题（Exposure Bias）。
*   **核心机制：** 
    *   模型 $G_\theta$ 在时间步 $t_j$ 接收自身生成的历史 KV Cache 和噪声输入。
    *   生成的输出被送入 Forward Process 加噪，作为下一去噪步或下一帧的输入。
    *   通过这种链式反应，当前帧的生成直接受到历史生成质量的影响。
*   **Challenge & Solution:** 自回归训练串行度高。作者发现结合 **Gradient Truncation**（不回传梯度到 KV Cache）和 **FlashAttention** 优化，Self Forcing 的训练速度（Wall-clock time）实际上比需要复杂 Mask 操作的 Teacher Forcing 还要快。

## module 2: Holistic Distribution Matching Loss
**为什么能work？motivation是什么？technical challenge是什么？**
*   **Motivation:** 传统的逐帧 MSE 损失无法直接优化整个视频序列的连贯性和质量，且在使用 Self-Rollout 后，生成的是一个完整分布，适合用 GAN 或蒸馏思路优化。
*   **具体选择：** 框架兼容多种分布匹配损失：
    *   **DMD (Distribution Matching Distillation):** 最小化 $D_{KL}(p_\theta || p_{data})$。
    *   **SiD (Score Identity Distillation):** 基于 Fisher Divergence。
    *   **GAN Loss:** 引入判别器区分生成视频和真实视频。
*   **优势：** 相比于 CausVid（匹配的是 DF 输出的分布），Self Forcing 匹配的是真实的推理时分布，因此监督信号更准确。

## module 3: Rolling KV Cache for Long Video Generation
**为什么能work？motivation是什么？technical challenge是什么？**
*   **Motivation:** 实现无限长视频生成，且保持低延迟和高吞吐。
*   **机制：** 类似于 LLM 的 Streaming Inference。当 Cache 满时，移除最早的 Frame Embedding。
*   **Technical Detail:** 
    *   **Naive实现的缺陷：** 首帧通常包含完整的图像信息且未经过度压缩。直接 Rolling 掉首帧会导致模型“迷失”，产生画面闪烁。
    *   **Local Attention Training:** 训练时强制模型在去噪最后一个 Chunk 时不看第一个 Chunk，迫使模型学会依赖“中间”的历史信息，从而适应 Rolling Cache 的环境。

# Experiment

## 资源消耗
*   **推理：** 单张 NVIDIA H100 GPU。
*   **训练：** Post-training 阶段非常高效，使用 64 张 H100 GPU 仅需约 1.5 小时即可收敛（DMD 损失）。

## 数据集/bench是什么
*   **Base Model:** Wan2.1-T2V-1.3B。
*   **训练数据:** 从 Base Model 采样的 16k ODE solution pairs 以及 VidProM 筛选的 prompts。用于 GAN/DMD 训练的数据集也是模型生成的（Data-free distillation）。
*   **Benchmark:** VBench (包含16个维度的评估指标)。
*   **User Study:** 使用 MovieGenBench 的 1003 个 prompt 进行人工对比。
*   **Baselines:** Wan2.1 (Base), LTX-Video, SkyReels-V2, MAGI-1, CausVid, Pyramid Flow。

## 结果如何
1.  **视频质量 (VBench):** Self Forcing (Chunk-wise) 取得了 **84.31** 的总分，超越了 Base Model Wan2.1 (84.26) 和所有对比的 AR/Diffusion 模型。在语义对齐（Semantic Score）和运动质量上表现尤为出色。
2.  **推理速度与延迟:**
    *   实现了 **17.0 FPS** 的吞吐量（Chunk-wise），达到了实时标准。
    *   首帧延迟（Latency）极低：Chunk-wise 为 0.69s，Frame-wise 仅为 **0.45s**。相比 Wan2.1 (103s) 速度提升巨大，且比其他 AR 模型（如 MAGI-1）更快。
3.  **用户偏好 (User Study):** 在人工评估中，Self Forcing 生成的视频在视觉质量和提示词一致性上均优于 CausVid、SkyReels-V2 和原始的 Wan2.1。
4.  **长视频生成:** 使用 Rolling KV Cache 成功生成了连贯且无闪烁的长视频，且保持了高 FPS，证明了该机制的有效性。


# 实现

self forcing解决train-test的gap的实现如下：

```
核心思想：训练时直接使用模型自己生成的帧作为上下文
SelfForcingTrainingPipeline.inference_with_trajectory():

1. 初始化:

   - noise ~ N(0, I), shape [B, F, C, H, W]
   - output = zeros([B, F, C, H, W])
   - kv_cache = zeros for each layer


1. For each block (e.g., [1, 3, 3, 3, 3, 3, 3]):
   a. 获取当前块的噪声
   b. 多步去噪 (denoising_step_list = [1000, 750, 500, 250]):
      - 随机选择一个步骤用于反向传播 (exit_flag)
      - 其他步骤使用 torch.no_grad()
      for idx, timestep in enumerate([1000, 750, 500, 250]):
          if idx != exit_flag:
              with torch.no_grad():
                  x0_pred = generator(noisy_input, timestep, kv_cache)
                  noisy_input = add_noise(x0_pred, next_timestep)
          else:
              # 这一步启用梯度
              x0_pred = generator(noisy_input, timestep, kv_cache)
              break
   c. 记录输出: output[block] = x0_pred
   d. 更新KV缓存 (关键!):
      - 使用 x0_pred (模型自己的预测) 更新缓存
      - 而非使用真实数据
      with torch.no_grad():
          generator(x0_pred, timestep=0, kv_cache)  # 缓存干净特征
1. 返回: output, denoised_timestep_from, denoised_timestep_to

关键点:
- 梯度只通过一个随机选择的去噪步骤反向传播
- KV缓存使用模型自己的预测更新，实现self-forcing
- 这消除了训练-推理不匹配
```

每个块进行四次去噪，但只选择其中的一次进行反向传播（节省显存），并送入kv-cache（这就保证了kv-cache里存的memory是自己生成的，而不是之前diffusion forcing和teacher forcing那种干净的，不符合推理过程的设置）

kv cache只对generator使用，学生模型和教师模型都是bidirection attn的，不用kv cache。

generator根据自己产的先前帧推理之后的视频流程如下：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Self-Forcing 完整处理流程                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  时间步骤1：处理 Chunk 0                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  输入: Chunk 0 的 noisy latent [B, 21, C, H, W]                     │     │
│  │       ↓                                                             │     │
│  │  Token化: [B, 21×h×w, D] (假设 h×w = 936)                           │     │
│  │       ↓                                                             │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ DiT Block 1  ← 处理完整的 [B, 21×936, D] token序列           │   │     │
│  │  │ DiT Block 2  ← 处理完整的 [B, 21×936, D] token序列           │   │     │
│  │  │ ...                                                          │   │     │
│  │  │ DiT Block 40 ← 处理完整的 [B, 21×936, D] token序列           │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  │       ↓                                                             │     │
│  │  输出: Chunk 0 的 denoised latent                                   │     │
│  │       ↓                                                             │     │
│  │  更新 KV Cache (保存 Chunk 0 的 K, V)                               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  时间步骤2：处理 Chunk 1                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  输入: Chunk 1 的 noisy latent + KV Cache (Chunk 0)                │     │
│  │       ↓                                                             │     │
│  │  Token化: 当前块 [B, 21×936, D]                                     │     │
│  │       ↓                                                             │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ DiT Block 1:                                                 │   │     │
│  │  │   - Q: 来自当前 Chunk 1 的 token                              │   │     │
│  │  │   - K,V: 来自 Cache(Chunk 0) + 当前 Chunk 1                   │   │     │
│  │  │   - 实际计算的序列长度: (21+21)×936 = 42×936                  │   │     │
│  │  │                                                              │   │     │
│  │  │ DiT Block 2: 同上                                            │   │     │
│  │  │ ...                                                          │   │     │
│  │  │ DiT Block 40: 同上                                           │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  │       ↓                                                             │     │
│  │  输出: 只有 Chunk 1 的 denoised latent                              │     │
│  │       ↓                                                             │     │
│  │  更新 KV Cache (追加 Chunk 1 的 K, V)                               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ... 后续 Chunk 类似 ...                                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```



