
# DreamZero: World Action Models are Zero-shot Policies

**论文信息**: Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng 等, NVIDIA, 2026年2月  
**arXiv**: [2602.15922](https://arxiv.org/abs/2602.15922)

---

## Introduction

### Task and Applications

本文聚焦于**机器人基础模型（Robot Foundation Model）**的构建，目标是让机器人在面对全新的任务、全新的环境、甚至全新的机器人形态时，仍能实现零样本（zero-shot）或少样本（few-shot）的泛化执行。应用场景覆盖双臂移动操作（bimanual mobile manipulation）、单臂操作（single-arm manipulation）、跨形态迁移（cross-embodiment transfer）等。

### Technical Challenges for Previous Approaches

当前主流方法是 Vision-Language-Action (VLA) 模型，即在预训练的视觉-语言模型（VLM）基础上微调以输出机器人动作。VLA存在以下核心问题：

1. **语义泛化 ≠ 物理泛化**：VLA继承了VLM的语义理解能力（比如识别"可乐罐"和"Taylor Swift"），但VLM的预训练数据是静态图像-文本对，缺乏时空动态先验。因此VLA可以泛化到新物体（"把可乐罐移到Taylor Swift旁边"），但无法泛化到训练时未见过的新动作或新技能（如"解鞋带"）。
2. **对重复性示范数据的依赖**：现有VLA为了学好策略，需要大量同一任务的重复性 teleoperation 数据。要覆盖所有可能的物理交互是不现实的。
3. **跨形态迁移困难**：VLA直接学习观测到动作的映射，跨形态迁移通常需要动作标注，扩展性差。

### 解决 Challenge 的 Pipeline

本文提出 **DreamZero**，一个 **World Action Model (WAM)**——基于预训练视频扩散模型（video diffusion model）的 14B 参数机器人基础模型。核心思路是：**联合预测未来视频帧和机器人动作**，利用视频生成作为隐式的视觉规划器来引导动作生成。视频扩散模型从网络规模的视频数据中继承了丰富的时空物理先验，从而弥补了VLA在物理动态理解上的缺失。

### Contribution 1: 从多样化、非重复性数据中有效学习

**怎么做的？** DreamZero 采用联合视频-动作预测的目标函数。由于视频预测部分已经从预训练视频模型中继承了大量世界物理知识，模型在学习动作时本质上是在学习一个隐式的逆动力学模型（Inverse Dynamics Model, IDM）——即从预测的视觉未来中提取对应动作。这意味着模型不需要从密集的状态-动作对中隐式推断动态，而是可以利用视频作为中间桥梁。

**Key Insight**: 联合建模视频和动作，将动作学习从"密集模仿"转化为"逆动力学"——对齐运动指令和预测的视觉未来。这使得模型可以高效利用异质性强、非重复性的 teleoperation 数据（约500小时，来自22个真实环境），而无需依赖传统的每个任务大量重复示范。消融实验证实，多样化数据显著优于等量的重复性数据（50% vs. 33%）。

```ad-note
传统的VLA模型训练的时候需要大量规范，重复的数据来让它学习特定的动作，记住状态A-> 动作B的映射
```
### Contribution 2: 在新任务和新环境上的零样本泛化

**为了解决什么问题？** VLA在未见过的动作/技能上几乎完全失败（<1% 任务进度）。

**具体怎么做的？** DreamZero 的视频扩散 backbone 从海量互联网视频中学到了各种物理交互的先验。在推理时，给定语言指令和当前观测，模型先通过视频生成进行"视觉规划"——想象出未来场景应该如何演变，然后从该视觉规划中提取对应动作。这使得即使训练时从未见过"解鞋带"、"握手"、"熨衣服"等任务，模型依然可以通过视频先验合成合理的行为。实验表明，DreamZero 在未见任务上达到 39.5% 平均任务进度，是最佳预训练VLA基线（16.3%）的 2× 以上。

### Contribution 3: 38× 推理加速，实现 7Hz 实时闭环控制

**为了解决什么问题？** 视频扩散模型需要迭代去噪，朴素实现一次推理需约 5.7 秒，无法满足实时控制需求。

**具体怎么做的？** 三层优化体系：(1) 算法层面：DreamZero-Flash 解耦视频和动作的噪声调度，使动作在仅 1 步去噪下即可产出高质量结果；(2) 系统层面：CFG 双 GPU 并行、DiT 缓存（利用速度向量余弦相似度跳过冗余计算）、异步执行；(3) 实现层面：torch.compile + CUDA Graphs、NVFP4 量化、cuDNN 注意力核、调度器 GPU 化。最终在 GB200 上实现 38× 加速，延迟从 5.7s 降至 150ms。

### Contribution 4: 跨形态迁移与少样本新形态适配

**为了解决什么问题？** 让一个机器人学习的知识迁移到另一个机器人，或利用人类视频数据提升机器人表现。

**具体怎么做的？** 两种形式：(1) **跨形态视频迁移**——仅使用其他机器人或人类执行任务的视频（无动作标注），通过视频预测目标对齐到目标机器人。仅需 10-20 分钟视频数据，在未见任务上带来 42%+ 的相对提升（38.3% → 55.4%）。(2) **少样本新形态适配**——用 AgiBot G1 上预训练的 DreamZero，仅用 30 分钟的 YAM 机器人 play data 进行微调，即可在新形态上保持零样本泛化能力。

```ad-note
不同机器人做一样的任务的时候视觉表示是相似的，所以WAM相比更有优势，他能直接学习这些相似的表示
```
### Contribution 5: 开源

开源模型权重、推理代码，以及在 RoboArena、PolaRiS 和 Genie Sim 3.0 基准上的评估代码。

```ad-note

三个泛化：对task的泛化（快速适应数据集中没见过的任务）、对数据的泛化（不需要大量重复数据），对embodiment的泛化（适应不同的机器人、硬件设备非常快）
**第一个进步：泛化能力的全面突破**

"unlocks new generalization capabilities beyond traditional VLAs and previous WAMs—across environments, across tasks, and across embodiments"

这里强调的是"三个维度的泛化"：新环境、新任务、新形态。之前的 VLA 主要在语义层面泛化（识别新物体），而之前的 WAM 虽然引入了视频预测，但泛化验证不够系统。DreamZero 在环境和任务泛化 benchmark 上比最好的预训练 VLA 高出 2× 以上。这个"2×"是个很强的数字——不是微调提升几个百分点，而是翻倍式的跨越。

---

**第二个进步：打破"重复示范"的传统范式**

这段话有个很重要的学术定位。作者指出一个"conventional wisdom"——业界普遍认为训练通用机器人策略需要每个任务大量重复示范。这段话分三层论述：

1. **DreamZero 可以从杂乱多样的数据中学**：不需要同一任务反复做几百遍，而是500小时数据覆盖244种技能、22个环境，每个任务平均只出现很少次。
2. **和其他 WAM 的区分**："Although other WAMs show that priors learned from video prediction improves sample efficiency... most works still focus on repeated demonstrations." 这句话承认了其他 WAM 确实也展示了视频先验能提高样本效率，但它们的实验验证仍然建立在重复示范的数据上。DreamZero 的贡献在于**真正在非重复数据上验证了这一点**。
3. **后训练后泛化不退化**："the environment generalization of DreamZero is retained even after task-specific post-training." 这一点很关键——通常模型在特定任务上微调后会丧失泛化能力（catastrophic forgetting），但 DreamZero 微调后仍比 VLA 高 10%。这说明视频先验提供了一种更鲁棒的知识表示。

---

**第三个进步：跨形态迁移，分两种形式**

**形式一：视频迁移（Video-only Transfer）**

"video-only demonstrations from another robot (YAM) or humans yield a relative improvement of over 42%"

用 YAM 机器人或人类执行任务的视频（注意：没有动作标注，只有视频）来增强 AgiBot G1 的表现。仅需 10-20 分钟的视频数据就带来 42%+ 的相对提升。这之所以可行，是因为 WAM 的学习目标本身就包含视频预测——不同形态做同一件事的视频在视觉层面是相通的，模型可以从中学到"这个任务应该产生什么样的视觉变化"，然后用自己的逆动力学模块转化为本体动作。

**形式二：少样本新形态适配（Few-shot Embodiment Adaptation）**

"a model pretrained on AgiBot G1 adapts to an entirely new robot (YAM) with only 30 minutes of play data, retaining zero-shot generalization"

这个更惊人——在 AgiBot G1 上预训练好的模型，迁移到一个**全新的机器人 YAM** 上，只需要 30 分钟的 play data（随意操作的数据，非特定任务示范），就能在 YAM 上保持零样本泛化。

为什么只需这么少？因为视频预测能力（"世界会怎么变"）已经从预训练中继承了，模型只需要额外学习一件事：**新形态的运动学映射**——即"看到这样的视觉未来，YAM 的关节应该怎么动"。这个逆动力学映射相对简单，30分钟数据足够。

作者最后用"To the best of our knowledge, this sets a new benchmark for data-efficient embodiment adaptation"强调这是目前已知最数据高效的形态适配方案
```

---

## Method

### Overview

**输入**: 当前及历史视觉观测 o₀:ₗ（通过 VAE 编码为潜在向量）、语言指令 c（通过文本编码器）、本体感受状态 qₗ（通过状态编码器）。

**输出**: 未来视频帧 oₗ:ₗ₊ₕ 和对应动作 aₗ:ₗ₊ₕ 的联合预测。

**Pipeline 组成**:

- **Backbone**: Wan2.1-I2V-14B-480P，一个 14B 参数的图像到视频自回归扩散 Transformer（Autoregressive DiT）。
- **额外模块**: 状态编码器、动作编码器、动作解码器（参数量极少，保留视频模型的泛化能力）。
- **训练目标**: 基于 Flow Matching 的联合视频-动作去噪，采用 Teacher Forcing 的 chunk-wise 训练。
- **推理**: 自回归逐 chunk 生成，执行后用真实观测替换预测帧更新 KV Cache，消除误差累积。

### Module 1: 联合视频-动作 Flow Matching 训练

**Motivation**: 将 WAM 的联合预测分解为"视频预测"和"逆动力学模型"两部分（公式 1），但用**单一端到端模型**同时学习两者，确保视频和动作之间的深度对齐。

**Technical Details**:

- 将轨迹分为 M 个 chunk，每个 chunk 包含 K=2 个潜在帧。
- 对每个 chunk 独立采样去噪时间步 tₖ，在干净的历史 chunk 条件下去噪当前 chunk 的 noisy 视频和动作（Teacher Forcing）。
- 损失函数为标准 flow matching 速度场回归（公式 3）。
- 视频和动作共享同一去噪时间步以加速收敛。

**Why it works**: 预训练视频模型已经在视频预测目标上优化过，DreamZero 只需额外学习(1)机器人视频的生成和(2)从预测视频中提取动作。相比VLA从VLM初始化，WAM显式学习时间动态，因此泛化更强。

### Module 2: 自回归架构设计

**Motivation**: 相比双向（Bidirectional）架构，自回归架构有三个优势：

1. **KV Cache 加速推理**: 自回归生成可缓存历史计算，推理速度比双向快 3-4×。
2. **保留原生帧率**: 双向模型需要视频下采样以匹配语言标注区间，导致帧率失真，破坏视频-动作对齐。自回归模型通过视频上下文条件化避免了这一问题。
3. **更平滑的运动**: 反向传播穿过整个动作序列，时间一致性更好。

**Technical Challenge**: 自回归视频生成会累积误差。

**解决方案**: 利用闭环控制的特性——每个 chunk 执行后，用真实观测替换 KV Cache 中的预测帧，从根本上消除误差累积。

### Module 3: DreamZero-Flash（解耦噪声调度）

**Motivation**: 即使有系统优化，扩散步数仍是延迟瓶颈。但朴素减少步数会导致动作质量下降——因为视频残余噪声传播到动作预测。

**Technical Challenge**: 标准 DreamZero 训练时视频和动作在同一噪声水平，但少步推理时动作需要去噪到干净状态，而视频仍然很嘈杂——存在训练-推理不匹配。

**解决方案**:

- 训练时将视频时间步偏向高噪声（t_video = 1 - η, η ~ Beta(7,1)，期望值仅 0.125），同时保持动作时间步均匀分布。
- 这迫使模型学会在视频仍然很嘈杂的条件下预测干净动作，直接匹配单步推理场景。
- 效果：扩散步数从 4 步降至 1 步，推理延迟从 ~350ms 降至 ~150ms，任务进度仅下降 9%（83% → 74%）。

### Module 4: 系统与实现层优化

**CFG 并行**: Classifier-Free Guidance 需要两次前向传播，分配到两个 GPU 并行，单步延迟降低 47%。

**DiT 缓存**: 监测连续速度预测的余弦相似度，超过阈值时复用缓存速度，有效 DiT 步数从 16 降至 4。

**Torch Compile + CUDA Graphs**: 消除 CPU 开销和算子启动延迟。

**NVFP4 量化**: 在 Blackwell 架构上将权重和激活量化到 NVFP4，敏感操作保持 FP8/FP16。

**动作 Chunk 平滑**: 上采样到 2× 分辨率 → Savitzky-Golay 滤波 → 下采样，抑制高频噪声。

### Module 5: 跨形态迁移机制

**Robot-to-Robot / Human-to-Robot Transfer**: 仅使用其他形态执行任务的视频（无动作标注），通过视频预测目标进行 co-training。视频数据作为额外的视觉经验强化世界模型对任务动态的理解。

**Few-shot Embodiment Adaptation**: 在新机器人上仅用 30 分钟 play data 微调，模型只需学习从视觉未来到新形态动作的映射（IDM），而视频预测能力已从预训练中继承——因此极其样本高效。

---

## Experiment

### 资源消耗

- 模型大小: 14B 参数（Wan2.1-I2V-14B-480P backbone）
- 训练: 100K 步，全局 batch size 128
- 推理硬件: 2× GB200 GPU，实时推理 ~150ms/chunk（7Hz）
- 消融实验: 50K 步，batch size 32

### 数据集与 Benchmark

**训练数据**:

- AgiBot G1: ~500 小时 teleoperation 数据，22 个真实环境，7193 个 episode，244 种技能，平均每 episode 42.4 个子任务
- DROID: 公开的 Franka 单臂机器人数据集

**评估设置**（默认在未见环境、未见物体下评估）:

|评估类别|说明|指标|
|---|---|---|
|AgiBot 已见任务|10 个任务，4 个机器人 × 每任务 8 rollout|任务进度 / 成功率|
|AgiBot 未见任务|10 个训练时不存在的任务|任务进度|
|DROID 已见/未见|各 20 个任务，每任务 2 rollout|任务进度 / 成功率|
|后训练|衬衫折叠、水果打包、桌面清理|平均任务进度|
|跨形态迁移|9 个未见任务|任务进度|
|Flash 消融|Table bussing|任务进度 + 推理速度|

**对比方法**: GR00T N1.6（from-scratch / from-pretrained）、π0.5（from-scratch / from-pretrained）

### 核心结果

|实验|DreamZero|最佳 VLA 基线|
|---|---|---|
|已见任务平均进度 (AgiBot)|**62.2%**|27.4% (π0.5 pretrained)|
|未见任务平均进度 (AgiBot)|**39.5%**|16.3% (π0.5 pretrained)|
|后训练平均进度|**90.5%**|79.8% (π0.5 pretrained)|
|DROID 已见任务进度|**82%**|69% (GR00T pretrained)|
|DROID 未见任务进度|**49%**|33% (π0.5 pretrained)|
|跨形态迁移 (Robot2Robot)|**55.4%**|38.3% (baseline DreamZero)|
|DreamZero-Flash (1步)|**74%**|83% (4步 DreamZero)|

**关键发现**:

1. From-scratch VLA 在多样化数据上几乎完全失效（~0%），而 DreamZero 有效学习。
2. 大多数 DreamZero 失败来源于视频生成错误而非动作提取——策略忠实执行视频预测的轨迹，意味着改进视频 backbone 可直接提升策略性能。
3. 14B 模型显著优于 5B（50% vs. 21%），展现清晰的缩放行为。
4. 多样化数据 > 重复性数据（50% vs. 33%），即使总时长相同。