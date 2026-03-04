[Cosmos Policy](https://arxiv.org/abs/2601.16163)

```ad-tldr
**将大规模预训练的视频生成模型转换为机器人控制策略，技能直接生成机器人动作，也可以预测未来状态和轨迹规划**
```


## Introduction

**Task and Applications**

Cosmos Policy 旨在将大规模预训练视频生成模型（NVIDIA Cosmos-Predict2-2B）适配为机器人视觉运动控制策略（visuomotor policy），用于机器人操作任务。应用场景包括单臂和双臂机械臂的物体抓取、放置、折叠衣服、打开袋子等长时域、高精度操作任务。

**Technical Challenges for Previous Problems**

1. 此前将视频模型用于机器人策略的工作需要**多阶段训练**（例如先微调视频模型，再单独训练动作模块），引入额外的架构组件（如独立的 action diffuser 或 inverse dynamics model），流程复杂。
2. 一些工作训练统一的视频-动作模型（如 UVA、UWM），但**未利用预训练视频模型**的时空先验，从头训练导致无法继承大规模视频数据中学到的物理规律和运动模式。
3. VLA（Vision-Language-Action）模型虽然语义泛化能力强，但其预训练主要基于**静态图文对**，缺乏视频模型所具备的时序因果性和隐式物理建模能力，在高动作多模态性和高精度操作任务上表现不佳。

**Contribution 1: 单阶段无架构修改的视频模型适配方法（Latent Frame Injection）**

解决的问题：如何在不修改预训练视频模型架构的前提下，让其支持机器人本体感受、动作、多视角图像、状态值等多种模态的输入输出。

怎么做的：提出 latent frame injection，将新模态（机器人本体感受、动作 chunk、状态值）编码为与图像latent frame 同形状的 latent volume，直接插入视频模型的 latent diffusion 序列中。模型通过单阶段微调即可学会生成这些新模态。

Key insight：视频扩散模型本身就擅长建模复杂的高维多模态分布和时序连贯序列，其学习算法天然适合在同一序列中同时表示动作和其他模态，无需额外设计。

**Contribution 2: 统一的策略、世界模型和价值函数联合训练**

解决的问题：仅训练动作预测的策略无法进行 test-time planning；单独训练世界模型和价值函数需要额外模块和训练阶段。

具体做法：在同一架构中，通过调整 latent diffusion 序列的 conditioning mask 来切换训练目标——50% batch 训练策略 p(a, s', V(s')|s)，25% 训练世界模型 p(s', V(s')|s, a)，25% 训练价值函数 p(V(s')|s, a, s')。辅助目标（如策略同时预测未来状态和价值）提供额外监督，提升策略性能。

**Contribution 3: 基于世界模型和价值函数的 Model-Based Planning**

解决的问题：仅在 demonstration 上训练的世界模型和价值函数只见过成功轨迹，难以泛化到分布外状态，无法有效 planning。

具体做法：先用基础 Cosmos Policy 收集 rollout 数据（含成功和失败），再用该 rollout 数据微调得到 "planning model"。部署时采用 dual deployment：原始 checkpoint 作为 policy model 提出动作候选，微调后的 checkpoint 作为 planning model 预测未来状态和价值。通过 best-of-N sampling 选择预测价值最高的动作执行。使用 ensemble（每个动作 3 次世界模型预测 × 5 次价值预测 = 15 个价值估计）和 "majority mean" 聚合策略来增强鲁棒性。

---

## Method

**Overview**

输入：当前多视角相机图像（如腕部相机、两个第三人称相机）、机器人本体感受（关节角或末端执行器位姿）、自然语言任务描述。

输出：(1) 动作 chunk（多步动作序列），(2) 预测的未来状态（未来本体感受 + 未来多视角图像），(3) 未来状态的价值估计（期望累积回报）。

Pipeline 组成：预训练视频扩散模型（Cosmos-Predict2-2B）→ latent frame injection 编码多模态 → 单阶段微调联合训练策略/世界模型/价值函数 → 可选的 rollout 数据收集与 planning model 微调 → 部署时直接策略或 best-of-N model-based planning。

**Module 1: Latent Frame Injection**

Motivation：预训练视频模型只接受单视角图像+文本输入、生成视频帧，不支持本体感受、动作、多视角等机器人所需模态。需要一种不改架构就能引入新模态的方法。

做法：将 latent diffusion 序列从纯图像 latent frame 扩展为包含 11 个 latent frame 的混合序列（以三相机平台为例）：blank placeholder、当前本体感受、3 个当前视角图像、动作 chunk、未来本体感受、3 个未来视角图像、未来状态价值。非图像模态通过归一化到 [-1, +1] 后 flatten 并 duplicate 填充到与图像 latent frame 同尺寸的 volume 中，直接覆盖 VAE 编码的 placeholder latent。

为什么能 work：视频扩散模型的 denoising score matching 目标对 latent frame 的语义内容是无关的——它只负责从噪声中恢复 clean latent。因此即使 latent frame 中编码的不是图像而是动作或标量值，模型依然可以通过微调学会生成它们。

Technical challenge：VAE 的时序压缩方案 (1 + T/4) 要求特殊处理——第一帧无时序压缩，后续每 4 帧压缩为 1 帧，因此需要在序列开头放置 blank placeholder，并将每张图像复制 4 份以保证每个模态对应一个 latent frame。

**Module 2: Joint Training of Policy, World Model, & Value Function**

Motivation：将多个训练目标直接统一到了一个框架里：$(s,a,s^{'},V(s^{'}))$, 分别对应当前的状态观测，当前将要采取的动作，预测的未来状态，对未来状态的奖励函数得分（实际上就是预测采取这个动作之后最终完成目标的概率）

做法：每个 batch 中，50% 样本用于策略训练（conditioning on s，target 为 a, s', V(s')），25% 用于世界模型训练（conditioning on s, a，target 为 s', V(s')），25% 用于价值函数训练（conditioning on s, a, s'，target 为 V(s')）。conditioning 的切换仅通过 noise mask 实现——clean 部分为条件，noised 部分为生成目标。

实际中发现奖励分数的设计有多种。如果给全$s,s^{'},a,$生成的就是$V(s^{'})$，是model-based planning，必须要模型输出未来预测才能估计；如果给的是信息中的子集，那么输出分数是$Q(s,a)$，是model-free planning。

支持两种解码模式：parallel decoding（一次性直接对a,s', V(s')全部加噪去噪，速度快，实际部署的时候预测状态和分数预测了也要扔掉，因此并行解码适合用在实际部署；）和 autoregressive decoding（质量高，用于 planning 时依次生成动作→未来状态→价值）。

**Module 3: 噪声分布调整**

Motivation：原始 Cosmos-Predict2 的 log-normal 噪声分布在高噪声水平上权重过低，导致扩散采样起始阶段（高 σ）去噪不准确，对动作生成的精度有害。

做法：训练时使用混合 log-normal-uniform 分布（0.7 概率采样原始 log-normal，0.3 概率从 [1.0, 85.0] 均匀采样），增加高 σ 区域训练权重。推理时将 σ_min 从 0.002 提高到 4，避免低信噪比区间的不准确去噪。

**Module 4: Model-Based Planning with Dual Deployment**

Motivation：仅训练在 demonstration 上的世界模型和价值函数只见过成功分布，泛化能力差。

做法：用基础策略收集 rollout 数据（含成功和失败），微调得到 planning model（90% batch 用于世界模型+价值函数，10% 用于策略）。部署时原始 checkpoint 提出 N 个候选动作，planning model 对每个候选预测未来状态（ensemble 3 次）和价值（ensemble 5 次），通过 majority mean 聚合后选择最高价值的动作执行。N 个候选并行在 N 个 GPU 上推理。

---

## Experiment

**资源消耗**

训练：LIBERO 用 64 张 H100 训练 40K 步（48 小时）；RoboCasa 用 32 张 H100 训练 45K 步（48 小时）；ALOHA 用 8 张 H100 训练 50K 步（48 小时）。推理：直接策略在 1 张 H100 上 5 步去噪耗时 0.61 秒，10 步 0.95 秒，1 步 0.16 秒；model-based planning 在 8 张 H100 并行上耗时 4.9 秒。

**数据集/Benchmark**

LIBERO（4 个 task suite，每 suite 10 任务 × 50 demo，评估指标为 success rate，对比 Diffusion Policy、Dita、π0、π0.5、UVA、UniVLA、OpenVLA-OFT、CogVLA 等）；RoboCasa（24 个厨房任务，仅用 50 demo/task，评估 50 trials × 3 seeds，对比 GR00T-N1/N1.5、UVA、DP-VLA、UWM、π0、Video Policy、FLARE 等）；Real-world ALOHA（4 个双臂任务，101 trials，评估 score 即任务完成百分比，对比 Diffusion Policy、OpenVLA-OFT+、π0、π0.5）。

**结果**

Cosmos Policy 在 LIBERO 上达到 98.5% 平均成功率（SOTA），在 RoboCasa 上以仅 50 个 demo 达到 67.1%（SOTA，而之前方法多使用 300+ demo），在 ALOHA 真实机器人上取得 93.6% 平均分（最高）。在三个任务上超过所有对比方法（"put X on plate" 100%、"fold shirt" 99.5%、"put candies in bowl" 89.6%），在 "put candy in ziploc bag" 上达到 85.4%。消融实验表明：去掉辅助损失降 1.5%，从头训练降 3.9%；去掉未来状态预测目标则大幅下降至 44.4%（RoboCasa）。Model-based planning 在两个困难 ALOHA 任务上平均提升 12.5 分，model-based (V(s')) 优于 model-free (Q(s,a)) 变体。