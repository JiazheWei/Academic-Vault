[Solaris: Building a Multiplayer Video World Model in Minecraft](https://arxiv.org/abs/2602.22208)

---

## Introduction

### Task and Applications

Video world model（视频世界模型）：给定过去的观测帧和动作，生成未来的智能体观测视频。应用场景包括合成训练数据、推理时规划、策略学习与评估。本文的核心任务是将其扩展到**多玩家（multiplayer）**设定——同时模拟多个智能体的视角，且视角之间必须保持一致。

### Technical Challenges

1. **缺乏多人数据采集系统**：现有 Minecraft AI 框架（Malmo、MineRL、MineDojo、Mineflayer）要么不支持多人、要么没有视觉渲染、要么不可控（需要 RL 训练才能采集有意义的数据），没有一个能同时满足"可控 + 多人 + 有画面"三个条件。
    
2. **多视角一致性建模极难**：不仅要保证时间上的连贯性，还要保证不同玩家视角的空间一致性。一个玩家放了一个方块，另一个玩家的视角必须同时准确反映这个变化。
    
3. **长序列自回归退化**：自回归 diffusion 模型在长时间生成时会因为 exposure bias 导致误差累积、画面退化。直接在滑动窗口设定下做 Self Forcing 会导致显存爆炸（内存复杂度 O(L_t · L_s)）。
    

### 解决 Challenge 的 Pipeline

分阶段训练：预训练双向视频模型 → 单人双向微调 → 多人双向微调 → 因果掩码微调 → Self Forcing（用双向模型做 teacher 蒸馏到因果自回归 student）。同时自建了完整的多人数据采集引擎 SolarisEngine。

---

### Contribution 1: SolarisEngine — 首个支持可控多人 Minecraft 游戏数据采集的框架

**解决什么问题？** 现有框架无法同时满足三个条件：程序化可控（不需要训练 RL agent）、多人协作、有真实 Minecraft 渲染画面。

**怎么做的？**

- 基于 Mineflayer（JavaScript Minecraft 客户端库）构建高层原语库（寻路、放置方块、战斗等），通过编程组合实现多样化的多人协作 episode。
- 为解决 Mineflayer 不支持多人协调的问题，构建了**通信层**让两个 bot 协作。
- 为解决 Mineflayer 没有渲染能力的问题，为每个 controller bot 配一个"camera bot"（运行官方 Java 客户端，headless GPU 渲染），通过自定义服务端插件实时同步状态。
- Docker 容器化编排，支持大规模并行数据采集，内建错误检测和自动恢复。

**Key insight：** 不用 RL 训练 agent 来收集数据（那样的数据太偏、不像人类），而是用高层 API 编程模拟人类行为，然后翻译成低层动作，既多样又真实。

**数据规模：** 9,240 episodes，每玩家 6.32M 帧，共 12.64M 帧，涵盖建造、战斗、移动、挖矿四大类。

---

### Contribution 2: 多人视频 DiT 架构

**解决什么问题？** 如何用最小改动将单人预训练视频 DiT 适配为多人模型？

**怎么做的？**

- 在 sequence 维度上做**视觉交错（visual interleaving）**：将多个玩家的 token 拼接在一起。
- **Multiplayer Self-Attention**：所有玩家的 token 共享同一个 self-attention 层，实现跨玩家信息交互。加入可学习的 **Player ID embedding** 区分不同玩家。对每个玩家独立施加 3D RoPE。
- FFN、Action Module、Cross-Attention 等其他模块**权重共享但独立按玩家运行**（player 维度折叠进 batch）。
- 扩展动作空间以支持完整的 Minecraft 动作（MineRL 格式）。

**Key insight：** 通过 shared self-attention 交换多玩家信息，其余模块保持独立，改动极小但有效。相比 Multiverse 的 channel concatenation 方案（把多个玩家帧在通道维度拼接），本文的 interleaving 方案在复杂场景下表现更好。

```ad-note
设置所有帧的噪声时间步相等--全局双向模型，用来做teacher；
一个帧一个噪声强度，前一帧去噪到噪声强度=0的时候再去噪下一帧--自回归，用作student.
```

### Contribution 3: 分阶段训练流水线（Staged Training Pipeline）

**解决什么问题？** 直接从头训练多人因果模型效果差，如何高效地将预训练的双向模型逐步适配到多人自回归生成？

**四个阶段：**

|阶段|内容|关键细节|
|---|---|---|
|Stage 1|双向单人微调|在 VPT 数据集（2000+小时人类游戏）上微调 Matrix Game 2.0，适配完整动作空间。120K steps|
|Stage 2|双向多人训练|引入多人架构改动，在自采多人数据上训练。120K steps。此 checkpoint 作为 Self Forcing 的 teacher|
|Stage 3|因果多人训练|从 Stage 2 的 60K 中间 checkpoint 分支，加因果滑动窗口掩码 + Diffusion Forcing。60K steps。作为 Self Forcing 的 student 初始化|
|Stage 4|Self Forcing|用双向 teacher 监督因果 student 在自己生成轨迹上的输出|

**Key insight：**

- 单人预训练（Stage 1）对多人建模至关重要（实验证实去掉后效果显著下降）。
- **因果模型的初始化不需要 CausVid 的复杂流程**（ODE regression + DMD 蒸馏），简单地用 Diffusion Forcing + 因果掩码微调就够了，效果还更好。这是对 Self Forcing 原论文假设的简化。可能的原因是Self-forcing的loss中有DMD loss，在DMD的过程中few-step能力涌现出来了。

---

### Contribution 4: Checkpointed Self Forcing — 内存高效的长上下文 Self Forcing

**解决什么问题？** 原始 Self Forcing 要求 teacher 和 student 上下文长度相同。作者希望 teacher 的上下文更长（更强的监督），但在滑动窗口生成中，naive 实现的内存是 O(L_t · L_s)，不可行。

**具体怎么做的？**

1. **阶段 A（无梯度 rollout）：** 先用滑动窗口自回归生成完整视频，缓存每帧的 clean estimate x̂₀ 和对应的 noisy state x_σ，全程 stop gradient。
2. **阶段 B（并行重计算 + 反向传播）：** 将所有帧的 clean context 和 noisy input 拼接成一个长序列，用特殊的 **Teacher Forcing Mask** 做一次并行 forward pass，模拟每帧的最后一步去噪。然后在这个 pass 上正常反向传播。

**Teacher Forcing Mask 的规则：**

- Noisy query 只能 attend 到同帧的 noisy key 和窗口内过去的 clean key
- Clean query 只能因果地 attend 到窗口内的 clean key
- Clean 不 attend noisy

**效果：** 内存从 O(L_t · L_s) 降到 O(L_t)。省出的内存还允许**对 KV cache 做反向传播**（原始 Self Forcing 中 KV 是 stop gradient 的），实验证明这进一步提升了视觉质量。

**Key insight：** 类比 gradient checkpointing 的思想——先不存中间计算图地跑一遍，缓存关键状态，再用一次并行 forward 重建计算图做反向传播。将顺序的 rolling cache 操作转化为一次并行操作。

```ad-note
简而言之，stage 4的阶段A只是输出clean latent和noisy latent，告诉teacher现在的student 的水平能产生什么视频，阶段B负责基于clean latent和noisy latent重新打一遍计算图，然后让teacher评判，接着反向传播更新参数。
```

---

### Contribution 5: 多人评估框架

**解决什么问题？** 多人视频世界模型没有现成的评估基准。

**怎么做的？** 设计了 5 类 held-out episode（训练中完全未见过的类型），结合 FID（视觉质量）和 **VLM-as-a-judge**（语义准确性，用 VLM 回答关于生成视频的可验证问题）：

- **Movement：** 测试动作跟随（WASD + 转头）的视觉一致性
- **Grounding：** 一个玩家转开再转回来，是否还能正确看到另一个玩家
- **Memory：** 两个玩家都转开再转回来，是否都能记住对方位置
- **Building：** 一个玩家建造，观察者视角是否正确反映了环境变化
- **Consistency：** 两个相邻玩家同时转向同侧/反侧，视野是否一致/不同

---

## Method

### Overview

**输入：** 多个玩家的历史观测帧序列 x^{<t} 和动作序列 a^{<t}（P=2 个玩家）

**输出：** 所有玩家的未来观测帧 x^t

**Pipeline：** 基于视频 Diffusion Transformer (DiT)，用 Flow Matching 训练。联合状态张量 shape 为 (B, P, T, H, W, C)。双向阶段用全局共享噪声，因果阶段用 Diffusion Forcing（每帧每玩家独立噪声）。推理时自回归 + 滑动窗口 KV cache。

### Module 1: 多人视觉交错注意力

**Motivation：** 多个玩家观察同一世界，需要交换信息以保证视角一致性。

**做法：** 将所有玩家的 visual tokens 在 sequence 维度拼接，加 Player ID embedding，共享 self-attention。其余模块（FFN、action module、cross-attention）按玩家独立执行。

**为什么 work：** Self-attention 天然可以处理变长序列中任意 token 对的关系，通过 Player ID 区分来源即可。比 channel concatenation 更灵活——后者在复杂场景下（第二玩家严重退化）表现不佳。

### Module 2: Diffusion Forcing 因果化

**Motivation：** 双向模型无法自回归生成，需要转为因果模型。

**做法：** 每帧每玩家采样独立噪声等级 σ_{p,t}，配合滑动窗口因果注意力掩码（窗口大小 6 latent frames = 24 real frames）。

**Technical insight：** 这比 CausVid 的初始化流程（ODE regression → DMD 蒸馏）简单得多，但效果不差甚至更好。因为 few-step 去噪能力可以在后续 Self Forcing 训练中同步习得。

### Module 3: Checkpointed Self Forcing

**Motivation：** 想让 teacher 上下文比 student 长，从而提供更强监督。但 naive 滑动窗口 Self Forcing 内存不可行。

**做法：** 两步法——无梯度 rollout 缓存关键状态 → 并行 forward + 特殊 mask 重建计算图。详见 Contribution 4。

**额外收益：** 内存节省后可对 KV 做反向传播，消除原始 Self Forcing 中 KV stop-gradient 的限制。

---

## Experiment

### 资源消耗

- 双向单人阶段：v5p-128 TPU
- 其余阶段：v5p-64 TPU
- Self Forcing generator 仅 240 steps，critic 1200 steps（训练量非常小）

### 数据集 / Benchmark

**训练数据：** 自采 12.64M 帧多人 Minecraft 数据 + VPT 单人数据（Stage 1）

**评估：** 自建 5 类 held-out episode benchmark（Movement / Grounding / Memory / Building / Consistency），指标为 FID + VLM accuracy。对比方法包括 Multiverse 的 frame concatenation、无单人预训练的消融、以及 Self Forcing 各组件消融（CausVid init vs Causal FT、有无 DMD 预蒸馏、有无 KV 反向传播）。

### 关键结果

**架构对比（Table 2）：**

- Solaris 在几乎所有任务上 FID 最优（38.5 / 38.0 / 55.1 / 83.6 / 99.4）
- Building 和 Consistency 的 VLM 指标显著领先（20.8 vs 0.0，71.4 vs 49.5）
- Frame concatenation 方案在第二玩家视角严重退化
- 无单人预训练导致生成出现不自然行为（重复玩家身体、错误弹窗、退化到水下场景）

**Self Forcing 消融（Table 3）：**

- Causal FT 初始化 > ODE Regression 初始化（更简单且更好）
- DMD 预蒸馏反而有害
- KV 反向传播带来全面 FID 提升（38.5 vs 60.3），在 Building 和 Consistency 这类难任务上 VLM 分数也最高

**定性结果：** Solaris 能生成 224 帧长序列保持稳定，展现出库存追踪、天气同步、火把放置、挖矿动画、PvP 等复杂游戏动态。