[DreamDojo](https://arxiv.org/abs/2602.06949)

![[image-2.png]]



# DreamDojo 论文阅读笔记

_A Generalist Robot World Model from Large-Scale Human Videos · NVIDIA 2026_

```ad-tldr
与DreamZero不同，DreamDojo的最终目的是训一个能够生成符合现实运动学规律和物理学定律视频的world model，而不直接产生policy。为了做到这一点，文章采用了力大砖飞的做法：用大量人类操作数据训world model，同时用latent continuous action 表示来解决人类操作视频缺少动作标注的问题。
```

---

## Introduction

### Task & Applications

DreamDojo 是一个通用机器人世界模型，核心任务是：给定当前状态帧与动作序列，预测未来视频帧，模拟机器人与环境交互的物理后果。主要应用场景有三个：

- **策略评估（Policy Evaluation）**：无需真实部署，用世界模型批量评估策略检查点，仿真成功率与真实成功率 Pearson r = 0.995。
- **基于模型的规划（Model-based Planning）**：推理时生成多个候选动作轨迹，通过外部价值模型选优执行，成功率提升近 2×。
- **实时遥操作（Live Teleoperation）**：以 10.81 FPS 实时预测虚拟机器人未来状态，用于数据采集前的预演与验证。

### 前人工作的技术挑战

- **数据覆盖不足**：现有机器人数据集场景单一，难以泛化到 OOD 场景，收集新轨迹需要昂贵的遥操作硬件。
- **缺乏动作标签**：互联网规模的人类视频没有动作标注，被动视频预测忽略了观测与动作之间的因果关系，模型无法学到可控的动力学。
- **动作空间统一难**：不同机器人形态与数据集的动作格式差异极大，难以用统一接口跨具身训练。
- **推理速度慢**：现有视频扩散模型采用双向注意力 + 50 步去噪，无法满足实时交互需求。

### 整体解决 Pipeline

三阶段训练流程：

1. **人类视频预训练**：在 44k 小时自中心视角人类视频上，用连续潜在动作作为代理标签进行预训练，学习通用物理知识。
2. **目标机器人后训练**：重置动作条件层，在少量目标机器人数据上微调，适配具体形态（GR-1 / G1 / AgiBot 等）。
3. **知识蒸馏**：将双向注意力教师模型蒸馏为因果注意力学生模型，4 步去噪即可达到 10.81 FPS 实时速度。

### 核心贡献

**Contribution 1：最大规模自中心人类视频数据集 DreamDojo-HV**

> 解决什么问题？机器人数据规模与多样性远不足以覆盖真实世界的物理交互分布。

怎么做的？Key insight 是：人类操作与机器人操作背后的物理规律一致，可将海量人类视频当作物理先验知识库，跨越具身差异进行迁移。具体收集了 43,827 小时众包人类日常活动视频，覆盖 6,015 种技能、1,135k 个场景，远超此前最大公开数据集 DROID（86 技能 / 564 场景）。最终数据混合总量 44,711 小时，是此前世界模型预训练最大数据集的 15 倍以上。

---

**Contribution 2：连续潜在动作（Continuous Latent Actions）作为统一代理标签**

> 解决什么问题？人类视频没有动作标注；被动视频预测无法学习动作因果性；不同具身的动作格式不统一。

怎么做的？训练一个 700M 参数的时空 Transformer VAE（潜在动作模型），输入相邻两帧 (f_t, f_{t+1})，通过信息瓶颈设计自监督地提取帧间运动向量 â_t（维度 32）。Key insight 是：该向量捕捉跨具身一致的运动语义（同一 latent action 下人手与机械臂执行相同类型操作）。实验证明性能接近有高精度设备采集 ground-truth 动作标签的理想情形。

---

**Contribution 3：面向精确动作可控性的模型架构设计**

> 解决什么问题？高维连续机器人动作难以被视频扩散模型精确跟随；因果混淆（未来动作干扰当前预测）导致可控性下降。

怎么做的？两项关键改进：（1）**相对动作变换**：将绝对关节角转为相对动作，压缩动作空间、提升跨轨迹泛化；（2）**分块动作注入（Chunked Action Injection）**：利用 WAN2.2 tokenizer 时间压缩比为 4 的先验，将 4 帧连续动作拼成 chunk，仅注入对应 latent 帧，消除因果混淆。此外引入**时间一致性损失 L_temporal**（λ=0.1）监督相邻帧预测速度差，加速动作可控性学习并减少物体形变伪影。

---

**Contribution 4：Self Forcing 蒸馏 Pipeline，实现实时自回归推理**

> 解决什么问题？教师模型双向注意力固定窗口、50 步去噪无法实时交互；长时滚动出现累积误差与上下文丢失。

怎么做的？两阶段蒸馏：**Warmup 阶段**用教师 ODE 轨迹监督学生回归；**Distillation 阶段**让学生基于自身历史输出生成，KL 散度分布匹配消除训练-推理分布差距，同时随机生成 N'>N 帧进一步模拟长时滚动。学生模型将双向注意力替换为滑动窗口因果注意力（窗口 12 帧），去噪步数缩至 4 步，获得 4× 加速（2.72 → 10.81 FPS）；多帧上下文还使学生在遮挡恢复上显著优于仅条件于单帧的教师。

---

## Method

![[image-3.png]]



```ad-tldr
采用latent action注入的方法，action为相对动作。首先训一个action vae，通过类似于信息瓶颈的设置强迫保留帮助预测下一个状态最有用的action 信息。 action embedding跟timestep embedding加在一起。
```

### Overview

- **输入**：初始条件帧 + 动作序列（机器人关节角或 latent action 向量）
- **输出**：未来视频帧序列（640×480，支持任意长度自回归续写）
- **Pipeline 组成**：潜在动作模型 → DreamDojo 世界模型（基于 Cosmos-Predict2.5 DiT）→ 自回归蒸馏学生模型

训练主要分为三个部分：在human-hv数据上的训练，在机器人数据上的后训练，以及通过self-forcing蒸馏成自回归模型。

### Module 1：潜在动作模型（Latent Action Model）

首先区分一下三种模型：普通video model，游戏world model，和机器人world model。第一种单纯接受输入生成视频；第二种如genie 2，生成的游戏动作是离散的，例如WASD和跳跃；而机器人world model需要面对现实世界高频多样连续的动作空间，并且设计接触丰富的物理交互。


**Motivation**：需要从无标注视频中提取语义一致、可跨具身迁移的动作表示，不依赖任何标注设备。技术挑战在于如何区分背景运动与主动操作动作，并让 embedding 对人手与机械臂具备相同语义。

**具体做法**：架构为时空 Transformer VAE（编码器 24 块 + 解码器 24 块，700M 参数）。编码器接收 (f_t, f_{t+1}) 并将全局特征投影到 32 维 latent；解码器以 latent + f_t 重建 f_{t+1}，通过**重建损失 + KL 散度**构造信息瓶颈，迫使动作VAE只保留最关键的信息，自动过滤无关背景变化。训练超参 β=1e-6 平衡表示容量与后训练迁移性。文章中这个值非常小，说明更重视重建质量。


### Module 2：DreamDojo 世界模型

**Motivation**：如何在大规模异构数据（人类视频 + 多具身机器人数据）上统一训练，同时保证精确的高维连续动作跟随，并让视频帧满足时间物理一致性。

**具体做法**：基础架构为 Cosmos-Predict2.5（WAN2.2 tokenizer + DiT blocks，流匹配损失）。

动作条件注入：轻量 MLP 投影 latent action，与 timestep embedding 相加，通过 AdaLN 调制每个 DiT block（最后一层初始化为 0 避免干扰预训练权重）。

叠加相对动作变换 + 分块注入 + 时间一致性损失。预训练采样比例 In-lab : EgoDex : DreamDojo-HV = 1:2:10；提供 2B 与 14B 两种规模，均在 256 块 H100 上训练 140k 步。

我们可以看到latent action的处理方式跟timestep是一样的，好处是不需要额外引入的结构和layer，坏处是Adalaynorm所能做的处理有限，比如scale, shift和gate，处理更精细复杂的动作时会丢失信息。

### Module 3：自回归蒸馏学生模型

**Motivation**：教师模型推理慢（2.72 FPS）且固定窗口，无法支持实时流式输出与长时上下文感知。

**具体做法**：将教师的双向注意力替换为滑动窗口（12 帧）因果注意力，去噪步数从 35 → 4。Warmup（10k ODE 轨迹，训练 10k 步 teacher-forcing 回归）→ Distillation（学生基于自身历史输出，KL 分布匹配，随机生成 13–49 帧取最后 13 帧计算损失，3k 步）。最终在单张 H100 达到 10.81 FPS，支持超过 1 分钟不降质的长时自回归。

---

## Experiments

### 资源消耗

- 潜在动作模型：700M 参数，256 块 H100，batch=256，训练 400k 步
- 世界模型预训练：256 块 H100，batch=1024，140k 步（2B / 14B）
- 后训练：128 块 H100，batch=512，50k 步
- 蒸馏：64 块 H100，warmup batch=256 / distill batch=64，共约 13k 步

### 数据集 / Benchmark

训练数据：In-lab（55h）+ EgoDex（829h）+ DreamDojo-HV（43,827h）+ 多具身机器人数据。

6 个评估 Benchmark（全为 OOD 场景）：In-lab Eval、EgoDex Eval、DreamDojo-HV Eval、Counterfactual Eval（反事实动作）、EgoDex-novel Eval、DreamDojo-HV-novel Eval（背景替换，Gemini 2.5 Flash 生成）。

定量指标：PSNR ↑ / SSIM ↑ / LPIPS ↓；定性指标：Physics Correctness 与 Action Following 人工偏好胜率（12 名志愿者盲评）。

### 核心结果

**① 潜在动作 vs. 其他动作条件（Table 2）**：Latent action 预训练接近甚至追上需要额外硬件采集标注的理想方案（Retargeted / MANO）；相比 action-free 预训练，EgoDex Eval PSNR 提升 19.924 → 20.344，说明因果动作信号对世界模型至关重要。

**② 数据多样性消融（Table 3）**：随数据逐步加入，所有 benchmark 持续提升；14B 模型在 DreamDojo-HV Eval 上 PSNR 达 18.924，优于 2B（18.813）和所有小数据变体。

**③ OOD 人工偏好评估（Table 4）**：DreamDojo-2B 物理正确性胜率 62.5%；DreamDojo-14B 进一步达到 73.5%（物理）和 72.55%（动作跟随）。

**④ 架构设计消融（Table 5）**：逐步加入相对动作 → 分块注入 → 时序一致性损失，反事实动作集 PSNR 从 19.448 → 20.980，每项改进均有效。

**⑤ 蒸馏（Table 6）**：4× 提速（2.72 → 10.81 FPS），长时滚动 PSNR 从 14.086 降至 13.146，略有下降但可接受；因果注意力多帧上下文显著改善遮挡后物体恢复。

**⑥ 下游应用**：策略评估 Pearson r=0.995、MMRV=0.003，可靠替代真实评估；模型规划对方差大的策略组提升 17%（超最佳单检查点），对任意策略组均约 2× 成功率提升。