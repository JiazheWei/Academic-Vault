# Paper Info

- **Title**: Beyond Pixel Histories: World Models with Persistent 3D State
- **Authors**: Samuel Garcin, Thomas Walker, Steven McDonagh, Tim Pearce, Hakan Bilen, Tianyu He, Kaixin Wang, Jiang Bian
- **Venue/Year**: Preprint, March 2026 (Under review)
- **Paper Link**: https://arxiv.org/abs/2603.03482
- **Code Link**: https://francelico.github.io/persist.github.io (Project Page)

```ad-tldr
PERSIST 提出了一种全新的 world model 范式：不再依赖像素历史帧作为记忆，而是维护一个**显式的、动态演化的 latent 3D 场景表示**（world-frame），结合相机模型和可微渲染，实现了具有**持久空间记忆**和**几何一致性**的交互式世界生成。在 Luanti（类 Minecraft）环境中，PERSIST 在空间记忆、3D 一致性和长时序稳定性上大幅超越 Oasis 和 WorldMem 等基线方法。
```

  

---

  

# Introduction

  

## Task and Applications

本文研究的是**交互式世界模型（Interactive World Models）**任务：模型需要根据用户的动作（如键盘/鼠标操作），实时、自回归地生成连续的视频帧，构建一个可交互的虚拟世界体验。

  

应用场景包括：

- 沉浸式交互体验（类似 AI 驱动的游戏引擎）

- 在学习到的模拟器中安全训练 embodied agents

- 数字孪生、3D 场景生成与编辑

  

## Technical Challenges

现有方法（如 Oasis、WorldMem）主要是基于像素历史的自回归生成，存在以下瓶颈：

  

1. **有限的上下文窗口导致空间记忆缺失**：模型只能看到最近 K 帧像素，一旦生成的 episode 超过这个窗口，时间一致性就会退化。当玩家回头看之前看过的区域时，模型已经"忘了"之前长什么样。

2. **3D 一致性必须从 2D 数据隐式学习**：现有方法没有显式 3D 表示，几何一致性完全依赖模型从数据中学到，这在复杂 3D 环境中非常困难。

3. **记忆检索困难**：即使像 WorldMem 那样用 key-frame retrieval 来增强记忆，随着 episode 变长，从不断增长的像素历史中检索相关信息变得越来越昂贵且不可靠。

  

## 与之前工作的区别/定位

| 维度 | 之前的方法（Oasis/WorldMem） | PERSIST |

|------|------|------|

| 记忆载体 | 像素历史帧（滑动窗口 / key-frame 检索） | 动态演化的 latent 3D world-frame |

| 3D 一致性 | 隐式从数据中学习 | 通过显式 3D 表示 + 几何投影来保证 |

| 记忆成本 | 随 episode 长度增长 | 固定成本（只维护以 agent 为中心的 3D 区域） |

| 可编辑性 | 仅支持像素级编辑 | 支持 3D 空间中的几何感知编辑 |

  

核心区别：PERSIST 受传统游戏引擎启发，将世界模拟分解为**3D 场景演化 + 相机 + 渲染**三个耦合组件，而非把所有信息都压缩在像素历史中。

  

## 解决 Challenge 的 Pipeline

  

### Contribution 1: 持久化 3D 环境表示（Persistent Environment Representation）

**解决什么问题？** 解决像素历史无法提供持久空间记忆的问题。

  

**Key insight**: 定义一个代理状态 $\tilde{s} = \langle w, c \rangle$，其中 $w$ 是以 agent 为中心的 3D world-frame（体素网格的 latent 表示），$c$ 是相机状态。world-frame 作为动态空间记忆模块持续演化，相机状态作为空间查询键（lookup key），通过几何投影从 world-frame 中检索当前视角所需的 3D 信息。

  

**具体做法**: 将 world-frame 投影到屏幕空间得到 depth-ordered 的 3D 特征栈 $W_{2D}$，作为像素生成的引导信号。

  

### Contribution 2: 三组件解耦的世界模拟框架

**解决什么问题？** 解决端到端像素生成难以保证几何一致性的问题。

  

**Key insight**: 将世界模拟分解为三个可独立训练的组件：

- **World-Frame Model** $W_\theta$：预测 3D 场景如何随时间演化

- **Camera Model** $C_\theta$：预测 agent 在场景中的视角

- **World-to-Pixel Model** $P_\theta$：将 3D 信息投影并渲染为像素（learned deferred shader）

  

### Contribution 3: 新涌现能力

**解决什么问题？** 传统 pixel-based world model 无法支持的功能。

  

**具体能力**:

- 从单张图片生成多样化的 3D 环境

- Episode 中途进行 3D 编辑（改地形、放置物体）

- 屏幕外事件的持续演化（如水流在看不到的地方蔓延，之后流到玩家面前）

  

---

  

# Method

  

## Overview

- **输入**: 初始 RGB 帧 $o_0$、初始相机状态 $c_0$（可选：初始 world-frame $w_0$），以及每一步的用户动作 $a_t$

- **输出**: 自回归生成的视频帧序列 $\{o_1, o_2, ..., o_T\}$

- **Pipeline 整体流程**:

  1. 用 3D-VAE 和 2D-VAE 分别编码 world-frame 和像素帧为 latent patches

  2. World-Frame Model 预测下一步的 3D 场景 latent $\bar{w}_t$

  3. 3D-VAE 解码得到 $w_t$

  4. Camera Model 预测当前相机参数 $c_t$

  5. 可微光栅化器将 $w_t$ 按 $c_t$ 投影为屏幕空间的 depth-ordered 3D 特征栈 $W_{2D}$

  6. Pixel Model 以 $W_{2D}$ 为引导，生成最终像素帧 $\bar{o}_t$

  7. 2D-VAE 解码得到 RGB 输出 $o_t$

  

- **模块连接关系**:

```

Action + History -> [World-Frame Model Wθ] -> w̄_t -> [3D-VAE Decode] -> w_t

                                                                          |

Action + History -> [Camera Model Cθ] -> c_t ----+                       |

                                                  |                       |

                                                  v                       v

                                            [Differentiable Rasterizer R]

                                                  |

                                                  v

                                               W_2D_t

                                                  |

                                                  v

              Action + History + W_2D -> [Pixel Model Pθ] -> ō_t -> [2D-VAE Decode] -> o_t

```

  

## Module 1: 数据预处理与 VAE 编码

  

- **这个模块做什么？** 将原始像素帧和 3D 体素网格编码为紧凑的 latent 表示。

- **Motivation**: 直接在高维原始数据上建模计算成本过高，需要先压缩到 latent 空间。

- **具体做法**:

  - **2D-VAE**: ViT 架构，将 $360 \times 640 \times 3$ 的 RGB 帧编码为 $36 \times 64 \times 16$ 的 latent patch $\bar{o}$（patch size = 10 像素），用 MSE 重建 + KL 散度训练，参数量 227M。

  - **3D-VAE**: 3D-ResNet 架构，将 $48 \times 48 \times 48 \times 2138$（2138 个体素类别）的 world-frame 编码为 $12 \times 12 \times 12 \times 48$ 的 latent $\bar{w}$（patch size = $4^3$ 体素），用交叉熵分类 + KL 散度训练，参数量 138M。

  - 动作编码为 23 维 multi-hot 向量（键盘按键 + 离散化鼠标移动）。

  

## Module 2: World-Frame Model ($W_\theta$) — 3D 场景动力学预测

  

- **这个模块做什么？** 预测以 agent 为中心的 3D 场景（world-frame）如何随时间演化。

- **Motivation / 为什么需要？** 这是 PERSIST 的核心——通过在 latent 3D 空间中建模场景动力学，提供持久的空间记忆和几何一致的引导信号。

- **Technical Challenge**: 需要在 3D 空间中做自回归生成，处理 3D spatial tokens + temporal tokens 的复杂注意力机制。

- **具体做法**:

  - 采样过程：$\bar{w}_t \sim W_\theta(\bar{w}_t | \bar{W}_{t-K}^{t-1}, A_{t-K}^t, C_{t-K-1}^{t-1}, \bar{O}_{t-K-1}^{t-1})$

  - 基于 Rectified Flow Matching 的 causal DiT backbone

  - 使用交错的 **3D spatial attention**（处理体素空间关系）、**temporal attention**（处理时间关系）和 **cross-attention**（注入像素帧信息）

  - 空间位置编码：使用每个 voxel token 质心的 XYZ 绝对坐标（而非 RoPE）

  - 动作和相机通过 MLP 嵌入，注入到 AdaLN 中

  - 像素帧通过 Plucker embeddings（编码像素 patch 到 3D 空间的投影关系）后经 cross-attention 注入

  - **关键能力**: 支持 $\bar{W} = \emptyset$ 的条件，即仅从 $\langle o_0, c_0 \rangle$ 就能生成初始 world-frame $w_0$

  - 两种配置：PERSIST-S（patch size=2, 216 spatial tokens）和 PERSIST-XL（patch size=1, 1728 spatial tokens），参数量 686M

  

- **为什么能 work？** 在 3D latent 空间中建模动力学，天然具备空间结构归纳偏置，使得模型能够追踪场景中物体的空间位置和状态变化。

  

## Module 3: Camera Model ($C_\theta$)

  

- **这个模块做什么？** 根据动作序列和 world-frame 历史，预测 agent 的相机参数。

- **Motivation**: 相机参数决定了从 3D 场景中"看到"什么，是连接 3D 场景和 2D 像素的桥梁。

- **具体做法**:

  - 相机表示为 10 维向量 $c = \langle pos, rot, fov \rangle$，其中 $pos \in \mathbb{R}^3$，$rot \in \mathbb{R}^6$（6D 连续旋转表示），$fov \in \mathbb{R}^1$

  - 采样过程：$c_t = C_\theta(C_{t-1-K_C}^{t-1}, W_{t-K_C}^t, A_{t-K_C}^t)$

  - Feed-forward Transformer，上下文窗口大小 8

  - 参数量 234M

  

## Module 4: 可微光栅化器（Differentiable Rasterizer $R$）

  

- **这个模块做什么？** 将 3D world-frame 按当前相机参数投影到屏幕空间，生成 depth-ordered 的 3D 特征栈 $W_{2D}$。

- **Motivation**: 通过几何投影而非学习来实现 3D→2D 的映射，从构造上保证几何一致性。

- **具体做法**:

  - 利用 GPU 原生三角形光栅化，将 voxel 特征分配到静态体素网格 mesh 的面上

  - 使用 **Depth Peeling** 技术生成逐像素、按深度排序的 3D 特征栈 $W_{2D} \in \mathbb{R}^{H \times W \times L \times C}$（$L$ 为最大层数）

  - Mesh 拓扑固定，仅更新顶点属性，避免重复构造 mesh

  

## Module 5: Pixel Model ($P_\theta$) — World-to-Pixel 渲染

  

- **这个模块做什么？** 以 $W_{2D}$ 为引导，生成最终的像素帧。本质上是一个 **learned deferred shader**。

- **Motivation**: $W_{2D}$ 提供了几何结构信息，但缺少纹理、光照、粒子效果等视觉细节。需要一个学习到的渲染器来补全这些信息。

- **Technical Challenge**: 既要充分利用 3D 引导信号保持几何一致，又要补全 3D 表示中缺失的视觉信息。

- **具体做法**:

  - 采样过程：$\bar{o}_t \sim P_\theta(\bar{o}_t | W_{2D,t-K}^t, A_{t-K}^t, \bar{O}_{t-K}^{t-1})$

  - Rectified Flow Model + causal DiT backbone（spatial + temporal attention）

  - $W_{2D}$ 通过 1D 卷积投影到 latent 空间后，与 $\bar{o}$ 进行 channel-wise concatenation

  - **关键设计**: 给 $W_{2D}$ 分配比 $\bar{o}$ **更多的 latent channels**（$W_{2D}$ embedder 输出 752 channels vs $\bar{o}$ embedder 输出 16 channels），强制模型以 3D latent frame 为主要信息来源

  - 上下文窗口 16 帧，参数量 460M

  

## 核心亮点深度解析

  

### 亮点 1: World-Frame 作为动态空间记忆

  

**Intuition**: 传统方法把"记忆"存在像素历史中，就像人类只靠照片来记住去过的地方。而 PERSIST 维护了一个 3D"心智地图"——world-frame 就是 agent 对周围环境的 3D 认知模型，相机参数则是"往哪看"的查询。

  

**与之前方法的关键区别**:

- Oasis：只看最近 K 帧像素，没有长期记忆

- WorldMem：从不断增长的像素历史中检索 key-frame，成本随 episode 长度线性增长

- PERSIST：维护固定大小的 3D 表示，记忆成本恒定，且信息检索通过几何投影完成（精确、高效）

  

**为什么更好？** 3D 表示天然支持空间去重（同一个物体从不同角度看到多次，在 3D 中只存储一份），且投影操作是精确的几何变换，不需要学习如何从像素中检索相关信息。

  

### 亮点 2: 通道数不对称设计强制 3D 引导

  

在 Pixel Model 中，$W_{2D}$ 被分配了 752 个 channels，而像素 latent $\bar{o}$ 只有 16 个 channels。这种极端不对称的设计迫使模型主要依赖 3D 信息来生成像素，有效地将几何一致性从 3D 空间传递到了 2D 像素空间。这是一个简单但非常有效的归纳偏置。

  

## Training

  

- **Loss Function**:

  - World-Frame Model $W_\theta$: Rectified Flow Matching / CFM objective: $\mathcal{L}(\theta) = \|V_\theta(x_\tau, \tau) - (x_0 - x_1)\|^2$

  - Pixel Model $P_\theta$: 同样的 CFM objective

  - Camera Model $C_\theta$: Feed-forward 预测（非 diffusion）

  - 2D-VAE: MSE 重建 + KL 散度（系数 $1 \times 10^{-6}$）

  - 3D-VAE: 交叉熵分类 + KL 散度（系数 $1 \times 10^{-6}$）

  

- **训练策略**:

  - **各组件独立训练**: $W_\theta$、$C_\theta$、$P_\theta$ 分别训练，推理时组合使用无需微调

  - **Diffusion Forcing**: 用于缓解 exposure bias（训练时对 context frames 加噪）

  - **噪声增强**: 训练 $W_\theta$ 时对 $\bar{O}$ 加 10% flat 随机噪声；训练 $P_\theta$ 时对 $\bar{W}$ 加 10% 噪声。这让各组件在训练时就习惯了不完美的输入，推理时组合起来不需要额外适配。

  

- **关键超参数**:

  - Flow matching 采样步数: 20

  - Denoising step schedule: $\tau_k = \frac{\eta k}{1 + (\eta - 1)k}$, $\eta = 3$

  - Context noise level: 0.02 ($W_\theta$) / 0.1 ($P_\theta$)

  - World-Frame Model 上下文窗口: 8

  - Pixel Model 上下文窗口: 16

  - Camera Model 上下文窗口: 8

  

---

  

# Experiment

  

## 资源消耗

- 论文未直接报告训练 GPU 数量和时长

- 模型参数量：

  - 2D-VAE: 227M

  - 3D-VAE: 138M

  - World-Frame Model $W_\theta$: 686M

  - Camera Model $C_\theta$: 234M

  - Pixel Model $P_\theta$: 460M

  - **总计约 1.75B 参数**

  

## 数据集 / Benchmark

- **训练环境**: Luanti（开源体素游戏引擎，类似 Minecraft），使用 Craftium 平台收集数据

- **训练数据**: ~40M 环境交互（玩家在程序化生成的世界中的游玩轨迹）

- **关键特点**: 使用**程序化生成**的多样化世界（而非单一固定地图），大幅增加了空间/时间一致性的建模难度

- **评估集**: 在 Craftium 和 MineDojo 上收集，涵盖四种行为模式：

  - Free Play（自由探索）

  - Move Forward（前进 + 周期性环顾）

  - Move Backward（后退 + 周期性环顾）

  - Orbit（绕圈 + 注视中心）

- **评估指标**: FVD（Frechet Video Distance）+ 人类评分（28 名参与者，800+ 次评估）

  

## 定量结果

  

| Method | FVD↓ | Visual Fidelity↑ | 3D Consistency↑ | Temporal Consistency↑ | Overall↑ |

|--------|------|-------------------|-----------------|----------------------|----------|

| Oasis | 706 | 2.1 | 1.9 | 1.8 | 1.9 |

| WorldMem | 596 | 1.7 | 1.7 | 1.5 | 1.5 |

| **PERSIST-S** | **209** | **2.8** | **2.7** | **2.5** | **2.6** |

| **PERSIST-XL** | **181** | **2.8** | **2.5** | **2.5** | **2.6** |

| **PERSIST-XL+w0** | **116** | **3.2** | **2.8** | **2.8** | **3.0** |

  

核心发现：

- PERSIST 所有配置在所有指标上**显著超越**基线方法

- FVD 从 596（WorldMem）降到 181（PERSIST-XL），降幅约 70%

- 提供 ground truth 初始 world-frame（+w0）进一步提升性能

- PERSIST-S 和 PERSIST-XL 人类评分差异不大，说明 3D 表示的有效性对空间分辨率具有鲁棒性

  

## 定性结果

- 600 步 episode 的可视化对比（Figure 5）显示 PERSIST 在长时序上保持了更好的环境一致性

- Oasis 和 WorldMem 在长 episode 中出现明显的场景退化和不一致

- PERSIST 支持的新能力可视化：

  - 从单张图片生成多样化 3D 环境（Figure 13）

  - Episode 中途 3D 编辑：地形修改、生物群落切换、放置树木等（Figure 6）

  - 屏幕外动态事件：洞穴中水位上升，水流到玩家面前（Figure 7）

- 2000 步 episode 中会出现一些 glitches（Figure 14），但模型能自我恢复，整体保持全局一致性

  

## Ablation Study

论文的消融分析主要体现在不同配置的对比中：

- **PERSIST-S vs PERSIST-XL**: 降低 world-frame 的空间分辨率（1728→216 spatial tokens）对人类评分影响很小，说明 3D 表示的效果不强依赖于分辨率

- **PERSIST-XL vs PERSIST-XL+w0**: 提供 ground truth 初始 3D 条件能显著提升性能，证明模型能有效利用额外的 3D 信息

- **Oasis 作为隐式消融**: Oasis 可看作去掉了 $W_{2D}$ 引导的 PERSIST，性能差距证明了 3D 引导的重要性

  

---

  

# Limitations & Future Work

  

- **作者提到的局限**:

  1. **依赖 ground truth 3D 监督训练**: 目前 PERSIST 需要从环境中获取 3D 体素网格作为训练数据，限制了在真实世界（in-the-wild）场景中的应用

  2. **Exposure bias 导致的长时序退化**: 各组件训练时用 GT 数据，推理时用自己的预测，分布不匹配导致误差累积。2000 步后 glitches 频率增加

  3. **有限的空间范围**: 当前只追踪以 agent 为中心的固定大小 3D 区域，远离的信息会丢失

  

- **作者提出的未来方向**:

  1. 利用 2D-to-3D foundation models（如 SAM 3D）生成合成 3D 标注，摆脱对 GT 3D 监督的依赖

  2. 端到端后训练（Self-Forcing 等方法）缓解 exposure bias

  3. 引入 3D memory bank 实现无限空间记忆（类似 chunk loading）

  

- **我观察到的局限/疑问**:

  1. 实验仅在体素环境（Luanti/Minecraft）中验证，体素环境有天然的离散 3D 结构，推广到连续 3D 场景（如真实世界）的难度可能更大

  2. 模型总参数量约 1.75B，各组件串行运行，推理速度和实时性未报告，实际交互体验的流畅度存疑

  3. 人类评分虽然优于基线，但绝对分数仍较低（最高 3.0/5.0），与 GT 仍有明显差距

  

# Personal Notes

- **3D 表示作为记忆载体**的思路非常有启发性：在任何需要长期空间一致性的生成任务中（如长视频生成、3D 场景漫游），都可以考虑引入显式 3D 中间表示

- **通道数不对称**的简单设计值得借鉴：通过控制信息带宽来引导模型的注意力偏好，比复杂的 loss 设计更直接有效

- **独立训练 + 噪声增强**的组合策略很实用：各模块可以独立迭代改进，不需要昂贵的端到端训练

- 未来值得探索的方向：如何将这种 persistent 3D state 的思想应用到非体素的连续 3D 场景中（如用 3D Gaussian Splatting 代替体素网格作为 world-frame）