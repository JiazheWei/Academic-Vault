# Paper Info

- **Title**: LiveWorld: Simulating Out-of-Sight Dynamics in Generative Video World Models
- **Authors**: Zicheng Duan*, Jiatong Xia*, Zeyu Zhang*, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen†, Xinyu Zhang‡, Lingqiao Liu†‡
- **Venue/Year**: arXiv preprint, March 2026（in submission）
- **Affiliations**: Adelaide University, ANU, Monash University, Zhejiang University, University of Auckland
- **Paper Link**: https://arxiv.org/abs/2603.07145
- **Code Link**: https://github.com/ZichengDuan/LiveWorld （Coming Soon）
- **Project Page**: https://zichengduan.github.io/LiveWorld/index.html

## TL;DR

这篇论文首次提出并形式化了视频世界模型中的 **"视野外动态缺失"（out-of-sight dynamics）** 问题——当物体离开观察者视野后，其状态会被"冻结"而无法持续演化。作者提出了 **LiveWorld** 框架，通过将世界演化与观测渲染解耦，利用 **Monitor 机制** 自主模拟视野外动态实体的时间演进，实现了真正持续演化的 4D 动态世界模拟。同时提出了专用 benchmark **LiveBench**，在多项指标上大幅超越现有方法。

---

# Introduction

## Task and Applications

本文研究的是 **生成式视频世界模型（Generative Video World Models）** 任务，即构建一个能够根据当前上下文和控制输入（如相机轨迹、文本提示）来预测未来世界状态的生成式系统。

**实际应用场景**包括：

- 智能体训练（Agent Training）
- 决策制定（Decision-making）
- 大规模合成环境生成
- 自动驾驶仿真
- 游戏交互（如 Matrix-Game、GameCraft 等）

## Technical Challenges

现有视频世界模型存在一个 **根本性缺陷**：

**世界演化与相机渲染被隐式地耦合在一起**。具体来说：

1. 现有方法将整个 4D 世界压缩为一系列 2D 观测快照（通过 KV cache 或 3D 空间记忆存储历史帧）
2. 一旦动态实体（如正在吃东西的狗）离开观察者的视野，其状态就被"冻结"在最后一次被观测到的时间戳
3. 当观察者重新回到该区域时，看到的依然是过去的静态快照，而不是应该发生的后续变化
4. 例如：观察者转头离开正在吃食物的狗，回头时应该看到狗已经吃完了，但现有方法会显示狗还在原来的姿态

这就是论文定义的 **"out-of-sight dynamics"（视野外动态缺失）** 问题。

## 与之前工作的区别/定位

- **KV cache 方法**（如 CausVID、Self-Forcing 等）：只存储 2D 视觉快照，本质是 static-world assumption
- **显式 3D 空间记忆方法**（如 Spatia、WorldPlay 等）：虽然有 3D 几何一致性，但存储的只是**观测时刻**的静态 3D 结构，时间维度被忽略
- **LiveWorld 的核心区别**：首次明确**解耦世界演化（Evolution）和观测渲染（Rendering）**，将世界状态分解为静态 3D 背景 + 动态实体，并通过 Monitor 机制自主维护视野外的时间演进

## 解决 Challenge 的 Pipeline

### Contribution 1: 问题形式化

**解决什么问题？** 现有视频世界模型社区并未意识到"out-of-sight dynamics"这一问题的存在和严重性。

**Key insight**：世界演化 $\mathcal{E}$ 和观测渲染 $\mathcal{R}$ 是两个**本质不同的过程**，应被显式分离，而非隐式耦合在一个黑盒生成器中。

**具体做法**：形式化定义了 out-of-sight dynamics 问题，提出了结构化的世界状态近似方案，将 4D 世界分解为静态 3D 背景 $\mathcal{M}_{static}$ 和动态实体 $\mathcal{M}_{dyn,t}$。

### Contribution 2: LiveWorld 框架

**解决什么问题？** 如何在有限计算资源下维护视野外的动态实体演化？

**Key insight**：不需要建模整个不可见世界——只需要在动态实体被检测到的位置放置虚拟"Monitor"，用 Monitor 来自主推进该实体的时间演化。

**具体做法**：设计了 Monitor 驱动的动态演化系统 + 统一的 state-conditioned video diffusion backbone，实现演化引擎和渲染器的共享架构。

### Contribution 3: LiveBench 评测基准

**解决什么问题？** 缺少专门评估 out-of-sight dynamics 的 benchmark。

**具体做法**：构建了包含 100 个场景、400 条评测序列的 LiveBench，设计了涵盖空间记忆、实体身份保持、事件演进一致性的多维度评估指标体系。

---

# Method

## Overview

- **输入**：初始场景图像、相机轨迹 $C^{cam}$、文本提示 $C^{text}$
    
- **输出**：具有持续 out-of-sight 动态的长时视频序列
    
- **Pipeline 整体流程**：
    
    1. 根据历史帧检测动态实体，注册 Monitor
    2. 静态背景通过 SLAM 累积为 3D 点云
    3. Monitor 利用 Evolution Engine 自主推进动态实体的时间演化
    4. 将静态背景 + 演化后的动态实体投影到目标相机轨迹
    5. State-aware Renderer 生成最终观测视频
- **模块连接关系**：
    

```
Input Scene
    ├── [Static Background] → SLAM → 3D Point Cloud (M_static)
    │                                        ↓
    └── [Dynamic Entity Detection] → Monitor Registration
                                           ↓
                                    Evolution Engine (G_evo)
                                           ↓
                                    4D Dynamic Point Cloud (M_dyn,t)
                                           ↓
                    M_static + M_dyn,t → Projection → State Projection P
                                                            ↓
                                    Appearance References + P → Renderer (G_render) → Output Frames
```

## Module 1: Problem Formulation & Structured World-State Approximation

- **这个模块做什么？** 将不可行的完整 4D 世界状态维护，转化为可计算的结构化近似。
    
- **Motivation**：直接维护全局显式 4D 世界状态在计算上不可行，需要一种简化但有效的近似。
    
- **具体怎么做？**
    
    现有方法的公式： $$F_t = \mathcal{V}_\theta(\text{F}_{<t}, C_t)$$ 其中 $\text{F}_{<t}$ 只是历史 2D 观测快照，世界演化和渲染被耦合在 $\mathcal{V}_\theta$ 中。
    
    LiveWorld 恢复显式分离： $$\mathcal{W}_t = \mathcal{E}(\mathcal{W}_{<t}), \quad F_t = \mathcal{R}(\mathcal{W}_t, C_t)$$
    
    **结构化世界状态近似**： $$\mathcal{W}_t \approx {\mathcal{M}_{static}, \mathcal{M}_{dyn,t}}$$
    
    - $\mathcal{M}_{static}$：沿时间轴投影得到的 **时间不变** 的 3D 静态背景
    - $\mathcal{M}_{dyn,t}$：通过演化函数 $G_\theta^{evo}$ 维护的 **动态实体**的当前状态
- **Key insight**：静态场景在时间上不变，而时间变化集中在稀疏的动态实体上。这种分解使得只需要对少数动态实体维护时间维度。
    

## Module 2: Unified State-Conditioned Video Backbone ($G_\theta$)

- **这个模块做什么？** 提供一个统一的架构，可以同时作为 Evolution Engine 和 Renderer 使用。
    
- **Motivation**：演化引擎和渲染器本质上都是"根据之前的状态和控制信号生成未来视觉内容"，共享相同的生成范式。
    
- **具体怎么做？**
    
    基于 **Wan2.1-14B-T2V** 的 latent Video Diffusion Transformer (DiT) backbone，加入双注入条件设计：
    
    1. **State Adapter**（基于 ControlNet 初始化自 Wan2.1-VACE-14B）：
        - 输入像素级投影张量 $\mathbf{P}_{t:t+T} \in \mathbb{R}^{T \times H \times W \times C}$
        - 提供严格的像素级几何引导
    2. **LoRA 参数**（注册在 DiT backbone 上）：
        - 接收拼接的历史参考帧（temporal anchor + appearance anchor）
        - 提供细粒度视觉纹理
    
    统一接口： $$V_{t:t+T} = G_\theta(\mathbf{P}_{t:t+T}, \mathbf{A}, C_{t:t+T}^{text})$$
    
- **Key insight**：通过灵活配置输入投影 $\mathbf{P}$ 和外观参考 $\mathbf{A}$，同一个 backbone 可以无缝扮演两个不同角色——无需训练两个独立模型。
    

![[image-6.png|820x740|820]]


![[image-7.png]]


## Module 3: Monitor-Driven Dynamic Evolution System

- **这个模块做什么？** 自主维护视野外动态实体的时间演化。
    
- **Motivation**：不需要对整个不可见世界建模，只需关注那些包含活跃动态实体的局部区域。
    
- **Technical Challenge**：
    
    1. 如何检测和注册需要追踪的动态实体？
    2. 如何在实体离开视野后继续推进其时间演化？
    3. 如何处理异步出现的新实体？
- **具体怎么做？**
    
    **Monitor 注册**：
    
    - 使用 VLM（Qwen3-VL）+ 分割器（SAM 3）检测前一轮生成帧中的动态实体
    - 若检测到新的动态实体且与已有 Monitor 覆盖区域重叠度低于阈值，则在该位置注册新 Monitor
    - 最大活跃 Monitor 数量 $M=3$，超限时丢弃距观察者最远的
    
    **Monitor 驱动演化**：
    
    - 将 Evolution Engine 实例化为固定视角（Monitor 锚定位置）
    - 输入锚定帧的静态背景作为 state projection：$\mathbf{P}^{bg_{anc}}_{t:t+T}$
    - 输入裁剪的实体参考作为 appearance reference：$\mathbf{A}^{entity}$
    - 输入动作描述文本作为 text prompt
    - 生成局部视频 $v^{monitor}_{t:t+T}$
    
    $$v_{t:t+T}^{monitor} = G_\theta^{evo}(\mathbf{P}^{bg_{anc}}_{t:t+T}, \mathbf{A}^{entity}, C_{t:t+T}^{text})$$
    
    **异步时间同步**：若新实体在一轮生成的中途出现（时刻 $t_a$），先用 $G_\theta^{evo}$ 合成从 $t_a$ 到 $t$ 的缺失帧，与全局时间线对齐。
    
    **动态记忆整合**：利用已知的 Monitor 锚定位姿和逐帧深度，将 2D 动态前景反投影回 3D 世界空间，形成 **时间演化的 4D Monitor 点云**。
    
- **Key insight**：Monitor 是一个"驻守在原地的虚拟观察者"，它在主观察者离开后继续"看着"那个区域，自主推进那里的时间演化。
    

## Module 4: State-Aware Renderer ($G_\theta^{render}$)

- **这个模块做什么？** 将更新后的世界状态（静态+动态）渲染为观察者视角的视频。
    
- **Motivation**：渲染器需要同时利用静态 3D 背景的几何一致性和动态实体的最新演化状态。
    
- **具体怎么做？**
    
    1. **世界状态投影**：将 $\mathcal{M}_{static}$ 和 $\mathcal{M}_{dyn,t:t+T}$ 投影到目标相机轨迹： $$\mathbf{P}_{t:t+T} = \text{Proj}({\mathcal{M}_{static}, \mathcal{M}_{dyn,t:t+T}}, C_{t:t+T}^{cam})$$
        
    2. **参考帧检索**：
        
        - Temporal anchor：最新的前一帧 $F_{t-1}$（运动连续性）
        - Appearance anchor：若重访已探索区域，检索最早的历史帧（最少视觉漂移）
    3. **渲染生成**： $$F_{t:t+T} = G_\theta^{render}(\mathbf{P}_{t:t+T}^{global}, \mathbf{A}^{history}, C_{t:t+T}^{text})$$
        

## 核心亮点深度解析

### 亮点 1：World Evolution vs. Observation Rendering 的解耦

- **Intuition**：现实世界中，世界在你不看的时候也在持续运转。但现有视频世界模型把"世界如何变化"和"你看到什么"混为一体。
- **与之前方法的关键区别**：之前的方法（无论是 KV cache 还是 3D 空间记忆）本质上都在做 "**记忆你看过什么**"，而 LiveWorld 在做 "**模拟世界如何演化**"。
- **为什么更好？** 因为解耦后，世界状态可以独立于观察者持续更新，从根本上解决了"out-of-sight dynamics"问题。

### 亮点 2：Monitor 机制的设计

- **Intuition**：与其维护整个不可见世界（计算不可行），不如只在"有事情发生"的地方放一个"摄像头"（Monitor）。
- **与之前方法的关键区别**：之前的方法要么没有处理动态变化，要么试图用 KV cache 隐式记忆（但无法推进时间）。Monitor 提供了一种**显式、可控、可扩展**的动态维护方式。
- **为什么更好？** 计算量可控（最多 M 个 Monitor），同时能覆盖场景中的主要动态事件。且 Monitor 和 Renderer 共享同一个 backbone，避免了训练多个模型的开销。

## Training

- **数据集**：论文在 Appendix 中提到了训练数据构建方法（PDF 中未完整展示）。
    
- **Loss Function**： 使用标准的 **Flow Matching** 目标函数： $$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, \boldsymbol{\epsilon}, t} \left| \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{P}, \mathbf{A}, C) - (\boldsymbol{\epsilon} - \mathbf{z}_0) \right|^2$$ 其中 $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\boldsymbol{\epsilon}$。Loss 只在目标帧上计算；前导帧和参考帧作为 clean conditioning tokens。
    
- **训练策略**：**两阶段训练**
    
    - **Stage 1**：训练 State Adapter，backbone 冻结（10k steps）
    - **Stage 2**：冻结 adapter，fine-tune backbone attention 层上的 LoRA 模块（5k steps）
    - 所有 encoder 全程冻结
    - Text prompt drop probability: 0.2
- **关键超参数**：
    
    - 基础模型：Wan2.1-14B-T2V
    - State adapter：初始化自 Wan2.1-VACE-14B
    - LoRA rank: 64
    - Learning rate: 1×10⁻⁴ with cosine decay
    - Global batch size: 16
    - 最大活跃 Monitor 数: M=3

---

# Experiment

## 资源消耗

- **训练**：16 × NVIDIA H200 GPU，bf16 FSDP
- **Stage 1**: 10k steps
- **Stage 2**: 5k steps
- 推理速度和总训练时间论文未明确给出
- 模型参数量：基于 Wan2.1-14B（~14B 参数）

## 数据集 / Benchmark

### LiveBench（自建 benchmark）

- 100 个多样场景图像（使用 VLM + T2I 模型生成的高质量 480×832 图像）
- 4 种轨迹变体/场景 → 共 400 条评测序列
- 每条序列 4 轮、260 帧（16 FPS）
- **两种重访模式**：
    - Same-Pose Revisit（A→B→A→B→A）
    - Different-Pose Revisit（A→B→C）
- 评估指标覆盖：
    - 空间记忆：PSNR↑, SSIM↑, LPIPS↓（背景一致性）
    - 动态实体几何：Chamfer Distance↓（3D 点云对齐）
    - 实体身份：DINOf_g↑（DINO v2 前景特征匹配）
    - 事件演进：VQA-Acc↑（VLM 驱动的视频 QA 准确率）
    - 时序平滑：CLIP_F↑（相邻帧 CLIP 相似度）

## 定量结果

### 核心对比结果（Same Pose Revisit）

|方法|PSNRbg↑|DINOfg↑|VQA-Acc↑|
|---|---|---|---|
|Matrix-Game-2.0|16.3/16.1|0.335/0.198|7.7/5.0|
|GameCraft-1.0|17.6/16.0|0.527/0.262|20.1/10.3|
|Spatia|20.1/19.0|0.440/0.416|19.2/14.7|
|**LiveWorld**|**20.1/20.0**|**0.760/0.721**|**59.1/54.6**|

（数值格式为 1st revisit / 2nd revisit）

**关键发现**：

1. **背景维护**：LiveWorld 和 Spatia 都受益于显式 3D 点云，背景指标相当。MG-2 和 GC-1 在长时域下背景严重崩溃。
2. **动态实体保持**：LiveWorld 是唯一能有效保持 out-of-sight 动态物体的方法。CDfg 从 Spatia 的 4.031 降至 0.068。
3. **事件演进**：VQA-Acc 从 baseline 的最高 ~20% 提升至 ~59%，说明解耦架构确实能成功完成文本脚本描述的事件。
4. **不同角度重访**：优势更加明显，baseline 在新角度重访时进一步退化。

### Late-appear Event Revisiting（用户研究）

|指标|w/o Event Evo|LiveWorld|
|---|---|---|
|E1 Event Succ.|2%|42%|
|E2 Event Succ.|3%|35%|
|Full Succ.（两个事件同时成功）|0%|26%|

去掉 Evolution Engine 后，系统完全无法维持多事件并行演化。

## 定性结果

论文 Fig. 5 展示了 260 帧长时域生成的可视化对比：

- **MG-2 和 GC-1**：相机多次往返后背景严重失真，前景实体消失或错位
- **Spatia**：背景较好但前景实体在重访时被冻结在原始状态
- **LiveWorld**：背景稳定、前景实体随文本脚本持续演化，重访时状态与时间进度一致

Fig. 6 展示了 late-appearing entity 的同步演化示例，证明 Renderer 和各 Monitor 之间能完美时间同步。

## Ablation Study

|消融设置|PSNRbg↑|DINOfg↑|VQA-Acc↑|
|---|---|---|---|
|w/o Event Evolution|20.0/19.0|0.425/0.401|18.5/14.0|
|w/o Spatial Memory|17.5/16.9|0.395/0.285|12.4/8.5|
|w/o Reference Frames|18.5/17.1|0.615/0.410|38.5/22.2|
|**Full LiveWorld**|**20.1/19.0**|**0.760/0.721**|**59.1/54.6**|

**核心发现**：

1. **去掉 Evolution Engine**：背景尚可但前景指标和事件完成率暴跌——这就退化为纯相机控制模型
2. **去掉 Spatial Memory**：相机控制失败、严重鬼影和空间抖动，背景指标大幅下降
3. **去掉 Reference Frames**：缺失密集视觉纹理导致背景不稳定，引发级联时序崩溃，尤其在第二次长时域重访时指标严重恶化

三个组件缺一不可，但 **Evolution Engine 对核心创新（out-of-sight dynamics）贡献最大**。

---

# Limitations & Future Work

- **作者提到的局限**：
    
    - 最大活跃 Monitor 数量限制为 M=3，对于非常复杂的多实体场景可能不足
    - 文本到视频的随机性有时无法成功触发新实体的出现（E2 Presence 仅 70%）
    - Full Succ.（两个并行事件同时成功）只有 26%，说明多事件同步仍有提升空间
- **我观察到的局限/疑问**：
    
    1. **动态实体的交互**：论文似乎只处理独立演化的实体，未讨论实体之间的交互（如两个人对话、物体碰撞）
    2. **文本 prompt 的依赖**：动态演化依赖文本描述实体应执行的动作，这在开放世界场景中可能难以自动化
    3. **计算开销**：基于 14B 参数模型 + 多个 Monitor 并行生成，推理成本可能较高，但论文未报告具体延迟
    4. **深度估计质量**：将 2D 演化结果反投影到 3D 依赖于逐帧深度估计的准确性，误差会在长时域累积
    5. **评估的 ground-truth 缺失**：由于没有真正的 ground-truth 视频，评估主要依赖 VLM-based QA，这本身的可靠性有待验证
    6. **静态/动态分离的鲁棒性**：现实场景中静态和动态的边界并不总是清晰的（如树叶摇摆、水流），如何处理这些"半动态"元素？

# Personal Notes

- **核心启发**：将一个复杂的生成问题**解耦为多个更简单的子问题**是一种非常有效的研究范式。LiveWorld 将 4D 世界模拟解耦为"静态 3D 累积 + 局部动态演化 + 状态投影渲染"三步，每步都更可控。
- **Monitor 机制**的设计思想可借鉴到其他需要维护"不在当前视野内的信息"的场景，如大规模 3D 场景生成、长时序视频理解、甚至多智能体系统中的局部状态维护。
- **统一 backbone 思路**值得学习：同一个 backbone 通过不同输入配置扮演不同角色，大幅减少训练/维护成本。
- **值得深入探索的方向**：
    1. 将 Monitor 机制扩展到支持实体间交互的场景
    2. 用视觉语言模型自动生成动态实体的演化脚本，减少对手动文本 prompt 的依赖
    3. 探索更轻量的 backbone（如蒸馏版本）以支持实时交互
    4. 将该思路迁移到 3D 世界模型或 embodied AI 中
    5. 