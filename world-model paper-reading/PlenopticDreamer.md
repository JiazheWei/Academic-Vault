# Abstract

## Task
相机控制的生成式视频重渲染,给定一个源视频和一组目标相机轨迹，目标是合成对应这些新轨迹的新视角视频 。生成的视频不仅需要保留源视频的内容保真度，还必须在多个视点之间保持严格的时空一致性（即如同在真实的 3D 场景中拍摄一样） 。

## Technical challenges for previous problems
- **多视角一致性缺失：** 现有方法主要在单视角设置下成功，但在多视角场景中表现挣扎，无法重建场景的整体一致性表示 。
- **幻觉区域的时空错位：** 视频生成模型本质具有随机性，且缺乏长距离的空间记忆。当生成源视角不可见的区域（hallucinated regions）时，不同视角的生成结果往往不一致，导致几何错位 。
- **独立生成的局限性：** 之前的“单次生成（single-shot）”方法独立地生成每个视图，没有利用先前生成的上下文来约束后续生成，导致视图之间缺乏同步 .

>task自己本身做的是给定**1**个video与**N**个相机轨迹产生**N**个新视角的视频，产生后面的video时候可以将之前产生的video加进来作为上下文。


## 解决问题的关键key insight与motivation
将视频生成视为对场景**全光函数（Plenoptic Function**的离散采样，因此必须通过显式的记忆机制来强制执行时空一致性 。
- 自回归记忆流（Autoregressive Memory）：将已产生的video作为condition，一个接一个的送入context中，一次只产一个video。
- 基于 3D 几何的检索（Geometry-guided Retrieval）：做了一个检索机制，从已经有的video context中通过3D视场检索得到最相关的video clip，作为条件产video。
- 基于 3D 几何的检索（Geometry-guided Retrieval）：自回归方式容易产生误差，并没有仅依赖完美的 Ground Truth 训练，而是让模型在训练中通过“自条件（self-conditioning）”策略去适应和修正自己生成的有瑕疵的输入。

## technical contributions
如上。

# Introduction
## task and applications

- **Task:** 相机控制的生成式视频重渲染。即沿着任意相机轨迹合成新视频，同时保留原始内容 。
- **Applications:** 沉浸式内容创作（immersive content creation）和具身智能（embodied AI）

## Technical challenges for previous problems

- **单视角局限性：** 现有方法（如 ReCamMaster, TrajectoryCrafter）主要在单视角设置下有效，难以应对需要重建场景整体表示的多视角场景 。
- **幻觉区域不一致：** 在源视图不可见的区域（unseen regions），无法保持一致的时空幻觉 。
- **根本原因：** 扩散模型固有的随机性（stochasticity），加上有限的长程空间记忆，导致了跨视角的几何错位和视图不同步 。
## Pipeline

提出 **PlenopticDreamer** 框架，显式地强制执行时空记忆以实现一致的场景生成
### Autoregressive Multi-Camera Generator with 3D FOV Retrieval（自回归多相机视频生成器与 3D FOV 检索机制）

- **具体做法：** 提出了 PlenopticDreamer 框架，采用“多入单出（multi-in-single-out）”的自回归架构 。通过引入 **3D 视场（Field-of-View, FOV）** 视频检索机制，计算空间共视性（spatial co-visibility），从而在每一步生成时能从记忆库中智能检索出最相关的历史“视频-相机对”作为条件 。
- **Key Insight：** 这一设计将视频生成视为对场景全光函数（plenoptic function）的采样 。通过检索几何上重叠的视角作为强约束，模型能够显式地维护长期时空记忆，从而在生成新视角（特别是源视频不可见的幻觉区域）时，保证与历史视角在几何和内容上的高度一致性 。
### 渐进式上下文缩放与自条件训练

大量上下文时训练不稳定，并且自回归范式容易误差积累 。

- **渐进式上下文缩放（Progressive Context-Scaling）：** 在训练初期使用较少的上下文视频（如 1 个），随着训练稳定逐渐增加数量直至目标大小（如 4 个），从而加速并稳定模型的收敛 。
- **自条件训练（Self-Conditioned Training）：** 在微调阶段，用模型自己生成的（带有瑕疵的）合成视频替换真实的 Ground Truth 作为条件输入。使模型学会了如何从不完美的历史输入中恢复高质量结果，增强了推理时的鲁棒性 。

### 长视频条件机制

一方面为了突破时间限制生成长视频，一方面测试framework能不能泛华到其他领域测试。

- **长视频条件机制（Long-Video Conditioning）：** 提出了一种重叠分块策略，在生成后续视频块时保留前一块的部分帧作为条件，确保了块与块之间的时间连续性 。
- **基准验证（Benchmarks）：** 在 **Basic**（自然场景）和 **Agibot**（机器人操作）基准上进行了广泛实验。

# Method
实现相对简单。
## overview
![[Pasted image 20260118160925.png]]
- 首先自建了一个memory bank，存储所有已经生成好的video和每个video对应的camera 轨迹。
- 将生成不同相机轨迹的video过程重新建模成一个自回归过程，generate 新视角video的时候从过往video中检索出FOV最高的几组加入到context中作为condition。
- 训练策略比较新。



# Experiment

## 资源消耗

## 数据集/bench是什么
包括计算哪些指标？每个指标如何计算？

## 结果如何


