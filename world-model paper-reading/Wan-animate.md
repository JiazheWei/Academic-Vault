主要聚焦于人物图像动画领域的几个问题：
- 人物的表情，动作不自然，面部特征细节失真严重；
- 人物与原视频中的环境格格不入，或者说环境光影不匹配。

针对这几点问题，wan-animate主要采取的措施有：
- 针对面部特征细节丢失，wan直接将人物脸部image作为输入，而不采用传统的手动定义关键信号节点，
- 做人物角色替换的时候，参考人物图像和driving video来自不同的环境，直接合成会导致环境光影不和谐。wan训了一个relightning LoRA模块，来适配参考image和driving video的光影差距。
- 对于人物做动作时的驱动适配问题，采用2D骨架表示形状，并采用姿态重定向（Pose Retargeting）在推理的时候协调骨骼长度。


# Task Definition

主要有两个：
- animate，给定character image和reference video，参考video中角色的动作，让image的character做出相同的动作，生成一段video。背景还是用character image中的背景。
- replacement, 输入和第一个范式相同，要求变成了将reference video中的人物换成character image中的人物，因此对环境背景一致性有了要求。


# Method

不妨先看下和wan相比，wan-animate的需求有什么变化。

Wan-Animate 采用 Wan-I2V 作为基础架构 。在标准的 Wan-I2V 任务中，输入由以下三个部分组成：
- **噪声潜变量（Noise Latent）：** 初始的随机噪声 。
- **条件潜变量（Conditional Latent）：** 由给定的图像（作为第一帧）与后续的零填充帧在时间维度上拼接而成 。
- **二值掩码（Binary Mask）：** 用于指示哪些帧是需要保留的（值为 1），哪些是需要生成的（值为 0）。在普通 I2V 任务中，只有第一帧的掩码设为 1 。

需要克服的几重挑战：
- 在I2V任务中，reference image只是初始第一帧起点，但是animate任务中需要将这张reference image作为参考贯穿始终。
- 长视频生成需要不断将前一段视频的最后几帧作为下一段视频的参考，提供temporal guidance。
- 框架最好能同时实现replacement 和animate两个任务，这样就不用重复训练。

为了以最小的成本解决这些问题，wan-animate修改了下wan的输入范式：

## Reference Formulation

处理保持character image在视频生成过程中始终作为监督的问题。

首先使用 Wan-VAE 将参考人物图像编码为**参考潜变量（Reference Latent）** 。为了利用 Wan 模型预训练好的帧间一致性能力，将这个参考潜变量在**时间维度**上与条件潜变量拼接，并将对应的二值掩码（Mask）设为 1 。这意味着模型在生成时会一直“盯着”这个参考帧看 。

为了生成长视频，模型会随机选择目标序列的前几帧作为**时间潜变量（Temporal Latents）** 。这些帧使用真实值（Ground-truth）作为条件输入，且掩码设为 1，从而引导模型生成在时间上连贯的后续画面 。

```ad-note
这里所指的拼接具体是怎么拼的？
- **参考图注入：** 将参考人物图像通过 VAE 编码成 **Reference Latent**。它被放在序列的最前面 。
- **时间引导注入（针对长视频）：** 从上一段生成的视频中选出最后几帧（通常是 1 或 5 帧），编码成 **Temporal Latents**，拼在参考图后面 。
- **目标序列注入：** 剩下的位置填入 **Noise Latents**（随机噪声），这是模型需要填充内容的地方 。
- **Mask（掩码）配合：** 与这些 Latent 形状完全一致的还有一个 **Binary Mask**。参考图和时间引导帧对应的 Mask 设为 **1**（代表已知，不要改动），噪声部分设为 **0**（代表未知，需要生成）
```


## Environment Formulation
这里解决如何统一replacement 和animation的问题。

对于动画模式，目标帧的掩码设置为0. 意即要求模型生成人物视频中的所有动作，表情等等。

对于替换模式，将目标帧中背景的掩码设置为1，人物的掩码设置为0.模型只在掩码为 0 的区域生成新人物，从而实现将新角色“缝合”到原视频背景中 。



## Body Control

## Face Control

本质上是非常强的脸部特征提取与数据增强。

![[Pasted image 20260114163056.png]]

- **特征提取：** 使用特定的编码器提取每帧的特征，并利用 **线性运动分解（Linear Motion Decomposition）** 将特征正交化，进一步分离表情信息 。
- **时间对齐：** 使用堆叠的 **1D 因果卷积层（Causal Convolutional Layers）** 对面部潜变量进行下采样，使其在时间序列长度上与主模型的噪声潜变量对齐 。
- **注入机制：**  
	- **专用模块：** 在 Transformer 内部设有专门的 **“Face Blocks（面部块）”** 。
    - **交叉注意力：** 采用**时间对齐的交叉注意力机制（Cross-attention）**，确保每一帧的表情特征只影响对应那一帧的生成结果 。
- **高效策略：** 并非每一层都注入。在 40 层的 Wan-14B 模型中，每隔 5 层注入一次，总共只有 8 层负责接收面部信号，从而减轻了计算负担 。


wan-animate将视频中的脸部特征翻译成了一种高维latent，然后在每一帧要求模型按照这个latent对齐人物脸部特征的生成。


## Relightning LoRA

通过IC-Light将reference image修改几个光影不同的版本，然后训练LoRA让模型依照修改光影之后的人物image生成原始光影的video。

这个LoRA专门应用在DiT的self-attn和cross-attn层。

# 原始的输入输出

driving video+人物脸部图像+人物整体图像+人物骨架

现在的：



# 与recammaster的框架不同之处

recammaster本质上在DiT block中加入camera 轨迹encoder与projector，重新训一遍self-attn层和其中的camera encoder。而wan-animate在patchif之前就构造了条件token，组成如下：

![[Pasted image 20260129170916.png|500]]

