首先定义了一个新的任务：Promptable Visual Segmentation。与传统的semi-supervised visual segmentation不同，后者约定在视频的第一帧一定会给出所需要分割的对象的mask，PVS不要求这一点，而是要求从视频的任一一帧开始都可以对对象进行分割。

![[Pasted image 20260108142334.png]]

核心是引入记忆机制处理视频这种带时序信息的数据。

# Overall Architecture

图像/video送入image encoder之后与memory bank进行交互得到带历史条件的条件化特征，接着**条件化特征**与prompt encoder的输出共同送入mask decoder得到该视频帧的分割掩码，最后memory encoder接受当前帧的预测结果与图像的无条件特征更新memory bank。


# Image encoder

采用MAE预训练的Hiera vit。

# Memory attention

由L个相同的transformer block组合成。数据流在每一个模块里经过三个流程：
- 首先进行self-attn, 聚合当前帧的上下文信息。
- 然后经过self-attn之后的feature作为query，从memory bank中取出的memory作为key 与value，做cross-attn。
- 最后一个FFN进行非线性特征变换。

# Prompt encoder

这里的prompt指的是用户给出的交互式指令，例如bbox，点击（positive or negative）和mask。其中bbox与点击属于sparse prompt，mask属于dense prompt。

SAM2的prompt encoder沿用了sam1的版本，没有用复杂的transformer结构。

针对sparse prompt，bbox和点击会被转换成位置编码，位置编码的计算过程：

```ad-note
**具体计算步骤如下（基于 SAM 的设计）：**

1. 坐标归一化：
    
    首先将用户输入的点或框的坐标 $(x, y)$ 归一化到 $[0, 1]$ 区间。
    
2. 随机高斯映射（Random Gaussian Projection）：
    
    使用一个固定的（非学习的）高斯随机矩阵 $B \in \mathbb{R}^{2 \times k}$ 对坐标进行线性映射。
    
    $$\text{coords} \times B$$
    
3. 正弦/余弦变换（Sinusoidal Transformation）：
    
    将映射后的结果通过正弦和余弦函数，生成高维向量。
    
    $$\text{PE}(x, y) = [\sin(2\pi \cdot \text{coords} \cdot B), \cos(2\pi \cdot \text{coords} \cdot B)]$$
    
    这种方法能让模型更好地感知高频的空间细节（即精确的位置信息）。
    
4. 与类型嵌入相加：
    
    计算出的位置编码向量最后会加上一个 “学习到的类型嵌入”（Learned Embedding）。
    
    - 例如：如果这是一个“正向点击”，就加上“正向点击向量”；如果是“框的左上角”，就加上“左上角向量”。
```

针对dense prompt，采用卷积结构，将带掩码的feature下采样成和latent一样的形状，然后与原始image latent相加。

# Mask decoder

接受两拨token：memory bank经过cross-attn之后输出的image latent，还有prompt latent。

mask decoder首先将output token与prompt token拼在一起，做self-attn, 再和image-attn 轮流当query做cross-attn。

```ad-info
### Output Tokens 到底是什么？
**一句话总结：它是模型自带的“空白填空题”占位符，不是用户输入的，也不是简单的空白 token，而是可学习的参数（Learnable Embeddings）。**

- **它的本质**：
    
    - 这些 Token 是模型参数的一部分。在模型初始化时，它们就是一组随机初始化的向量，随着训练过程不断更新（学习）。
        
    - 在推理（Inference）时，无论你输入什么图片或提示，这些 Output Tokens 的**初始值**都是固定的、一样的。
        
- **它的位置**：
    
    - 它们被**拼接到**用户输入的 Prompt Tokens（点击、框的 Embedding）后面，组成一个完整的序列输入到 Mask Decoder 中。
        
    - 输入序列 = `[Prompt Tokens (用户输入), Output Tokens (模型自带)]`
        
- **它的作用（类似于 DETR 的 Object Queries）**：
    
    - 你可以把它们想象成是模型手中的“探针”或者“提问者”。
        
    - **Mask Token** 的潜台词是：“根据前面那些提示（点击/框）和图像特征，最终的分割掩码应该长什么样？请把特征浓缩到我身上。”
        
    - **IoU Token** 的潜台词是：“生成的这个掩码质量大概有多少分？”
        
    - **Occlusion Token**（SAM 2 新增）的潜台词是：“这个物体现在是被遮挡了吗？”
        
- **变化过程**：
    
    - 刚进去时，它们只包含通用的“提问”语义。
        
    - 经过 Transformer 层层处理后，它们通过注意力机制“吸取”了图像和提示的信息。
        
    - 最后，这些吸饱了信息的 Token 会被拿出来，分别送入不同的 MLP（多层感知机）头，映射成最终的掩码图、分数和遮挡判断。
```

整体流程：

```ad-summary
- **第一步：Token Self-Attention（Token 自玩）**
    
    - **Prompt Tokens** 和 **Output Tokens** 之间互相交流。
        
    - 比如：如果你给了两个点击，模型需要知道这两个点是指同一个物体还是不同部分。
        
- **第二步：Token-to-Image Cross-Attention（Token 查图）—— 第一路**
    
    - **Query**: Tokens
        
    - **Key/Value**: Image Embeddings
        
    - **含义**：这是最直观的一步。提示 Token 去图像里“找”对应的特征。比如提示是“左上角的点”，Token 就会去图像左上角提取视觉特征更新自己。
        
- **第三步：MLP 更新 Token**
    
    - 对 Token 进行一次非线性变换。
        
- **第四步：Image-to-Token Cross-Attention（图查 Token）—— 第二路（这才是 Two-way 的精髓）**
    
    - **Query**: Image Embeddings
        
    - **Key/Value**: Tokens
        
    - **含义**：这是反直觉的一步。**图像特征本身被更新了**。图像特征会根据提示信息调整自己。
        
    - **为什么需要这一步？** 如果没有这一步，图像特征就是死的（虽然经过了 Image Encoder，但不知道你要分什么）。有了这一步，图像特征会“意识到”用户的关注点。例如，如果你点击了“猫头”，图像特征中关于“猫头”的区域可能会被强化，而背景区域被抑制。
```


mask decoder输出的token会被分别送入几个head得到不同的输出：
- 分割掩码，是最关键的输出，通过将 Transformer 输出的 Mask Token 转换为动态权重，与上采样后的高分辨率图像特征图进行点乘得到 。
- IoU 分数 (IoU Scores)，表示模型对预测结果的置信度，每一个mask都有自己的置信度。
- 遮挡分数 (Occlusion Score) —— **SAM 2 新增**，一个标量分数（Score），指示**目标对象在当前帧是否可见**（即是否存在）。由专门的 Occlusion token 经过一个 MLP 头预测得到 。
- 对象指针 (Object Pointers) —— **SAM 2 新增**， 一个轻量级的向量，它代表了当前帧中该对象的**高层语义信息**。这个向量会被存入**记忆库 (Memory Bank)**，作为后续帧检索目标的线索，通常直接是用来生成最终掩码的那个mask token。

