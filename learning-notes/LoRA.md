# 什么是LoRA
LoRA本身就是一种在模型自身参数之外并联一个旁路的adapter。

举一个简单例子：

原来模型参数一层是一个$1000\times1000$ 的矩阵，LoRA将这个形状拆分成$1000\times8$和$8\times1000$两个小矩阵相乘，这里的8就是rank。可训练的参数显著下降，所以LoRA节省显存。

# LoRA的参数怎么用

LoRA 的参数并不是“插入”在两层之间，而是**并联**在原有的权重矩阵旁边。

*   **原有路径：** 输入 $x$ 经过冻结的原权重 $W$，得到 $W \cdot x$。
*   **LoRA 路径：** 输入 $x$ 同时经过两个新增的小矩阵 $A$ 和 $B$（先降维再升维），得到 $B \cdot A \cdot x$。
*   **最终输出：** 将两条路径的结果相加：
    $$h = W \cdot x + \Delta W \cdot x = W \cdot x + (B \cdot A) \cdot x$$

# LoRA加在哪

最常见的添加位置包括：
*   **注意力机制（Self-Attention）部分：**
    这是 LoRA 最经典也是最常用的注入位置。
    *   $W_q$ (Query Projection)
    *   $W_v$ (Value Projection)
    *   $W_k$ (Key Projection) 和 $W_o$ (Output Projection) 也可以加，但在早期论文中通常只选 Q 和 V。
*   **前馈神经网络（FFN / MLP）部分：**
    随着研究深入（如 QLoRA 等），大家发现把 LoRA 加在 MLP 层也能显著提升效果。
    *   $W_{gate}$ (Gate Projection)
    *   $W_{up}$ (Up Projection)
    *   $W_{down}$ (Down Projection)

# 什么task适合上LoRA

从LoRA的公式中可以看出，LoRA只能接受模型见过的$x$，$x$的形状，模态必须是模型原本就认识的。

