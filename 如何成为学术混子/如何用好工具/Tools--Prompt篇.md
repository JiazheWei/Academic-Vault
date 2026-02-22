
# 数据流动与模型框架
让claude给你画详细的模型框架/数据流动mermadi图：

```
**Role**: 你现在的身份是深度学习系统架构师。

**Task**: 请阅读我提供的代码（特别是 forward 函数），梳理出数据流向，以及该模型框架内所有模块的拼接关系与内部组成，并用 **Mermaid** 语言绘制一个详细的 **"Tensor Flow Architecture Diagram"（张量流架构图）**。

**Requirements**:

1. **节点区分**：
    
    - 使用**矩形框**表示**数据/张量 (Tensor)**，并在框内明确写出该步骤的张量形状（例如：Input: (B, C, T, H, W) 或 Shape: {1, 16, 20, h, w}）。
        
    - 使用**圆角框**或**菱形**表示**操作/模块 (Operations)**（例如：Conv3d, VAE, Attention, Reshape, Concat）。
        
2. **数据流向**：
    
    - 用箭头连接各步骤，严格遵循代码的执行顺序。
        
    - 如果在某一步发生了 Reshape、Permute 或 View 操作，请在箭头上或操作节点中注明。
        
3. **详细程度**：
    
    - 请像这张图一样（如果不方便发图，就描述：请像论文中的架构图一样），从输入（Driving Video, Ref Image, Text）开始，一直画到最终输出。
        
    - 标注清楚 Cross-Attention 和 Self-Attention 的输入来源。
        
4. **Mermaid 样式**：
    
    - 使用 graph TD (Top-Down) 布局。
        
    - 使用 subgraph 将不同的逻辑模块（例如 Face Encoder 部分、VAE 部分、Transformer 主干部分）框起来，方便阅读。
      
输出： 输出mermaid代码块，要求文字颜色纯黑或纯白或浓度高，方便阅读
```


让claude画ascii结构图：

```
# Role
You are an expert AI Architect and Deep Learning Engineer. Your task is to analyze the provided source code repository and reverse-engineer the detailed data flow of the neural network.

# Objective
Create a comprehensive ASCII visualization representing the model's pipeline. I need to clearly understand how modules are connected, what happens inside them, and crucially, **how the Tensor Shapes change** at each step.

# Constraints & Formatting
1.  **ASCII Style**: Use standard ASCII characters (box-drawing characters like ┌ ┐ │ └ ┘ are preferred) to draw boxes around modules.
2.  **Orientation**: Use a Top-to-Bottom flow.
3.  **Tensor Notation**: At every connection line (arrow), annotate the Tensor Shape using the format `[Batch, Dim1, Dim2, ...]`.
    - If exact numbers are known from the config, use them (e.g., `[32, 768]`).
    - If dynamic, use symbolic variables (e.g., `[B, Seq_Len, H]`) and define a Legend.
4.  **Granularity**: Do not just draw "Transformer Block". Break it down into "Self-Attention -> Add & Norm -> FFN -> Add & Norm" if space permits, or detailed sub-blocks.

# Task Breakdown

## Part 1: Configuration Analysis
First, briefly analyze the configuration files (e.g., config.py/yaml) to establish the base dimensions. List the key variables:
- Batch Size (B)
- Input Resolution/Sequence Length (T/H/W)
- Hidden Dimension (D/C)
- Vocab Size (V), etc.

## Part 2: Training Flow Diagram
Draw the flow from `Input Data` to `Loss Calculation`.
- Include Data Augmentation/Preprocessing steps if visible.
- Show the Forward Pass through the layers.
- Show where the Loss is calculated and what tensors are involved (Prediction vs Label).

## Part 3: Inference/Generation Flow Diagram
Draw the flow for the Inference stage.
- Highlight how this differs from training (e.g., KV-Cache, autoregressive loops, beam search, or removal of dropout).
- Show the Input prompt/image processing and the final Output decoding.

# Example Style Reference
(Please follow this level of detail)

[Input Image]
     │
     ▼ [B, 3, 224, 224]
┌─────────────────────────────┐
│ Patch Embedding             │
│ (Conv2d: k=16, s=16)        │
└────────────┬────────────────┘
             ▼ [B, 196, 768] (Flattened)
┌────────────┴────────────────┐
│ Positional Encoding (Add)   │
└────────────┬────────────────┘
             ▼ [B, 196, 768]
      ... (Layers) ...
             ▼ [B, 196, 768]
┌────────────┴────────────────┐
│ Classifier Head (Linear)    │
└────────────┬────────────────┘
             ▼ [B, 1000]
        [Logits]

# Action
Please begin the analysis. If you need me to clarify any specific file for dimension inference, ask me first. Otherwise, produce the diagrams based on the uploaded code.
```