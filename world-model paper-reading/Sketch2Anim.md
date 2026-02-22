
![[image.png]]


# Introduction
## task and applications
**任务**：自动将2D手绘故事板（Sketch Storyboards）转换为3D动画。
**应用**：辅助3D动画制作流程。传统的动画流程中，动画师使用2D故事板作为参考，通过试错法手动调整3D角色关节和轨迹，这是一项耗时且需要高专业技能的工作。该技术旨在通过自动化这一过程，降低创作门槛，提高制作效率。

## Technical challenges for previous problems
1.  **巨大的域差距（Domain Gap）**：2D草图是在像素空间中的稀疏、抽象表达，而3D运动是在4D空间（空间+时间）中的密集、精确表达。
2.  **直接从2D学习困难**：由于2D输入的模糊性和缺乏深度信息，直接基于2D输入训练条件生成模型很难生成高质量的3D运动。
3.  **多条件控制的冲突**：故事板包含关键姿势（Keypose，静态、局部约束）、关节轨迹（Trajectory，动态、全局约束）和动作词（Action Word，语义约束）。现有的条件生成方法通常难以同时有效平衡这三种不同性质的控制信号。
4.  **检索方法的局限性**：早期的基于检索的方法受限于数据库规模，难以处理复杂的自定义草图轨迹，且难以设计合适的匹配评分函数。

## 解决challenge 的pipeline是什么
文章提出了一种名为 **Sketch2Anim** 的两阶段方法：首先基于3D数据训练一个多条件运动生成器（作为2D输入的替代），然后设计一个神经映射器将2D草图输入对齐到3D特征空间，从而实现用2D输入驱动生成器。

### contribution 1: 2D-3D 神经映射器 (Neural Mapper)
**怎么做的？key insight是什么？**
*   **Key Insight**：与其尝试将2D草图显式地“提升（Lift）”为不准确的3D坐标，不如在**特征嵌入空间（Embedding Space）**中将2D输入与3D输入对齐。
*   **做法**：在训练好基于3D条件（3D关键帧和3D轨迹）的运动生成器后，冻结其3D编码器。然后训练额外的2D编码器，使其输出的特征向量与对应的3D特征向量在共享空间中尽可能接近。这样，推理时输入的2D草图特征就能无缝替代3D特征被生成器接受。

### contribution 2: 多条件运动生成器设计 (Trajectory ControlNet + Keypose Adapter)
**为了解决什么问题？具体怎么做的？**
*   **问题**：解决同时控制“静态局部姿势”和“动态全局轨迹”时的冲突问题。简单的将条件拼接或使用多个ControlNet往往会导致控制效果相互干扰或无法收敛。
*   **做法**：设计了一个**轨迹感知关键姿势适配器（Trajectory-aware Keypose Adapter）**并与**轨迹ControlNet**结合。
    *   **ControlNet** 负责处理动态的关节轨迹。
    *   **Adapter** 位于ControlNet和扩散模型之间，它接收ControlNet的残差输出，并注入关键姿势（Keypose）信息。
    *   这种设计让模型在建立全局动态模式的基础上，再对特定时间步进行局部姿势的精细修正，减少了不同条件间的干扰。

### contribution 3: 端到端的草图到动画系统
**做法**：结合上述技术，实现了第一个能够直接从2D故事板生成3D动画的扩散模型框架，并开发了一个配套的基于Blender的草图动画设计界面，支持用户绘制关键帧、轨迹和输入动作词，并能自动连接生成的片段形成完整动画。

# Method

## overview
*   **输入**：单帧或多帧草图故事板，包含：
    1.  **2D Keypose**：角色骨架的关节点（2D Joint Points）。
    2.  **2D Trajectory**：关键关节的运动轨迹线（2D Trajectory Points）。
    3.  **Action Word**：动作描述词（如 "Kick", "Jump"）。
*   **输出**：高质量的3D人体运动片段（最终融合为完整动画）。
*   **Pipeline组成**：
    1.  **多条件运动生成器（Multi-conditional Motion Generator）**：基于Latent Diffusion Model，负责在3D空间生成运动。
    2.  **2D-3D神经映射器（2D-3D Mapper）**：负责将用户输入的2D草图特征映射到生成器可理解的3D特征空间。
    3.  **推理与融合**：使用映射后的特征进行生成，并通过推理引导（Inference Guidance）和运动融合（Motion Blending）得到最终结果。

## module 1: 多条件运动生成器 (Multi-conditional Motion Generator)
**为什么能work？motivation是什么？technical challenge是什么？**
*   **Motivation**：扩散模型在条件生成方面表现出色，但需要针对特殊的草图约束进行架构调整。
*   **Technical Challenge**：如何融合Action（文本）、Trajectory（时空曲线）和Keypose（特定帧静态姿态）。Keypose强调局部静态，Trajectory强调全局动态，两者容易打架。
*   **核心设计**：
    *   **Trajectory ControlNet**：复制扩散模型的编码器作为ControlNet，专门用于提取和注入密集的轨迹控制信号。它计算出的残差特征（residuals）代表了轨迹对运动的引导。
    *   **Trajectory-aware Keypose Adapter**：这是一个轻量级的适配器模块。它不仅仅接收Keypose嵌入，还接收Trajectory ControlNet输出的残差。它在轨迹控制的基础上，计算额外的修正项来满足Keypose约束。
    *   **Grounded Action Embedding**：将Keypose的Embedding加到Action Word的Embedding上，增强语义理解。

## module 2: 2D-3D 神经映射器 (2D-3D Alignment)
**为什么能work？具体怎么做的？**
*   **Motivation**：训练数据中拥有完美的3D运动数据（3D Keypose/Trajectory），但推理时只有粗糙的2D草图。直接用2D训练模型难以收敛且效果差。
*   **做法**：
    *   **利用共享嵌入空间**：利用已经训练好的3D编码器（来自Module 1），训练新的2D编码器。
    *   **损失函数**：
        1.  **Matching Loss**：强制2D Embedding和对应的3D Embedding的距离最小化。
        2.  **Contrastive Loss (CLIP-style)**：拉近匹配的2D-3D对，推远不匹配的对。
        3.  **Reconstruction Loss (Regularizer)**：确保对齐后的2D Embedding送入扩散模型后，依然能重建出合理的噪声（即能生成合理的动作），防止过拟合于对齐任务而丧失生成能力。

```ad-info 
# 3D-2D数据集是怎么造的
- **提取 3D Ground Truth (3D GT)**：
    
    - 直接从运动数据中提取 **3D Keypose**（根据文本描述中的动词选定最代表性的一帧）。
        
    - 直接提取 **3D Trajectory**（关键关节在时间轴上的3D坐标路径）。
        
    - 这就是Module 2中3D编码器的输入。
        
- **合成 2D Sketch (2D Input)**：
    
    - **投影 (Projection)**：将上述提取的3D Keypose和3D Trajectory通过虚拟摄像机投影到2D平面上，得到基础的2D坐标。
        
    - **模拟手绘缺陷 (Simulating Sketch Imperfections)**：为了缩短“完美投影的2D线条”与“真实用户手绘草图”之间的域差距（Domain Gap），作者对投影后的2D数据进行了关键的**数据增强（Data Augmentation）**：
        
        1. **相机视角增强 (Camera View Augmentation)**：随机采样不同的相机视角（Pitch, Yaw）和缩放比例。这意味着模型需要学会将不同视角的2D草图映射到统一的规范化3D空间中。
            
        2. **关节扰动 (Joint Perturbation)**：在2D关节坐标上添加高斯噪声。这模拟了用户手绘线条时的抖动和不精确（crooked strokes）。
            
        3. **身体比例扰动 (Body Proportion Perturbation)**：随机缩放2D骨架的特定身体部位（如手臂、腿的长度）。这模拟了用户画火柴人时常见的比例失调问题（disproportional body parts）。
```

## module 3: 推理引导与融合 (Inference Guidance & Blending)
**具体怎么做的？**
*   **推理引导**：在去噪过程中，通过计算生成动作的投影轨迹与用户绘制的2D轨迹之间的误差，计算梯度并更新Latent Code，从而在像素级别进一步细化轨迹跟随精度（使用二阶优化器 L-BFGS）。
*   **运动融合**：对于故事板中的连续片段，使用确定性DDIM反转（Inversion）和线性混合来生成平滑的过渡动作，将多个片段无缝连接。

# Experiment

## 资源消耗
*   **硬件**：单张 NVIDIA RTX 4090 GPU。
*   **时间**：训练Trajectory ControlNet和Keypose Adapter约需12小时（1000 epochs）；训练2D-3D对齐映射约需3小时（100 epochs）。
*   **推理时间**：生成一个40帧的动作片段平均约需0.5秒（包含推理引导）。

## 数据集/bench是什么
*   **数据集**：**HumanML3D**。作者对其进行了预处理，从3D运动中提取对应的关键帧和轨迹，并通过投影、加噪、视角变换等数据增强手段合成“伪造”的2D草图数据用于训练映射器。
*   **对比基线 (Baselines)**：
    1.  **Motion Retrieval (TMR based)**：基于文本和草图特征检索最接近的运动。
    2.  **Lift-and-Control**：先通过网络将2D草图“提升（Lift）”为3D坐标，再输入生成器。
    3.  **Direct 2D-to-Motion**：直接使用2D坐标作为条件训练扩散模型。
*   **评价指标**：
    *   **Realism**: FID (Frechet Inception Distance), Foot Skating Ratio.
    *   **Control Accuracy**: MPJPE-2D/3D (Keypose误差), Avg. Err.-2D/3D (轨迹误差).
    *   **Text-Motion Matching**: MM Dist, R-precision.

## 结果如何
1.  **定量对比**：
    *   **Sketch2Anim** 在几乎所有指标上都显著优于基线方法。
    *   相比Direct 2D-to-Motion，FID分数大幅降低（生成质量更高），轨迹控制误差降低了约5倍。
    *   相比Lift-and-Control，证明了隐空间对齐（Embedding Alignment）比显式坐标提升（Coordinate Lifting）更有效。
2.  **定性结果**：
    *   生成的动画能够精确遵循用户绘制的复杂轨迹（如后空翻、弯腰等），同时保持自然的人体运动。
    *   对于用户手绘的粗糙、比例失调的草图具有很好的鲁棒性。
3.  **消融实验**：
    *   证明了“ControlNet + Adapter”的架构优于“Single ControlNet”或“Double ControlNet”架构，能更好地平衡局部和全局约束。
    *   证明了推理引导（Inference Guidance）能进一步提升轨迹的像素级对齐精度。
4.  **用户调研**：在与基线方法的成对比较中，用户在87%-90%的情况下认为Sketch2Anim的结果在真实感和控制准确性上更好。

