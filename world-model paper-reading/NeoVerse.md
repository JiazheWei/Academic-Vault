[arxiv](https://arxiv.org/abs/2601.00393)
这是一种旨在利用大量自然场景（in-the-wild）单目视频来增强 4D 世界模型的框架。

---

# Introduction

## task and applications

**Task:** NeoVerse 的核心任务是**4D 重建（4D Reconstruction）**与**新轨迹视频生成（Novel-trajectory Video Generation）** 。它旨在从单目视频中构建 4D 世界模型，并能够生成符合物理和几何一致性的新视角视频。

**Applications:** 得益于其架构的通用性，NeoVerse 支持多种下游应用，包括：

- **4D 重建：** 从单目视频恢复动态 3D 场景 。

- **新视角/轨迹视频生成：** 改变相机路径生成视频 。
    
- **视频编辑（Video Editing）：** 编辑视频中的对象（如改变颜色、移除物体） 。
    
- **视频稳像（Stabilization）：** 平滑相机轨迹 。
    
- **视频超分辨率（Super-resolution）：** 生成更高分辨率的视频 。
    
- **3D 追踪（3D Tracking）：** 追踪场景中的点或物体 。
    

## Technical challenges for previous problems

目前的 4D 世界建模方法主要受限于**可扩展性（Scalability）**，具体表现为两方面 ：

1. **数据可扩展性受限（Limited Data Scalability）：**
    
    - 现有的方法（如 SynCamMaster 等）通常依赖昂贵且难以获取的**多视角动态视频**数据进行训练，难以扩展到自然场景 。
    - 有些方法（如 ViewCrafter）只能处理静态场景，无法扩展到 4D 动态场景 。
2. **训练可扩展性受限（Limited Training Scalability）：**
    - 部分方法（如 TrajectoryCrafter, FreeSim）虽然可以使用灵活的数据，但依赖**繁琐的离线预处理**（如离线深度估计、离线 Gaussian 重建），导致计算负担重、存储消耗大，且无法进行在线数据增强 。  
    - 这种离线流程阻碍了利用海量且廉价的自然场景单目视频进行大规模训练 。

## 解决challenge 的pipeline是什么

NeoVerse 的核心理念是打造一个**全流程可扩展（fully scalable）**的管道，从而利用海量的 in-the-wild 单目视频（论文使用了 1M 个视频片段）。

### contribution 1: Pose-Free Feed-Forward 4DGS Reconstruction (免位姿前馈 4DGS 重建)

- **为了解决什么问题？** 解决传统优化方法速度慢、且无法处理动态单目视频中缺乏多视角信息的问题，实现高效的在线重建。
    
- **怎么做的？** 提出了一种基于 Transformer 的前馈网络，直接从视频帧预测 4D Gaussian Splatting (4DGS) 参数 。
    
- **Key Insight:** 引入了**双向运动建模（Bidirectional Motion Modeling）**机制。不仅仅是提取帧特征，还显式地预测 $t \to t+1$ 和 $t \to t-1$ 的瞬时速度 。这使得模型能够在稀疏关键帧之间进行时间插值，大大提高了在线训练的效率 。
    

### contribution 2: Reconstruction-guided Video Generation with Online Degradation Simulation (带有在线退化模拟的重建引导视频生成)

- **为了解决什么问题？** 如何在没有 Ground Truth 新视角数据的单目视频上训练生成模型？
    
- **怎么做的？** 将重建模型作为条件输入到视频生成模型中。
    
- **Key Insight:** **在线单目退化模拟（Online Monocular Degradation Simulation）**。由于单目重建在变换视角时必然会出现遮挡空洞、边缘飞逸（flying pixels）等伪影，作者故意设计了算法（如基于可见性的剔除、平均几何滤波）来**模拟**这些退化模式 。这样，生成模型（Diffusion Model）就能学会从“崩坏”的渲染图中恢复出高质量、逼真的视频，从而利用单目视频实现自我监督训练 。
    

---

# Method

## overview

- **输入：** 一个单目视频片段（Monocular Video） 。
    
- **输出：** 动态 4D Gaussians 表示，以及生成的任意新视角视频 。
    
- **Pipeline 组成：** 主要包含两个阶段（如图 2 所示）：
    
    1. **重建阶段（Reconstruction）：** 一个免位姿的前馈网络，从视频中预测 4D 高斯原语和相机参数 。
        
    2. **生成阶段（Generation）：** 一个视频扩散 Transformer（Video Diffusion Transformer），接收重建阶段生成的（退化的）渲染图作为条件，生成高质量视频 。
        

## module 1: Pose-Free Feed-Forward 4DGS Reconstruction

**Motivation:** 现有的前馈方法（如 VGGT）缺乏对时间动态的感知。为了处理视频，模型必须理解运动 。

**Technical Details:**

- **Backbone:** 基于 VGGT ，使用 DINOv2 提取特征 。
    
- **双向运动编码（Bidirectional Motion-Encoding）：**
    
    - 将特征在时间维度错开，通过 Cross-Attention 计算前向特征（$t$ 做 Query，$t+1$ 做 Key/Value）和后向特征 。
        
    - 分别预测前向速度 $v_i^+$ 和后向速度 $v_i^-$ 。
        
- **Gaussianizing:** 直接回归预测高斯的 3D 属性（位置、旋转、缩放、不透明度、球谐系数）以及生命周期 $\tau_i$ 。
    

## module 2: Scalable Reconstruction-guided Generation

**Motivation:** 为了利用 1M+ 的单目视频训练，必须解决两个问题：1. 每一帧都做重建太慢；2. 单目视频没有新视角真值（GT）。

**Technical Details:**

1. **稀疏关键帧在线重建（Sparse Key Frames Reconstruction）：**
    
    - **做法：** 假设输入 $N$ 帧，只选取 $K$ 个关键帧进行网络推理，非关键帧的高斯状态通过预测的**双向速度**进行插值得到（Eq. 3-5）。这极大提升了训练效率。
        
2. **单目退化模拟（Monocular Degradation Simulation）：**
    
    - **Visibility-based Gaussian Culling (针对遮挡):** 在新轨迹下，剔除那些被遮挡的 Gaussians，然后渲染回原视角。这模拟了视角变化导致的空洞 。
        
    - **Average Geometry Filter (针对飞逸像素):** 对深度图应用平均滤波，并调整 Gaussian 位置。这模拟了深度不连续边缘产生的“拉丝”或畸变 。
        
    - **Training:** 生成模型接收这些“人造的劣质渲染图”作为条件 $c_{render}$，以原始清晰视频作为目标 $v_t$ 进行去噪训练 。
        

## module 3: Inference Strategies (Global Motion Tracking)

**Technical Challenge:** 在推理时，简单的逐帧预测会导致物体在静止和运动状态切换时出现不一致 。

**Technical Details:**

- **全局运动追踪（Global Motion Tracking）：** * 计算每个 Gaussian 在整个视频序列中的**最大速度幅度** $m_i$（考虑可见性）。
    
    - 根据这个全局指标将 Gaussians 划分为**静态集合 S** 和**动态集合 D** 。
        
    - **策略：** 静态集合在全时段聚合（去噪），动态集合只在局部时段聚合，防止运动漂移 。
        

---

# Experiment

## 资源消耗

- **训练资源：** 32块 NVIDIA A800 GPU 。
    
    - 第一阶段（重建）：150K iterations。
        
    - 第二阶段（生成）：50K iterations。
        
- **推理速度：**
    
    - 重建：约 0.18秒 。
        
    - 生成：约 20-28秒（取决于关键帧数量，Full frames 较慢，Sparse key frames 较快）。
        

## 数据集/bench是什么

- **训练数据集：**
    
    - **18个公开数据集**（包含 Static, Dynamic, Incomplete 3D info 等类别），如 DL3DV, RealEstate10K, Kubric 等 。
        
    - **自收集数据集：** **100万个（1M）** in-the-wild 单目视频片段，用于生成模型的训练 。
        
- **Benchmarks & Metrics:**
    
    - **静态重建：** Scannet++, VRNeRF (PSNR, SSIM, LPIPS) 。
        
    - **动态重建：** DyCheck, ADT (PSNR, SSIM, LPIPS) 。
        
    - **视频生成：** VBench (Subject Consistency, Motion Smoothness, Aesthetic Quality 等) 。
        
- **对比基线：** AnySplat, 4DGT, MonST3R, TrajectoryCrafter, ReCamMaster 等 。
    

## 结果如何

- **重建性能：** 在静态和动态 Benchmark 上均超越了 SOTA（如 4DGT 和 MonST3R）。例如在 DyCheck 上，LPIPS 从 4DGT 的 0.208 降低到 0.120 。
    
- **生成性能：**
    
    - 相比 TrajectoryCrafter，NeoVerse 生成的视频伪影（ghosting patterns）更少，画面更清晰，这归功于退化模拟策略 。
        
    - 相比 ReCamMaster，NeoVerse 具有更精确的相机轨迹控制能力 。
        
    - 在 VBench 评分中，NeoVerse 在多项指标（如 Imaging Quality, Temporal Flickering）上优于竞品 。