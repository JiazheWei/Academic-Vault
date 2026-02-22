# Abstract

## Task

Human image animation, 输入是：一张refernece image，其中包含人物，一段driving video，video中有另一个主体在运动，任务是生成一段合成视频，主角被换成reference image中的人物，但是动作和节奏与driving video中的人物动作完全一致。

## Technical challenges for previous problems

以往方法提供人物动作引导时，通常都是用2D图像例如骨骼图、深度图或渲染的SMPL网格图，但是有几个关键问题：
- 丢失关键3D时空信息，这些2D结构图像虽然能提供结构信息，但2D图像不可避免地丢弃了现实3D世界中丰富的时空运动信息。例如骨架图很难表示街舞中肢体之间的遮挡关系，与四肢的折叠。
- 当姿态以图像形式提供时，生成模型往往会倾向于逐像素地“盲目复制”这些固定形状的姿态，而不是去理解动作背后的语义 。这就导致如果image中的一个主人公体型是巨人，而driving video中的人物是正常人体型，生成的视频中巨人的身体很可能发生不正常的肢体折叠，消失，以强制适应driving video中人物的特征。

## 解决问题的关键key insight与motivation


## technical contributions



# Introduction
## task and applications


## Technical challenges for previous problems


## 解决challenge 的pipeline是什么

### contribution 1
**怎么做的？key insight是什么？**

### contribution 2
**为了解决什么问题？具体怎么做的？**



# Method

## overview

输入是什么？输出是什么？
大概pipeline的组成是什么？

## module 1
为什么能work？motivation是什么？technical challenge是什么？

## module 2
。。。。。。


# Experiment

## 资源消耗

## 数据集/bench是什么
包括计算哪些指标？每个指标如何计算？

## 结果如何


