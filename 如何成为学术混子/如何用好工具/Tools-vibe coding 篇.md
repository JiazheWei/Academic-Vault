首先我们要知道coding AI最大的特点是**顺从**，而**不是智能**。当你跟他说“帮我修好这个bug”的时候，它可能选择性忽略很多限制条件，又或者直接在源代码中硬编码一堆东西，或者删掉先前放的assert检查。人说话的时候习惯模糊性表达，因为人有一个默认的上下文，但是AI没有这东西，他能看到的就只有100M token的上下文窗口，挤在窗口外和没告诉他的东西，AI都不知道。

# Lazar的vibe coding
[Red note](https://www.xiaohongshu.com/explore/698d2280000000000a029d54?note_flow_source=wechat&xsec_token=CBkFGDxiSyWfkNgHnDAfNofB-ZqE_kiNF_21V4dq_jW1E=)

关键是平行宇宙和文档记忆。
## 启动想法
有一个点子，不要在一个窗口里死磕，启动新项目的时候打开至少5个标签页。第一个标签页，直接将想法碎碎念出来；第二个标签页，用文本认真地写一段prompt；第三个标签页，传张设计图；第四个标签页，找一段方向相近的开源代码交上去；第五个标签页，混合使用所有的素材。

之后就能看到几个不同的结果，整理出最终的想法。

## 实现

通过多个md文档协作。
- master plan.md：写清楚project的宏观目标，这是AI实现过程中的大方向。
- implementation plan.md: 具体的实现步骤
- task .md: 当前的任务清单

同时在prompt中要求AI在做任何事之前都先读一遍task .md和master plan.md。

## 改bug

当然不可能文档写完之后一次性就可以roll出完美的结果。当出现bug之后，首先让ai自己尝试修复，如果他自己修不好，就让他在代码中加入大量的console.log，再把输出的log内容给ai看，让自己看问题到底在哪。最后解决了问题之后，记得问一嘴：“我应该怎么问，才能让你一步就能解决这个问题？”最后将内容写到rules.md文档中。




