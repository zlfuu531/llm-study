# 简历项目打磨与 STAR 话术

更新时间：2026-06-18

这份文档解决一个问题：简历上的大模型项目怎么写，才能在面试里经得起追问。2026 的 AI 应用、RAG、Agent、推理部署面试越来越强调“生产信号”，不是“我用了 LangChain / vLLM / 向量数据库”。

## 项目质量标准

一个能撑住面试的大模型项目，至少要讲清 8 件事：

| 证据 | 面试官想确认什么 |
| --- | --- |
| 场景 | 你解决的是真问题，不是套壳 demo |
| baseline | 你知道原方案为什么不够 |
| 方案 | 你能拆系统链路，不只会堆框架 |
| 我的工作 | 你真实参与了关键模块 |
| 指标 | 你能证明效果、延迟或成本变化 |
| bad case | 你真的调过系统，不是只跑通 |
| 上线/评估 | 你有工程意识和风险意识 |
| 复盘 | 你知道局限和下一步 |

一句话标准：

> 简历写项目结果，面试讲项目证据。

## 简历 bullet 公式

### 公式 1：动作 + 技术 + 指标

```text
使用 A 技术解决 B 问题，将 C 指标从 x 提升/降低到 y。
```

例子：

```text
设计知识库问答 RAG 链路，结合结构化 chunk、BM25+向量混合召回和 cross-encoder rerank，将离线 Recall@5 从 72% 提升到 86%，并通过引用校验降低无证据回答比例。
```

### 公式 2：问题 + 取舍 + 结果

```text
针对 A 场景中 B 难点，对比 C/D 方案后选择 E，并通过 F 评估验证效果。
```

例子：

```text
针对企业文档中编号、专有名词和语义改写混杂导致的召回不稳问题，对比纯向量检索和 BM25 后采用混合召回，并用人工标注 query-doc 集评估 Recall@k 和 MRR。
```

### 公式 3：工程能力 + 风险控制

```text
为 A 模块补充 B 机制，解决 C 风险，支持 D 监控/回滚。
```

例子：

```text
为 Agent 工具调用链路增加参数 schema 校验、最大步数、权限白名单和 trace 日志，降低工具误调用与死循环风险，支持 bad case 回放和人工审计。
```

## STAR 话术

STAR 不要写成作文，要压缩成面试能说的结构：

```text
S - Situation：什么业务场景，原来有什么痛点
T - Task：你的目标是什么，成功标准是什么
A - Action：你做了哪些关键动作，为什么这样做
R - Result：指标、case、上线结果、复盘
```

2 分钟版：

```text
这个项目面向 X 场景，原来的问题是 Y。我的目标是把 Z 指标做好，同时控制延迟和成本。
我主要负责 A/B/C 三块：第一是……第二是……第三是……
方案上我没有只依赖 prompt，而是先建立评估集，再优化检索/工具/模型/服务链路。
最后在离线评估中，指标从 x 到 y；bad case 主要集中在 z，后续我会通过 m 继续改。
```

8 分钟版：

```text
背景 -> baseline -> 指标定义 -> 架构 -> 我的模块 -> 技术选择
-> 实验/ablation -> bad case -> 上线/监控 -> 复盘
```

## RAG 项目怎么写

简历可写：

```text
构建面向 X 业务的知识库问答系统，完成文档解析、结构化 chunk、向量索引、BM25+embedding 混合召回、rerank、证据引用和拒答策略；构造 query-doc-answer 评估集，分层评估 Recall@k、faithfulness、引用正确率和端到端解决率。
```

面试追问准备：

- 为什么不用微调？
- chunk 为什么这样切？
- embedding 模型怎么选？
- rerank 为什么有效？
- 如果召回为空怎么办？
- 如果召回互相矛盾怎么办？
- 如何防 prompt injection？
- 线上 bad case 怎么回流？

项目证据：

- 至少 20-50 条 query-doc 评估样本。
- 至少 3 类 bad case：召回漏、rerank 错、生成幻觉。
- 至少 2 个指标：Recall@k、faithfulness、引用正确率、P95 延迟、单次成本。

## Agent 项目怎么写

简历可写：

```text
实现面向 X 任务的工具调用 Agent，设计任务状态、tool schema、参数校验、权限白名单、最大步数、失败重试和 trace 日志；构造任务级评估集，统计 task success、tool call accuracy、argument accuracy、平均步骤数和安全违规率。
```

面试追问准备：

- 为什么用 Agent，不用固定 workflow？
- tool schema 怎么设计？
- 参数错了怎么办？
- 多工具冲突怎么选？
- memory 怎么设计？
- 怎么防死循环？
- 高风险工具怎么人工确认？
- 怎么评估 Agent 不是 demo？

项目证据：

- 工具调用 trace。
- 失败任务样例。
- 权限和审计设计。
- 任务成功率和平均步骤数。

## 微调 / 对齐项目怎么写

简历可写：

```text
基于 X 场景数据完成指令微调/偏好优化流程，负责数据清洗、脱敏、chat template、loss mask、LoRA/QLoRA 配置和验证集评估；对比 prompt/RAG/SFT 方案，分析过拟合、格式遵循和安全退化 bad case。
```

面试追问准备：

- 为什么必须微调？
- 数据怎么清洗和去重？
- loss mask 怎么做？
- LoRA rank、alpha 怎么选？
- DPO 为什么需要 reference model？
- GRPO reward 怎么设计？
- 微调后幻觉或安全变差怎么办？

项目证据：

- 数据规模和样例。
- 训练/验证/测试划分。
- loss 曲线或验证集结果。
- 失败样本和修复策略。

## 推理部署项目怎么写

简历可写：

```text
参与大模型推理服务优化，围绕 TTFT、TPOT、tokens/s、P95/P99、显存和单次成本建立监控；通过量化、prefix cache、continuous batching、上下文长度控制、模型路由和限流降级优化在线服务稳定性。
```

面试追问准备：

- prefill 和 decode 的瓶颈分别是什么？
- TTFT 高怎么排查？
- TPOT 高怎么排查？
- KV Cache 显存怎么估算？
- PagedAttention 和 FlashAttention 区别？
- 量化后如何评估质量下降？
- 高并发如何限流、降级和熔断？

项目证据：

- 压测配置。
- 吞吐/延迟/显存指标。
- 降级策略。
- 灰度和回滚方案。

## VLM / 多模态项目怎么写

简历可写：

```text
实现面向 X 场景的多模态理解/问答系统，完成图像解析、OCR、视觉特征接入、文本检索和答案生成；针对 OCR 错误、图文对齐和多图长上下文构造评估集，分析幻觉和定位错误 bad case。
```

面试追问准备：

- 视觉 encoder 怎么接 LLM？
- projector/adapter 解决什么？
- OCR 错误怎么定位？
- 多图/视频上下文为什么贵？
- 多模态幻觉怎么评估？

## 简历项目禁忌

不要写：

```text
负责 RAG 模块开发。
使用 LangChain 实现智能问答。
调用 OpenAI API 完成对话机器人。
使用向量数据库提高效果。
```

要改成：

```text
负责知识库问答系统的检索与生成链路，完成结构化 chunk、混合召回、rerank、引用校验和拒答策略；构造离线评估集，按 Recall@k、faithfulness、引用正确率和 P95 延迟定位 bad case。
```

## 数字怎么写才安全

可以写：

- 离线评估集规模：`构造 120 条 query-doc-answer 样本`。
- 相对提升：`Recall@5 从 72% 到 86%`。
- 工程指标：`P95 延迟从 3.2s 降到 2.1s`。
- 成本指标：`平均输入 token 降低 35%`。

不要编：

- 没有证据的 DAU、GMV、转化率。
- “提升 90%”但说不清分母。
- “达到工业级”这种空话。

如果是课程项目或个人项目：

```text
这是个人/课程项目，没有真实线上流量，所以我用离线评估集和压测结果证明效果。评估集来自公开文档和手工构造问题，主要看 Recall@k、引用正确率、P95 延迟和 bad case 类型。
```

## 一页项目卡

每个项目最后压成一页：

```text
项目名：
岗位主线：算法 / 应用 / 推理 / 搜广推 / 多模态
一句话：
业务场景：
baseline：
我的工作：
核心技术：
指标：
bad case：
上线/评估：
如果重做：
最怕被问：
对应复习文件：
```

## 本轮参考来源

- AgentGuide 求职项目与公司面经：https://github.com/adongwanai/AgentGuide
- AI Engineer Portfolio Projects 2026：https://jobsbyculture.com/blog/ai-engineer-portfolio-projects-2026
- 2026 AI Agent 岗位求职项目讨论：https://gitcode.csdn.net/6a0c335a10ee7a33f273b587.html
- AI Agent 工程师面试指南：https://www.cnblogs.com/limingqi/p/20068242
- AgentGuide 公司面试案例：https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/12-company-interview-cases.md
