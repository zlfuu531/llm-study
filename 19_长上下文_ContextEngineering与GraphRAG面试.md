# 长上下文、Context Engineering 与 GraphRAG 面试

这一章面向大模型应用、RAG/Agent、系统设计、算法评测和热点追问。面试官问“为什么不直接用长上下文”“Context Engineering 和 Prompt Engineering 区别”“GraphRAG 什么时候有用”时，真正考的是你能不能把模型上下文当成一个可设计、可评估、可控成本的信息系统。

你需要能回答：

- 长上下文为什么贵，为什么还会忽略中间信息。
- RAG 和长上下文不是谁替代谁，而是怎么组合。
- Context Engineering 到底工程化在哪里。
- Context Builder 怎么做排序、压缩、去重、权限过滤和预算控制。
- GraphRAG、RAPTOR、HyDE、query rewrite 解决什么问题。
- Agent memory 怎么设计，怎么避免污染和泄露。
- 长上下文/RAG/GraphRAG 怎么评估。

如果面试官继续追 RoPE 公式、Position Interpolation、NTK-aware、YaRN、LongRoPE、Ring Attention、StreamingLLM、attention sink、LongBench/RULER/NoLiMa 或长上下文底层成本，跳到进阶专题：[41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md](41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md)。

## 一句话总览

```text
用户任务 -> 查询理解 -> 检索/记忆/工具/历史
-> 权限过滤 -> 排序/去重/压缩 -> context packaging
-> 生成/工具调用 -> 引用/校验 -> eval/bad case 回流
```

一句话背诵：

> Context Engineering 不是写更长 prompt，而是设计模型运行时的信息供应系统。长上下文能放更多信息，但成本、位置偏置、噪声和权限风险都会上升；RAG 负责选择证据，GraphRAG 负责处理实体关系和全局摘要，Context Builder 负责把检索、记忆、工具结果和用户状态压成有顺序、有预算、有权限边界的上下文。

## 长上下文为什么不是万能

长上下文的好处：

- 能处理长文档、多文档、多轮对话。
- 可以减少部分检索失败。
- 适合代码仓库、合同、报告、视频转写、会议纪要等长材料。
- 给模型更多全局信息。

问题：

- attention 成本高，prefill 慢。
- KV Cache 随上下文长度增长，decode 阶段显存压力大。
- 无关上下文会干扰答案。
- 模型可能忽略中间位置的信息，也就是 lost in the middle。
- 上下文越长，权限过滤、隐私泄露和 prompt injection 风险越大。
- 长上下文不保证证据排序正确，也不保证会引用。

面试口语：

> 长上下文只是扩大输入窗口，不等于自动选择正确信息。真实系统仍然要做检索、排序、去重、压缩、权限过滤和引用校验。否则上下文越长，成本越高，噪声越多，关键证据还可能被模型忽略。

## Lost in the Middle

Lost in the Middle 指模型在长上下文中对开头和结尾的信息更敏感，中间位置的信息更容易被忽略。

为什么会发生：

- 训练数据和位置分布偏差。
- 注意力和位置编码对长距离信息利用不均。
- prompt 组织不合理。
- 关键信息被大量噪声稀释。
- 长上下文 eval 和真实业务分布差异。

缓解方式：

- 把关键证据放在靠前或靠近问题的位置。
- 先 rerank，再 context packaging。
- 对长文档做结构化摘要和章节索引。
- 使用引用约束，让模型逐条对证据回答。
- 对多证据任务做分步检索和分步推理。
- 对上下文顺序做 ablation，评估位置敏感性。

面试口语：

> lost in the middle 说明长上下文不是越长越好。我的处理方式是先检索和 rerank，把最关键证据放到更显著的位置；长文档先按结构摘要；评估时还要测试证据在不同位置时答案是否稳定。

## RAG 和长上下文怎么组合

不要把 RAG 和长上下文对立。

| 场景 | 更适合 |
| --- | --- |
| 文档很多、更新频繁、需要权限过滤 | RAG |
| 单个文档很长，需要全局理解 | 长上下文 + 结构化摘要 |
| 多文档、多跳、实体关系复杂 | GraphRAG / RAG + 图 |
| 高风险问答，需要引用来源 | RAG + 引用校验 |
| 代码仓库/长报告问答 | 长上下文 + 检索定位 |
| 成本敏感线上服务 | RAG + 压缩 + 模型路由 |

常见组合：

```text
先 RAG 找候选证据
-> rerank 排序
-> 取 top evidence + 结构化摘要
-> 长上下文模型综合回答
```

面试口语：

> RAG 负责从大规模资料里选证据，长上下文负责在选出的证据内做综合理解。长上下文可以减少检索切碎带来的信息损失，但不能替代权限、版本、引用和评估闭环。

## Context Engineering 是什么

Prompt Engineering：

- 更关注指令怎么写。
- 例如角色、格式、few-shot、约束语。

Context Engineering：

- 更关注模型运行时拿到什么信息。
- 包括检索结果、记忆、工具返回、用户状态、历史对话、系统规则。
- 要解决选择、排序、压缩、权限、预算、缓存、评估。

对比：

| 维度 | Prompt Engineering | Context Engineering |
| --- | --- | --- |
| 核心问题 | 怎么问 | 给模型什么信息 |
| 对象 | 指令文本 | 信息供应链 |
| 典型组件 | system prompt、few-shot | retriever、memory、tool results、context builder |
| 风险 | 指令不清 | 噪声、泄露、污染、成本、过期信息 |
| 评估 | 输出格式和质量 | context recall、faithfulness、成本、权限正确性 |

面试口语：

> Prompt Engineering 更像写说明书，Context Engineering 更像设计模型的工作台。生产系统里模型看到的上下文来自检索、工具、记忆、用户状态和历史对话，怎么选、怎么排、怎么压缩、怎么防泄露，决定了系统可靠性。

## Context Builder 怎么设计

Context Builder 输入：

- 用户当前问题。
- 系统规则。
- 用户画像和权限。
- 对话历史。
- 检索文档。
- 工具结果。
- Agent memory。
- 当前任务状态。

处理流程：

```text
收集候选上下文
-> 权限过滤
-> relevance scoring
-> 去重/冲突检测
-> 排序
-> 摘要/压缩
-> token budget 分配
-> context packaging
-> 引用和 trace 记录
```

预算分配例子：

```text
system/developer rule: 10%
current query/task state: 10%
top evidence: 50%
conversation summary: 15%
tool results: 10%
reserved output budget: 5%
```

实际不一定按这个比例，但要有预算意识。

常见策略：

- 当前问题和任务状态优先。
- 高置信证据优先。
- 权限不确定的信息不进入上下文。
- 工具结果要标注来源和时间。
- 长历史先摘要，最近几轮保留原文。
- 冲突信息要显式标记，不要混在一起。
- 给每段上下文加 source id，便于引用和审计。

## Context Compression

为什么要压缩：

- token budget 有限。
- 长上下文贵。
- 无关文本会干扰。
- Agent 长任务历史会爆炸。

压缩类型：

| 类型 | 方法 | 风险 |
| --- | --- | --- |
| 抽取式 | 只保留原文关键句 | 可能漏掉隐含关系 |
| 摘要式 | 用模型总结 | 可能引入幻觉 |
| 结构化 | 表格、字段、事件、实体 | 需要解析和 schema |
| 层级式 | 章节摘要 -> 文档摘要 -> 全局摘要 | 更新和一致性复杂 |
| 查询相关压缩 | 只保留和 query 相关内容 | 依赖 query 理解 |

压缩要评估：

- 是否保留 gold evidence。
- 是否引入错误。
- 是否降低 faithfulness。
- 是否降低成本和延迟。

面试口语：

> 上下文压缩不是随便 summarize。要看压缩后 gold evidence 是否还在、是否引入新事实、答案是否仍忠实证据。高风险场景更适合抽取式或带引用的结构化压缩。

## Query Rewrite、HyDE 和 Multi-Query

Query rewrite：

- 把用户问题改写成更适合检索的形式。
- 补全省略、代词、业务别名。
- 适合多轮问答和企业术语。

HyDE：

- 先让模型生成一个 hypothetical answer/document。
- 用这个假设答案去检索。
- 适合用户问题很短、语义稀疏时。
- 风险是生成假答案导致检索偏。

Multi-query：

- 生成多个不同角度的 query。
- 提高 recall。
- 风险是召回噪声增加、成本上升。

面试口语：

> query rewrite 是把问题变得更可检索，HyDE 是用假设答案增强语义召回，multi-query 是从多个角度扩召回。它们都要用 Recall@k、MRR、噪声率和端到端答案质量评估，不能只看召回数量变多。

## GraphRAG 什么时候有用

普通 RAG 更像找片段，GraphRAG 更像找实体和关系。

适合：

- 多跳问答。
- 跨文档关系。
- 企业组织、金融、法律、风控、医学等实体关系复杂场景。
- 需要全局摘要和社区发现。
- 问题不是“某段文字在哪里”，而是“多个实体之间有什么关系”。

不适合：

- 文档很少。
- 关系不复杂。
- 普通 chunk 检索已经能解决。
- 图谱更新成本过高。
- 实体抽取质量很差。

GraphRAG 典型流程：

```text
文档 -> 实体/关系抽取 -> 图构建
-> 社区发现 / 层级摘要
-> query 到实体/社区/子图
-> 图证据 + 原文证据 -> answer
```

面试口语：

> GraphRAG 适合实体关系和全局问题，不是所有 RAG 都要上图。它的价值是把跨文档关系显式建模，让模型能围绕实体、边和社区摘要回答多跳问题。代价是抽取错误、图更新、权限和评估都更复杂。

## GraphRAG 错了会怎样

错误来源：

- 实体抽取错。
- 同名实体没有消歧。
- 关系边抽错。
- 社区划分不合理。
- 摘要引入幻觉。
- 图谱更新滞后。
- 权限过滤只过滤原文，没过滤图节点或摘要。

影响：

- 多跳路径错。
- 答案引用错。
- 全局摘要偏。
- 错误关系被放大。
- 用户更难发现错误，因为图答案看起来很“结构化”。

治理：

- entity linking / canonicalization。
- 图节点和边保留 provenance。
- 图证据回链到原文。
- 图谱版本化。
- 对实体、关系、社区摘要分别评估。
- 高风险关系人工抽检。

## RAPTOR 和层级检索

RAPTOR 思路：

```text
把文档切块
-> 对块聚类
-> 对每个聚类生成摘要
-> 递归形成树
-> 检索时可检索叶子块和高层摘要
```

适合：

- 长文档和多文档总结。
- 问题既可能需要细节，也可能需要全局摘要。
- 普通 flat chunk 检索丢全局结构。

风险：

- 摘要层可能幻觉。
- 上层摘要过粗导致细节丢失。
- 更新成本和版本管理复杂。

面试口语：

> RAPTOR 不是图谱，而是层级摘要检索。它通过树状摘要同时支持细粒度 chunk 和全局信息。适合长文档总结和跨章节问题，但摘要质量和版本管理要重点评估。

## Agent Memory 怎么设计

记忆类型：

| 类型 | 内容 | 生命周期 |
| --- | --- | --- |
| short-term memory | 当前对话原文 | 当前 session |
| working memory | 当前任务状态、计划、工具结果 | 当前任务 |
| long-term memory | 用户偏好、历史事实、稳定知识 | 跨 session |
| retrieval memory | 可检索历史、文档、经验 | 按需召回 |

写入原则：

- 用户明确表达的偏好可以写。
- 稳定事实可以写。
- 一次性临时信息不要写长期记忆。
- 不确定、敏感、越权、被注入的信息不要写。
- 写入前可以让用户确认。

读取原则：

- 和当前任务相关才读。
- 读取结果要标注来源和时间。
- 长期记忆要允许删除和更正。
- 记忆进入 prompt 前要权限和安全过滤。

面试口语：

> Agent memory 不是把聊天记录全塞进向量库。要区分短期、工作、长期和检索记忆；写入要有准入规则，读取要按任务相关性和权限过滤，还要允许用户更正和删除，避免错误记忆长期污染系统。

## 长上下文和 Context Engineering 怎么评估

长上下文评估：

- LongBench / RULER 类 benchmark 看长文本能力。
- needle-in-a-haystack 看定位能力。
- lost-in-the-middle 测证据位置敏感性。
- 业务长文档 eval 看真实任务。

Context Engineering 指标：

- context recall：关键证据是否进入上下文。
- context precision：上下文有多少是有用的。
- token cost：输入 token 和输出 token。
- faithfulness：答案是否忠实上下文。
- citation accuracy：引用是否支持答案。
- permission correctness：上下文是否只包含有权限信息。
- latency：检索、压缩、生成总耗时。

评估方式：

```text
baseline: naive top-k RAG
+ rerank
+ query rewrite
+ compression
+ long context packaging
+ GraphRAG / RAPTOR
```

不要只看最终答案平均分，要分桶看：

- 单文档 / 多文档。
- 短问题 / 长问题。
- 单跳 / 多跳。
- 有答案 / 无答案。
- 权限敏感 / 非敏感。
- 证据在开头 / 中间 / 结尾。

## 项目里怎么讲

如果你做的是知识库或 Agent 项目，可以这样讲：

```text
我没有直接把所有检索结果塞给模型，而是做了 Context Builder。
先根据用户问题和权限召回候选证据，再 rerank、去重、压缩，
把高置信证据、最近任务状态、必要历史和工具结果按 token budget 组合。
长文档场景下会先做结构化摘要，关系复杂场景尝试 GraphRAG，
最后用 context recall、faithfulness、引用正确率、P95 和成本评估。
```

## 面试常见高压问题

### Q1：为什么不直接把所有文档塞进长上下文？

因为成本高、延迟高、KV Cache 压力大，而且长上下文也会受 lost in the middle 和噪声干扰影响。更重要的是，直接塞文档没有权限过滤、版本管理、引用和评估闭环。生产系统通常用 RAG 选择证据，再用长上下文做综合。

### Q2：Context Engineering 和 RAG 有什么关系？

RAG 是 Context Engineering 的一部分。Context Engineering 还包括历史对话、用户状态、工具结果、Agent memory、权限过滤、上下文压缩、token budget 和上下文评估。RAG 解决找证据，Context Engineering 解决模型运行时看到什么。

### Q3：GraphRAG 比普通 RAG 好在哪里？

GraphRAG 对实体关系、多跳问题和全局摘要更强。普通 RAG 找 chunk，GraphRAG 能找实体、关系、社区和路径。但它更复杂，依赖实体抽取、关系抽取、图更新和图证据评估，不适合简单问答硬上。

### Q4：上下文压缩会不会导致答案错？

会。摘要可能丢掉关键证据或引入新事实。所以压缩后要评估 gold evidence 是否保留、答案是否忠实、引用是否正确。高风险场景优先保留原文证据或抽取式压缩。

### Q5：Agent 记忆怎么防污染？

写入前做准入：只写稳定、明确、允许保存的信息；敏感信息、注入内容、工具错误结果不写长期记忆。读取时按相关性和权限过滤，给记忆加来源和时间，允许用户删除和更正，并对记忆命中做日志审计。

## 面试前背诵版

长上下文不是 RAG 的替代品。长上下文扩大窗口，但成本、KV Cache、lost-in-the-middle、噪声和权限风险都会上升；RAG 负责选择证据，GraphRAG 负责实体关系和全局摘要，Context Engineering 负责把检索、工具、记忆、用户状态和历史对话组织成可控上下文。生产系统要有 Context Builder：权限过滤、排序、去重、压缩、预算分配、引用和 trace。评估时看 context recall、precision、faithfulness、citation accuracy、权限正确性、延迟和成本。

## 本轮参考来源

- Microsoft GraphRAG：https://www.microsoft.com/en-us/research/project/graphrag/
- GraphRAG 论文：https://arxiv.org/abs/2404.16130
- Lost in the Middle：https://arxiv.org/abs/2307.03172
- RAPTOR 论文：https://arxiv.org/abs/2401.18059
- LongBench 论文：https://arxiv.org/abs/2308.14508
- RULER 论文：https://arxiv.org/abs/2404.06654
- Ring Attention 论文：https://arxiv.org/abs/2310.01889
- LongRoPE 论文：https://arxiv.org/abs/2402.13753
- HyDE 论文：https://arxiv.org/abs/2212.10496
- 本地外部资料中的 `LLM-Agent-Interview-Guide` RAG/Agent/长上下文资料和 `TorchLeet` Ring Attention 题目
