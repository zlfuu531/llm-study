# 完整学习章 03：RAG、Agent 与 MCP

## 你学完要能做到什么

这一章面向大模型应用岗、Agent 工程岗和算法岗项目追问。学完后你要能做到：

- 讲清 RAG 全链路。
- 回答 RAG 和微调怎么选。
- 解释 chunk、embedding、hybrid retrieval、rerank。
- 说明 RAG 为什么还会幻觉，以及怎么评估。
- 讲清 Agent、ReAct、Function Calling、MCP。
- 把自己的 RAG/Agent 项目讲成系统，而不是“用了 LangChain”。

一句话总览：

> RAG 是给模型找证据，Agent 是让模型调用工具做事，MCP 是把工具和上下文标准化接入模型应用。

## 1. RAG 的本质

RAG，即 Retrieval-Augmented Generation，检索增强生成。

它把问题拆成两件事：

1. 检索系统负责找到证据。
2. 大模型负责基于证据组织答案。

RAG 不是简单的“向量数据库 + prompt”。它真正难的地方在：

- 文档解析是否干净。
- chunk 是否完整。
- 检索是否召回正确证据。
- rerank 是否把答案片段排前面。
- prompt 是否约束模型基于证据回答。
- 评估是否能发现幻觉。

**面试答案：RAG 解决什么问题？**  
RAG 解决模型参数知识不够新、不够私有、不可溯源的问题。企业知识库、政策制度、产品文档、客服问答都适合 RAG。

## 2. RAG 和微调怎么选

| 场景 | 优先方案 |
| --- | --- |
| 知识频繁更新 | RAG |
| 企业私有知识 | RAG |
| 需要引用来源 | RAG |
| 需要固定输出格式 | 微调 |
| 模型不会某种任务风格 | 微调 |
| 模型缺少领域表达习惯 | 微调 |
| 既要知识又要稳定格式 | RAG + 微调 |

**面试答案：**  
RAG 更适合补知识，微调更适合改行为。如果问题是“模型不知道公司制度”，用 RAG；如果问题是“模型知道但总是不按格式答”，用 SFT/LoRA。

## 3. RAG 全链路

离线建库：

```text
文档采集 -> 解析清洗 -> chunk 切分 -> embedding -> 建索引 -> 版本管理
```

在线问答：

```text
query -> rewrite -> retrieve -> rerank -> context packaging -> generation -> citation -> evaluation
```

你在面试中要讲得像系统：

> 我们离线侧先做文档解析和结构化切分，保留标题、页码、来源等 metadata，然后用 embedding 建向量索引，同时保留 BM25 做混合召回。在线侧用户问题会先做 query rewrite，然后走向量召回和关键词召回，候选片段再用 reranker 精排，最后把 top 证据拼进 prompt，让模型基于证据回答并返回引用。

## 4. chunk 为什么重要

chunk 太小：

- 信息不完整。
- 答案跨 chunk 时召回不到完整证据。
- 模型拿到片段后无法回答。

chunk 太大：

- 向量语义被稀释。
- 检索不精准。
- prompt 成本上升。
- 噪声变多。

好的 chunk：

- 是完整语义单元。
- 保留标题层级。
- 带 metadata。
- 对表格、代码、PDF 特殊处理。

**面试答案：chunk size 怎么设置？**  
不是拍脑袋。先根据文档结构按标题、段落或语义切分，再用 Recall@k、答案命中率和 bad case 调整。chunk 的目标是包含能回答一个问题的最小完整语义单元。

**overlap 的作用：**  
缓解切分边界问题，防止答案刚好被切断。overlap 太大也会导致冗余和召回重复。

## 5. embedding、召回与混合检索

Dense retrieval：

- 用 embedding 做语义召回。
- 能处理同义表达。

Sparse retrieval：

- 如 BM25。
- 擅长关键词、编号、专有名词、错误码。

Hybrid retrieval：

- dense + sparse 融合。
- 企业 RAG 常见。

**面试答案：为什么要混合召回？**  
单独向量召回可能漏掉精确词，单独 BM25 不懂语义。混合召回同时覆盖语义相似和关键词精确匹配，适合企业文档。

**embedding 模型怎么选？**  
看语言、领域、文本长度、向量维度、推理速度和业务评测。中文场景优先选中文或多语言效果好的模型，并用自己的 query-doc 测试集评估 Recall@k。

## 6. rerank

召回阶段像海选，rerank 阶段像复试。

Bi-encoder 召回：

- query 和 doc 分别编码。
- 快。
- 交互弱。

Cross-encoder rerank：

- query 和 doc 一起输入模型。
- 判断更细。
- 更慢。

**面试答案：rerank 的作用是什么？**  
召回负责捞得全，rerank 负责排得准。rerank 能把真正包含答案的片段排到前面，减少无关上下文进入 prompt。

## 7. RAG 为什么还会幻觉

检索侧：

- 没召回正确文档。
- 召回相关但无答案的片段。
- 文档过时或冲突。
- chunk 切断上下文。

排序侧：

- rerank 排错。
- top-k 过少。

生成侧：

- prompt 没要求基于证据。
- 模型使用常识补全。
- 上下文太长导致忽略关键证据。

系统侧：

- 没有引用校验。
- 没有拒答策略。
- bad case 没回流。

**治理：**

- 提升召回。
- 加 rerank。
- 基于证据回答。
- 引用来源。
- 证据不足时拒答。
- 答案-证据一致性检查。
- bad case 驱动迭代。

## 8. RAG 评估

检索指标：

- Recall@k
- Precision@k
- MRR
- NDCG

上下文指标：

- context relevance
- context precision
- 是否包含答案

生成指标：

- faithfulness
- answer relevance
- citation correctness
- hallucination rate

业务指标：

- 解决率
- 转人工率
- 人工满意度
- P95 延迟
- 单次成本

**面试答案：如何评估 RAG？**  
要分层评估：先看检索是否找到 gold 文档，再看上下文是否包含答案，最后看生成是否忠于证据。只看最终回答“感觉对”不够。

长上下文、Context Builder、上下文压缩、GraphRAG、RAPTOR 和 Agent memory 的完整深挖见：[../19_长上下文_ContextEngineering与GraphRAG面试.md](../19_长上下文_ContextEngineering与GraphRAG面试.md)。

Embedding 模型选择、BM25、hybrid retrieval、RRF、HNSW/IVF/PQ、reranker、multi-vector、向量库权限和召回低排查见：[../22_Embedding_Reranker与向量检索面试.md](../22_Embedding_Reranker与向量检索面试.md)。

## 9. GraphRAG

普通 RAG 更像“找片段”，GraphRAG 更像“找实体关系”。

适合：

- 多跳问题。
- 实体关系复杂。
- 跨文档总结。
- 法规、金融、组织关系、知识图谱。

**面试答案：GraphRAG 适合什么场景？**  
当问题需要跨多个实体和关系推理时，普通 chunk 检索可能找不到完整链路，GraphRAG 通过图结构表达实体和关系，更适合多跳推理和全局总结。

## 10. Agent 的本质

普通 Chatbot：

```text
用户问题 -> 模型回答
```

Agent：

```text
目标 -> 思考 -> 工具调用 -> 观察结果 -> 更新状态 -> 下一步 -> 完成任务
```

**面试答案：Agent 和 Chatbot 区别？**  
Chatbot 是回答器，Agent 是任务执行器。Agent 能规划、调用工具、读取环境反馈，并根据状态决定下一步。

## 11. ReAct 与 Plan-and-Execute

ReAct：

```text
Thought -> Action -> Observation -> Thought -> ... -> Final
```

适合：

- 需要边查边想。
- 需要工具反馈。
- 路径不完全固定。

Plan-and-Execute：

1. Planner 拆任务。
2. Executor 执行。
3. 必要时 re-plan。

适合：

- 长任务。
- 多步骤任务。
- 需要明确阶段的任务。

**追问：什么时候不用 Agent？**  
流程固定、规则明确、风险高、强一致性要求高、延迟敏感时，优先工作流，不要硬上 Agent。

## 12. Function Calling / Tool Calling

Function Calling 的本质：

- 模型输出函数名和参数。
- 外部系统执行函数。
- 结果返回给模型。

模型不真正执行函数。

工具 schema 要写清：

- 工具用途。
- 参数类型。
- 必填字段。
- 枚举范围。
- 返回格式。
- 错误情况。

**面试答案：工具调用失败怎么办？**  
参数错误就让模型修正或让用户补充；工具超时就重试/降级；权限不足就拒绝；结果为空就换检索策略或说明无结果；高风险操作需要人工确认。

## 13. MCP

MCP，即 Model Context Protocol。

它解决：

> 模型应用如何以标准方式连接外部工具、数据源和上下文。

Function Calling 更关注模型如何表达“我要调用哪个工具”；MCP 更关注工具和上下文如何标准化接入模型应用。

**面试答案：MCP 和 Function Calling 的关系？**  
不是同一层。Function Calling 是模型结构化表达工具调用意图，MCP 是工具、资源和上下文暴露给模型应用的协议。

## 14. Context Engineering

Prompt Engineering：

- 怎么写提示词。

Context Engineering：

- 给模型什么上下文。
- 检索结果怎么选。
- 工具列表怎么暴露。
- 记忆怎么管理。
- 长上下文怎么压缩。
- 成本和缓存怎么控制。

**面试答案：为什么 Context Engineering 变热？**  
RAG 和 Agent 进入生产后，关键不只是 prompt 怎么写，而是如何稳定、低成本地给模型提供正确上下文。

## 15. Agent 稳定性和评估

Agent 常见失败：

- 工具选错。
- 参数填错。
- 结果没读懂。
- 死循环。
- 状态丢失。
- 权限越界。
- 成本失控。

工程措施：

- 清晰 tool schema。
- 参数校验。
- 最大步数。
- 状态机约束。
- 失败重试。
- 高风险操作确认。
- trace 日志。

评估指标：

- task success rate。
- tool call accuracy。
- argument accuracy。
- average steps。
- cost per task。
- latency。
- safety violation rate。

## 16. 2026 面试新增重点：Harness Engineering

2026 的 Agent 面试里，越来越少只问“ReAct 是什么”，更多会问“你怎么把 Agent 做成一个可控系统”。这类问题可以用 Harness Engineering 来回答。

Harness 可以理解成包住大模型的工程外壳：

```text
用户目标
  -> 任务理解
  -> 规划/路由
  -> 上下文组装
  -> 工具选择
  -> 参数校验
  -> 工具执行
  -> 观察结果
  -> 状态更新
  -> 终止/回退/人工确认
  -> 结果生成
```

它和 Prompt Engineering 的区别：

- Prompt Engineering：主要关心“怎么提示模型”。
- Context Engineering：主要关心“给模型什么上下文”。
- Harness Engineering：主要关心“模型、工具、状态、评估、权限和失败恢复怎么被一个系统管理起来”。

**面试答案：为什么 Agent 需要 Harness？**  
因为模型本身不可靠，工具也会失败。生产级 Agent 不能只靠一段 prompt 让模型自由发挥，而要有任务状态、工具注册表、参数校验、权限边界、最大步数、trace 日志、评估集和降级策略。Harness 的价值是把不稳定的大模型行为约束在可观测、可回滚、可评估的工程流程里。

**容易被追问：Agent Loop 怎么设计？**

```text
while not done and steps < max_steps:
    state = read_state()
    context = build_context(state, memory, tool_results)
    action = model.decide(context, available_tools)
    checked_action = validate(action)
    result = execute_or_reject(checked_action)
    state = update_state(state, action, result)
    done = judge_finish(state, result)
```

关键不是循环本身，而是每一步都要有边界：

- `available_tools` 不能无限暴露，按任务和权限动态筛选。
- `validate(action)` 要检查工具名、参数类型、必填项、权限和风险。
- `execute_or_reject` 要处理超时、重试、幂等、回滚和人工确认。
- `judge_finish` 不能只靠模型一句“完成了”，要结合任务条件、工具结果和输出校验。

## 17. RAG 与 Agent 评估：从“感觉对”到可复盘

RAG 评估要拆成三层：

```text
检索是否找到证据 -> 上下文是否足够回答 -> 答案是否忠于证据
```

最小评估集应该包含：

- `query`：用户真实问题或改写后的问题。
- `gold_docs`：应该命中的文档、段落或实体。
- `retrieved_chunks`：系统实际召回的 top-k chunk。
- `answer`：模型最终回答。
- `reference_answer`：人工或业务认可的参考答案。
- `citation`：回答引用了哪些证据。

常用指标：

```text
Recall@k = top-k 召回中命中的 gold 文档数 / gold 文档总数
Precision@k = top-k 召回中命中的 gold 文档数 / k
MRR = 1 / 第一个正确文档的排名
Faithfulness ≈ 被证据支持的回答声明数 / 回答声明总数
```

注意：

- Recall@k 低，说明召回没捞到证据，先改 query rewrite、embedding、chunk 或 hybrid retrieval。
- Recall@k 高但答案错，说明 rerank、context packaging 或生成阶段有问题。
- Faithfulness 低，说明模型没有忠于证据，要加证据约束、引用校验、拒答和后验检查。

Agent 评估要看任务轨迹，而不是只看最终答案：

- task success rate：任务是否完成。
- tool call accuracy：工具是否选对。
- argument accuracy：参数是否填对。
- step efficiency：步骤是否过多。
- recovery rate：工具失败后是否能恢复。
- cost per task：单任务 token、工具、延迟成本。
- safety violation rate：是否越权、误操作、泄露敏感信息。

**面试答案：怎么做 RAG/Agent 线上闭环？**  
先建小而准的离线评估集，每次改 chunk、embedding、rerank、prompt 或工具 schema 都跑一遍；线上记录 query、召回片段、rerank 分数、prompt 版本、工具轨迹、答案、引用、用户反馈和人工判定。bad case 要回流成评估样本，而不是只临时改 prompt。

如果面试继续追 Agent Harness、Tool Calling、流式工具解析、MCP 安全边界、多智能体协作、HITL、代码 Agent 和 Agent eval，转到 [21_Agent工程化_ToolCalling与MCP面试.md](../21_Agent工程化_ToolCalling与MCP面试.md) 深挖。

## 18. MCP 讲深一点：不是“又一个 Function Calling”

MCP 面试最容易答浅，只说“连接工具的协议”。更完整的说法：

```text
MCP Host
  -> MCP Client
      -> MCP Server
          -> Tools / Resources / Prompts
```

核心角色：

- MCP Host：承载模型应用的宿主，比如 IDE、桌面助手、Agent 平台。
- MCP Client：Host 内部维护连接的组件，一个 server 通常对应一个 client。
- MCP Server：对外暴露工具、资源和提示模板的服务，可以本地运行，也可以远程运行。

两层协议：

- Data layer：基于 JSON-RPC 的数据交换，包含生命周期、能力协商、工具/资源/提示模板、通知等。
- Transport layer：负责通信方式和认证，例如本地 stdio、远程 HTTP/SSE 等。

三类核心 primitives：

- Tools：可执行动作，例如查数据库、调用 API、读文件、发工单。
- Resources：只读或可读取的上下文数据，例如文件内容、数据库 schema、文档片段。
- Prompts：可复用的交互模板，例如某类任务的 system prompt、few-shot 模板。

**面试答案：MCP、Function Calling、Agent 框架怎么区分？**  
Function Calling 是模型输出结构化工具调用意图；MCP 是工具、资源、提示模板如何被标准化发现、连接和调用；Agent 框架负责规划、状态、记忆、工具编排和任务闭环。三者可以组合：Agent 框架用 Function Calling 表达动作，再通过 MCP server 执行具体工具。

**MCP 的工程风险：**

- 工具列表暴露过多，模型容易误用。
- 工具描述不清，参数容易填错。
- 本地工具权限过大，可能误删文件或泄露数据。
- 远程 server 要考虑鉴权、审计、速率限制和超时。
- server 版本变化后，client 的工具缓存和 prompt 可能过期。

## 19. RAG 项目 2 分钟讲法

```text
我做的是一个面向 X 场景的知识库问答系统。离线侧对文档做解析、清洗和结构化切分，保留标题、来源和页码等 metadata，然后用 embedding 建向量索引，同时保留 BM25 做混合召回。在线侧用户问题先做 query rewrite，再走向量召回和关键词召回，候选片段用 reranker 精排，最后把 top 证据拼进 prompt，让模型基于证据回答并返回引用。评估上我看 Recall@k、faithfulness、引用正确率、人工满意度和延迟成本。主要难点是 chunk 粒度、专有名词召回和幻觉控制。
```

## 20. Agent 项目 2 分钟讲法

```text
我做的是一个能调用工具完成 X 任务的 Agent。系统分成任务理解、规划、工具调用、状态管理和结果生成几部分。模型根据用户目标生成下一步动作，按 tool schema 调用检索、数据库或业务 API。每次工具返回后进入 observation，模型决定下一步。为了稳定性，我加了参数校验、最大步数、失败重试、权限控制和 trace 日志。评估上看任务完成率、工具调用准确率、平均步骤数、失败率和成本。
```

## 21. 本章高频问答

### Q1：RAG 和微调怎么选？

RAG 补知识，微调改行为。知识更新快、私有、要引用，用 RAG；输出格式和领域行为不稳定，用微调；复杂业务常组合。

### Q2：RAG 幻觉怎么治理？

从检索、排序、生成、评估四层治理：提升召回、rerank、基于证据回答、引用来源、证据不足拒答、答案一致性校验、bad case 回流。

### Q3：Agent 为什么容易死循环？

目标不清、工具结果不明确、缺少终止条件、模型误读 observation。解决是最大步数、状态机约束、失败次数限制和人工兜底。

### Q4：MCP 解决什么问题？

标准化模型应用和外部工具、资源、上下文的连接，减少每个工具单独适配的成本。

### Q5：多 Agent 有必要吗？

只有任务复杂且角色分工明确时有必要。否则会增加通信、延迟、成本和错误传播风险。

### Q6：RAG 评估为什么不能只看最终答案？

因为最终答案错了不代表生成模型错，可能是召回没命中、rerank 排错、上下文拼接丢证据；最终答案对了也不代表系统可靠，可能是模型凭常识猜对。要分层看 Recall@k、context precision/recall、faithfulness、引用正确率和业务解决率。

### Q7：Agent Harness 和 Agent 框架有什么区别？

框架是实现工具，Harness 是工程设计思想。你可以用 LangGraph、AutoGen 或自己写状态机，但核心都要解决状态、工具、权限、评估、trace、失败恢复和成本控制。

### Q8：MCP 为什么 2025-2026 面试变高频？

因为 Agent 应用需要连接大量外部系统。如果每个工具都单独适配，开发和维护成本高；MCP 把工具、资源和提示模板的发现、调用、通知和传输标准化，适合问“AI 应用怎么接企业系统”这类工程题。

### Q9：工具调用参数错了怎么办？

先在 schema 层减少歧义，再在执行前做参数校验；可修复错误让模型基于错误信息重试，不可修复错误让用户补充；高风险动作要人工确认；多次失败后降级或终止，避免无限循环。

### Q10：RAG 和 Agent 的日志要记什么？

RAG 记 query、rewrite、召回 chunk、rerank 分数、prompt 版本、答案、引用和反馈；Agent 还要记每一步 thought/action/observation 的结构化轨迹、工具参数、工具返回、错误、重试、成本、耗时和终止原因。

## 22. 面试前背诵版

RAG 是检索增强生成，核心是先找证据再让模型基于证据回答。RAG 难点在 chunk、embedding、hybrid retrieval、rerank、幻觉治理和分层评估。Agent 是任务执行器，不只是聊天，会规划、调用工具、观察结果并更新状态。生产级 Agent 要讲 Harness：状态、工具、权限、评估、trace、失败恢复和成本控制。Function Calling 是模型输出工具名和参数，真正执行在应用侧。MCP 是标准化工具、资源和提示模板接入模型应用的协议。生产级 RAG/Agent 要重点讲评估、失败处理、权限和成本。
