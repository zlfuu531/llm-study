# 深挖 03：RAG、Agent 与 MCP

## 这一章解决什么问题

2025-2026 的大模型应用岗，很容易围绕这些问题追问：

- RAG 和微调怎么选？
- RAG 为什么还会幻觉？
- rerank 到底解决什么？
- Agent 和工作流有什么区别？
- Function Calling 和 MCP 是一回事吗？
- Context Engineering 为什么变成新词？

这章把应用岗面试的理论和工程链路讲透。

## 1. RAG 的本质

RAG 不是“向量数据库 + prompt”这么简单。

它本质上是把回答问题拆成两件事：

1. 从外部知识源找到证据。
2. 让模型基于证据组织答案。

所以 RAG 的关键不是模型多会说，而是证据找得对不对、拼得好不好、模型是否遵守证据。

## 2. RAG 和微调怎么选

优先 RAG 的场景：

- 知识经常更新。
- 企业私有知识。
- 需要引用来源。
- 不希望模型记住敏感数据。
- 数据主要是事实型知识。

优先微调的场景：

- 需要改变模型行为风格。
- 需要固定输出格式。
- 需要学习领域任务模式。
- 低延迟，不想每次检索。
- 有高质量训练数据。

组合使用：

- RAG 提供知识。
- SFT/LoRA 改善指令遵循和领域格式。

面试表达：

> RAG 更适合补知识，微调更适合改行为。企业知识库问答通常先用 RAG，因为知识更新快、需要溯源；如果模型格式遵循差或领域表达不稳定，再考虑 SFT/LoRA。

## 3. 为什么 chunk 是 RAG 的第一难点

chunk 太小：

- 信息碎。
- 召回片段不完整。
- 模型拿不到上下文。

chunk 太大：

- 向量语义被稀释。
- 召回不准。
- prompt 成本高。
- 噪声增加。

好的 chunk：

- 尽量是一个完整语义单元。
- 能独立回答某类问题。
- 保留标题、层级和来源。
- 对表格、代码、PDF 有特殊处理。

面试表达：

> chunk size 不是拍脑袋定的。我会根据文档结构和问题类型选择切分策略，再用 Recall@k、答案命中率和人工 bad case 来调。

## 4. 为什么只用向量召回不够

向量召回擅长语义相似：

- “报销流程”能召回“费用 reimbursement policy”。

但它不擅长：

- 精确编号。
- 人名、产品名、错误码。
- 代码符号。
- 表格字段。
- 否定条件。

所以常用 hybrid retrieval：

- dense retrieval：语义。
- BM25/sparse：关键词和精确匹配。
- rerank：精排。

## 5. rerank 的价值

第一阶段召回像海选：

- 快。
- 覆盖广。
- 噪声多。

rerank 像复试：

- 慢一点。
- 判断 query 和文档是否真正匹配。
- 能把有答案的片段排到前面。

面试表达：

> 向量召回通常用 bi-encoder，query 和 doc 各自编码后算相似度，速度快但交互弱。reranker 常用 cross-encoder，让 query 和 doc 一起输入模型做细粒度匹配，所以排序更准但成本更高。

## 6. RAG 为什么还会幻觉

RAG 幻觉通常不是一个点的问题。

检索侧：

- 没召回正确文档。
- 召回了相关但无答案的文档。
- 文档过时或冲突。
- chunk 切断了关键上下文。

生成侧：

- prompt 没要求基于证据。
- 模型把常识和证据混在一起。
- 上下文太长，模型忽略关键片段。
- 多跳问题缺少中间推理。

系统侧：

- 没有引用校验。
- 没有拒答机制。
- 没有 bad case 回流。

## 7. RAG 的评估要分层

只问“回答对不对”太粗。

检索评估：

- gold 文档是否被召回。
- Recall@k。
- MRR。
- NDCG。

上下文评估：

- context relevance。
- context precision。
- 是否包含答案。

生成评估：

- faithfulness。
- answer relevance。
- citation correctness。
- hallucination rate。

业务评估：

- 解决率。
- 转人工率。
- 成本。
- 延迟。
- 用户满意度。

## 8. Agent 的本质

Agent 不是“会聊天的模型”，而是一个执行循环。

典型 loop：

```text
目标 -> 思考 -> 选择动作 -> 调用工具 -> 观察结果 -> 更新状态 -> 下一步
```

所以 Agent 的核心问题不是“模型能不能答”，而是：

- 能不能选对工具。
- 参数能不能填对。
- 失败能不能恢复。
- 状态能不能保持。
- 什么时候停止。

## 9. Agent 和工作流的区别

工作流：

- 步骤固定。
- 可控性强。
- 适合规则明确的任务。

Agent：

- 步骤由模型动态决定。
- 灵活性强。
- 适合开放、多变、多工具任务。

面试表达：

> 如果流程稳定、规则明确，我会优先用工作流，因为更可靠、更便宜。Agent 适合任务路径不固定、需要模型根据环境反馈动态决策的场景。

## 10. Function Calling 的本质

Function Calling 不是模型真的执行函数。

模型做的是：

```json
{
  "name": "search_docs",
  "arguments": {
    "query": "..."
  }
}
```

外部系统负责：

- 校验参数。
- 执行函数。
- 处理权限。
- 捕获错误。
- 把结果返回给模型。

## 11. MCP 和 Function Calling 的关系

Function Calling 更偏“模型输出工具调用意图”。

MCP 更偏“工具、资源、上下文如何标准化暴露给模型应用”。

类比：

- Function Calling：模型说“我要调用哪个函数”。
- MCP：外部工具和上下文用统一协议接入，让模型应用能发现和使用。

面试表达：

> MCP 和 Function Calling 不完全是同一层。Function Calling 解决模型如何结构化表达工具调用，MCP 解决工具和上下文如何以标准协议接入模型应用。

## 12. Context Engineering 为什么变热

Prompt Engineering 关注：

- 怎么写单次 prompt。

Context Engineering 关注：

- 给模型什么上下文。
- 上下文怎么分层。
- 什么进入短期上下文。
- 什么写入长期记忆。
- 工具列表怎么选择。
- RAG 结果怎么压缩。
- 长任务状态怎么总结。
- 成本和缓存怎么控制。

面试表达：

> Prompt Engineering 更像写提示词，Context Engineering 更像设计模型运行时的信息供应系统。Agent 和 RAG 走向生产后，关键问题变成如何稳定、低成本地给模型提供正确上下文。

## 13. Agent 为什么容易失败

常见失败：

- 工具选错。
- 参数填错。
- 工具结果没读懂。
- 进入循环。
- 任务状态丢失。
- 记忆污染。
- 权限越界。
- 成本失控。

工程措施：

- 工具 schema 写清楚。
- 参数校验。
- 最大步数。
- 状态机约束。
- 失败重试。
- 人工确认高风险操作。
- trace 日志和回放评估。

## 14. Agent 评估怎么做

不要只看 demo。

指标：

- task success rate。
- tool call accuracy。
- argument accuracy。
- average steps。
- cost per task。
- latency。
- human intervention rate。
- safety violation rate。

评估集：

- 正常任务。
- 边界任务。
- 工具失败任务。
- 权限敏感任务。
- 长任务。
- 多轮任务。

## 15. 面试官追问路线

RAG 追问：

1. 为什么不用微调？
2. chunk 怎么切？
3. embedding 怎么选？
4. 为什么要 hybrid retrieval？
5. rerank 成本怎么控制？
6. 幻觉怎么定位？
7. 怎么评估？

Agent 追问：

1. Agent 和工作流怎么选？
2. ReAct 有什么问题？
3. Function Calling 失败怎么办？
4. MCP 解决什么问题？
5. 记忆怎么设计？
6. 怎么防止死循环？
7. 怎么做观测和评估？

## 参考来源

- RAG paper: https://arxiv.org/abs/2005.11401
- ReAct paper: https://arxiv.org/abs/2210.03629
- Model Context Protocol: https://modelcontextprotocol.io/
- AgentGuide: https://github.com/adongwanai/AgentGuide
- Datawhale LLM Cookbook: https://github.com/datawhalechina/llm-cookbook
- 小林面试笔记 AI: https://xiaolinnote.com/ai/
- 卡码笔记 2026 大模型面经: https://notes.kamacoder.com/interview/llm/

