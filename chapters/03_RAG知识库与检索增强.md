# RAG 知识库与检索增强

## 面试目标

这一章要达到的状态：

- 能完整讲出 RAG 链路。
- 能解释 chunk、embedding、召回、重排、生成、评估。
- 能回答“RAG 为什么还会幻觉”。
- 能把自己的项目讲成系统，而不是“我用了 LangChain”。

更深入的通俗版讲解见：[../deepdives/03_RAG_Agent_MCP_深挖.md](../deepdives/03_RAG_Agent_MCP_深挖.md)

## RAG 是什么

RAG，即 Retrieval-Augmented Generation。

核心思想：

- 模型参数里不一定有最新或私有知识。
- 先从外部知识库检索相关内容。
- 再把检索结果放进 prompt，让模型基于上下文回答。

面试表达：

> RAG 把生成问题拆成检索和生成两步。检索负责把外部知识找出来，生成模型负责基于这些证据组织答案。它适合知识更新频繁、企业私有知识、需要可追溯来源的场景。

## 标准链路

离线建库：

1. 文档采集
2. 清洗和解析
3. chunk 切分
4. embedding
5. 写入向量库或混合索引

在线问答：

1. 用户 query
2. query 改写或扩展
3. 向量召回 / 关键词召回 / 混合召回
4. rerank
5. 上下文拼接
6. LLM 生成
7. 引用、后处理、评估、日志

## Chunk

chunk 是把长文档切成小段。

影响：

- 太小：上下文不完整。
- 太大：召回不准、浪费 token。
- 切分不合理：答案跨 chunk，模型拿不到完整证据。

常见策略：

- 固定长度 + overlap。
- 按标题、段落、语义切分。
- 结构化文档按章节/表格/字段切。

面试表达：

> chunk 的核心不是越小越好，而是让一个 chunk 尽量包含可回答一个问题的最小完整语义单元。实际项目里我会根据文档结构、问题类型和召回评估来调 chunk size 和 overlap。

## Embedding

embedding 把文本映射到向量空间，用相似度找语义相关内容。

常见追问：

- embedding 模型怎么选？
- 中文场景要注意什么？
- query 和 document 是否用同一个模型？
- dense embedding 和 sparse retrieval 的区别？

回答要点：

- 选模型要看语言、领域、长度、评测指标和部署成本。
- 企业知识库常用 dense + BM25 混合召回。
- embedding 不是万能，关键词、编号、专有名词常需要稀疏检索补充。

## 召回

常见方式：

- Dense retrieval：语义召回。
- Sparse retrieval：BM25 / 关键词。
- Hybrid retrieval：二者融合。
- Query rewrite：改写用户问题，提高召回。
- Multi-query：生成多个查询角度。

常见问题：

- 召回不到。
- 召回太多噪声。
- 召回内容和问题相关但不含答案。
- 召回内容过长，超过上下文窗口。

## Rerank

rerank 用更强的模型对候选文档重新排序。

为什么需要：

- 向量召回追求快和广。
- reranker 追求精确排序。

面试表达：

> RAG 常用两阶段检索。第一阶段召回尽量多找候选，第二阶段 rerank 用 cross-encoder 或 reranker 模型判断 query 和文档片段的细粒度相关性，从而把真正有答案的片段排到前面。

## 生成

prompt 设计重点：

- 明确要求基于上下文回答。
- 要求引用来源。
- 上下文不足时允许回答“不知道”。
- 控制格式。
- 避免把无关 chunk 一股脑塞进去。

常见策略：

- context compression
- answer with citation
- map-reduce
- refine
- multi-hop retrieval

## RAG 为什么还会幻觉

原因：

- 检索没召回正确证据。
- 召回证据有噪声或过时。
- rerank 排错。
- prompt 没约束模型基于证据。
- 问题需要多跳推理，单次检索不够。
- 模型倾向补全看似合理但无证据的内容。

解决：

- 提高召回率。
- rerank。
- 引用来源。
- 答案和证据一致性检查。
- 不足证据时拒答。
- 记录 bad case，迭代知识库和 prompt。

## RAG 评估

不要只看“感觉不错”。

检索指标：

- Recall@k
- Precision@k
- MRR
- NDCG

生成指标：

- faithfulness
- answer relevance
- context relevance
- hallucination rate
- 人工满意度

业务指标：

- 解决率。
- 转人工率。
- 平均响应时间。
- 单次成本。
- 引用命中率。

## 项目讲法模板

> 我做的是一个面向 X 场景的 RAG 系统。离线侧先对文档做解析、清洗和结构化切分，再用 embedding 建向量索引，同时保留 BM25 做混合召回。在线侧用户问题会先做 query rewrite，然后走向量召回和关键词召回，候选文档再用 reranker 排序，最后把 top 文档拼进 prompt，让模型基于证据回答并返回引用。评估上我分别看检索 Recall@k、答案 faithfulness、人工满意度和延迟成本。主要难点是 chunk 粒度、专有名词召回和幻觉控制。

## 高频问题

1. RAG 和微调怎么选？
2. chunk size 怎么设置？
3. overlap 有什么作用？
4. embedding 模型怎么选？
5. 为什么要混合召回？
6. rerank 的作用是什么？
7. RAG 为什么仍然会幻觉？
8. 如何评估 RAG？
9. 文档更新怎么处理？
10. 多轮对话 RAG 怎么做上下文管理？

## 延伸阅读

- RAG paper: https://arxiv.org/abs/2005.11401
- Datawhale LLM Universe: https://github.com/datawhalechina/llm-universe
- Datawhale All-in-RAG: https://datawhalechina.github.io/all-in-rag/
- Datawhale LLM Cookbook: https://github.com/datawhalechina/llm-cookbook
- 小林面试笔记 AI: https://xiaolinnote.com/ai/
