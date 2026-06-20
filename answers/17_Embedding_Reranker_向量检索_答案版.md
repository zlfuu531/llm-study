# 答案版 17：Embedding、Reranker 与向量检索

对应题号：361-380。建议先读 [22_Embedding_Reranker与向量检索面试.md](../22_Embedding_Reranker与向量检索面试.md)，再用本文件做口语复述。

## 361. RAG 里的 embedding 和 LLM token embedding 有什么区别？

LLM token embedding 是模型内部把 token id 映射成 hidden state 输入，每个 token 一个向量。RAG embedding 是专门用于检索的文本表示模型，把 query、句子、段落或文档 chunk 映射成语义向量。

面试要说清：RAG 不是直接拿 LLM 的词表 embedding 检索，而是用经过对比学习、检索任务训练或指令优化的 embedding 模型，让相关 query-doc 在向量空间更近。

## 362. Bi-encoder 检索怎么工作？

Bi-encoder 分别编码 query 和 doc：`q = E_q(query)`，`d = E_d(doc)`，再用 cosine、dot product 或 L2 计算相似度。文档向量可以离线预计算，在线只需要编码 query 并查向量索引。

它的优点是快、可扩展，适合百万/亿级召回；缺点是 query 和 doc 分开编码，细粒度交互弱，所以常配 reranker。

## 363. cosine、dot product、L2 在向量检索里怎么选？

cosine 看方向相似度，dot product 看内积，L2 看欧氏距离。如果 query 和 doc 向量都做了 L2 normalize，那么 cosine 等于 dot product，L2 排序也能和 cosine 互相转换。

选择时优先遵循 embedding 模型和向量库官方推荐。不要随便换 metric，因为模型训练时可能就是按某种相似度优化的。

## 364. 为什么常对 embedding 做 normalize？

normalize 后向量长度为 1，相似度主要由方向决定，减少向量范数对排序的干扰。它还方便用 inner product index 实现 cosine search。

但不是所有模型都必须 normalize。如果模型输出分数依赖向量范数，强行 normalize 可能损失效果，所以要看模型说明并在自己的 eval 集上比较。

## 365. embedding 模型怎么选？

从语言、领域、输入长度、向量维度、Recall@k、MRR/NDCG、延迟、成本、部署方式、许可证和数据安全选择。

流程是收集真实 query 和 gold evidence，固定 chunk、index、rerank，对多个候选 embedding 做离线评估，再按业务分桶看错误。不要只看公开榜单，因为企业文档、专有名词和中文 query 往往和榜单数据差异很大。

## 366. dense retrieval 和 BM25 各自擅长什么？

Dense retrieval 擅长语义相似、同义改写、口语 query，比如“怎么退钱”召回“售后退费流程”。BM25 擅长关键词、编号、专有名词、代码和错误码，比如“ERR-4521”。

所以生产 RAG 常用 hybrid retrieval。dense 解决语义，BM25 保住精确词匹配，两路结果融合后再 rerank。

## 367. BM25 公式和直觉是什么？

BM25 大致是对 query 中每个词求和：`IDF * 词频饱和项 * 文档长度归一化项`。稀有词更重要，词在文档中出现越多相关性越高但收益会饱和，长文档因为更容易包含词要做惩罚。

口语版：BM25 不是简单数关键词次数，而是考虑“这个词有多稀有、出现多少次、文档是不是太长”。

## 368. hybrid retrieval 怎么融合？

常见做法是 dense top-k 和 BM25/sparse top-k 各自召回，再用 RRF 或分数归一化加权融合，最后交给 reranker。

RRF 按排名融合：`score = Σ 1/(k + rank)`，好处是不用让不同检索器的原始分数可比。也可以按 query 类型动态调权，编号类 query 偏 BM25，语义类 query 偏 dense。

## 369. 为什么向量检索需要 ANN？

暴力搜索要把 query 和所有向量算相似度，成本约为 `N*d`。当 N 到百万、亿级时，延迟和成本太高。

ANN 用近似换速度，只搜索最可能相关的区域或图邻居。面试要强调 trade-off：延迟下降，但可能损失 Recall@k；需要调索引参数来平衡准确率、延迟、内存和更新成本。

## 370. HNSW、IVF、PQ 怎么区分？

HNSW 是图搜索，通过小世界图快速导航，低延迟高召回，但内存占用高。IVF 是先聚类分桶，查询时只搜部分桶，适合大规模，但召回受 nprobe 影响。PQ 是产品量化，把向量压缩成短码，显著省内存，但有量化误差。

一句话：HNSW 用内存换速度和召回，IVF 用分桶减少搜索范围，PQ 用压缩换存储。

## 371. 向量数据库生产里要关注什么？

不只存向量。要关注 metadata、租户权限、文档版本、索引类型、过滤能力、增删改、写入一致性、sharding、replica、备份恢复、查询延迟、空召回率和 eval 抽检。

项目里不要只说“用了 Milvus/Qdrant/Pinecone”，要说你怎么设计 metadata、权限过滤、索引参数、更新流程和监控。

## 372. metadata filter 应该放检索前还是检索后？

权限相关过滤尽量 pre-filter，即先按租户和权限缩小候选，再做检索，这样更安全。post-filter 实现简单，但 top-k 里如果很多无权限文档，过滤后可能为空，也可能增加泄露风险。

生产建议是：能 pre-filter 就 pre-filter；最终 context 前一定再做强权限校验；如果必须 post-filter，就扩大候选池并记录过滤前后数量。

## 373. reranker 为什么更准但更慢？

Bi-encoder 分开编码 query 和 doc，用向量相似度打分。Cross-encoder reranker 把 query 和 doc 拼在一起输入 Transformer，让两边 token 充分交互，所以更能判断细粒度相关性、否定、数字和条件。

慢是因为每个 query-doc pair 都要跑一次模型，不能像 doc embedding 那样全部离线预计算。因此一般只 rerank 初召回的 top 50-200。

## 374. cross-encoder 和 bi-encoder 怎么取舍？

Bi-encoder 用于大规模召回，速度快、可离线建索引，但精排能力弱。Cross-encoder 用于候选集重排，效果强但成本和延迟高。

常见架构是 bi-encoder/BM25 先召回，再用 cross-encoder rerank。面试可以说：召回阶段要“找得到”，rerank 阶段要“排得准”。

## 375. retrieval_top_k、rerank_top_k、context_top_k 怎么选？

retrieval_top_k 决定 gold evidence 是否有机会进入候选池；rerank_top_k 决定 reranker 处理多少候选；context_top_k 决定最终塞进 prompt 的 chunk 数。

调参顺序：先调大 retrieval_top_k 看 Recall@k 上限，再加 reranker 看 MRR/NDCG 是否提升，最后调 context_top_k 看 faithfulness、延迟和 token 成本。

## 376. ColBERT / multi-vector / BGE-M3 解决什么问题？

单向量把整段文本压成一个向量，可能丢 token 级细节。ColBERT 这类 multi-vector 方法保留多个 token/片段向量，用 late interaction 做更细粒度匹配，比如 `score = Σ max sim(q_i, d_j)`。

BGE-M3 这类模型把 dense、sparse、multi-vector 能力统一起来，适合多语言、多粒度、多检索方式融合。面试不要只背模型名，要讲它解决的是单一表示不够稳的问题。

## 377. RAG 召回率低怎么排查？

按链路排：query 是否表达清楚，query rewrite 是否改错，chunk 是否切丢，metadata 是否缺失，embedding 是否领域不匹配，BM25 分词/同义词是否失败，ANN 参数是否牺牲召回，filter 是否过严，fusion 是否压低正确文档，reranker 是否误排。

如果完全召回不到，多半是 query/chunk/embedding/index/filter；如果召回到了但排后，多半是 fusion/rerank/top-k。

## 378. 检索评估要看哪些指标？

看 Recall@k、Precision@k、MRR、NDCG。RAG 还要看 context recall、context precision、citation accuracy 和 faithfulness。

关键是检索和生成分开评估。否则最终答案错了，你不知道是没召回证据、证据没进 prompt，还是模型没忠于证据。

## 379. 向量库投毒和权限问题怎么防？

入库前做来源校验、清洗、安全扫描和版本记录；metadata 里保留 tenant、owner、permission、source、version；检索前后做权限过滤；对 prompt injection 文档做隔离或清洗；删除和更新要同步索引；bad case 回流到 eval。

最重要的一句：向量相似不代表有权限，也不代表可信。召回结果进入 prompt 前必须做权限和安全检查。

## 380. Embedding/Reranker/向量检索项目怎么讲 8 分钟？

按背景、离线链路、在线链路、评估、难点、结果讲。

背景：企业知识库文档多、更新频繁、专有名词多、权限复杂。离线：解析、清洗、chunk、metadata、embedding、BM25/sparse index、版本管理。在线：鉴权、query rewrite、dense + sparse hybrid retrieval、RRF 融合、cross-encoder rerank、context packing、带引用生成。

评估：Recall@k、MRR、NDCG、context recall、context precision、citation accuracy、人工满意度、P95 和成本。难点：embedding 领域不匹配、chunk 粒度、BM25 分词、ANN 参数、metadata filter、rerank 延迟和权限过滤。结果要说清指标提升和 bad case 闭环。
