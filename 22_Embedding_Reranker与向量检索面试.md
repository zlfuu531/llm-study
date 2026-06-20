# Embedding、Reranker 与向量检索面试

这一章面向 RAG 应用、搜索推荐 + LLM、算法应用、AI 应用后端和系统设计。很多同学面试时会说“我用了向量库”，但面试官真正会追的是：

- embedding 模型怎么选？
- 为什么要 hybrid retrieval？
- BM25 和 dense retrieval 各自擅长什么？
- HNSW、IVF、PQ 分别解决什么问题？
- reranker 为什么更准但更慢？
- 召回率低时怎么分层排查？
- 权限过滤放在检索前还是检索后？
- 怎么评估检索，而不是只看最终回答？

先背一句：

```text
RAG 的检索底座 = query 理解 + chunk/metadata + sparse/dense retrieval + ANN index + rerank + eval + 权限与更新。
```

不要把 RAG 说成：

```text
PDF -> embedding -> vector db -> prompt
```

更像生产系统的说法是：

```text
Offline:
  parse -> clean -> chunk -> metadata -> embedding/sparse vector
  -> vector index + lexical index -> version

Online:
  query -> rewrite/expand -> dense + sparse retrieval
  -> filter/fusion -> rerank -> context packing
  -> generation with citation -> eval/trace/bad-case loop
```

## 1. Embedding 到底是什么

Embedding 是把文本、图片或其他对象映射到向量空间。RAG 文本检索里，常见是把 query 和 chunk 分别编码成向量，然后用相似度找最近的文档。

```text
q = encoder(query)  -> R^d
d = encoder(chunk)  -> R^d
score(q, d) = similarity(q, d)
```

这和大模型内部 token embedding 不是一回事：

| 概念 | 用途 | 输出 |
| --- | --- | --- |
| LLM token embedding | 模型内部把 token id 映射成 hidden state 输入 | 每个 token 一个向量 |
| Retrieval embedding | 把句子/段落/文档映射成可检索向量 | 一段文本一个或多个向量 |

面试表达：

> RAG embedding 不是拿 LLM 的词表 embedding 直接检索，而是用专门训练过的文本表示模型，把 query 和文档 chunk 映射到语义空间，让相关内容距离更近。

## 2. Bi-Encoder 召回

大规模检索通常用 bi-encoder：

```text
query_encoder(query) -> q
doc_encoder(doc)    -> d
score = q · d 或 cosine(q, d)
```

优点：

- 文档向量可以离线预计算。
- 在线只编码 query，再查向量索引。
- 可扩展到百万/亿级文档。

缺点：

- query 和 doc 分开编码，细粒度交互弱。
- 对数字、专有名词、否定、复杂条件可能不如关键词或 cross-encoder。
- embedding 模型和业务领域不匹配时，召回会虚高或漏召。

一句话：

> Bi-encoder 用速度换取可扩展性，是 RAG 召回阶段的主力。

## 3. 相似度：Cosine、Dot Product、L2

常见相似度：

```text
dot(q, d) = q^T d
cos(q, d) = (q^T d) / (||q|| ||d||)
L2(q, d) = ||q - d||_2
```

如果向量已经 L2 normalize：

```text
||q|| = ||d|| = 1
cos(q, d) = q^T d
||q - d||_2^2 = 2 - 2 cos(q, d)
```

所以归一化后，cosine、inner product、L2 排序可以相互转换。面试常问“为什么要 normalize embedding”，可以答：

- 减少向量长度对相似度的影响。
- 让相似度主要由方向决定。
- 便于用 inner product index 做 cosine search。
- 让不同 query/doc 的分数更稳定。

但不是所有模型都必须 normalize，要看模型训练方式和官方建议。

## 4. Embedding 模型怎么选

不要只说“选效果好的”。实际要看：

| 维度 | 追问点 |
| --- | --- |
| 语言 | 中文、英文、多语言、跨语言 |
| 领域 | 法律、金融、医疗、代码、客服、企业内部文档 |
| 输入长度 | 短句、长文档、表格、代码、OCR 文本 |
| 向量维度 | 维度越高通常存储和检索成本越高 |
| 效果 | Recall@k、MRR、NDCG、业务命中率 |
| 延迟 | query 编码耗时、batch 能力 |
| 成本 | API 成本、GPU/CPU 部署成本、索引存储 |
| 更新 | 是否支持本地部署、版本稳定性、许可证 |
| 安全 | 数据能否出外部 API、是否需要私有化 |

选择流程：

```text
收集真实 query + gold evidence
  -> 选 2-4 个候选 embedding
  -> 固定 chunk/index/rerank
  -> 比 Recall@k、MRR、NDCG、延迟、成本
  -> 按业务场景分桶看错误
```

面试高分点：embedding 模型不要只看公开榜单，要在自己的文档和 query 上评估。

## 5. Dense Retrieval vs BM25

Dense retrieval 用向量语义相似度，BM25 用关键词匹配和统计权重。

| 方法 | 擅长 | 不擅长 |
| --- | --- | --- |
| Dense retrieval | 同义改写、语义相似、口语 query | 精确编号、专有名词、短关键词、符号 |
| BM25 / sparse retrieval | 关键词、代码、错误码、产品名、法规条款 | 同义词、语义改写、跨语言 |

例子：

- 用户问“怎么申请退款”，文档写“售后退费流程”：dense 更可能命中。
- 用户问“ERR-4521 是什么”：BM25 更可靠。

所以企业 RAG 常用 hybrid retrieval：

```text
dense top-k + sparse/BM25 top-k -> score fusion -> rerank
```

## 6. BM25 公式和直觉

BM25 常见形式：

```text
BM25(q, D) =
Σ IDF(q_i) * f(q_i, D) * (k1 + 1)
  / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
```

直觉：

- `IDF(q_i)`：词越稀有越重要。
- `f(q_i, D)`：词在文档中出现越多越相关，但收益会饱和。
- `|D| / avgdl`：长文档天然包含更多词，要做长度归一化。
- `k1`：控制词频饱和速度。
- `b`：控制文档长度归一化强度。

面试口语：

> BM25 是关键词检索的经典打分。它不是简单数词频，而是同时考虑稀有词更重要、词频收益会饱和、长文档要惩罚。

## 7. Hybrid Retrieval 和 RRF

混合检索要解决 dense 和 sparse 分数不可比的问题。一个常用方法是 Reciprocal Rank Fusion，按排名融合，而不是直接加原始分数。

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档在第 i 路检索中的排名，`k` 是平滑常数。

优点：

- 不需要不同检索器的分数在同一尺度。
- 对多路召回比较稳。
- 实现简单。

常见融合方式：

- score normalization 后加权。
- RRF 按排名融合。
- dense/sparse 先各取 top-k，再交给 reranker。
- 按 query 类型动态调权：编号类 query 偏 BM25，语义类 query 偏 dense。

## 8. 为什么需要 ANN

如果有 1 亿个向量，每次 query 都和所有向量算相似度，成本太高。

```text
brute force cost ≈ N * d
```

ANN，即 Approximate Nearest Neighbor，用近似换速度：

```text
更快搜索 + 可接受的召回损失
```

面试要讲 trade-off：

- 准确率：Recall@k。
- 延迟：P95/P99 query latency。
- 内存：索引结构和原始向量存储。
- 构建时间：index build time。
- 更新成本：新增、删除、重建。
- 过滤能力：metadata filter 是否高效。

## 9. Flat、HNSW、IVF、PQ 怎么区分

| 索引 | 直觉 | 优点 | 代价 |
| --- | --- | --- | --- |
| Flat | 暴力精确搜索 | 最准，适合小数据或离线评估 | 慢，规模大不可用 |
| HNSW | 图搜索，小世界导航 | 低延迟、高召回 | 内存高，构建成本高 |
| IVF | 先聚类到桶，再搜少量桶 | 大规模搜索快 | 召回依赖 nlist/nprobe |
| PQ | 向量压缩，减少内存 | 存储低，适合超大规模 | 有量化误差，精度下降 |
| IVF-PQ | 聚类 + 压缩 | 大规模低内存 | 调参和精度折中复杂 |

HNSW 参数：

- `M`：每个节点连接数，越大召回高、内存高。
- `efConstruction`：建图搜索宽度，越大构建慢、质量高。
- `efSearch`：查询搜索宽度，越大召回高、延迟高。

IVF 参数：

- `nlist`：聚类桶数量。
- `nprobe`：查询时搜索多少桶，越大召回高、延迟高。

面试表达：

> HNSW 更像在图上导航找近邻，常见于低延迟高召回场景；IVF 先把空间分桶再搜候选桶，更适合大规模；PQ 通过压缩降低内存，但会损失精度。

## 10. 向量数据库生产能力

向量数据库不是只存向量。生产里要关心：

- 向量字段：dense、sparse、multi-vector。
- metadata：文档 id、租户、权限、时间、来源、页码、版本。
- filter：按租户、权限、类型、时间过滤。
- index：HNSW、IVF、PQ、磁盘索引等。
- update/delete：文档更新、删除、重建索引。
- consistency：写入后多久可查。
- sharding/replica：容量和可用性。
- backup/restore：索引和原始文档可恢复。
- observability：query latency、Recall@k 抽检、filter 命中、空召回率。

向量库项目别只讲“我用了 Milvus/Qdrant/Pinecone/FAISS”，要讲你怎么做 metadata、权限、索引参数、更新和评估。

## 11. Metadata Filter 放前还是放后

权限过滤是 RAG 的高频追问。

两种方式：

```text
pre-filter: 先按权限/metadata 缩小候选，再做向量搜索
post-filter: 先向量搜索，再过滤无权限结果
```

pre-filter：

- 优点：不会召回无权限内容，更安全。
- 缺点：过滤条件太窄可能影响 ANN 索引效率或召回。

post-filter：

- 优点：实现简单，不影响初始向量搜索。
- 缺点：top-k 里很多无权限文档时，过滤后可能为空；也有泄露风险。

生产建议：

- 权限必须在最终 context 前强校验。
- 能 pre-filter 就尽量 pre-filter。
- 如果 post-filter，需要扩大候选池并重新填满 top-k。
- trace 里记录过滤前后数量，便于排查。

## 12. Reranker 为什么更准但更慢

Bi-encoder 分开编码：

```text
q = E(query)
d = E(doc)
score = q · d
```

Cross-encoder reranker 联合编码：

```text
score = f([query; document])
```

它能让 query token 和 document token 在 Transformer 内部充分交互，所以更擅长判断细粒度相关性、否定、条件、数字和上下文关系。

但它不能提前离线算好所有文档分数。每个 query-doc pair 都要跑一次模型：

```text
cost ≈ top_k_candidates * transformer_forward
```

所以常见架构是：

```text
dense/BM25 召回 top 50-200
  -> cross-encoder rerank
  -> 取 top 3-10 进 prompt
```

## 13. Top-k 怎么选

有三个 k：

```text
retrieval_top_k: 初始召回多少候选
rerank_top_k: reranker 处理多少候选
context_top_k: 最终进 prompt 多少 chunk
```

取舍：

- retrieval_top_k 太小：gold evidence 进不了候选池。
- rerank_top_k 太大：延迟和成本上升。
- context_top_k 太大：上下文噪声和 token 成本上升。

调参方法：

```text
先提高 retrieval_top_k 看 Recall@k 上限
再加 reranker 看 NDCG/MRR 是否提升
最后调 context_top_k 看 faithfulness 和成本
```

面试别说“取 top3/top5 就行”，要说基于 eval 和延迟预算调。

## 14. Multi-Vector 和 ColBERT

单向量表示把一段文本压成一个向量，可能丢失细粒度信息。Multi-vector 方法会为一个文档保留多个向量，典型思路是 late interaction。

ColBERT 的直觉：

```text
query token vectors 和 document token vectors 做最大相似匹配
再聚合成文档分数
```

简化公式：

```text
score(q, d) = Σ_i max_j sim(q_i, d_j)
```

优点：

- 比单向量保留更多 token 级信息。
- 比 full cross-encoder 更容易预计算文档表示。

代价：

- 存储更多向量。
- 检索系统更复杂。
- 延迟和工程成本高于普通 dense retrieval。

BGE-M3 这类模型把 dense、sparse、multi-vector 等能力统一到一个模型里，适合面试时作为“多功能检索表示”的例子，但不要只背模型名，要讲清它解决的是多语言、多粒度、多检索方式融合的问题。

## 15. Query Rewrite、HyDE、Multi-Query

当 query 太短、太口语、上下文省略严重时，可以做 query 增强。

| 方法 | 直觉 | 风险 |
| --- | --- | --- |
| Query rewrite | 把用户问题改写成更适合检索的 query | 改错意图 |
| HyDE | 先生成假设答案，用假设答案去检索 | 假设答案幻觉会带偏 |
| Multi-query | 从多个角度生成多个 query 扩召回 | 成本高，噪声变多 |

评估时要比较：

- Recall@k 是否提升。
- MRR/NDCG 是否提升。
- 噪声和上下文冲突是否增加。
- 延迟和 token 成本是否可接受。

## 16. 检索评估指标

检索要单独评估，不要只看最终回答。

Recall@k：

```text
Recall@k = top-k 命中的 gold evidence 数 / gold evidence 总数
```

Precision@k：

```text
Precision@k = top-k 中相关文档数 / k
```

MRR：

```text
MRR = mean(1 / 第一个相关结果排名)
```

DCG/NDCG：

```text
DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

RAG 还要看：

- context recall：gold evidence 是否进入最终上下文。
- context precision：进入上下文的内容有多少真的有用。
- citation accuracy：引用是否指向正确证据。
- faithfulness：答案是否被证据支持。

## 17. 召回率低怎么排查

排查链路：

```text
query 是否表达清楚
  -> query rewrite 是否改错
  -> chunk 是否切碎/切丢
  -> metadata 是否缺失
  -> embedding 是否领域不匹配
  -> BM25 分词/同义词是否失败
  -> ANN 参数是否牺牲召回
  -> filter 是否过严
  -> fusion 是否压低正确文档
  -> reranker 是否误排
```

按现象定位：

| 现象 | 可能原因 |
| --- | --- |
| 完全召回不到 | query/chunk/embedding/index/filter 问题 |
| 召回到了但排很后 | rerank/fusion/score normalization 问题 |
| 编号/专名查不到 | BM25/分词/metadata 问题 |
| 语义改写查不到 | dense embedding 或 query rewrite 问题 |
| 多租户查不到 | 权限过滤过严或 metadata 错 |
| 更新后查不到 | 索引构建、写入一致性、版本问题 |

## 18. 向量库投毒和权限风险

风险：

- 恶意文档进入索引，诱导模型输出错误。
- 文档中藏 prompt injection。
- 无权限文档被召回进上下文。
- response cache 跨租户泄露。
- embedding API 发送敏感内容到外部。
- 删除文档后向量索引未同步删除。

治理：

- 文档入库前做来源校验、清洗和安全扫描。
- metadata 记录 owner、tenant、permission、version、source。
- 检索前后都做权限过滤。
- prompt injection 检测和上下文隔离。
- 删除和更新要同步 index。
- bad case 回流到 eval 和安全样本。

## 19. 项目 8 分钟讲法

```text
背景：
我们做的是企业知识库/RAG 问答，难点是文档多、更新频繁、专有名词多、权限复杂，并且用户问题很口语。

离线链路：
文档解析、清洗、结构化 chunk，保留标题、来源、页码、租户、权限、时间等 metadata。
同时建立 dense embedding 向量索引和 BM25/sparse 索引，索引和 embedding 模型都做版本管理。

在线链路：
用户 query 先做鉴权和必要的 query rewrite，然后 dense + sparse hybrid retrieval。
两路结果用 RRF 或加权融合，取候选 top-k 交给 cross-encoder reranker。
最终 top chunks 进入 context builder，按 token budget、去重、权限和引用格式打包给模型。

评估：
离线看 Recall@k、MRR、NDCG、context recall、context precision、citation accuracy。
线上看空召回率、无答案率、人工满意度、P95、成本、bad case 类型。

难点：
embedding 模型领域不匹配、chunk 粒度、BM25 分词、ANN 参数、metadata filter、rerank 延迟和权限过滤。

结果：
通过 hybrid retrieval + rerank 提升证据命中率，用 trace 和 eval 把召回、排序、生成问题分开定位。
```

## 20. 面试前背诵版

RAG 检索底座不是“向量库 + prompt”，而是 query 理解、chunk/metadata、dense retrieval、BM25/sparse retrieval、ANN index、hybrid fusion、rerank、权限过滤和 eval 的组合。Embedding 用 bi-encoder 把 query 和文档映射到向量空间，适合大规模召回；BM25 擅长关键词、编号和专有名词；hybrid retrieval 把语义召回和关键词召回结合起来。ANN 用近似换速度，HNSW 是图搜索，IVF 是聚类分桶，PQ 是向量压缩。Reranker 通常用 cross-encoder 联合看 query 和 doc，更准但更慢，所以只 rerank 初召回的 top-k。检索评估要看 Recall@k、MRR、NDCG、context recall、context precision 和 citation accuracy。召回低时要从 query、chunk、embedding、BM25、ANN、filter、fusion、rerank 分层排查。

## 21. 和搜广推 / Ranking 的衔接

这一章重点是检索底座，适合回答“候选怎么找、证据怎么排”。如果面试官继续追到搜索推荐广告的完整排序链路，要转到：

- 召回、粗排、精排、重排的漏斗。
- CTR/CVR/GMV/eCPM 目标。
- Pointwise、Pairwise、Listwise LTR。
- LambdaMART、NDCG、MRR、MAP、GAUC。
- 位置偏差、曝光偏差和 A/B 实验。
- LLM query rewrite、LLM rerank 与成本控制。

完整整理见：[32_搜索推荐广告_LLM_Ranking与LTR面试.md](32_搜索推荐广告_LLM_Ranking与LTR面试.md)

## 本轮参考

- Faiss 文档：[https://faiss.ai/index.html](https://faiss.ai/index.html)
- Faiss Indexes Wiki：[https://github.com/facebookresearch/faiss/wiki/Faiss-indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- Milvus Index 文档：[https://milvus.io/docs/index.md](https://milvus.io/docs/index.md)
- Milvus HNSW 文档：[https://milvus.io/docs/hnsw.md](https://milvus.io/docs/hnsw.md)
- Qdrant Hybrid Queries：[https://qdrant.tech/documentation/search/hybrid-queries/](https://qdrant.tech/documentation/search/hybrid-queries/)
- Elasticsearch BM25 解释：[https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- BGE-M3 论文：[https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)
- BGE-M3 文档：[https://bge-model.com/bge/bge_m3.html](https://bge-model.com/bge/bge_m3.html)
- BGE Reranker 文档：[https://bge-model.com/tutorial/5_Reranking/5.2.html](https://bge-model.com/tutorial/5_Reranking/5.2.html)
- Sentence Transformers Retrieve & Re-Rank：[https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- Cohere Rerank 文档：[https://docs.cohere.com/docs/rerank-overview](https://docs.cohere.com/docs/rerank-overview)
