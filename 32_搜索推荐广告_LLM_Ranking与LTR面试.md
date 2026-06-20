# 搜索推荐广告 + LLM、Ranking 与 LTR 面试

这一章面向搜索、推荐、广告、内容理解、风控和大模型应用算法岗。它不是把 RAG 里的“向量检索”再讲一遍，而是把工业排序系统的主线补齐：召回、粗排、精排、重排、Learning to Rank、CTR/CVR/GMV 目标、偏差校正、A/B 实验，以及 LLM 如何嵌入这些链路。

如果时间紧，先掌握这句话：

> 搜广推系统的本质，是在有限延迟内从海量候选里选出最符合用户、场景和业务目标的一组结果；LLM 的价值通常不是替代整条链路，而是增强 query/item 理解、语义召回、复杂重排、解释生成、样本构造和评估闭环。

相关答案版：[answers/27_搜索推荐广告_LLM_Ranking与LTR_答案版.md](answers/27_搜索推荐广告_LLM_Ranking与LTR_答案版.md)

相邻章节：

- [22_Embedding_Reranker与向量检索面试.md](22_Embedding_Reranker与向量检索面试.md)：检索、BM25、向量召回、reranker、RAG 评估。
- [28_ML_DL数学基础_损失函数优化器与指标面试.md](28_ML_DL数学基础_损失函数优化器与指标面试.md)：CE/BCE、AUC/PR-AUC、校准、loss 与 metric 排查。
- [16_大模型评测与实验设计面试.md](16_大模型评测与实验设计面试.md)：离线评估、ablation、线上回归、LLM-as-judge。

## 1. 搜索、推荐、广告的区别

三者都做“候选集合排序”，但用户意图、反馈信号和约束不同。

| 方向 | 输入 | 用户意图 | 优化目标 | 典型指标 | 难点 |
| --- | --- | --- | --- | --- | --- |
| 搜索 | query + 用户/场景 | 显式需求强 | 相关性、满意度、转化 | NDCG、MRR、Recall@K、CTR、无结果率 | query 理解、拼写纠错、召回覆盖、时效性 |
| 推荐 | user + context + item pool | 隐式兴趣强 | 长期兴趣、留存、消费、满意度 | CTR、CVR、观看时长、留存、diversity | 兴趣漂移、冷启动、信息茧房、探索利用 |
| 广告 | user + context + ad + bid | 商业目标强 | 收入、转化、体验、合规 | pCTR、pCVR、eCPM、ROI、投诉率 | 竞价机制、预算、合规、作弊、用户体验 |

面试里不要只说“搜索有 query，推荐没有 query”。更好的说法是：

```text
搜索更强调显式意图和相关性；
推荐更强调用户兴趣建模、长期价值和多样性；
广告更强调 pCTR/pCVR、出价、ROI、预算和合规。
三者底层都有召回、排序、重排、评估和线上实验。
```

## 2. 召回、粗排、精排、重排链路

工业链路通常是漏斗：

```text
用户/请求
-> query/user/context 理解
-> 多路召回 thousands/millions -> hundreds/thousands
-> 粗排 hundreds/thousands -> hundreds
-> 精排 hundreds -> dozens
-> 重排/策略 dozens -> final slate
-> 展示、点击、转化、反馈回流
```

为什么要分层？

- 海量 item 无法全部交给大模型或复杂 ranker。
- 不同阶段的目标不同：召回要“别漏”，排序要“排准”，重排要“整体体验和业务约束”。
- 延迟预算不同：召回和粗排要快，精排可稍复杂，LLM rerank 通常只能处理很小的候选集。

各层职责：

| 层级 | 输入规模 | 常见模型 | 重点 |
| --- | --- | --- | --- |
| 召回 | 百万到十亿级 | 倒排、ANN、双塔、协同过滤、图召回、热门召回 | 高覆盖、低延迟、多样来源 |
| 粗排 | 千级 | GBDT、LR、轻量 DNN、双塔打分 | 快速过滤明显差候选 |
| 精排 | 百级 | Wide&Deep、DeepFM、DIN、Transformer ranker、多任务模型 | 精细特征交叉、CTR/CVR 预估 |
| 重排 | 十级 | 规则、多目标优化、MMR、约束优化、LLM rerank | 多样性、新鲜度、业务规则、安全 |

## 3. 多路召回怎么设计

召回层的面试重点是“多路互补”。不要只说“向量召回”。

常见召回通道：

- 关键词召回：倒排索引、BM25，适合专有名词、编号、品牌词、实体词。
- 语义召回：embedding + ANN，适合同义表达、长尾 query、语义匹配。
- 双塔召回：user tower 和 item tower 分别编码，score 通常是 dot product，适合大规模推荐。
- 协同过滤：基于 user-item 行为矩阵，适合“相似用户喜欢什么”。
- 图召回：user-item、item-item、作者、话题、商品图谱，适合关系扩展。
- 热门/新鲜召回：补冷启动、热点、时效内容。
- 规则召回：运营、合规、必出、过滤、类目约束。
- LLM 增强召回：query rewrite、query expansion、意图识别、实体归一、item 文本理解。

召回不要只看平均 Recall@K，也要分桶：

```text
头部 query / 长尾 query
新用户 / 老用户
新 item / 老 item
热门类目 / 稀疏类目
中文专名 / 英文缩写 / 编号
有点击 query / 无点击 query
```

## 4. 排序模型目标

排序模型不是“预测点击”这么简单。业务里常见目标包括：

```text
pCTR = P(click | user, item, context)
pCVR = P(conversion | click, user, item, context)
pCTCVR = P(click and conversion | user, item, context)
eCPM = bid * pCTR * pCVR * calibration_factor
score = w1 * relevance + w2 * pCTR + w3 * pCVR + w4 * value - penalties
```

搜索更偏相关性，推荐更偏兴趣和长期价值，广告更偏商业收益和体验约束。面试里要主动说：

- 训练 label 来自曝光后的点击、停留、收藏、购买、转化等行为。
- label 有偏，因为用户只会点击被曝光且排在某些位置的结果。
- offline loss 不等于 online business metric，要用 A/B test 验证。
- 多目标模型需要校准和权重调参，否则一个目标会压倒另一个目标。

## 5. CTR/CVR 模型直觉

常见模型可以按“是否显式做特征交叉”和“是否建模用户历史兴趣”理解。

### LR / GBDT / FM

- LR：线性可解释，适合强基线，但特征交叉依赖人工。
- GBDT：能挖非线性和特征组合，常用于传统排序、特征筛选或 teacher。
- FM：用低维向量建模二阶特征交叉，适合稀疏类别特征。

FM 的二阶交叉直觉：

```text
y = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j
```

其中 `<v_i, v_j>` 表示两个稀疏特征的隐向量交互，比如“用户城市”和“商品类目”的组合。

### Wide&Deep

Wide 侧记忆强，Deep 侧泛化强：

```text
score = wide_features * w + DNN(embedding_features)
```

适合回答：

> Wide 记住高频共现规则，Deep 泛化到没见过的组合。工业排序常需要两者兼顾。

### DeepFM

DeepFM 把 FM 的低阶交叉和 DNN 的高阶交叉一起学：

```text
y = sigmoid(y_FM + y_DNN)
```

优点是减少人工交叉特征，适合稀疏特征很多的 CTR 预估。

### DIN

DIN 的关键是 attention over user history。不是把用户所有历史简单平均，而是对当前候选 item 做兴趣激活：

```text
interest = sum_i attention(candidate_item, history_item_i) * history_item_i
```

面试表达：

> 用户兴趣不是一个固定向量。看鞋时激活鞋相关历史，看手机时激活数码相关历史。

## 6. Learning to Rank

Learning to Rank 的核心是直接学习排序，而不是只学习单点分类。

### Pointwise

把每个 item 当独立样本：

```text
loss = BCE(y, sigmoid(score))
```

优点：简单、可用 CTR/CVR label。缺点：没有直接优化列表顺序。

### Pairwise

学习同一个 query/user 下 item 两两之间谁更靠前：

```text
P(i > j) = sigmoid(s_i - s_j)
loss = -log sigmoid(s_i - s_j)
```

优点：更贴近排序相对关系。缺点：pair 数多，采样和偏差处理重要。

### Listwise

把整组候选当一个列表优化，比如 ListNet、ListMLE，或者围绕 NDCG 设计 surrogate loss。

优点：最贴近最终排序指标。缺点：实现复杂，对候选集和 label 质量敏感。

### LambdaRank / LambdaMART

LambdaRank 的直觉：

```text
如果交换两个 item 会让 NDCG 变化很大，就给这对 item 更大的梯度。
```

LambdaMART = LambdaRank 思想 + MART/GBDT。它很适合传统 LTR，因为：

- 能处理稠密、稀疏、统计、规则等多种特征。
- 对非线性和特征交叉友好。
- 训练稳定、可解释性相对好。
- 直接把 NDCG 位置权重融入 pairwise 梯度。

## 7. 常用指标和公式

### AUC / GAUC

AUC 直觉：

```text
AUC ~= 随机抽一个正样本和一个负样本，正样本得分高于负样本的概率
```

推荐/广告常看 GAUC，因为不同用户的行为量差异很大：

```text
GAUC = sum_u impressions_u * AUC_u / sum_u impressions_u
```

它比全局 AUC 更关注每个用户内部的排序质量。

### DCG / NDCG

```text
DCG@K = sum_{i=1..K} (2^{rel_i} - 1) / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

适合多级相关性排序。搜索里尤其常见，因为排在前面的错误比排在后面的错误更伤用户体验。

### MRR

```text
MRR = mean(1 / rank_first_relevant)
```

适合“第一个正确结果很重要”的任务，比如问答检索、导航型搜索。

### MAP

```text
AP = mean(Precision@k for each relevant item position k)
MAP = mean(AP over queries)
```

适合一个 query 有多个相关结果的检索任务。

### Recall@K

```text
Recall@K = topK 命中的相关 item 数 / 全部相关 item 数
```

召回层常看 Recall@K，但不要只看一个 K。K 太大可能掩盖排序问题，K 太小可能被标注不完整影响。

## 8. 位置偏差、曝光偏差和反馈闭环

搜广推的 label 不是天然真实。用户点击某 item，既可能因为 item 好，也可能因为它排在前面。

常见偏差：

- Position bias：位置越靠前越容易被看见和点击。
- Exposure bias：没曝光的 item 没有反馈。
- Selection bias：日志来自旧策略，训练新策略时分布不一致。
- Popularity bias：热门 item 获得更多曝光，进一步变热门。
- Presentation bias：图片、标题、价格、角标影响点击，不完全是相关性。

处理方法：

- 随机探索或小流量打散，估计 propensity。
- IPS/SNIPS 做反事实评估或训练加权。
- 点击模型估计 exam probability。
- 对训练样本做 position/context 特征建模。
- 在线 A/B test 最终验证。

IPS 直觉：

```text
unbiased_reward ~= observed_reward / propensity
```

propensity 是某 item 在某位置被展示的概率。概率越小，观测样本权重越大，但方差也会变大。

## 9. LLM 在搜广推里的位置

LLM 不一定直接做最终排序。常见位置有：

| 位置 | 用法 | 价值 | 风险 |
| --- | --- | --- | --- |
| Query 理解 | 意图识别、改写、扩展、纠错、实体归一 | 提升召回覆盖 | 改写漂移、成本、延迟 |
| Item 理解 | 商品/视频/文档摘要、标签、属性抽取 | 补足结构化特征 | 幻觉、属性错误 |
| 语义召回 | embedding、LLM 生成扩展 query | 长尾和同义表达 | 召回噪声 |
| Rerank | 对 topN 候选做语义重排 | 复杂相关性判断更强 | 延迟高、吞吐低、位置不稳定 |
| Explanation | 推荐理由、广告文案、搜索摘要 | 用户理解和转化 | 合规和真实性 |
| 样本构造 | 合成 query、偏好对、hard negative | 降低标注成本 | 数据污染、风格偏差 |
| Eval/Judge | 相关性判定、bad case 聚类 | 快速诊断 | judge 偏差、不可替代线上指标 |

好的面试回答要强调：

```text
LLM 适合处理语义理解和复杂判断；
传统 ranker 适合高吞吐、低延迟、稳定线上排序；
生产中通常是 LLM 生成特征、增强召回、rerank 小候选集或做评估辅助。
```

## 10. LLM rerank、cross-encoder rerank 和传统 ranker

| 方法 | 输入 | 优点 | 缺点 | 适合 |
| --- | --- | --- | --- | --- |
| 传统 ranker | 稠密/稀疏特征 | 快、稳定、可控 | 语义理解有限 | 主链路精排 |
| Cross-encoder | query/item 拼接 | 相关性强、比 LLM 便宜 | 只能 rerank topN | RAG/search rerank |
| LLM rerank | query + 候选列表 + 指令 | 复杂语义、可解释、多约束 | 慢、贵、格式和稳定性问题 | 小候选集、复杂 query、离线标注 |

LLM rerank 的工程控成本：

- 只对 top20/top50 做 rerank。
- 先用轻量模型判断 query 难度，只对困难 query 触发。
- 结构化输出候选 id，不让模型生成自由文本。
- 缓存 query 和候选列表。
- 蒸馏 LLM 偏好到小 ranker。
- 离线用 LLM 标注 pair/listwise 数据，线上用传统 ranker。

## 11. 离线评估和线上 A/B

离线评估：

```text
召回: Recall@K、coverage、空召回率、分桶召回
排序: AUC/GAUC、NDCG@K、MRR、MAP、logloss、calibration
推荐: CTR、CVR、时长、留存、多样性、新鲜度
广告: pCTR/pCVR、eCPM、ROI、预算消耗、投诉/违规
LLM: 改写准确率、rerank win rate、judge 一致性、延迟和成本
```

线上 A/B：

```text
实验假设 -> 实验分桶 -> guardrail 指标 -> 小流量 -> 观察周期
-> 显著性/置信区间 -> 分桶分析 -> bad case -> 灰度扩大或回滚
```

Guardrail 很重要：

- 延迟 P95/P99 不显著变差。
- 投诉率、违规率、不满意率不升。
- 短期 CTR 提升不能严重伤害长期留存和多样性。
- 广告收入提升不能明显损害自然结果体验。

## 12. 离线涨、线上不涨怎么排查

常见原因：

| 现象 | 排查方向 |
| --- | --- |
| AUC 涨、CTR 不涨 | AUC 对头部位置不敏感，看 NDCG/TopK、校准和分桶 |
| NDCG 涨、转化不涨 | 相关性不是转化，检查价格、库存、落地页、商业目标 |
| 离线 rerank 好、线上慢 | 候选太多、prompt 太长、模型太大、并发不足 |
| 长尾 query 变好、整体不涨 | 流量占比小，需要分桶看 |
| 点击涨、满意度降 | 标题党、低质量内容、短期刺激 |
| 搜索改写召回多但差 | query drift，扩展词引入噪声 |
| 广告收益涨、留存降 | 商业目标压过体验，需要 guardrail |

面试回答模板：

```text
先确认实验和数据是否可靠；
再看指标是否和目标一致；
然后分桶看用户、query、item、位置、类目；
接着排查召回、排序、重排、策略和延迟；
最后用 bad case 和回放定位具体链路。
```

## 13. 多目标重排

最终展示不是只按一个 score 排。重排要处理：

- 多样性：避免连续相同类目、作者、品牌。
- 新鲜度：给新内容探索机会。
- 公平性：避免某些供给方长期无曝光。
- 安全：过滤违规、低质、敏感内容。
- 商业规则：广告位、库存、预算、运营活动。
- 用户体验：频控、去重、已读过滤、疲劳控制。

简单重排可用 MMR：

```text
score(item) = lambda * relevance(item) - (1 - lambda) * max_similarity(item, selected)
```

直觉：既要相关，也要和已选结果不要太重复。

## 14. 搜广推 + LLM 项目 8 分钟讲法

```text
背景：
我们做的是搜索/推荐场景里的语义增强排序，问题是长尾 query 和口语化表达下召回不稳，传统 ranker 对复杂语义相关性判断不足。

基线：
原链路是关键词/BM25 + 向量召回，多路召回后用轻量 ranker 精排，线上看 CTR/NDCG/无结果率/P95。

方案：
先用 LLM 做 query 意图识别、实体归一和 query rewrite，只在低置信或长尾 query 触发；
召回层保留关键词、向量、热门、协同等多路互补；
排序层用传统 ranker 保证吞吐和稳定；
对 topN 候选用 cross-encoder 或 LLM rerank 做复杂语义重排；
最后把 LLM rerank 的偏好蒸馏成训练样本，减少线上调用成本。

评估：
离线看 Recall@K、NDCG@K、MRR、改写准确率、rerank win rate、分桶指标；
线上看 CTR/CVR/停留、无结果率、P95/P99、成本、投诉和安全违规。

难点：
query rewrite 会漂移，LLM rerank 慢且贵，离线 NDCG 涨不等于线上转化涨，点击日志还有位置偏差。

结果：
通过触发策略、缓存、topN 限制、蒸馏和 A/B guardrail，把语义能力引入链路，同时控制延迟和成本。
```

## 15. 高频追问答题骨架

### Q: LLM 能不能直接替代推荐系统？

不能简单替代。LLM 擅长语义理解、复杂推理、解释和样本构造，但推荐系统主链路要求毫秒级延迟、高吞吐、稳定校准和长期反馈闭环。更现实的是：

```text
LLM 做理解、特征、召回增强、rerank 小候选集和离线标注；
传统推荐系统负责主链路排序、探索、校准、A/B 和长期优化。
```

### Q: 为什么排序不能只看 AUC？

AUC 衡量整体正负样本排序，但线上用户只看前几个结果，且每个用户内部排序更重要。搜索推荐还要看 NDCG@K、MRR、Recall@K、GAUC、校准、CTR/CVR、延迟和 guardrail。

### Q: query rewrite 怎么评估？

不要只看改写文本是否“像人话”。要看：

- rewrite 后 Recall@K 是否提升。
- NDCG/MRR 是否提升。
- query drift 是否变多。
- 长尾 query 是否改善。
- 线上无结果率、点击、转化、延迟和成本。
- bad case 是否集中在实体、数字、否定、时间、地点。

### Q: LTR 和 CTR 预估是什么关系？

CTR 预估通常是 pointwise 学一个点击概率；LTR 更强调同一 query/user 下候选之间的相对顺序。实际系统可能同时用 CTR/CVR 作为排序分数的一部分，再用 NDCG、业务规则和多目标重排调整最终列表。

## 16. 面试前背诵版

搜广推系统是一个候选漏斗：多路召回保证覆盖，粗排快速过滤，精排用 CTR/CVR/相关性模型精细打分，重排处理多样性、新鲜度、业务规则和安全。搜索、推荐、广告的区别在于意图和目标：搜索有显式 query，推荐偏隐式兴趣，广告还要考虑 bid、ROI、预算和合规。排序指标不能只看 AUC，搜索常看 NDCG/MRR/Recall@K，推荐广告看 AUC/GAUC、CTR/CVR、校准和线上业务指标。Learning to Rank 分 pointwise、pairwise、listwise，LambdaMART 用 NDCG 变化给 pairwise 梯度加权。点击日志有位置偏差、曝光偏差和反馈闭环，最终要靠 A/B test 验证。LLM 在这个体系里更适合 query/item 理解、语义召回、rerank 小候选集、解释、合成数据和 eval，而不是无脑替代低延迟主链路。

## 本轮参考

- Deep Neural Networks for YouTube Recommendations：https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- Wide & Deep Learning for Recommender Systems：https://arxiv.org/abs/1606.07792
- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction：https://arxiv.org/abs/1703.04247
- Deep Interest Network for Click-Through Rate Prediction：https://arxiv.org/abs/1706.06978
- RankNet / LambdaRank / LambdaMART 论文：https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- RankGPT: Large Language Models are Zero-Shot Listwise Document Rankers：https://arxiv.org/abs/2304.09542
- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models：https://arxiv.org/abs/2104.08663
- Counterfactual Learning to Rank from Biased Data：https://dl.acm.org/doi/10.1145/3159652.3159737
- scikit-learn NDCG 文档：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
