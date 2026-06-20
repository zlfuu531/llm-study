# 答案版 27：搜索推荐广告 + LLM、Ranking 与 LTR

对应题目：`03_高频题单100题.md` 的 561-580。

用法：每题先说 30 秒版，再按 2 分钟版补链路、公式、指标和项目例子。搜广推题最怕只背模型名，要主动讲“目标、链路、指标、偏差、上线”。

## 561. 搜索、推荐、广告三者有什么区别？

30 秒版：

搜索有显式 query，重点是相关性和满足用户明确需求；推荐更多来自用户历史和上下文，重点是兴趣、留存和长期价值；广告还要考虑出价、ROI、预算、合规和用户体验。三者底层都有召回、排序、重排、评估和 A/B。

2 分钟版：

搜索的输入通常是 query + 用户/场景，指标常看 NDCG、MRR、Recall@K、CTR、无结果率。推荐没有明确 query 或 query 较弱，更依赖用户画像、行为序列、item 特征和上下文，指标可能是 CTR、观看时长、留存、多样性、新鲜度。广告除了 pCTR/pCVR，还要考虑 bid、eCPM、预算消耗、ROI、投诉率和合规风险。

面试句：

> 搜广推不是三个完全割裂的系统，而是目标和约束不同的候选排序问题。

## 562. 召回、粗排、精排、重排链路怎么讲？

30 秒版：

召回从海量 item 里找可能相关的候选，粗排用轻量模型快速过滤，精排用复杂特征和模型精细打分，重排再处理多样性、新鲜度、商业规则、安全和最终展示体验。

2 分钟版：

典型链路是：

```text
request -> query/user/context 理解 -> 多路召回
-> 粗排 -> 精排 -> 重排/策略 -> 展示 -> 反馈回流
```

召回追求覆盖和低延迟，宁可多召一些；粗排把几千候选降到几百；精排用 CTR/CVR/相关性模型做高质量排序；重排考虑列表整体，不只是单 item 分数，比如去重、多样性、广告位、频控、合规。

## 563. 多路召回怎么设计？

30 秒版：

多路召回要互补：关键词召回保专名和精确匹配，向量召回保语义，双塔/协同过滤保个性化，图召回保关系扩展，热门/新鲜召回保冷启动和时效，LLM 可做 query rewrite、实体归一和扩展。

2 分钟版：

我会先按业务场景拆 query/user/item 的主要信号，再设计召回通道：

- 搜索：BM25、语义向量、同义词、实体、类目、query rewrite。
- 推荐：双塔、协同过滤、item2item、user2item、图召回、热门/新鲜。
- 广告：定向召回、关键词、类目、人群包、预算和合规过滤。

评估看 Recall@K、coverage、空召回率、分桶召回和延迟。多路召回后要做去重、融合和来源打点，否则不知道哪一路真的有效。

## 564. 双塔召回和 cross-encoder rerank 有什么区别？

30 秒版：

双塔把 query/user 和 item 分别编码，点积打分，适合大规模 ANN 召回；cross-encoder 把 query 和 item 一起输入模型，交互更充分、更准，但慢，适合 rerank topN。

2 分钟版：

双塔：

```text
q = E_q(query/user)
d = E_d(item)
score = q · d
```

优点是 item embedding 可离线预计算，线上 ANN 检索快；缺点是 query-item 细粒度交互不足。Cross-encoder 把 `[query, item]` 一起编码，能看 token 级交互，相关性判断更强，但每个候选都要跑一次模型，所以只适合 top50/top100 级别重排。

LLM rerank 可以看更复杂的上下文和列表约束，但成本更高，工程上通常只对小候选集触发。

## 565. CTR、CVR、GMV 模型怎么建？

30 秒版：

CTR 预测点击概率，CVR 预测点击后转化概率，GMV/收益通常把概率、价格、出价或价值结合起来。模型输入包括用户、item、上下文、历史行为和交叉特征，线上还要校准和 A/B 验证。

2 分钟版：

常见定义：

```text
pCTR = P(click | user, item, context)
pCVR = P(conversion | click, user, item, context)
pCTCVR = P(click and conversion | user, item, context)
score = w1 * pCTR + w2 * pCVR + w3 * value - penalty
广告里常见 eCPM = bid * pCTR * pCVR * calibration
```

训练上常用 BCE/logloss。难点是样本选择偏差、类别不平衡、延迟反馈、校准、多目标冲突和线上指标不一致。面试时要说清 label 是曝光后产生的，天然有偏。

## 566. Wide&Deep、DeepFM、DIN 的直觉是什么？

30 秒版：

Wide&Deep 是记忆和泛化结合；DeepFM 是 FM 二阶交叉加 DNN 高阶交叉；DIN 是对用户历史做 attention，让当前候选 item 激活相关兴趣。

2 分钟版：

Wide&Deep：

```text
score = wide_linear(cross_features) + DNN(embedding_features)
```

Wide 记住高频规则，Deep 泛化到新组合。DeepFM：

```text
y = sigmoid(y_FM + y_DNN)
```

FM 学低阶交叉，DNN 学高阶非线性。DIN 的核心是 activation unit：

```text
interest = sum_i attention(candidate, history_i) * history_i
```

它认为用户兴趣和候选相关，不是一个固定平均向量。

## 567. Pointwise、Pairwise、Listwise LTR 有什么区别？

30 秒版：

Pointwise 把排序当单样本分类/回归，Pairwise 学两个 item 谁该排前，Listwise 直接看整个列表。越往后越贴近排序指标，但实现和数据要求也更高。

2 分钟版：

Pointwise：

```text
loss = BCE(y, sigmoid(score))
```

简单，适合 CTR label，但没有直接优化相对顺序。Pairwise：

```text
loss = -log sigmoid(s_i - s_j)
```

关注同一 query/user 下正负 item 的相对位置。Listwise 把整个候选列表作为训练对象，目标更接近 NDCG/MAP。工业里常混用：CTR 模型做 pointwise 预估，LTR/rerank 再考虑列表指标和业务约束。

## 568. LambdaMART 为什么适合排序？

30 秒版：

LambdaMART 把 LambdaRank 的排序梯度思想和 GBDT/MART 结合，能用 NDCG 变化给重要 pair 更大梯度，同时处理复杂非线性特征，传统 LTR 里很强。

2 分钟版：

LambdaRank 的直觉是：交换两个 item 如果会让 NDCG 变化很大，那这对 item 的训练权重要更大。LambdaMART 用 GBDT 来拟合这些 lambda 梯度，所以既能利用树模型处理非线性和特征交叉，又能贴近排序指标。

适合：

- 搜索/推荐/广告里有大量人工、统计、行为和内容特征。
- label 是相关性等级或点击偏好。
- 需要相对稳定、可解释、上线成本低的 ranker。

## 569. AUC、GAUC、NDCG、MRR、MAP 怎么选？

30 秒版：

AUC 看整体正负排序，GAUC 看用户内排序，NDCG 看多级相关性且重视头部位置，MRR 看第一个正确结果，MAP 看多个相关结果的平均排序质量。

2 分钟版：

公式直觉：

```text
AUC ~= P(score_pos > score_neg)
GAUC = 按用户曝光量加权的用户内 AUC
DCG@K = sum((2^rel_i - 1) / log2(i + 1))
NDCG@K = DCG@K / IDCG@K
MRR = mean(1 / 第一个相关结果位置)
MAP = mean(AP)
```

搜索常看 NDCG/MRR/Recall@K；推荐广告常看 AUC/GAUC/logloss/校准，再结合 CTR/CVR/GMV；RAG 检索也常看 Recall@K、MRR、NDCG 和 citation accuracy。

## 570. 位置偏差和曝光偏差怎么处理？

30 秒版：

点击日志有偏，排前面更容易被看见，没曝光就没反馈。处理方法包括随机探索、估计 propensity、IPS/SNIPS 加权、点击模型、位置特征建模和最终 A/B 验证。

2 分钟版：

位置偏差是 item 排前更容易点击，不代表它一定更相关。曝光偏差是只观察到被展示 item 的反馈，没展示的 item 没 label。处理方法：

- 小流量随机打散或探索，估计展示概率。
- 用 `reward / propensity` 做 IPS 估计。
- 用 SNIPS 降低方差。
- 训练模型加入 position/context 特征。
- 点击模型估计 examination probability。
- 线上 A/B 验证实际效果。

提醒：IPS 能减偏，但 propensity 很小时方差大。

## 571. 离线指标涨，线上不涨怎么排查？

30 秒版：

先确认实验和数据，再看离线指标是否对齐业务目标，然后分桶看用户、query、item、位置、类目，最后拆召回、排序、重排、策略、延迟和 bad case。

2 分钟版：

排查模板：

```text
数据可靠性 -> 指标一致性 -> 分桶 -> 链路定位 -> bad case -> A/B guardrail
```

可能原因：

- AUC 对 top 位置不敏感，NDCG/CTR 不涨。
- 离线 label 有偏，线上分布变化。
- 长尾改善但流量占比低。
- rerank 提升相关性但延迟变差。
- 点击涨但转化或满意度下降。
- LLM rewrite 引入 query drift。
- 商业规则或过滤覆盖了模型收益。

## 572. LLM 怎么用于搜索推荐？

30 秒版：

LLM 可做 query 理解、query rewrite、实体归一、item 摘要和标签、语义召回、topN rerank、推荐理由、合成训练样本和评估辅助，但主链路排序通常仍需要低延迟 ranker。

2 分钟版：

按链路讲：

- 召回前：意图识别、纠错、扩展、实体归一。
- 离线建库：item 摘要、属性抽取、多模态理解、embedding。
- 召回后：cross-encoder 或 LLM rerank topN。
- 展示层：解释、摘要、广告文案。
- 训练评估：生成 hard negative、偏好对、LLM-as-judge、bad case 聚类。

工程重点是触发策略、延迟、缓存、结构化输出、蒸馏和安全合规。

## 573. LLM rerank 和传统 ranker、cross-encoder 有什么区别？

30 秒版：

传统 ranker 快、稳、可大规模服务；cross-encoder 更准但只能 rerank 小候选；LLM rerank 能处理复杂语义和多约束，但慢、贵、格式稳定性和线上吞吐都是问题。

2 分钟版：

传统 ranker 输入结构化特征和 embedding 特征，适合主链路。Cross-encoder 输入 query-item 对，适合 top50/top100。LLM rerank 可以输入 query、候选列表和规则，让模型输出排序 id 或偏好解释，适合复杂 query、小候选集、离线标注和高价值流量。

成本控制：

- 只对困难 query 触发。
- 限制候选 topN。
- 缓存和 batch。
- 结构化输出候选 id。
- 把 LLM 偏好蒸馏给小 ranker。

## 574. Query rewrite / expansion 怎么评估？

30 秒版：

看 rewrite 是否提升召回和排序，而不是只看文字是否好看。指标包括 Recall@K、NDCG、MRR、无结果率、query drift、长尾分桶、线上 CTR/CVR、延迟和成本。

2 分钟版：

评估分三层：

1. 改写质量：意图是否保留，实体、数字、时间、否定是否正确。
2. 检索效果：Recall@K、NDCG、MRR、空召回率、长尾 query 改善。
3. 线上效果：CTR/CVR/满意度、无结果率、延迟、成本、投诉。

坏 case 常在实体归一、否定词、范围条件、地名、品牌、数字和时间上。可以用 LLM 做初筛，但必须有人工抽检和线上实验。

## 575. RAG 检索评估和搜索指标怎么打通？

30 秒版：

RAG 检索本质也是搜索。可以用 Recall@K、MRR、NDCG 看证据召回和排序，再加 context precision、citation accuracy、answer faithfulness，把检索指标和最终回答质量连起来。

2 分钟版：

RAG 里 query 是用户问题，doc/chunk 是候选文档。搜索指标可直接用：

- Recall@K：金标准证据是否进 topK。
- MRR：第一个正确证据排第几。
- NDCG：多个证据按相关性排序是否合理。
- MAP：多个相关 chunk 的整体排序。

RAG 还要多看：

- context recall / precision。
- answer correctness。
- citation accuracy。
- hallucination rate。
- latency/cost。

检索好不代表生成好，所以要把 retrieval eval 和 answer eval 分开打点。

## 576. 推荐系统怎么兼顾相关性、多样性、新鲜度和商业目标？

30 秒版：

主排序分数负责相关性和转化，重排层再加入多样性、新鲜度、频控、去重、探索和商业规则。核心是多目标权衡，并用 guardrail 防止一个指标压倒用户体验。

2 分钟版：

常见做法：

- 分数融合：`score = relevance + ctr/cvr/value - penalty`。
- MMR 或相似度惩罚做多样性。
- 新内容探索流量池处理冷启动。
- 频控和去重减少疲劳。
- 类目、作者、品牌配额控制。
- 广告和商业目标通过位置、预算、ROI、体验指标约束。

线上必须看 guardrail：停留、留存、投诉率、负反馈、违规率、P95 延迟。

## 577. 广告/推荐中的安全合规怎么做？

30 秒版：

要做内容安全、广告法合规、隐私保护、反作弊、敏感人群保护、解释真实性和日志权限控制。LLM 生成文案或解释时尤其要防夸大、幻觉和违规承诺。

2 分钟版：

治理链路：

- 入库前审核 item/ad 内容、资质、类目和敏感词。
- 召回和排序前做合规过滤。
- LLM 生成摘要、推荐理由、广告文案时用 schema、敏感词、规则和人工抽检。
- 隐私上避免把个人敏感信息送入外部 API 或日志明文。
- 监控投诉率、违规率、误伤率、申诉和灰度回滚。

广告还要关注夸大宣传、医疗金融等高风险行业、未成年人和地域法规。

## 578. LLM 带来的延迟和成本怎么控？

30 秒版：

用触发策略、topN 限制、小模型优先、缓存、batch、结构化短输出、蒸馏和异步离线处理。高吞吐主链路不要无条件调用大模型。

2 分钟版：

控制思路：

- 只对长尾、低置信、复杂 query 触发。
- 先用轻量模型或规则判断是否需要 LLM。
- LLM rerank 只看 top20/top50。
- prompt 压缩，输出候选 id 而不是长文本。
- 缓存 query rewrite、item 摘要和 rerank 结果。
- 离线用 LLM 标注 pair/listwise 数据，线上用小 ranker。
- 监控 P95/P99、token 成本、超时率和降级比例。

面试句：

> LLM 能力要放在高价值、低候选数、可缓存或可蒸馏的环节。

## 579. 搜广推 + LLM 项目 8 分钟怎么讲？

30 秒版：

按背景、基线、方案、指标、难点、结果讲。重点突出为什么用 LLM、放在哪一层、怎么评估、怎么控成本和怎么处理 query drift、偏差、延迟。

2 分钟版：

模板：

```text
背景：长尾 query/语义相关性/推荐解释不足。
基线：BM25/向量/双塔召回 + 传统 ranker。
方案：LLM 做 query rewrite/item 理解，cross-encoder 或 LLM rerank topN，偏好蒸馏到小 ranker。
指标：Recall@K、NDCG、MRR、AUC/GAUC、CTR/CVR、无结果率、P95、成本。
难点：rewrite drift、位置偏差、离线线上不一致、LLM 延迟和合规。
结果：分桶改善、A/B 结果、bad case 回流、灰度和回滚。
```

如果没有真实搜广推项目，可以把 RAG 检索项目改写成“搜索排序链路”，但不要假装做过广告竞价。

## 580. 搜广推 + LLM 面试前最后怎么复习？

30 秒版：

最后只背一条主线：召回、粗排、精排、重排；三类模型：CTR/CVR、LTR、LLM rerank；五类指标：AUC/GAUC、NDCG/MRR/MAP、Recall@K、业务指标、延迟成本；两类风险：偏差和合规。

2 分钟版：

面试前清单：

- 能画出召回、粗排、精排、重排链路。
- 能区分搜索、推荐、广告。
- 能解释 AUC、GAUC、NDCG、MRR、MAP、Recall@K。
- 能说 pointwise、pairwise、listwise、LambdaMART。
- 能讲 Wide&Deep、DeepFM、DIN 的直觉。
- 能回答位置偏差、曝光偏差、离线线上不一致。
- 能讲 LLM 在 query rewrite、item 理解、rerank、样本构造和 eval 里的作用。
- 能说清 LLM 成本延迟怎么控。
- 准备一个 8 分钟项目版本，至少包含指标、bad case 和上线取舍。

背诵版：

> 搜广推 + LLM 不是把大模型塞进每一步，而是在传统高吞吐排序系统里，把大模型用于语义理解、复杂重排和数据闭环，同时用触发策略、蒸馏、缓存和 A/B guardrail 控制成本与风险。
