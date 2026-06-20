# 答案版 31：解码策略、Sampling、Beam Search 与 Logits Processor

对应题目：`03_高频题单100题.md` 的 641-660。

用法：先把每题 30 秒版背顺，再用 2 分钟版补公式、工程取舍和项目排查。解码策略题最怕答成“调 top_p 就行”，要主动讲 logits、分布截断、长度控制、重复惩罚、约束解码、评估和成本。

## 641. LLM 生成 loop 怎么讲？

30 秒版：

LLM 生成是自回归 next-token loop：prompt tokens 输入模型，得到最后一个位置的 vocab logits，经解码策略选出 next token，把它拼回上下文，直到 EOS、stop 或 max_new_tokens。

2 分钟版：

流程是：

```text
tokens -> model forward -> logits[:, -1, :]
-> logits processor / warper
-> argmax or sampling
-> append next token
-> update KV cache
-> stop condition
```

工程里还要管 KV Cache、streaming、EOS/stop string、max token、tokenizer 和安全过滤。面试要说清模型给的是分布，策略负责选 token。

## 642. Temperature 的公式和作用是什么？

30 秒版：

Temperature 用 `p_i = softmax(z_i / T)` 调整概率分布尖锐程度。`T` 小更确定，`T` 大更多样，`T -> 0` 接近 greedy。

2 分钟版：

Temperature 不改变 logits 排序，但改变概率差距。低温适合分类、抽取、JSON、RAG 忠实回答；高温适合创意生成和多候选探索。高温会增加幻觉、格式错误和安全风险，所以不能把它当提升能力的开关。

## 643. Greedy decoding 和 sampling 有什么区别？

30 秒版：

Greedy 每步选概率最大 token，稳定但保守；sampling 按概率随机抽样，更多样但不稳定，可能抽到低质量 token。

2 分钟版：

Greedy 适合确定性任务、短答案和格式任务，但容易局部最优和重复。Sampling 适合开放式生成、代码 pass@k、Self-Consistency 和创意任务，但要配合 temperature、top-k/top-p、重复控制、seed 和 eval。面试一句话：greedy 追稳定，sampling 追探索。

## 644. Top-k sampling 怎么讲？

30 秒版：

Top-k 只保留概率最高的 k 个 token，其他 token 置为 `-inf`，重新 softmax 后从这 k 个候选里采样。

2 分钟版：

Top-k 控制候选集合大小，能截掉低概率长尾 token。优点是简单稳定、易手写；缺点是 k 固定，不随分布形状变化。分布很尖时 k 可能太大，分布很平时 k 可能太小。

## 645. Top-p / nucleus sampling 怎么讲？

30 秒版：

Top-p 按概率降序保留累计概率达到 `p` 的最小 token 集合，再归一化采样。候选集合大小是动态的。

2 分钟版：

如果分布很尖，top-p 会保留很少 token；如果分布很平，会保留更多 token。它比 top-k 更自适应。核心作用是截掉不可靠长尾，同时保留一定多样性。开放式生成里常用 temperature + top_p。

## 646. Top-k 和 Top-p 怎么区分？

30 秒版：

Top-k 控制固定候选数量，Top-p 控制累计概率质量。Top-k 的集合大小固定，Top-p 的集合大小随分布动态变化。

2 分钟版：

Top-k 像“只看前 k 名”，Top-p 像“看到累计概率够了就停”。当模型很确定时，top-p 候选很少；当模型不确定时，top-p 候选会变多。面试常见答法：top-k 更简单，top-p 更自适应。

## 647. Temperature、Top-k、Top-p 的顺序和组合怎么理解？

30 秒版：

通常先用 temperature 改变 logits 分布尖锐程度，再用 top-k/top-p 截断候选集合，最后从归一化后的分布采样。

2 分钟版：

Temperature 会改变概率形状，因此会影响 top-p 的 nucleus 集合。Top-k 和 top-p 都是在避免从长尾坏 token 里采样。组合参数要按任务评估，不能背一个通用神参。RAG/JSON 通常低温，创意任务可以更高温，代码/推理可以低温 pass@1 或多采样 pass@k。

## 648. Beam Search 的核心流程是什么？

30 秒版：

Beam Search 每一步保留 `num_beams` 条最高分候选路径，扩展它们的下一个 token，再继续保留总分最高的若干条，直到 EOS 或长度上限。

2 分钟版：

序列分数常用 logprob 累加：

```text
score(seq) = sum_t log p(x_t | x_<t)
```

它比 greedy 探索更多路径，适合翻译、摘要这类目标较确定任务。代价是计算和 KV Cache 近似随 beam 数放大，开放式对话容易重复、模板化。

## 649. Beam Search 为什么需要 length penalty？

30 秒版：

因为 logprob 累加通常会偏向短序列，length penalty 用长度归一化缓解短答案偏置。

2 分钟版：

常见形式：

```text
score = sum_logprob / length^alpha
```

`alpha` 太小仍偏短，太大可能偏长和啰嗦。它只能缓解长度偏置，不能解决 beam search 在开放生成里无聊、重复和缺少多样性的问题。

## 650. 为什么 Beam Search 不一定适合开放式聊天？

30 秒版：

Beam Search 近似找高概率序列，高概率文本往往安全、普通、模板化，所以开放式聊天里容易无聊、重复，缺少多样性。

2 分钟版：

开放式聊天没有唯一标准答案，过度追高概率会导致 generic response。Sampling 可以探索多种合理回答，更适合创作、聊天、brainstorming。Beam 更适合翻译、摘要、结构化任务这类目标较确定场景。

## 651. Repetition penalty / no-repeat ngram 解决什么？

30 秒版：

它们用于抑制重复。Repetition penalty 惩罚已生成 token，no-repeat ngram 禁止形成重复 n-gram。

2 分钟版：

重复可能来自模型、prompt、上下文、解码参数和 stop 条件。惩罚过强会伤害必要术语，特别是代码、数学、表格、JSON 和专有名词。面试要说：这是取舍，不是质量万能开关，要按任务分桶调参。

## 652. Presence penalty 和 frequency penalty 有什么区别？

30 秒版：

Presence penalty 是 token 出现过就惩罚，frequency penalty 是出现次数越多惩罚越重。

2 分钟版：

Presence 更像鼓励引入新内容，frequency 更像减少重复次数。它们适合开放式生成，但在事实问答、代码和结构化输出里要谨慎，因为重复术语可能是必要表达。

## 653. LogitsProcessor / LogitsWarper 是什么？

30 秒版：

它们是在生成前修改 logits 的组件。Processor 常做硬约束和惩罚，Warper 常做 temperature、top-k、top-p 这类采样分布变换。

2 分钟版：

抽象流程：

```text
raw logits
-> processors: bad words / min length / no-repeat / forced tokens
-> warpers: temperature / top-k / top-p
-> argmax or sampling
```

面试重点不是背类名，而是理解生成前可以对 vocab logits 做 mask、惩罚、强制和截断。

## 654. Constrained decoding 和 prompt 约束有什么区别？

30 秒版：

Prompt 是告诉模型应该怎么输出，constrained decoding 是在每一步屏蔽不合法 token，强制只从合法集合里选。

2 分钟版：

结构化输出里，prompt 只能提高倾向，不能保证 JSON/schema 合法。Constrained decoding 可以根据当前前缀和语法规则把非法 token 设为 `-inf`。它能提高语法合法率，但不保证业务正确，而且复杂 schema 会拖慢解码。

## 655. EOS、stop strings、max_new_tokens 怎么区分？

30 秒版：

EOS 是模型生成的结束 token，stop strings 是文本后处理停止条件，max_new_tokens 是最多生成多少新 token。

2 分钟版：

`max_length` 通常是输入 + 输出总长度，`max_new_tokens` 是新增输出长度。stop string 可能跨 token，streaming 时要跨 chunk 检测。EOS/BOS/chat template 加错会导致停不下来或过早停止。

## 656. 设置 seed 就一定可复现吗？

30 秒版：

不一定。Seed 只控制随机源，还受模型版本、tokenizer、generation config、batch、GPU kernel、框架版本和服务路由影响。

2 分钟版：

强一致任务应记录 model revision、tokenizer、chat template、prompt、generation config、seed、框架版本和服务版本。开放式任务通常追统计稳定，不追逐字一致；分类/JSON 等任务可以用低温或 constrained decoding 提高稳定性。

## 657. 解码策略怎么影响延迟和成本？

30 秒版：

Beam、多采样、Self-Consistency、Best-of-N 会增加 forward 次数和 token 成本；输出越长 decode step 越多；复杂 logits processor 和 constrained decoding 也可能拖慢每步生成。

2 分钟版：

Beam Search 近似按 beam 数放大计算和 KV Cache。多候选生成按候选数增加成本。Constrained decoding 每步要算合法 token 集合，复杂 schema 会增加开销。生产里要看 TTFT、TPOT、output tokens、P95/P99 和 cost per solved task。

## 658. 代码生成里的 pass@k 和解码策略有什么关系？

30 秒版：

pass@k 依赖多次采样候选，只要 k 个候选里有一个通过测试就算成功。采样越多探索越强，但成本越高。

2 分钟版：

代码任务常用低温 pass@1 看单次可靠性，用多采样 pass@k 看搜索空间覆盖。temperature/top_p 太低会候选太相似，太高会语法和逻辑错误变多。真实项目还要看测试质量、patch 安全、重复候选和运行成本。

## 659. 手写 Top-k / Top-p sampling 注意什么？

30 秒版：

Top-k 用 `torch.topk` 保留 k 个 logits，其他置 `-inf`；Top-p 先按 logits 排序，softmax 后算累计概率，mask 掉超过阈值的 token，再 scatter 回原词表顺序采样。

2 分钟版：

Top-p 易错点：至少保留一个 token，`cumulative > top_p` 后要右移一位，保留第一个超过阈值的 token；mask 用 `-inf`；batch 维别丢；temperature 不能除 0；采样前检查 NaN。

## 660. 解码策略相关项目 8 分钟怎么讲？

30 秒版：

按“任务目标 -> 参数问题 -> 分桶策略 -> 离线 eval -> 线上灰度 -> 成本延迟 -> bad case 回流”讲，不要只说调了 top_p。

2 分钟版：

示例：

```text
我们把任务分成 RAG 问答、JSON 抽取、代码生成和复杂推理。
RAG 用低温 + top_p 控制幻觉和输出长度；
JSON 用低温 + schema 约束提高格式有效率；
代码生成用 pass@1/pass@k 比较不同采样参数；
复杂推理用多采样 + verifier，但加预算路由。
上线时把 model、prompt、tokenizer、chat template、generation config 和 stop tokens 一起版本化，
通过灰度看解决率、格式有效率、P95、output tokens 和成本。
```
