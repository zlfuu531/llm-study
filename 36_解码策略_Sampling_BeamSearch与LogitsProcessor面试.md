# 解码策略、Sampling、Beam Search 与 Logits Processor 面试

这一章面向所有大模型岗位，尤其是算法、推理服务、AI 应用开发、代码模型、结构化输出和评测岗位。很多同学会背 `temperature/top_p/top_k`，但面试官追两层就会暴露：你到底知不知道模型每一步怎么从 logits 变成下一个 token，为什么 beam search 在开放式对话里容易无聊，为什么 top-p 比 top-k 更动态，为什么 repetition penalty 可能伤害事实性，为什么同样参数有时结果还不完全复现。

如果时间很紧，先背这句：

> 解码策略是在每一步 next-token logits 上做选择和约束：先得到 logits，再经过 temperature、logits processor/warper、mask 或 top-k/top-p 截断，最后 argmax 或 multinomial sampling 得到下一个 token。不同策略是在质量、多样性、稳定性、延迟、成本和可控性之间取舍。

相关答案版：[answers/31_解码策略_Sampling_BeamSearch_LogitsProcessor_答案版.md](answers/31_解码策略_Sampling_BeamSearch_LogitsProcessor_答案版.md)

相邻章节：

- [24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md](24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md)：decode 阶段、KV Cache、TTFT/TPOT、streaming 和 speculative decoding。
- [38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md](38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md)：draft model、EAGLE、Medusa、MTP、accept rate、无损验证和低延迟 decode。
- [27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md](27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md)：结构化输出、constrained decoding、parser/retry。
- [29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md](29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md)：Best-of-N、Self-Consistency、verifier、test-time scaling。
- [35_Tokenizer_BPE_SentencePiece与Token预算面试.md](35_Tokenizer_BPE_SentencePiece与Token预算面试.md)：EOS/stop tokens、token budget、输出长度和 tokenizer 边界。

## 1. 先从 next-token 分布开始

Decoder-only LLM 的生成是自回归过程：

```text
prompt tokens x_1...x_t
-> model forward
-> logits z_t over vocab
-> decoding strategy
-> next token x_{t+1}
-> append to context
-> repeat until EOS / stop / max_new_tokens
```

每一步模型输出的是词表大小维度的 logits：

```text
z_t: [vocab_size]
p_i = softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

面试要强调：模型只负责给出“下一个 token 的打分分布”，最终选哪个 token 是解码策略决定的。训练目标常是最大似然，但推理时如果每一步都最大化概率，可能会导致重复、保守、模板化。

## 2. Generation loop 怎么写

简化生成循环：

```python
tokens = prompt_ids
for step in range(max_new_tokens):
    logits = model(tokens).logits[:, -1, :]
    logits = apply_processors(logits, tokens)
    next_token = decode_next(logits)
    tokens = concat(tokens, next_token)
    if next_token == eos_token_id:
        break
```

真实工程里还会有：

- KV Cache：decode 时只计算新 token，复用历史 K/V。
- attention mask：padding 和 causal mask。
- tokenizer：stop strings 需要映射到 token 序列。
- streaming：每生成一个或几个 token 就发给客户端。
- logits processors：禁词、重复惩罚、最小长度、schema 约束等。
- stopping criteria：EOS、stop words、max_new_tokens、超时、工具调用结束。

## 3. Temperature

Temperature 是最常问的公式：

```text
p_i = softmax(z_i / T)
```

直觉：

- `T = 1`：原始分布。
- `T < 1`：分布变尖，更确定，更保守。
- `T > 1`：分布变平，更多样，也更容易跑偏。
- `T -> 0`：接近 greedy argmax。

注意：temperature 不改变 logits 的排序，只改变概率分布的尖锐程度。如果后面接 top-p，temperature 会改变累计概率，从而改变 nucleus 集合。

面试句：

> Temperature 控制随机性，不是控制模型智商。低 temperature 更稳定，高 temperature 更多样，但事实性、格式和安全任务通常不能盲目拉高。

## 4. Greedy decoding

Greedy 每一步选概率最大的 token：

```text
x_{t+1} = argmax_i p_i
```

优点：

- 确定性强。
- 快，简单。
- 适合格式固定、分类、短答案、代码补全中的保守场景。

缺点：

- 容易局部最优。
- 开放式生成会无聊、重复。
- 不会探索多个可能路径。

常见追问：greedy 一定可复现吗？

理论上同模型、同输入、同实现、同硬件确定性设置下更接近可复现；但线上系统可能因为不同 batch、不同 kernel、浮点非确定性、模型版本和后处理差异产生小变化。面试不要把“temperature=0”说成绝对保证。

## 5. Multinomial sampling

Sampling 是按概率分布随机抽样：

```text
x_{t+1} ~ Categorical(p)
```

优点：

- 有多样性。
- 可以生成多个候选。
- 适合创意写作、开放问答、代码 pass@k、reasoning self-consistency。

缺点：

- 可能抽到低质量长尾 token。
- 输出不稳定。
- 需要 seed、评估和安全控制。

纯 sampling 通常太冒险，所以会配合 temperature、top-k、top-p 或其他 truncation。

## 6. Top-k sampling

Top-k 只保留概率最高的 k 个 token，其他 token 置为 `-inf`，再重新归一化采样：

```text
S = top_k_indices(p, k)
p'_i = p_i / sum_{j in S} p_j, if i in S
p'_i = 0, otherwise
x_{t+1} ~ Categorical(p')
```

直觉：固定只从“最可能的 k 个词”里抽。

优点：

- 简单稳定。
- 避免极低概率长尾 token。
- 易手写，面试常考。

缺点：

- k 是固定数量，不随分布形状变化。
- 当分布很尖时，保留 k 个可能太多。
- 当分布很平时，保留 k 个可能太少。

适合说法：

> Top-k 是固定候选数截断，控制的是候选集合大小，不直接控制概率质量。

## 7. Top-p / Nucleus sampling

Top-p 也叫 nucleus sampling。它按概率从高到低排序，保留累计概率达到 `p` 的最小 token 集合：

```text
sort tokens by p_i descending
S = smallest prefix such that sum_{i in S} p_i >= top_p
sample from normalized distribution over S
```

直觉：候选集合大小是动态的。

- 分布尖：可能只保留少数 token。
- 分布平：会保留更多 token。

这也是它比 top-k 更适合开放式生成的原因之一。Neural Text Degeneration / Nucleus Sampling 论文的核心直觉是：语言模型尾部分布有不可靠 token，直接从全分布采样会导致退化；nucleus sampling 通过截掉低概率尾部，在多样性和流畅性之间折中。

## 8. Top-k vs Top-p

| 维度 | Top-k | Top-p |
| --- | --- | --- |
| 控制对象 | 固定 token 数 | 累计概率质量 |
| 候选集合大小 | 固定 | 动态 |
| 分布很尖 | 仍保留 k 个 | 候选很少 |
| 分布很平 | 只保留 k 个 | 候选变多 |
| 面试一句话 | 控制候选数量 | 控制概率质量 |

常见组合：

```text
temperature + top_p
temperature + top_k
temperature + top_k + top_p
```

组合时要说明顺序通常是先处理 logits，再截断，再采样；不同框架实现可能细节不同，但直觉是：temperature 改分布形状，top-k/top-p 裁掉候选，sampling 再抽样。

## 9. Beam Search

Beam Search 同时保留 `num_beams` 条最高分路径。每一步扩展每条 beam 的候选 token，然后选总分最高的若干条：

```text
score(sequence) = sum_t log p(x_t | x_<t)
```

简化流程：

```text
start with one beam
for each step:
  expand each beam with candidate next tokens
  score = old_score + logprob(next_token)
  keep top num_beams beams
stop when EOS / max length
```

优点：

- 比 greedy 探索更多路径。
- 输出稳定。
- 适合机器翻译、摘要、结构较固定的任务。

缺点：

- 计算更贵，近似 `num_beams` 倍。
- 开放式对话容易重复、模板化、无聊。
- 长度偏置，需要 length penalty。
- 多样性不一定好，多个 beam 可能很相似。

面试句：

> Beam Search 是近似最大化序列概率，不是提升创造力的方法。开放式聊天更常用 sampling，翻译/摘要等目标较确定任务更适合 beam。

## 10. Length penalty 和长度偏置

Beam Search 用 logprob 求和时，序列越长，累加的负 logprob 越多，天然偏向短序列。常见修正：

```text
score = sum_logprob / length^alpha
```

或其他长度归一化方式。

直觉：

- `alpha` 太小：仍偏短。
- `alpha` 太大：可能偏长、啰嗦。
- early stopping 和 EOS 处理也会影响最终长度。

面试里不要把 length penalty 当万能。它只是缓解长度偏置，不能解决 beam search 开放式生成重复和无聊的问题。

## 11. Repetition penalty / no-repeat ngram

常见重复控制：

### Repetition penalty

对已生成 token 的 logits 做惩罚。常见实现会根据 logit 正负做不同缩放：

```text
if token already generated:
  if logit > 0: logit = logit / penalty
  else:         logit = logit * penalty
```

### Frequency / presence penalty

- presence penalty：出现过就惩罚，鼓励引入新 token。
- frequency penalty：出现次数越多惩罚越重。

### No-repeat ngram

禁止生成会形成重复 n-gram 的 token。

风险：

- 惩罚过强会让模型不用必要术语，影响事实性。
- 代码、数学、表格、JSON 里重复 token 很正常，不能粗暴禁。
- 中英文 tokenizer 差异会改变惩罚效果。

面试句：

> 重复惩罚不是质量提升开关，它是在“少重复”和“别伤害必要表达”之间取舍，必须按任务分桶调参。

## 12. LogitsProcessor / LogitsWarper

Hugging Face 里常把生成前的 logits 处理分成两类：

- LogitsProcessor：根据规则修改 scores，例如最小长度前禁止 EOS、no-repeat ngram、bad words、forced BOS/EOS。
- LogitsWarper：改变采样分布，例如 temperature、top-k、top-p。

面试不用死背类名，但要讲清抽象：

```text
raw logits
-> processors: hard constraints / penalties / forced tokens
-> warpers: temperature / top-k / top-p
-> sampling or argmax
```

常见用途：

- 禁止敏感 token。
- 强制输出某些 token。
- 最小长度前不允许 EOS。
- JSON/schema constrained decoding。
- repetition penalty。
- top-k/top-p/temperature。

## 13. Constrained decoding 和结构化输出

Constrained decoding 是在每一步屏蔽不合法 token。它和 prompt 的关系：

- Prompt：告诉模型“应该输出什么”。
- Constrained decoding：从解码层限制“只能输出什么”。
- Parser/retry：输出后检查和修复。

例子：

```text
JSON schema -> 当前前缀可接受 token 集合
logits[illegal_tokens] = -inf
sample/argmax from legal tokens
```

优点：

- 语法合法率高。
- 适合 JSON、SQL、工具调用参数。

代价：

- 解码速度可能下降。
- schema 太复杂可能导致卡住或质量下降。
- 语法合法不等于业务正确。

更完整结构化输出见 [27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md](27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md)。

## 14. EOS、stop strings 和长度控制

停止条件常见几类：

- `eos_token_id`：模型生成 EOS token。
- `max_new_tokens`：最多生成多少新 token。
- `max_length`：输入 + 输出总长度上限。
- stop strings：生成文本包含某些字符串后截断。
- timeout：服务端超时。
- tool call boundary：工具调用 JSON 完成。

常见坑：

- `max_length` 和 `max_new_tokens` 混用。
- tokenizer 不同导致 stop string 跨 token，检测不稳定。
- EOS 被 repetition penalty 或 min length 影响。
- chat template 里 EOS/BOS 加错，导致停不下来。
- streaming 时 stop string 要处理跨 chunk。

面试句：

> 生产里长度控制不是只设一个 max token，要同时管理 EOS、stop、输出预算、超时、流式截断和安全过滤。

## 15. 解码参数怎么按任务选

没有一组参数适合所有任务。可以按任务分桶：

| 任务 | 常见策略 | 原因 |
| --- | --- | --- |
| 分类 / 抽取 / JSON | 低温或 greedy + constrained decoding | 追求稳定和格式 |
| RAG 问答 | 低温 + top-p，限制输出长度 | 要忠于证据 |
| 创意写作 | 较高 temperature + top-p | 需要多样性 |
| 代码生成 | 低温 pass@1；多候选 pass@k | 平衡正确性和探索 |
| 数学推理 | 低温或多采样 + verifier | 需要正确性和搜索 |
| 翻译/摘要 | beam 或低温 sampling | 目标较确定 |
| Agent 工具调用 | 低温 + schema/工具约束 | 参数必须可解析 |

别在面试里给固定神参。更好的说法是：

> 我会先定义任务和评估集，再按质量、格式、成本、延迟和安全分桶调参，用离线 eval 和线上灰度确定参数，而不是相信某个通用 top_p。

## 16. 解码和推理性能

解码策略会影响性能：

- Beam Search：每步保留多个 beam，计算和 KV Cache 近似按 beam 数放大。
- Sampling：本身开销通常小，但 logits processing、CPU 后处理、禁词和 schema 约束可能带来额外开销。
- Constrained decoding：每步计算合法 token 集合，复杂 schema 可能拖慢。
- 多采样：Best-of-N、Self-Consistency、pass@k 会成倍增加 token 成本。
- 输出长度：输出越长，decode step 越多，TPOT 和总延迟越高。

和推理引擎的关系：

```text
prefill: 处理 prompt，影响 TTFT
decode: 每步选 token，影响 TPOT / ITL
sampling/logits processor: decode loop 里的小算子和后处理
```

## 17. 可复现性和 seed

常见误区：设置 seed 就一定复现。

实际还受这些影响：

- 模型版本和 tokenizer 版本。
- generation config。
- batch 形态和 padding。
- GPU kernel 非确定性。
- distributed serving 路由到不同实例。
- floating point 精度。
- 流式截断、stop string 后处理。

面试表达：

> 我会把 model revision、tokenizer、chat template、prompt、generation config、seed、框架版本和服务版本一起记录。对强一致任务尽量用低温或约束解码，对开放式任务接受统计意义上的稳定。

## 18. 评估解码策略

不要只看“感觉更好”。可以按任务评估：

### 质量

- exact match / F1。
- pass@1 / pass@k。
- win rate。
- LLM-as-judge + 人工抽检。
- faithfulness / citation accuracy。

### 多样性

- distinct-n。
- self-BLEU。
- 多候选去重率。
- answer diversity。

### 稳定性

- 同 prompt 多次采样一致性。
- 格式有效率。
- 工具参数准确率。
- 拒答一致性。

### 成本和延迟

- input/output tokens。
- TTFT / TPOT / P95/P99。
- cost per solved task。
- verifier / rerank / parser 重试次数。

## 19. 手撕 Top-k / Top-p 的关键点

Top-k 常见写法：

```python
values, indices = torch.topk(logits, k, dim=-1)
mask = torch.full_like(logits, float("-inf"))
mask.scatter_(dim=-1, index=indices, src=values)
probs = torch.softmax(mask / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

Top-p 常见写法：

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
sorted_probs = torch.softmax(sorted_logits, dim=-1)
cumulative = torch.cumsum(sorted_probs, dim=-1)

remove = cumulative > top_p
remove[..., 1:] = remove[..., :-1].clone()
remove[..., 0] = False

sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
filtered_logits = torch.full_like(logits, float("-inf"))
filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
probs = torch.softmax(filtered_logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

易错点：

- top-p 要至少保留一个 token。
- `cumulative > top_p` 后要右移一位，保留第一个超过阈值的 token。
- mask 用 `-inf`，再 softmax。
- batch 维度别丢。
- temperature 接近 0 时不要直接除 0。
- 采样前要确保概率没有 NaN。

## 20. 项目里怎么讲解码策略

8 分钟项目表达可以这样说：

```text
背景：
业务需要稳定回答 / 创意生成 / 代码生成 / 工具调用。

问题：
默认参数下出现重复、格式不稳、输出过长、成本高或答案太保守。

方案：
按任务分桶设置 generation config：
RAG 低温 + top_p + 引用约束；
JSON/tool 低温 + schema constrained decoding；
代码任务用低温 pass@1 和多候选 pass@k；
复杂推理用多采样 + verifier。

评估：
离线看准确率、格式有效率、faithfulness、pass@k、重复率；
线上看解决率、转人工、P95、output tokens、成本和投诉。

上线：
把 model、prompt、chat template、tokenizer、generation config、stop tokens 一起版本化；
灰度新参数，监控 bad case 和成本。
```

## 21. 高频快答

### Top-p 为什么叫 nucleus？

因为它保留当前分布里累计概率达到阈值的“核心集合”，集合大小随分布动态变化。

### Temperature 越高越好吗？

不是。高 temperature 增加多样性，也增加幻觉、格式错误和安全风险。

### Beam Search 为什么开放式聊天不好？

它偏向高概率路径，容易模板化、重复、缺少多样性；开放式生成更常用 sampling。

### Repetition penalty 能解决所有重复吗？

不能。重复可能来自 prompt、模型、上下文、解码参数、stop 条件和训练数据。惩罚过强还会伤害必要术语。

### Constrained decoding 是不是保证答案正确？

不是。它主要保证语法或 schema 合法，业务正确性仍要靠模型能力、上下文、校验和评估。

## 22. 面试背诵版

LLM 生成时每一步先输出 vocab 上的 logits，再通过解码策略选择下一个 token。Greedy 选最大概率，稳定但容易局部最优和重复；sampling 从概率分布抽样，有多样性但可能抽到长尾坏 token；temperature 用 `softmax(logits / T)` 控制分布尖锐程度；top-k 固定保留 k 个最高概率 token，top-p 保留累计概率达到阈值的动态 nucleus。Beam Search 保留多条高分路径，适合翻译摘要这类目标较确定任务，但开放式聊天容易无聊和重复，还要处理长度偏置。生产里还会用 repetition penalty、no-repeat ngram、bad words、min length、EOS、stop strings 和 constrained decoding 等 logits processor/stopping criteria。参数不能靠玄学，要按任务分桶，用私有 eval、线上灰度、延迟和成本指标来确定。

## 本轮参考

- Hugging Face Generation strategies：[https://huggingface.co/docs/transformers/en/generation_strategies](https://huggingface.co/docs/transformers/en/generation_strategies)
- Hugging Face Utilities for Generation / LogitsProcessor：[https://huggingface.co/docs/transformers/en/internal/generation_utils](https://huggingface.co/docs/transformers/en/internal/generation_utils)
- Hugging Face text generation config：[https://huggingface.co/docs/transformers/en/main_classes/text_generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
- The Curious Case of Neural Text Degeneration / Nucleus Sampling：[https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)
- Fast Inference from Transformers via Speculative Decoding：[https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
