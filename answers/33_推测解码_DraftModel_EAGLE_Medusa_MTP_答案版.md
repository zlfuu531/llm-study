# 答案版 33：推测解码、Draft Model、EAGLE/Medusa/MTP

对应题目：`03_高频题单100题.md` 的 681-700。

用法：先把每题 30 秒版背顺，再用 2 分钟版补公式、工程取舍和线上排查。推测解码题最怕答成“小模型先生成”，要主动讲 target 验证、分布保持、accept rate、TPOT、QPS、EAGLE/Medusa/MTP 和什么时候会变慢。

## 681. Speculative decoding 解决什么瓶颈？

30 秒版：

它解决 decode 阶段逐 token 串行等待的问题。用便宜 proposer 先草拟多个 token，再让目标模型一次 forward 并行验证，从而尽量每次目标模型调用推进多个 token。

2 分钟版：

普通 decode 每生成一个 token 都要 target model forward 一次，强依赖上一步输出。小 batch / 中低 QPS 时 GPU 可能没有被充分利用，TPOT/ITL 高。Speculative decoding 把多个未来 token 的候选一次交给 target 验证，目标是减少 target forward 次数。它主要改善 decode，不直接解决长 prompt prefill 导致的 TTFT。

## 682. 推测解码基本流程怎么讲？

30 秒版：

draft/proposer 先生成 `gamma` 个候选 token，target model 对这些候选一次并行验证，从左到右接受连续正确 token；遇到拒绝就修正采样并停止本轮，全部接受时还可额外生成一个 target token。

2 分钟版：

流程：

```text
prefix
-> draft proposes y_1...y_gamma
-> target verifies prefix + draft tokens in one pass
-> accept prefix of draft tokens
-> reject point uses target/corrected distribution
-> append accepted tokens
-> repeat
```

核心是 target 仍然是裁判，draft 只是降低候选生成成本。

## 683. 为什么 speculative sampling 可以保持目标模型分布？

30 秒版：

因为 draft token 不是直接无条件输出，而是按目标分布 `p` 和 draft 分布 `q` 做接受/拒绝；接受概率常写作 `min(1, p(y)/q(y))`，拒绝后从修正分布采样，所以最终等价于 target sampling。

2 分钟版：

如果 draft 抽到 `y`，目标模型给出 `p(y)`，draft 给出 `q(y)`：

```text
a(y) = min(1, p(y) / q(y))
```

拒绝时从：

```text
r(x) ∝ max(0, p(x) - q(x))
```

采样。这个 correction 保证输出分布不偏离 target。面试要强调：无损来自验证和修正，不是来自 draft 天然准确。

## 684. Greedy 验证和 sampling 验证有什么区别？

30 秒版：

Greedy 场景通常看 draft token 是否等于 target argmax；sampling 场景要用概率接受/拒绝和修正采样，才能保持目标模型采样分布。

2 分钟版：

Greedy 更简单：draft 猜中 target argmax 就接受，否则输出 target argmax 并停止本轮。Sampling 更复杂，因为目标是保持 `softmax(logits)` 下的随机分布，不是只匹配最大 token。temperature、top-p、top-k、logits processor 都会影响 `p` 和 `q`，也会影响接受率。

## 685. Speculative decoding 的加速条件和粗略公式是什么？

30 秒版：

粗略看：

```text
speedup ≈ ((A + 1) * C_T) / (gamma * C_D + C_V + overhead)
```

`A` 是平均接受 token 数。draft 要便宜，target 验证要能并行，接受率要高。

2 分钟版：

普通 decode 生成 `A+1` 个 token 需要约 `(A+1)C_T`。推测解码一轮需要 draft 草拟 `gamma C_D`，target 验证 `C_V`，再加调度和 KV 开销。真正加速需要 `C_D` 小、`C_V` 接近一次 target forward、`A` 大。否则只是多跑了一个 draft。

## 686. accept rate 怎么定义？为什么会低？

30 秒版：

accept rate 通常是接受的 draft token 数除以提议的 draft token 数，也会看每轮平均接受长度。低的原因是 draft 太弱、任务难预测、temperature 高、gamma 太大、约束解码冲突或数据分布不匹配。

2 分钟版：

指标：

```text
token acceptance rate = accepted / proposed
mean accepted length = 每轮平均接受 token 数
tokens per target step = 每次 target 验证推进 token 数
```

开放写作、复杂推理、高温采样、长摘要通常更难预测；代码补全、模板化输出、重复上下文更容易高接受。

## 687. Draft model 怎么选？

30 秒版：

Draft 要足够便宜，也要足够像目标模型。太小不准接受率低，太大草拟成本高，最好 tokenizer/vocab 一致，训练域和目标流量接近。

2 分钟版：

选择标准：

- 同 tokenizer / vocab，方便 token 对齐和验证。
- 同模型族或蒸馏自 target，分布更接近。
- 低延迟、低显存，可量化。
- 对业务 prompt 有高接受率。
- 不让双模型调度和 KV Cache 管理压垮系统。

一句话：draft 不是越小越好，而是便宜和准确之间的平衡。

## 688. 为什么 draft 和 target 最好同 tokenizer / vocab？

30 秒版：

因为 target 验证的是 draft 提出的 token id。如果 tokenizer 不一致，同一段文本可能对应不同 token 序列，验证、对齐、概率计算和 KV Cache 更新都会复杂甚至不可用。

2 分钟版：

Speculative decoding 是 token 级算法，接受概率和 target logits 都针对词表 token。如果 draft token id 与 target token id 语义不同，就不能直接比较 `p(y)` 和 `q(y)`。不同 tokenizer 可以做文本级转码，但会引入对齐、长度变化、概率不可比和额外开销。

## 689. EAGLE 的核心思想是什么？

30 秒版：

EAGLE 是 feature-level speculative decoding。它不只是预测未来 token，而是预测目标模型接近输出层的 feature，并结合 shifted token 降低 feature uncertainty，再由 target 验证。

2 分钟版：

EAGLE 的直觉是：token 层未来不确定性大，feature 层更容易外推，但 feature 也有 uncertainty，所以引入提前一位的 token 序列帮助预测 next feature。它通常需要 EAGLE draft 模型，适合在 vLLM/SGLang/TensorRT-LLM 这类 runtime 中作为强 speculator。

## 690. Medusa 的核心思想是什么？

30 秒版：

Medusa 给 LLM 增加多个 decoding heads，让这些 heads 并行预测未来多个位置的 token，再用 tree attention 构造候选路径并验证，从而减少 decode 步数。

2 分钟版：

它不依赖独立小 draft model，而是在目标模型上加 heads。多个 heads 预测 `t+2/t+3/...` 的候选，tree attention 把多个候选路径合并验证。优点是少维护一个 draft 模型；代价是要训练 heads，tree 分支会增加计算和显存，不同实现对采样和无损保证要具体看。

## 691. MTP 和 DeepSeek 里的 MTP 怎么讲？

30 秒版：

MTP 是 Multi-Token Prediction，让模型不只预测 next token，还预测后续 token。它可以作为训练辅助目标，也可以在模型原生支持时作为 speculative decoding 的 proposer。

2 分钟版：

在 DeepSeek-V3/R1 语境里，MTP 是高效架构和训练目标的一部分。训练上，它提供更远未来 token 的监督信号；推理上，如果模型带 MTP 模块，runtime 可以利用它草拟未来 token，不必额外维护独立 draft model。但是否加速仍取决于验证机制、accept rate、框架支持和负载。

## 692. N-gram / suffix speculative decoding 适合什么？

30 秒版：

它从 prompt 或已生成文本中找重复片段作为 draft，不需要额外模型。适合代码、日志、模板文本和重复上下文，不适合完全开放的新内容。

2 分钟版：

优点是便宜、易开、无额外模型显存；缺点是候选只来自已有上下文，命中率依赖重复性。它更像保守优化：如果文本有明显重复，能减少 target decode；如果没有重复，收益有限。

## 693. vLLM / SGLang / TensorRT-LLM 里 speculative decoding 怎么讲？

30 秒版：

vLLM 强调中低 QPS、memory-bound 场景降低 inter-token latency，支持 EAGLE、MTP、draft、n-gram、suffix 等；SGLang 支持 EAGLE-2/3、MTP、standalone、NGRAM 等；TensorRT-LLM 支持 draft-target、Medusa、EAGLE、n-gram、lookahead 等。

2 分钟版：

不要背命令，要讲选择逻辑：先确认瓶颈在 decode，再按模型和负载选择 proposer。EAGLE/MTP 通常是更强模型化方法；n-gram 不需要额外模型但适用面窄；standalone draft 需要维护小模型；tree/topk/num draft tokens 越大潜在接受更多，也更吃显存和计算。

## 694. 为什么 speculative decoding 可能变慢？

30 秒版：

accept rate 低、draft 太重、tree/gamma 太大、高 QPS 下 target 已饱和、瓶颈在 prefill/排队/网络、结构化约束导致大量拒绝，都可能让它变慢。

2 分钟版：

Speculative decoding 多了 proposer、验证、KV Cache 和调度开销。只有当平均每次 target forward 推进多个 token 时才赚。如果每轮只接受 0-1 个 token，draft 就是额外负担。多卡并行、LoRA、量化、guided decoding 混用时还要看 runtime 兼容性。

## 695. 它对 TTFT、TPOT、QPS 分别有什么影响？

30 秒版：

主要改善 TPOT/ITL，因为它优化 decode；对长 prompt 的 TTFT 改善有限；QPS/吞吐收益取决于负载，高 QPS 大 batch 时可能不明显。

2 分钟版：

TTFT 大多由 prefill、排队、tokenizer 和调度影响。Speculative decoding 通常在首 token 后发挥作用。低 QPS latency-sensitive 场景收益更直接；高 QPS throughput 场景要看 GPU 是否已经饱和、draft 是否抢占资源、batching 是否受影响。

## 696. 它和 structured output / guided decoding 会冲突吗？

30 秒版：

可能。结构化输出会限制合法 token，如果 proposer 不知道 grammar/schema，草拟 token 很容易非法或被拒；CPU grammar 状态和 GPU 验证也会增加协调开销。

2 分钟版：

理想情况下 proposer 和 target 都要感知同一套约束。否则 draft 生成不合法 token，accept rate 下降。JSON schema、tool call、正则/CFG 约束还可能影响 CUDA graph、batching 和 logits processor 顺序。面试要说：结构化输出场景要单独评估，不要默认开。

## 697. 线上评估看哪些指标？

30 秒版：

看 accept rate、mean accepted length、TPOT/ITL、tokens/s、P95/P99、显存、OOM、draft latency、verification latency、质量分和格式有效率。

2 分钟版：

分桶评估：

```text
短问答 / 长输出 / 代码 / RAG / JSON / 工具调用 / 数学推理
```

对比 baseline normal decoding，记录不同 gamma/topk/proposer 下的性能和质量。只看 tokens/s 不够，因为可能质量掉、格式错或 P99 变差。

## 698. num_speculative_tokens / topk / tree size 怎么调？

30 秒版：

先小后大。gamma 或 num steps 太小收益有限，太大拒绝级联和显存开销增加；topk 增加候选多样性但扩大 tree 计算和内存。

2 分钟版：

调参顺序：

```text
固定 target 和 generation config
-> gamma/num_steps 从 3-5 起
-> 看 accept rate 和 mean accepted length
-> 再调 topk/tree size
-> 分桶看质量和 P95/P99
-> 不适合的任务动态关闭
```

参数不是越大越好，关键是每次 target 验证平均能多推进多少 token。

## 699. accept rate 低、OOM、质量异常怎么排查？

30 秒版：

accept rate 低查 draft 匹配、temperature、gamma、约束解码和任务类型；OOM 降 tree/num_draft_tokens/concurrency；质量异常查是否使用了非严格 acceptance、logits processor、stop 条件和框架实现。

2 分钟版：

排查链路：

```text
固定 tokenizer / template / generation config
-> baseline 对照
-> 看 accept rate 分桶
-> 调低 gamma/topk
-> 换 draft/EAGLE/MTP/ngram
-> 检查 structured output 和 sampling
-> 看显存、KV Cache、CUDA graph、batching
-> 必要时按任务路由或关闭
```

如果理论上 lossless 但质量变了，要怀疑实现路径、近似 acceptance、随机性、processor 顺序或 stop tokens。

## 700. 推测解码面试前最后怎么复习？

30 秒版：

背基本流程、无损接受公式、加速条件、accept rate、Draft/EAGLE/Medusa/MTP/n-gram 区别，以及“为什么可能变慢”的排查答案。

2 分钟版：

最后清单：

- 基本流程：draft proposes，target verifies，accept prefix。
- 公式：`a(y)=min(1,p(y)/q(y))`，拒绝后 `max(0,p-q)` 修正。
- 条件：draft 便宜、验证并行、accept rate 高、decode 是瓶颈。
- 方法：draft model、n-gram/suffix、Medusa、EAGLE、MTP。
- 指标：accept rate、mean accepted length、TPOT、tokens/s、P95/P99、显存。
- 风险：高 QPS、低接受率、draft 过重、结构化输出、多卡/量化/LoRA 兼容。
- 项目讲法：baseline、ablation、分桶评估、动态开关、fallback。
