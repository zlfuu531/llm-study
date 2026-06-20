# 答案版 19：推理引擎、vLLM、SGLang 与 TensorRT-LLM

对应题号：401-420。建议先读 [24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md](../24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 401. 推理引擎和普通模型 API 封装有什么区别？

30 秒版：

推理引擎不是简单包一层 `model.generate()`，而是面向在线并发的执行系统。它要管理请求队列、调度、prefill/decode、KV Cache、动态 batching、采样、流式输出、指标、限流和故障恢复。

2 分钟版：

普通 API 封装更像单请求视角：输入 prompt，调用模型，返回结果。推理引擎是多请求视角：很多请求动态到达，每个请求 prompt 长度、输出长度、优先级和租户都不同，GPU 显存又有限，所以要做 admission control、scheduler、KV Cache manager、continuous batching 和 metrics。

一个完整链路是：gateway 做鉴权和限流，请求进入 queue，scheduler 决定哪些请求进入 prefill 或 decode，KV Cache manager 分配和释放显存，decode loop 动态维护 batch，sampler 采样 token，streamer 流式返回，metrics 记录 TTFT、TPOT、queue time、tokens/s 和错误。

面试加分句：

> 推理引擎解决的是“在线并发下如何高效、稳定、可观测地使用 GPU”，不是“单个 prompt 怎么生成答案”。

## 402. prefill 和 decode 的瓶颈分别是什么？

30 秒版：

Prefill 处理整个 prompt，通常更偏计算密集，影响 TTFT；decode 每次生成一个 token，要反复读写 KV Cache，更偏显存带宽和调度，影响 TPOT/ITL。

2 分钟版：

Prefill 阶段把所有输入 token 一次性过模型，生成首 token 所需的 hidden states 和后续 decode 的 KV Cache。prompt 越长、RAG 拼接越多、system prompt 越复杂，prefill 越慢，所以它主要决定用户多久看到第一个 token。

Decode 阶段是自回归生成，每一步只输入上一个 token，但要读历史所有 token 的 KV Cache，并追加新的 K/V。每一步计算量不大，但步数多、显存读写频繁，所以常受 KV Cache 长度、显存带宽、batch 调度和 kernel 效率影响。

对比表：

| 阶段 | 影响指标 | 典型优化 |
| --- | --- | --- |
| Prefill | TTFT | prompt 裁剪、prefix cache、chunked prefill |
| Decode | TPOT/ITL | continuous batching、PagedAttention、量化、spec decoding |

## 403. TTFT、TPOT、ITL、tokens/s、P95/P99 分别看什么？

30 秒版：

TTFT 看首 token 延迟，TPOT/ITL 看生成过程中每个 token 的间隔，tokens/s 看吞吐，P95/P99 看尾延迟。线上不能只看平均值，必须分 input tokens、output tokens 和请求类型看。

2 分钟版：

TTFT 是从请求发出到第一个 token 返回的时间，用户体感非常敏感。它包括网关、排队、tokenizer、RAG、prefill、首 token decode 和网络 flush。

TPOT 是每个输出 token 平均耗时，ITL 是流式输出中相邻 token 间隔。它们主要反映 decode 阶段是否顺畅。

tokens/s 可以分 input tokens/s、output tokens/s 和 total tokens/s。生产排查最好分开看，因为 input tokens/s 高可能只是 prompt 很长，并不代表用户体验好。

P95/P99 是尾延迟，能暴露长 prompt、长输出、排队、batch 调度和下游阻塞。平均值好看但 P99 很差，线上仍然会被用户感知为不稳定。

## 404. KV Cache 显存怎么估算？

30 秒版：

常用公式是 `2 * layers * active_requests * seq_len * kv_heads * head_dim * bytes_per_element`。其中 2 是 K 和 V，GQA/MQA 会减少 kv_heads，长上下文和高并发会线性放大 KV Cache。

2 分钟版：

每层 attention 都要保存历史 token 的 K/V，decode 时后续 token 会读这些缓存。估算公式：

```text
KV Cache bytes
≈ 2 * num_layers * active_requests * seq_len
  * num_kv_heads * head_dim * bytes_per_element
```

举例：32 层、`kv_heads=8`、`head_dim=128`、FP16、单请求 4096 token：

```text
2 * 32 * 4096 * 8 * 128 * 2 bytes ≈ 512 MB
```

16 个并发接近 8 GB KV Cache，还没算权重和临时 buffer。

追问点：

- MHA 的 `kv_heads` 通常等于 attention heads。
- GQA/MQA 会减少 K/V head 数，所以 KV Cache 更省。
- 长上下文、长输出、高并发都会线性增加 KV Cache。
- KV Cache 量化能省显存，但要验证长上下文质量。

## 405. PagedAttention 解决什么问题？

30 秒版：

PagedAttention 解决在线 serving 中 KV Cache 动态增长和长短请求混合带来的显存碎片、预留浪费问题。它像操作系统分页一样，把 KV Cache 切成 block/page 管理。

2 分钟版：

朴素做法会给每个请求预留较大的连续 KV Cache 区域，但真实请求长度差异很大，有的很短，有的很长，还会逐 token 增长，容易造成碎片和浪费。

PagedAttention 把 KV Cache 分成固定大小 block，请求的逻辑 token 顺序由映射表维护，物理显存 block 不必连续。这样一个请求需要多少 block 就分配多少，结束后释放 block，能提升显存利用率，也让更多请求并发进入同一块 GPU。

面试可补：

- 它主要优化 serving 中的 KV Cache 管理。
- 对 continuous batching、beam search、prefix sharing 等场景有帮助。
- 它不是 attention 数学本身的改变，而是缓存管理和 kernel 访问方式的工程优化。

## 406. FlashAttention 和 PagedAttention 为什么不是一回事？

30 秒版：

FlashAttention 优化 attention 计算，减少中间矩阵写回和显存访问；PagedAttention 优化在线 serving 的 KV Cache 管理，减少碎片和预留浪费。一个管“怎么算”，一个管“缓存怎么放”。

2 分钟版：

FlashAttention 发生在单次 forward 的 attention kernel 里，它通过 tiling 和 IO-aware 设计减少 HBM 读写，不显式保存完整 attention matrix，从而提升训练/推理 attention 计算效率。

PagedAttention 发生在 LLM serving 的 KV Cache 管理里。在线服务有很多请求，每个请求长度不同，还会动态生成 token。PagedAttention 用 page/block 管理历史 K/V，减少显存碎片和浪费。

对比：

| 技术 | 关键词 | 主要解决 |
| --- | --- | --- |
| FlashAttention | IO-aware attention kernel | attention 计算效率 |
| PagedAttention | KV Cache block/page | serving 显存管理 |

一句话：

> FlashAttention 是计算优化，PagedAttention 是 serving 缓存管理优化。

## 407. Continuous Batching 为什么能提高吞吐？

30 秒版：

因为 LLM 请求输出长度不同，静态 batch 会被最长请求拖住。continuous batching 在 decode loop 中动态移出已完成请求、加入新请求，让 GPU 持续有活干。

2 分钟版：

传统 static batching 是凑一批一起跑，等这批都结束再跑下一批。LLM 在线推理中，每个请求输出长度不同，短请求结束后如果还要等长请求，GPU 资源和排队时间都会浪费。

Continuous batching 每个 decode step 都重新维护活跃序列集合。完成的请求退出，新请求在合适时机加入。这样 GPU 的 batch 更稳定，吞吐更高，queue time 更低。

代价：

- scheduler 复杂。
- KV Cache 动态分配和释放更复杂。
- 要处理公平性、优先级和长请求保护。

面试不要只说“batch size 变大”。重点是“动态到达、动态退出、decode step 级调度”。

## 408. Chunked Prefill 解决什么问题，代价是什么？

30 秒版：

Chunked prefill 把长 prompt 的 prefill 切成多个 chunk，避免长请求长时间独占 GPU，从而改善短请求 TTFT 和尾延迟。代价是调度更复杂，chunk 太小可能降低整体吞吐。

2 分钟版：

在 RAG 或长上下文场景里，一个请求可能有几千到几万 input tokens。完整 prefill 会占用 GPU 较长时间，其他短请求只能等，导致 P95/P99 TTFT 上升。

Chunked prefill 把长输入切块，在 chunk 之间插入其他请求的 prefill 或 decode，使短请求不用等长请求全部 prefill 完。它本质是调度优化，用一点调度开销换更好的公平性和尾延迟。

代价：

- chunk 太小会增加 scheduler overhead。
- 长 prompt 单请求完成时间可能变长。
- 要和 continuous batching、KV Cache 管理和优先级策略配合。

上线建议：

> 用真实流量按 input length 分桶压测，看短请求 TTFT、长请求完成时间和整体 tokens/s 是否符合目标。

## 409. Prefix Cache / RadixAttention 什么时候有效？

30 秒版：

当请求有大量 token 级完全相同的前缀时有效，比如固定 system prompt、工具 schema、few-shot 示例、重复 RAG 上下文。RadixAttention 用树状结构复用公共前缀，适合多轮和 agentic workflow。

2 分钟版：

Prefix cache 的前提是前缀 token 序列完全一致，才能复用已经计算好的 KV Cache。它适合 system prompt 很长且固定、RAG 模板稳定、工具说明重复、代码 Agent 在同一仓库反复迭代等场景。

命中低的原因包括：

- prompt 中加入动态时间戳、随机 ID。
- 用户信息放在前缀开头。
- RAG 文档排序变化。
- 工具列表动态变化。
- cache 太小频繁淘汰。
- 多租户隔离限制跨用户复用。

RadixAttention 可以把共享前缀组织成树，公共前缀只存一份，不同请求在分叉位置继续增长。它对多轮对话、Agent、结构化调用和重复 prompt 模板更友好。

## 410. Speculative Decoding 怎么加速？

30 秒版：

用小的 draft model 先草拟多个 token，再用大的 target model 一次性验证。如果接受率高，就能让 target model 一次前进多个 token，减少逐 token decode 的等待。

2 分钟版：

普通 decode 每次大模型 forward 只生成一个 token，串行性强。Speculative decoding 让便宜路径先提出多个候选 token，然后 target model 并行验证这些候选。被接受的 token 可以一次提交，遇到拒绝位置再由 target model 采样修正。

收益条件：

- draft model 明显便宜。
- draft 分布接近 target，接受率高。
- target 验证多个 token 的并行效率高。
- decode 是主要瓶颈。

面试可补：

> 标准 speculative decoding 在算法设计正确时可以保持 target model 的采样分布，不是简单用小模型替代大模型。

## 411. Speculative Decoding 接受率低为什么可能变慢？

30 秒版：

接受率低意味着 draft 生成的大量 token 会被 target 拒绝，草拟成本白花了，还额外占显存和调度资源。如果 draft 太大或验证不高效，就可能比普通 decode 更慢。

2 分钟版：

Speculative decoding 的总成本可以粗略看成：

```text
总成本 ≈ draft 生成成本 + target 验证成本 + 拒绝后的修正成本
```

如果接受率高，target 一次验证能提交多个 token，平均每 token target forward 成本下降。如果接受率低，大部分草拟 token 被拒绝，target 还是要频繁修正，draft 成本就变成额外负担。

还可能变慢的原因：

- draft model 太大，省不了多少。
- draft 占显存导致主模型 batch size 降低。
- prompt 分布变化，draft 和 target 不匹配。
- 验证 kernel 没有高效并行。
- 瓶颈不在 decode，而在 RAG、排队、网络或 tokenizer。

上线判断：

> 看接受率、每 token 成本、显存占用、P95/P99 和真实业务输出质量，不只看实验室 demo tokens/s。

## 412. vLLM 的核心设计怎么讲？

30 秒版：

vLLM 是通用 LLM serving 引擎，核心是 PagedAttention 管 KV Cache、continuous batching 管动态请求、prefix cache 和 chunked prefill 优化长 prompt 和重复前缀，并提供 OpenAI-compatible API、指标和多种部署优化。

2 分钟版：

vLLM 解决在线推理的几个痛点：KV Cache 显存浪费、长短请求混合导致 GPU 利用率低、重复前缀重复计算、长 prompt 阻塞短请求，以及应用侧需要稳定 API 和监控。

可以按四个关键词讲：

1. PagedAttention：把 KV Cache 分块管理，减少碎片和预留浪费。
2. Continuous batching：decode 阶段动态加入和移出请求，提高吞吐。
3. Prefix cache / chunked prefill：复用相同前缀，改善长 prompt 场景 TTFT。
4. Serving 能力：OpenAI-compatible server、metrics、量化、多 GPU、spec decoding 等。

适合场景：

- 通用聊天、RAG、Agent 后端。
- 需要快速搭建高吞吐在线推理。
- 对模型和量化支持有多样需求。

## 413. SGLang 适合什么场景？

30 秒版：

SGLang 适合多步 LLM workflow、Agent、结构化输出和重复前缀很多的场景。它既有 LLM program 的表达能力，也有 RadixAttention 等 serving 优化。

2 分钟版：

如果只是单次 chat completion，vLLM 很常用；如果你的业务是多轮、多工具、多分支、结构化输出和大量 prompt 模板复用，SGLang 的优势会更明显。

SGLang 可以把 LLM 调用组织成程序，比如先抽取字段、再调用工具、再根据结果继续生成。它的 serving 后端通过 RadixAttention 复用公共前缀 KV Cache，适合 system prompt、tool schema、few-shot 示例和上下文模板重复的 agentic workflow。

典型场景：

- Agent 工程化。
- 工具调用和结构化 JSON 输出。
- 长工具说明/格式约束复用。
- 同一任务流程中多次模型调用。

面试对比句：

> vLLM 更像通用高吞吐 serving 引擎，SGLang 更强调把 LLM 调用写成可优化的程序，并对重复前缀和结构化 workflow 做优化。

## 414. TensorRT-LLM 和 vLLM/SGLang 怎么区分？

30 秒版：

TensorRT-LLM 更偏 NVIDIA 生态下的高性能推理 runtime 和优化工具链，关注 kernel、量化、多 GPU、IFB、paged KV 和硬件效率；vLLM 更偏通用 serving；SGLang 更偏 LLM program 和 agentic workflow。

2 分钟版：

三者定位不同：

| 维度 | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| 重点 | 通用在线 serving | LLM program + serving | NVIDIA 高性能 runtime |
| 关键词 | PagedAttention、continuous batching | RadixAttention、structured generation | IFB、kernel、quantization |
| 优势 | 上手快，生态广 | 多步 workflow 和前缀复用 | 极致性能和硬件优化 |
| 成本 | 相对低 | 需要设计 workflow | 工程门槛较高 |

如果公司要快速搭通用推理服务，我会优先看 vLLM；如果业务是复杂 Agent 和结构化生成，我会评估 SGLang；如果是 NVIDIA GPU 集群、模型较稳定、团队追求极致成本和吞吐，会考虑 TensorRT-LLM。

## 415. Prefill/Decode disaggregation 为什么出现？

30 秒版：

因为 prefill 和 decode 的资源特征不同：prefill 更偏大块计算，decode 更偏逐 token、KV Cache 和显存带宽。拆开后可以分别优化资源池和调度，但会引入 KV 传输和系统复杂度。

2 分钟版：

在同一组 GPU 上同时跑 prefill 和 decode，会互相干扰。长 prompt prefill 可能阻塞短请求的 decode，decode 的细碎步骤也可能打断大 prefill 的吞吐。

PD disaggregation 把 prefill worker 和 decode worker 分开。prefill worker 处理 prompt 并生成 KV，decode worker 接收或加载 KV 后继续逐 token 生成。

收益：

- prefill 和 decode 独立扩缩容。
- 更好控制 TTFT 和 TPOT。
- 不同 worker 可以用不同 batch 和调度策略。

挑战：

- KV Cache 传输成本高。
- 调度、容错和 cache locality 更难。
- 网络/RDMA/GPU-GPU 传输成为新瓶颈。

结论：

> 它适合规模较大、长上下文和混合流量明显的系统；小团队不一定一开始就需要。

## 416. KV Cache 量化有什么收益和风险？

30 秒版：

KV Cache 量化能降低 decode 阶段显存占用和带宽压力，提高并发或长上下文能力；风险是数值误差影响生成质量，尤其长上下文、数学、代码和结构化输出要重点评估。

2 分钟版：

推理服务显存通常由模型权重、KV Cache、临时 buffer 和框架开销组成。长上下文和高并发下，KV Cache 可能成为主要显存压力。把 KV 从 FP16/BF16 量化到 FP8/INT8 等格式，可以减少显存和带宽，提升并发上限或降低 TPOT。

风险：

- 长上下文中误差累积。
- 注意力分布变化，导致引用错误或格式不稳。
- 某些 kernel 没有真实加速。
- 与 prefix cache、PagedAttention、多卡通信、spec decoding 的兼容性要验证。

评估：

- 私有业务 eval。
- 长上下文 eval。
- 代码/数学/结构化输出。
- 安全策略和拒答一致性。
- P95/P99、tokens/s、显存峰值和成本。

## 417. 线上 TTFT 变高怎么排查？

30 秒版：

把 TTFT 拆成 gateway、queue、tokenizer、RAG、prefill、首 token decode 和 network。先看 queue time、input tokens 分布、RAG 耗时和 prefill time，再决定是扩容、裁剪 prompt、开 prefix cache/chunked prefill，还是优化上游。

2 分钟版：

排查顺序：

1. 看是否是所有请求变慢，还是长 prompt / 某租户 / 某模型变慢。
2. 看 queue time 是否上升，判断容量和限流问题。
3. 看 input tokens 分布是否变长，RAG 是否拼了更多上下文。
4. 看 tokenizer CPU 和 RAG/rerank 耗时。
5. 看 prefill time 和 GPU 利用率。
6. 看 prefix cache 命中率是否下降。
7. 看 chunked prefill 是否配置改变。
8. 看首包网络、代理和流式 flush 是否异常。

对应动作：

- queue 高：扩容、限流、优先级、admission。
- input 长：prompt 裁剪、RAG top-k 调整、上下文压缩。
- prefill 高：prefix cache、chunked prefill、模型/量化优化。
- 上游慢：拆 RAG、rerank、工具调用。

## 418. 线上 TPOT 变高怎么排查？

30 秒版：

TPOT 主要看 decode。排查 output length、活跃 batch、KV Cache 长度、显存带宽、kernel、量化、多卡通信和 speculative decoding 接受率。

2 分钟版：

TPOT 变高说明生成过程中每个 token 慢了。先看请求分布：是不是输出更长、长上下文更多、并发更多。然后看 engine 指标：active seqs、batched tokens、KV Cache block 使用率、GPU util、memory bandwidth、SM util。

如果多卡推理，要看 TP/PP 通信；如果启用量化，要确认 kernel 是否走到了预期路径；如果启用 speculative decoding，要看 draft 成本和接受率。如果 batch 不饱，要看请求到达模式和 scheduler 参数；如果 batch 过大，则可能尾部延迟变差。

回答模板：

> TTFT 看 prefill 链路，TPOT 看 decode 链路。TPOT 高我优先看 output token 分布、KV Cache 长度、batch 形态、GPU 带宽和 kernel，然后再看通信、sampler、streamer 和 spec decoding。

## 419. 显存够但吞吐上不去怎么排查？

30 秒版：

显存够只是容量够，不代表算力、带宽或调度用满。要看请求量、batch 形态、GPU util、memory bandwidth、CPU tokenizer、RAG 上游、多卡通信、kernel 路径和网络流式输出。

2 分钟版：

可能原因分几类：

1. 流量不足：请求太少，batch 不饱。
2. 调度问题：`max_num_batched_tokens`、`max_num_seqs` 太保守。
3. CPU 瓶颈：tokenizer、采样、日志、JSON 序列化慢。
4. 上游瓶颈：RAG、rerank、工具调用拖慢。
5. Decode memory-bound：显存带宽满但 SM util 不高。
6. 多卡通信：TP all-reduce 或 KV 传输慢。
7. kernel 问题：量化或算子没有走高效 kernel。
8. IO 问题：网络、流式 flush、监控埋点阻塞。

排查指标：

- GPU SM util / memory bandwidth。
- input/output tokens/s。
- queue time。
- tokenizer time。
- prefill/decode time。
- active seqs / batched tokens。
- P95/P99 latency。

## 420. 推理服务项目怎么讲 8 分钟？

30 秒版：

按背景、流量、架构、引擎选择、优化手段、指标结果、事故排查和复盘讲。重点给出真实 token 分布、TTFT/TPOT/P95、GPU 利用率、成本和一次问题定位。

2 分钟版：

8 分钟结构：

1. 背景：业务为什么需要在线 LLM 服务，目标是延迟、吞吐、成本还是稳定性。
2. 流量：QPS、input/output token 分布、长 prompt 占比、是否流式。
3. 架构：gateway、queue、推理引擎、KV cache、监控、灰度。
4. 选型：为什么选 vLLM/SGLang/TensorRT-LLM。
5. 优化：continuous batching、prefix cache、chunked prefill、量化、限流。
6. 指标：TTFT、TPOT、P95/P99、tokens/s、GPU util、成本。
7. 排查：举一次 TTFT 高、TPOT 高或 OOM 的定位过程。
8. 复盘：如果重做，会补压测、eval、隔离、回滚和自动扩缩容。

示例说法：

> 我们用 vLLM 搭了 OpenAI-compatible 推理服务，网关负责鉴权、限流和 trace。上线初期 RAG 请求 TTFT P95 偏高，我们把 input tokens 分桶后发现长文档请求占比上升，于是做了上下文裁剪、prefix cache 和 chunked prefill；decode 侧通过限制 max output、调整 batched tokens 和量化方案控制 TPOT。最后用私有压测回放验证 P95、tokens/s 和单次成本都达到目标。

## 本组题的复习顺序

1. 先背 402、403、404：prefill/decode、指标、KV 公式。
2. 再背 405-409：PagedAttention、continuous batching、chunked prefill、prefix cache。
3. 然后背 410-416：spec decoding、vLLM、SGLang、TensorRT-LLM、PD disaggregation、KV 量化。
4. 最后练 417-420：线上排查和项目表达。

## 延伸阅读

- vLLM 官方文档：[https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/)
- vLLM PagedAttention 设计：[https://docs.vllm.ai/en/latest/design/paged_attention.html](https://docs.vllm.ai/en/latest/design/paged_attention.html)
- vLLM 论文：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- SGLang 官方文档：[https://docs.sglang.ai/](https://docs.sglang.ai/)
- SGLang 论文：[https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- TensorRT-LLM 官方文档：[https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- Speculative Decoding 论文：[https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
- Medusa：[https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- EAGLE：[https://arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)
