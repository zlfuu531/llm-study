# 推理引擎、vLLM、SGLang 与 TensorRT-LLM 面试

这一章面向推理部署、AI Infra、模型服务平台和大模型应用后端岗位。你不要只会说“用 vLLM 部署一下”，面试官真正想听的是：一个在线 LLM serving 系统如何把请求排队、调度、prefill、decode、KV Cache、batching、采样、流式输出、监控和限流串起来，以及线上 TTFT / TPOT 出问题时怎么排查。

推荐读法：

1. 先把 `Prefill / Decode / KV Cache / TTFT / TPOT` 这条主线吃透。
2. 再学 `PagedAttention / Continuous Batching / Chunked Prefill / Prefix Cache / Speculative Decoding`。
3. 最后比较 `vLLM / SGLang / TensorRT-LLM`：它们不是互斥概念，而是不同层次的工程实现和优化取舍。

如果被继续追问 FlashAttention、Triton、自定义 kernel、GPU 利用率或 NCCL，跳到 [25_GPU_CUDA_Triton与FlashAttention面试.md](25_GPU_CUDA_Triton与FlashAttention面试.md)。本章讲 serving engine，GPU 专题讲底层算子和硬件瓶颈。

如果被追问 tokenizer 为什么会影响 TTFT、token 预算、prefix cache 命中或模型迁移，跳到 [35_Tokenizer_BPE_SentencePiece与Token预算面试.md](35_Tokenizer_BPE_SentencePiece与Token预算面试.md)。

如果被追问 temperature、top-k/top-p、beam search、LogitsProcessor、EOS/stop strings 或解码参数如何影响质量和成本，跳到 [36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md](36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md)。

如果被追问 GPTQ、AWQ、SmoothQuant、FP8、KV Cache 量化、GGUF 或 INT4 为什么不一定提速，跳到 [37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md](37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md)。本章讲 serving engine，新章讲低比特推理的误差、kernel 和评估。

如果被追问 speculative decoding 为什么无损、accept rate 怎么看、EAGLE/Medusa/MTP/n-gram 有什么区别，跳到 [38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md](38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md)。本章讲 serving 全局，新章讲推测解码细节和线上取舍。

如果被追问 Prefill/Decode disaggregation、KV Cache transfer、NIXL/RDMA、TP/PP/EP 或 MoE serving，跳到 [39_分布式推理_PD分离_KVCache传输与MoEServing面试.md](39_分布式推理_PD分离_KVCache传输与MoEServing面试.md)。本章讲单个 serving engine 的主线，新章讲分布式资源拆分和多卡系统取舍。

## 一句话总览

推理引擎不是 `model.generate()` 的薄封装，而是一个面向在线并发的执行系统。它的核心目标是：

- 在显存有限的情况下承载更多并发。
- 在请求长短不一时尽量提高 GPU 利用率。
- 在吞吐、首 token 延迟、每 token 延迟和成本之间做取舍。
- 对外提供稳定 API、流式输出、指标、限流、隔离和故障恢复能力。

一个简化的 serving 链路可以这样记：

```text
Client
  -> API Gateway / Auth / Rate Limit
  -> Request Queue / Admission Control
  -> Scheduler
  -> Tokenizer
  -> Prefill Worker
  -> KV Cache Manager
  -> Decode Loop / Continuous Batching
  -> Sampler
  -> Streamer
  -> Metrics / Trace / Autoscaling
```

面试表达时可以说：

> 普通推理代码只关心一个请求怎么生成，推理引擎关心一批动态到达的请求如何共享 GPU、共享前缀、复用 KV Cache、控制延迟尾部，并在显存、吞吐和稳定性之间做调度。

## Prefill 和 Decode

LLM 在线推理通常分成两个阶段。

### Prefill

Prefill 是把用户 prompt 一次性喂给模型，计算所有 prompt token 的 hidden states，并生成后续 decode 需要的 KV Cache。

特点：

- 输入 token 多，矩阵乘法规模大。
- 更偏 compute-bound，算力利用率通常更高。
- 直接影响 TTFT，因为首 token 必须等 prompt 处理完。
- prompt 越长，RAG 拼接越多，prefill 越重。

### Decode

Decode 是自回归逐 token 生成。每一步只新增一个 token，但要读取历史 KV Cache，并把新 token 的 K/V 追加到 cache。

特点：

- 每步计算量相对小，但要反复读写 KV Cache。
- 更容易受显存带宽、KV Cache 长度、batch 调度和 kernel 效率影响。
- 直接影响 TPOT / ITL，也影响用户看到流式输出的速度。
- 并发越高、输出越长，decode 阶段越重要。

### 为什么这两个阶段要分开讲

因为它们的瓶颈不同：

| 阶段 | 典型瓶颈 | 主要指标 | 常见优化 |
| --- | --- | --- | --- |
| Prefill | prompt 长、RAG 上下文多、排队、tokenizer、算力 | TTFT、queue time、prefill time | prefix cache、chunked prefill、prompt 压缩、RAG 裁剪 |
| Decode | KV Cache 读写、显存带宽、batch 不饱、长输出 | TPOT、ITL、output tokens/s | continuous batching、PagedAttention、量化、speculative decoding |

面试官问“TTFT 高怎么办”，不要只答“上 vLLM”。要先判断 TTFT 是排队高、tokenizer 慢、RAG 拼太长、prefill 慢，还是网络/流式 flush 慢。

## 核心指标

### TTFT

TTFT 是 time to first token，用户从发请求到收到第一个 token 的时间。

近似拆解：

```text
TTFT ≈ 网关/鉴权时间
     + 排队时间
     + tokenizer 时间
     + RAG/工具前置时间
     + prefill 时间
     + 首 token decode 与采样时间
     + 网络 flush 时间
```

TTFT 对聊天体验特别敏感。首 token 慢，用户会觉得系统“卡住了”。

### TPOT / ITL

TPOT 是 time per output token，ITL 是 inter-token latency，关注流式输出过程中相邻 token 的间隔。

近似拆解：

```text
TPOT ≈ decode 单步 forward
     + KV Cache 读取/写入
     + batch 调度
     + sampler
     + streamer flush
```

TPOT 高时，用户会看到输出一顿一顿。

### Throughput

Throughput 可以看：

- requests/s：每秒完成多少请求。
- input tokens/s：每秒处理多少输入 token。
- output tokens/s：每秒生成多少输出 token。
- total tokens/s：input + output，但生产排查时最好分开看。

只看 total tokens/s 容易误判。比如长 prompt 场景 input tokens/s 很高，但用户体感仍然可能很差。

### Tail Latency

P95 / P99 比平均值更重要。在线服务里，一个长 prompt 或一个超长输出请求可能拖慢同 batch 的其他请求。

面试表达：

> 平均延迟只能看整体，P95/P99 才能暴露排队、长请求、显存碎片、batch 调度和下游阻塞问题。推理系统要同时看平均吞吐和尾延迟。

## KV Cache 显存估算

KV Cache 是推理部署最常被追问的公式。每层 self-attention 都要缓存历史 token 的 K 和 V。

常用估算：

```text
KV Cache bytes
≈ 2 * num_layers * batch_or_active_requests * seq_len
  * num_kv_heads * head_dim * bytes_per_element
```

解释：

- `2`：K 和 V 两份。
- `num_layers`：每一层都有 KV。
- `batch_or_active_requests`：在线服务里可以理解为活跃序列数。
- `seq_len`：每个请求当前上下文长度，包括输入和已生成 token。
- `num_kv_heads`：如果是 MHA，通常等于 attention heads；如果是 MQA/GQA，会更小。
- `head_dim`：每个 head 的维度。
- `bytes_per_element`：FP16/BF16 是 2 bytes，FP8/INT8 更小。

### 例子

假设：

- 32 层。
- GQA 后 `num_kv_heads = 8`。
- `head_dim = 128`。
- FP16 KV，2 bytes。
- 单请求当前上下文 4096 token。

单请求 KV 约为：

```text
2 * 32 * 4096 * 8 * 128 * 2 bytes
= 536,870,912 bytes
≈ 512 MB
```

如果 16 个并发请求都接近这个长度，KV Cache 就接近 8 GB。注意这还没算模型权重、临时 buffer、框架开销和显存碎片。

### 面试关键点

- 长上下文不仅让 attention 算得久，更会让 KV Cache 显存持续增长。
- 输出越长，decode 阶段 KV Cache 也越长。
- GQA/MQA/MLA 的一个重要价值是减少 KV Cache 的 head 数或表示规模。
- 线上要限制 `max_model_len`、`max_output_tokens`、并发和 admission，否则很容易显存被长请求吃掉。

## PagedAttention

PagedAttention 的直觉类似操作系统分页：不要给每个请求预留一整段连续的大 KV Cache，而是把 KV Cache 切成固定大小的 block/page，需要多少分配多少。

它解决的核心问题是：

- 减少 KV Cache 内存碎片。
- 避免为最大序列长度过度预留。
- 让不同长度、动态增长的请求更容易被调度到同一块 GPU 上。
- 方便做 prefix sharing、beam search 等场景的 KV 共享。

可以这样对比：

```text
朴素方式：
request A: [预留很大连续区域，实际只用一部分]
request B: [预留很大连续区域，实际只用一部分]
碎片和浪费明显

PagedAttention：
request A: page 1 -> page 2 -> page 9
request B: page 3 -> page 4
物理 block 不必连续，逻辑顺序由映射表维护
```

### PagedAttention 和 FlashAttention 的区别

这组区别非常高频：

| 技术 | 解决什么 | 发生在哪 |
| --- | --- | --- |
| FlashAttention | attention 计算和显存读写更高效，减少中间矩阵写回 | 单次 forward 的 attention kernel |
| PagedAttention | 在线 serving 的 KV Cache 管理更高效，减少碎片和浪费 | 多请求、多步 decode 的 KV Cache 管理 |

一句话：

> FlashAttention 优化“attention 怎么算”，PagedAttention 优化“很多请求的 KV Cache 怎么放、怎么复用、怎么增长”。

## Continuous Batching

普通 static batching 是凑一批请求一起跑，等这一批都完成后再跑下一批。在线 LLM 不适合这么做，因为每个请求输出长度不同，有的 20 token 就结束，有的要生成 1000 token。

Continuous batching 的核心思想是：

- decode 每一步都重新组织活跃请求。
- 已完成的请求退出 batch。
- 新请求可以在合适时机加入 batch。
- GPU 不必等最长请求结束才处理新请求。

直观例子：

```text
t0: batch = A, B, C
t1: A 结束，batch = B, C, D
t2: C 结束，batch = B, D, E
```

收益：

- 提高 GPU 利用率。
- 降低新请求排队时间。
- 对长短请求混合的真实流量更友好。

代价：

- scheduler 更复杂。
- batch 内序列长度和状态不同，KV 管理更难。
- fairness、priority、抢占和长请求保护需要额外策略。

面试表达：

> Continuous batching 不是简单把 batch size 调大，而是在 decode loop 中动态维护活跃序列集合，让 GPU 一直有活干，同时用调度策略控制尾延迟和公平性。

## Chunked Prefill

长 prompt 的 prefill 会占用很长一段 GPU 时间。如果一个超长 prompt 正在 prefill，短请求可能只能排队等待，导致 TTFT 变差。

Chunked prefill 的做法是：

- 把长 prompt prefill 切成多个 token chunk。
- 在 chunk 之间插入 decode 或其他短请求。
- 让长输入请求不要长时间独占 GPU。

收益：

- 改善短请求 TTFT。
- 改善混合流量下的 tail latency。
- 对 RAG 长上下文场景很有用。

代价：

- chunk 太小会增加调度开销。
- 可能降低单个长 prompt 的纯吞吐。
- 需要和 continuous batching、KV Cache 管理配合。

面试表达：

> Chunked prefill 是用调度换公平性和尾延迟，不是免费的加速。它适合长 prompt 和短 prompt 混合、短请求体验重要的在线场景。

## Prefix Cache 和 RadixAttention

很多生产请求会共享前缀：

- 相同 system prompt。
- 相同 few-shot 示例。
- 同一个文档、同一个代码仓库、同一个 Agent 任务模板。
- 多轮对话的历史上下文。

Prefix cache 的思路是：如果前缀 token 完全相同，就复用已经算好的 KV Cache，避免重复 prefill。

### 什么时候有效

有效场景：

- system prompt 固定。
- RAG prompt 模板稳定，文档片段重复率高。
- Agent 工具说明和格式约束很长且复用。
- 代码 Agent 反复在同一仓库、同一文件上下文里迭代。

不太有效场景：

- 每个请求前缀都不同。
- prompt 里有时间戳、随机 ID、用户特征，破坏共同前缀。
- cache 太小，频繁淘汰。
- 多租户隔离要求高，不能跨用户复用。

### RadixAttention

SGLang 的 RadixAttention 可以理解成把多个请求的前缀 KV Cache 组织成 radix tree，公共前缀只存一份，后续分叉部分各自继续增长。

直观结构：

```text
system prompt
  -> tool schema
      -> user A context
      -> user B context
      -> user C context
```

面试表达：

> Prefix cache 关注“相同前缀复用”，RadixAttention 把这种复用做成更系统的树状 KV 管理，对多轮、agentic workflow、结构化调用和重复 prompt 模板更友好。

## Speculative Decoding

Speculative decoding 的目标是减少大模型 decode 的逐 token 串行等待。

基本流程：

1. 用一个更小、更快的 draft model 先草拟多个 token。
2. 用 target model 一次性并行验证这些 token。
3. 如果草拟 token 被接受，就一次前进多个 token。
4. 如果某个 token 被拒绝，从拒绝位置重新采样，继续下一轮。

简化伪代码：

```text
while not finished:
    draft_tokens = small_model.generate(prefix, gamma)
    accepted = target_model.verify(prefix, draft_tokens)
    append accepted tokens
    if rejected:
        append target sampled token
```

### 为什么能加速

普通 decode 是：

```text
target model -> 1 token
target model -> 1 token
target model -> 1 token
```

Speculative decoding 是：

```text
draft model -> 多个候选 token
target model -> 一次验证多个 token
```

如果 draft model 很便宜，而且接受率高，就能减少 target model 的 forward 次数。

### 什么时候会变慢

可能变慢的情况：

- draft model 太大，草拟成本不低。
- draft 和 target 分布差距大，接受率低。
- target model 验证阶段没有真正并行起来。
- decode 已经不是主要瓶颈，瓶颈在排队、网络、RAG、KV 读写或 CPU tokenizer。
- 显存被 draft model 额外占用，导致 batch size 下降。

### 常见变体

| 方法 | 思路 | 适合场景 |
| --- | --- | --- |
| Draft model | 小模型草拟，大模型验证 | 通用 speculative decoding |
| Prompt lookup / n-gram | 从 prompt 或历史里找候选 | 复制、摘要、代码、长上下文复用 |
| Medusa | 在大模型上加多个预测 head | 希望减少额外 draft 模型 |
| EAGLE | 用特征级 extrapolation 预测后续 token | 追求更高接受率和吞吐 |

面试不要背成“spec decoding 一定加速”。更好的回答是：

> 它把多个 token 的候选生成交给便宜路径，再由大模型批量验证。收益取决于接受率、draft 成本、验证并行度和是否牺牲 batch / 显存。上线前必须用真实 prompt 和输出长度分桶压测。

推测解码的无损接受/拒绝公式、EAGLE、Medusa、MTP、n-gram/suffix 和 accept rate 排查见：[38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md](38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md)。

## vLLM 怎么讲

vLLM 是通用 LLM serving 里最常被问到的开源引擎之一。面试时重点讲它解决的问题，而不是背 API。

核心关键词：

- PagedAttention：KV Cache block/page 管理。
- Continuous batching：动态请求调度。
- Prefix caching：复用相同前缀的 KV。
- Chunked prefill：长 prompt 不长时间独占 GPU。
- OpenAI-compatible server：方便替换调用方。
- Metrics：服务指标、延迟、吞吐、cache 等。
- Quantization / distributed serving：支持多种部署优化。
- Speculative decoding：在合适场景降低 decode 延迟。

一句话：

> vLLM 的价值是把 LLM 在线 serving 中最难的 KV Cache 管理、动态 batching 和常见性能优化封装成生产可用的引擎，让应用侧不用自己手写复杂 scheduler。

适合场景：

- 通用聊天、RAG、Agent 后端。
- 需要 OpenAI-compatible API。
- 想快速获得较好的吞吐和显存利用率。
- 对模型种类、量化方式和部署形态有较多组合需求。

## SGLang 怎么讲

SGLang 可以从两个角度理解：

1. 面向大模型程序的前端：支持结构化生成、多步调用、工具/API workflow。
2. 面向 serving 的后端：通过 RadixAttention、cache-aware scheduling 等优化重复前缀和 agentic workflow。

核心关键词：

- RadixAttention：树状复用 KV Cache。
- Structured generation：对 JSON、regex、schema 等结构化输出更友好。
- Agentic workflow：多次模型调用、工具调用、分支和状态管理。
- Cache-aware serving：调度时考虑 prefix/cache 命中。

适合场景：

- Agent、工具调用、结构化输出很多。
- prompt 模板长且重复率高。
- 多轮 workflow 中大量共享系统提示词和工具描述。
- 需要把 LLM 调用写成程序，而不只是单次 chat completion。

面试表达：

> vLLM 更像通用高吞吐 serving 引擎，SGLang 更强调 LLM program 和 agentic serving，尤其适合有大量重复前缀、结构化输出和多步调用的工作流。

## TensorRT-LLM 怎么讲

TensorRT-LLM 是 NVIDIA 生态里的高性能 LLM 推理优化工具链和 runtime。面试时不要只说“它更快”，要说为什么和适合什么。

核心关键词：

- NVIDIA GPU 深度优化 kernel。
- Tensor parallel / pipeline parallel 等多 GPU 推理支持。
- In-flight batching：类似在线动态 batching 的思想。
- Paged KV Cache：更高效的 KV 管理。
- Chunked context / chunked prefill：改善长上下文调度。
- Quantization：FP8、INT8、INT4 等部署优化。
- Speculative decoding：支持多种推测解码路径。

适合场景：

- 明确部署在 NVIDIA GPU 集群。
- 追求极致性能和成本优化。
- 有专门平台/推理团队维护构建、kernel、engine 和 benchmark。
- 模型形态相对稳定，愿意为性能投入工程复杂度。

和 vLLM / SGLang 的区分：

| 对比项 | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| 核心定位 | 通用 LLM serving 引擎 | LLM program + serving runtime | NVIDIA 高性能推理 runtime |
| 面试关键词 | PagedAttention、continuous batching | RadixAttention、structured generation | IFB、kernel、quantization、GPU 优化 |
| 上手速度 | 较快 | 适合 workflow 场景 | 工程门槛更高 |
| 典型价值 | 快速搭在线服务 | 多步/结构化/Agent 复用 | 极致性能和硬件效率 |

## Prefill/Decode Disaggregation

Prefill 和 decode 的资源特征不同，所以近两年很多 serving 系统会讨论 prefill/decode disaggregation。

直觉：

- Prefill：长 prompt，计算密集，适合更大 batch / 算力利用。
- Decode：逐 token，小步多次，KV Cache 和显存带宽压力大，更看尾延迟。

如果把两者放在同一组 GPU 上，它们会互相影响：

- 长 prefill 会阻塞短请求 decode。
- decode 细碎步骤会打断大 prefill 的吞吐。

Disaggregation 的思路：

```text
Prefill worker:
  prompt -> hidden states / KV Cache
         -> KV transfer

Decode worker:
  load / receive KV Cache
  -> decode loop
  -> stream output
```

收益：

- 不同 worker 针对不同阶段优化。
- 更容易控制 TTFT 和 TPOT。
- 可以按流量特征独立扩缩 prefill 和 decode 资源。

挑战：

- KV Cache 传输成本高。
- 网络、RDMA、GPU-GPU 传输和序列化复杂。
- 调度和故障恢复更难。
- prefix cache 和 KV locality 要重新设计。

面试表达：

> PD disaggregation 是在承认 prefill 和 decode 瓶颈不同后，把资源池拆开做专门优化。它不是所有团队都必须上，只有当混合流量、长上下文和规模足够大时，收益才可能覆盖 KV 传输和调度复杂度。

PD 分离、KV Cache 传输、TP/PP/EP 和 MoE serving 的完整专题见：[39_分布式推理_PD分离_KVCache传输与MoEServing面试.md](39_分布式推理_PD分离_KVCache传输与MoEServing面试.md)。

## 量化在 Serving 里的位置

推理服务里常见量化不止一种：

| 类型 | 减什么 | 主要风险 |
| --- | --- | --- |
| 权重量化 | 模型权重显存、加载成本、矩阵乘带宽 | 精度下降、kernel 支持、兼容性 |
| KV Cache 量化 | decode 阶段 KV 显存和带宽 | 长上下文质量下降、数值误差累积 |
| activation 量化 | 中间激活显存和算子带宽 | 校准复杂、误差更敏感 |

上线前评估：

- 通用 benchmark。
- 业务私有 eval。
- 长上下文 eval。
- 代码/数学/结构化输出 eval。
- 安全和拒答策略 eval。
- 延迟、吞吐、显存、P95/P99、成本。

面试表达：

> 量化不是只看显存省多少，还要看业务质量、长上下文稳定性、kernel 是否真实加速、是否影响 batch size、是否和 speculative / prefix cache / 多卡通信兼容。

## 调度、限流和公平性

推理引擎的 scheduler 通常要同时考虑：

- 请求到达时间。
- prompt 长度。
- 已生成 token 数。
- 最大输出长度。
- 用户优先级。
- 当前 GPU 显存。
- 当前 KV Cache block 占用。
- 是否命中 prefix cache。
- 是否要抢占或暂停低优先级请求。

常见控制参数：

- `max_model_len`：最大上下文长度。
- `max_num_batched_tokens`：一个调度步最多处理多少 token。
- `max_num_seqs`：最多活跃序列数。
- `max_output_tokens`：最大输出 token。
- queue timeout：排队超时。
- request priority：高优任务优先。

生产原则：

- 不要让一个超长请求拖垮整批请求。
- 不要只追求最大吞吐，忽略 P99。
- 对不同租户、不同业务做隔离和配额。
- 对长 prompt 和长输出分别限流。
- 监控 queue time，queue time 常常比模型本身更早暴露容量问题。

## TTFT 变高怎么排查

按链路拆：

1. 网关：鉴权、限流、队列是否变慢。
2. 上游：RAG 检索、rerank、工具调用是否变慢。
3. Tokenizer：CPU 是否打满，batch tokenizer 是否成为瓶颈。
4. Prompt：input tokens 是否变长，系统提示词/RAG 是否膨胀。
5. Queue：请求是否排队，GPU 是否满载。
6. Prefill：prefill time 是否上升，长 prompt 是否堵住短请求。
7. Prefix cache：命中率是否下降。
8. Chunked prefill：是否关闭或 chunk 配置不合理。
9. 网络：流式 flush、代理、客户端是否变慢。

回答模板：

> 我会先把 TTFT 拆成 gateway、queue、tokenizer、RAG、prefill、first decode 和 network。先看 queue time 和 input tokens 分布，如果输入变长或长请求比例升高，就考虑 prompt 裁剪、prefix cache 和 chunked prefill；如果 queue 上升，则看并发、GPU 利用率、admission 和扩容；如果 RAG 上游慢，就拆检索和 rerank。

## TPOT 变高怎么排查

按 decode 链路拆：

1. Output tokens 是否变长。
2. 活跃序列数是否过高或过低。
3. KV Cache 长度是否增长，长上下文比例是否上升。
4. KV Cache 显存是否接近上限，是否频繁 eviction / recompute。
5. GPU 显存带宽、SM 利用率是否异常。
6. batch 调度是否不饱。
7. kernel / quantization / cuda graph 是否变化。
8. Tensor parallel 通信是否变慢。
9. Speculative decoding 接受率是否下降。
10. streamer / sampler / CPU 后处理是否变慢。

回答模板：

> TPOT 主要看 decode。先看输出长度和活跃 batch，再看 KV Cache 长度、显存带宽、kernel 和多卡通信。如果 speculative decoding 开着，还要看 draft 成本和接受率；如果 batch 不饱，就看请求到达模式和 scheduler 参数。

## 显存够但吞吐上不去

这种问题很常见。显存够不代表 GPU 算力或显存带宽用满。

可能原因：

- batch 太小，请求不够。
- batch 太大，但长短混合导致尾部拖慢。
- CPU tokenizer 或上游 RAG 是瓶颈。
- decode memory-bound，SM 利用率不高。
- KV Cache 访问不连续或碎片影响效率。
- 多卡 TP 通信占比高。
- 量化 kernel 没有真正走高效路径。
- 网络流式输出或日志阻塞。
- scheduler 参数保守。
- prefix cache 命中低，prefill 重复计算多。

面试表达：

> 我会把瓶颈分成流量不足、CPU/上游瓶颈、prefill 瓶颈、decode 瓶颈、通信瓶颈和 IO 瓶颈。显存只是容量约束，吞吐要看 GPU util、memory bandwidth、tokens/s、batch 形态和 queue time。

## 项目里怎么讲推理服务

8 分钟讲稿结构：

1. 背景：业务需要在线调用大模型，目标是降低成本、控制延迟、提升稳定性。
2. 流量：请求 QPS、input/output token 分布、长 prompt 占比、是否流式。
3. 架构：网关、队列、推理引擎、缓存、监控、灰度。
4. 引擎：为什么选 vLLM / SGLang / TensorRT-LLM。
5. 优化：batching、prefix cache、chunked prefill、量化、限流。
6. 指标：TTFT、TPOT、P95/P99、tokens/s、GPU 利用率、成本。
7. 排查：一次真实延迟或 OOM 问题怎么定位。
8. 复盘：如果重做，会补哪些 eval、压测和调度策略。

项目表达示例：

> 我们先用 vLLM 搭了 OpenAI-compatible 的推理服务，接入网关做鉴权、限流和 trace。上线后发现 RAG 长 prompt 导致 TTFT 尾部变高，所以我们把 input tokens 分桶，开启 prefix cache，裁剪低价值上下文，并用 chunked prefill 改善短请求等待。decode 侧主要看 TPOT 和 output tokens/s，通过控制 max output、调整 batched tokens 和量化方案，把 P95 延迟和单次成本压到目标范围内。

## 高频快答

### PagedAttention 解决什么？

解决在线 serving 中 KV Cache 动态增长、长短请求混合带来的显存碎片和预留浪费，用 block/page 方式管理 KV。

### Continuous batching 为什么有用？

因为 LLM 请求输出长度不同，动态把完成请求移出、新请求加入，能让 GPU decode loop 更持续地工作。

### Chunked prefill 什么时候开？

长 prompt 和短请求混合、短请求 TTFT 重要时考虑。它改善尾延迟，但 chunk 太小会影响吞吐。

### Prefix cache 为什么命中低？

前缀必须 token 级完全一致。时间戳、随机 ID、用户特征、动态工具列表、RAG 排序变化都可能破坏命中。

### Speculative decoding 一定加速吗？

不一定。它依赖高接受率、低 draft 成本和有效并行验证。接受率低或 draft 太重会变慢。

### vLLM、SGLang、TensorRT-LLM 怎么选？

通用 serving 先看 vLLM；多步 agentic workflow、结构化输出和前缀复用重的场景看 SGLang；NVIDIA 集群追求极致性能和硬件优化看 TensorRT-LLM。

## 面试背诵版

推理引擎的核心不是单请求生成，而是在线并发调度。LLM 推理分 prefill 和 decode，prefill 处理整个 prompt、影响 TTFT，decode 逐 token 生成、影响 TPOT 和流式体验。KV Cache 显存大约是 `2 * layers * active_requests * seq_len * kv_heads * head_dim * bytes`，所以长上下文和高并发会迅速吃显存。

vLLM 的 PagedAttention 用 block/page 管理 KV Cache，减少碎片；continuous batching 在 decode loop 中动态加入和移除请求，提高 GPU 利用率；chunked prefill 把长 prompt 切开，避免长请求独占 GPU；prefix cache 复用相同前缀的 KV。SGLang 更强调 LLM program 和 agentic serving，用 RadixAttention 复用树状前缀。TensorRT-LLM 更偏 NVIDIA 高性能 runtime，关注 kernel、IFB、paged KV、量化和多 GPU。

线上排查时，TTFT 要拆 gateway、queue、tokenizer、RAG、prefill 和网络；TPOT 要拆 decode batch、KV Cache、显存带宽、kernel、通信和 speculative 接受率。所有优化都要用真实流量分桶压测，不能只看平均 tokens/s。

## 延伸阅读

- vLLM 官方文档：[https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/)
- vLLM PagedAttention 设计：[https://docs.vllm.ai/en/latest/design/paged_attention.html](https://docs.vllm.ai/en/latest/design/paged_attention.html)
- vLLM 论文 Efficient Memory Management for Large Language Model Serving with PagedAttention：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- SGLang 官方文档：[https://docs.sglang.ai/](https://docs.sglang.ai/)
- SGLang 论文 Efficiently Programming Large Language Models using SGLang：[https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- TensorRT-LLM 官方文档：[https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- Speculative Decoding 论文 Fast Inference from Transformers via Speculative Decoding：[https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
- Medusa 论文：[https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- EAGLE 论文：[https://arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)
