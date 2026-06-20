# 答案版 34：分布式推理、PD 分离、KV Cache 传输与 MoE Serving

对应题目：`03_高频题单100题.md` 的 701-720。

用法：先把每题 30 秒版背顺，再用 2 分钟版补架构、指标和排查。分布式推理题最怕答成“加 GPU”，要主动讲 prefill/decode 资源差异、KV Cache transfer、路由、队列、并行策略、MoE all-to-all 和 SLO。

## 701. 为什么需要分布式推理？

30 秒版：

因为单卡可能放不下权重或 KV Cache，QPS、TTFT、TPOT、长上下文和多租户 SLO 也可能超出单机能力。分布式推理通过多副本、TP/PP/EP、PD 分离和路由调度来扩显存、扩吞吐和稳延迟。

2 分钟版：

分布式推理不是卡越多越快。它解决的是权重显存、KV Cache、算力、尾延迟、SLO 和多租户隔离。代价是通信、同步、调度和故障恢复。面试要先说瓶颈，再选拆法：模型放不下看 TP/PP/量化，QPS 不够看多副本，MoE 看 EP，prefill/decode 干扰看 PD 分离。

## 702. Prefill 和 decode 的资源差异是什么？

30 秒版：

Prefill 处理完整 prompt，偏大矩阵和算力，主要影响 TTFT；decode 逐 token 生成，读写历史 KV Cache，偏内存带宽和调度，主要影响 TPOT/ITL。

2 分钟版：

Prefill 输入 token 多，矩阵乘更大，长 prompt、RAG 文档会使 TTFT 变高。Decode 每步只新增一个 token，但要读写所有历史 KV Cache，输出越长、并发越高，TPOT 和尾延迟越敏感。资源画像不同，是 PD disaggregation 的根本原因。

## 703. Aggregated serving 和 PD disaggregation 有什么区别？

30 秒版：

Aggregated 是同一组 GPU 同时跑 prefill 和 decode，简单但两阶段互相干扰；PD disaggregation 把 prefill 和 decode 放到不同 worker/pool，能独立扩缩容，但要传 KV Cache，架构更复杂。

2 分钟版：

Aggregated 没有跨 worker KV transfer，适合小规模和短 prompt。PD 分离适合长 prompt、混合流量、严格 TTFT/TPOT SLO。它能降低 prefill 干扰 decode，但新增 gateway、两套队列、KV connector、预分配、清理、故障恢复和网络瓶颈。

## 704. PD 分离的一次请求流程怎么讲？

30 秒版：

请求先到 gateway，选 prefill 和 decode worker；prefill 计算 prompt KV Cache；decode 侧预留 KV slots；KV connector 把 KV 传过去；decode 跳过 prompt prefill，直接流式生成；结束后清理两侧资源。

2 分钟版：

流程：

```text
Client -> Gateway
-> Prefill Queue/Worker
-> KV Cache generated
-> KV Transfer / Connector
-> Decode Queue/Worker
-> Stream tokens
-> Cleanup
```

重点是 decode 必须拿到 prompt KV，gateway 要维护请求状态，处理取消、超时、失败和资源释放。

## 705. KV Cache 传输为什么是核心难点？

30 秒版：

KV Cache 很大，跨 worker 传输可能吃掉 PD 分离收益；同时 prefill/decode 的 TP size、KV layout、page/block 映射可能不同，需要预分配、gather/scatter 和清理。

2 分钟版：

KV 估算：

```text
KV bytes ≈ B * S * L * H_kv * D * 2 * bytes
```

长 prompt 和长上下文下，KV 传输可能是 GB 级。传输慢、layout 转换慢、decode 侧 slots 不够、请求取消后资源没清理，都会让系统卡住或 OOM。

## 706. KV transfer backend 怎么讲？

30 秒版：

常见关键词包括 NIXL、UCX、libfabric、EFA、RDMA、Mooncake、MPI/UCX 和 vLLM connector。核心目标是低延迟、高带宽、少拷贝地把 KV Cache 从 prefill 传到 decode。

2 分钟版：

面试不用背命令，要讲抽象：connector 负责握手、预分配、传输、状态同步和清理；底层可能用 RDMA、UCX、NIXL、Mooncake 等。KV block/page 元数据也要传对，否则 decode 侧无法接上缓存。

## 707. prefill 和 decode 为什么可以用不同并行策略？

30 秒版：

因为两阶段瓶颈不同。Prefill 更适合大矩阵和更大 TP，decode 更需要稳定 TPOT、少通信和更多副本。PD 分离后可以分别配置资源和并行度。

2 分钟版：

例子：prefill 用 TP=4 吃长 prompt，decode 用 TP=1/2 加更多 replicas 减少通信和提升并发。但 TP 不同会导致 KV layout 不同，传输时要 gather/scatter，所以它是收益和复杂度的交换。

## 708. TP、PP、DP、EP 在推理里怎么选？

30 秒版：

TP 切层内矩阵，PP 切层，DP/replica 扩 QPS，EP 切 MoE experts。模型放不下先看 TP/PP/量化，QPS 不够看副本，MoE experts 多看 EP。

2 分钟版：

TP 会带来层内通信，batch 小时开销明显；PP 有 bubble，在线小 batch 不一定划算；多副本最简单但每副本都要模型显存；EP 需要 all-to-all token dispatch。面试要从瓶颈出发，而不是背“上多卡”。

## 709. Chunked prefill、PD 分离和 prefix cache 怎么区分？

30 秒版：

Chunked prefill 是把长 prompt 的 prefill 切小，避免阻塞；PD 分离是把 prefill 和 decode 放不同资源池；prefix cache 是复用相同前缀的 KV，减少重复 prefill。

2 分钟版：

三者可以组合。prefix cache 命中时少做 prefill；未命中的长 prompt 可以 chunked prefill；当 prefill/decode 干扰严重时做 PD 分离。不要把 chunked prefill 说成 PD 分离，它还可能在同一 GPU 池里执行。

## 710. PD 分离里 gateway / router 要做什么？

30 秒版：

它要选择 prefill worker、decode worker、KV transfer path，维护请求状态，做 admission control、SLO 路由、取消/超时处理和 fallback。

2 分钟版：

路由依据包括 prompt length、max_new_tokens、租户、SLO、prefill/decode queue depth、KV capacity、prefix cache 命中和网络带宽。复杂点是请求跨两个 worker，任何一边失败都要清理另一边资源。

## 711. Admission control 怎么设计？

30 秒版：

不能只看请求数，要看 prompt tokens、预计输出 tokens、prefill queue、decode queue、KV capacity、KV transfer 带宽和 SLO。

2 分钟版：

一个长 prompt 短输出请求和短 prompt 长输出请求压力完全不同。Admission 可以按 token budget 和 KV budget 估算，超过容量时排队、降级、拒绝、缩短 max_new_tokens 或 fallback 到其他模型/路径。

## 712. PD 分离怎么扩缩容？

30 秒版：

TTFT 高、prefill backlog 高就加 prefill；TPOT 高、decode backlog 高就加 decode；KV transfer 高就优化 placement、connector 或减少跨节点传输。

2 分钟版：

扩缩容比例取决于流量形态：长 prompt 多需要更多 prefill；长输出多需要更多 decode；短请求多可能不用 PD。还要看 prefix cache 命中、KV transfer latency、GPU 利用率和 P95/P99，而不是只看平均 QPS。

## 713. PD 分离监控看哪些指标？

30 秒版：

看 TTFT、TPOT、P95/P99、prefill queue、decode queue、KV transfer latency/bytes/bandwidth、KV capacity、OOM、timeout、connector error 和 goodput。

2 分钟版：

分层监控：

```text
prefill: queue depth / tokens/s / GPU util / cache hit
decode: running reqs / TPOT / KV utilization / output tokens/s
transfer: latency / bytes / bandwidth / pending / errors
end-to-end: TTFT / TPOT / P99 / error / cost
```

PD 系统要监控两套队列和 transfer，否则只看端到端延迟很难定位。

## 714. 什么时候 PD 分离不划算？

30 秒版：

短 prompt、短输出、低流量、网络带宽不足、KV transfer 太慢、prefix cache 已解决大部分 prefill、系统复杂度不可控时，PD 分离可能不划算。

2 分钟版：

PD 分离是用架构复杂度换 SLO 和资源解耦。它适合长 prompt 或混合流量明显的服务。如果 KV 传输跨很远网络，或者请求本来很轻，额外 gateway、传输和状态管理可能让系统更慢、更不稳定。

## 715. MoE serving 的流程和难点是什么？

30 秒版：

MoE 每个 token 经 router 选 top-k experts，再 dispatch 到对应 experts 做 FFN，最后 combine。难点是 expert parallel、all-to-all、热门专家、负载不均和动态 shape 带来的尾延迟。

2 分钟版：

MoE 参数总量大但每 token 激活少量 experts。Serving 难点不只是算专家，还包括 token dispatch、跨卡通信、expert placement、load balance 和 grouped GEMM。热门 expert 会拖慢整批请求，导致 P99 上升。

## 716. Expert parallel 和 tensor parallel 有什么区别？

30 秒版：

TP 切 dense 层矩阵，所有 token 通常都参与；EP 切 MoE experts，token 根据 router 被发到不同专家，核心通信是 all-to-all。

2 分钟版：

TP 的通信更像层内矩阵并行需要的 all-reduce/all-gather；EP 的通信是 token dispatch/combine。EP 的负载随输入动态变化，所以尾延迟和负载均衡更难。MoE serving 常要同时考虑 TP + EP + replica。

## 717. MoE serving 怎么优化尾延迟？

30 秒版：

看 per-expert load，做负载均衡、热门 expert replication、expert placement、grouped GEMM、all-to-all overlap、按模型/租户路由和 capacity 控制。

2 分钟版：

优化方向：

- router 层面减少极端倾斜。
- 热门专家复制到多卡。
- 把常一起访问的 experts 放近。
- token batching / grouped GEMM 提升 expert 计算效率。
- 通信和计算 overlap。
- 监控 per-expert P95/P99 和 all-to-all 时间。

## 718. 多卡推理 TPOT 突然变高怎么排查？

30 秒版：

先看 decode queue、KV Cache 长度、HBM、TP/NCCL 通信、KV transfer、MoE all-to-all、输出长度和 GPU 利用率。

2 分钟版：

排查链路：

```text
TPOT high
-> decode queue?
-> output length / batch?
-> KV cache utilization / HBM bandwidth?
-> TP communication / NCCL?
-> PD KV transfer pending?
-> MoE hot expert / all-to-all?
-> kernel / CUDA graph / quantization path?
```

不要只说加卡，加卡可能增加通信。

## 719. PD 分离系统设计 8 分钟怎么讲？

30 秒版：

按“流量特征和 SLO -> baseline -> 为什么干扰 -> PD 架构 -> KV transfer -> 路由和扩缩容 -> 指标 -> fallback 和故障恢复”讲。

2 分钟版：

示例：

```text
我们有长 prompt + 长输出混合流量，TTFT/TPOT 都有 SLO。
baseline aggregated serving 下，长 prefill 干扰 decode，P99 抖动。
引入 gateway + prefill pool + decode pool + KV connector。
长 prompt 走 PD，短请求或 transfer 不划算时走 aggregated。
监控 prefill/decode queue、KV transfer latency、KV capacity、TTFT、TPOT、goodput。
失败时 fallback 到普通 serving，并清理两侧资源。
```

## 720. 分布式推理面试前最后怎么复习？

30 秒版：

背 prefill/decode 差异、PD 分离架构、KV transfer 难点、TP/PP/DP/EP 区别、MoE serving 难点、指标和排查链路。

2 分钟版：

最后清单：

- prefill -> TTFT，decode -> TPOT。
- PD 分离：两池资源、gateway、KV connector、decode 流式输出。
- KV transfer：大、慢、layout/page/预分配/清理复杂。
- 并行：TP 切矩阵，PP 切层，DP 扩副本，EP 切 experts。
- MoE：router、dispatch、all-to-all、hot experts、tail latency。
- 指标：prefill/decode queue、KV transfer、KV capacity、P95/P99、goodput。
- 取舍：短请求不一定适合 PD，网络差不适合远程 KV。
