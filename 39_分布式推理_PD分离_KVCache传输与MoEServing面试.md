# 分布式推理、PD 分离、KV Cache 传输与 MoE Serving 面试

这一章面向推理部署、AI Infra、模型服务平台、多卡推理、云厂商平台和高阶系统设计岗位。2025-2026 年的推理系统面试越来越常把问题从“会不会部署 vLLM”推进到“prefill/decode 为什么要分离、KV Cache 怎么传、TP/PP/EP 怎么选、MoE serving 为什么尾延迟高、怎么按 TTFT/TPOT/SLO 做扩缩容”。

如果时间很紧，先背这句：

> 分布式推理的核心不是把模型随便切到多张卡，而是按 prefill、decode、KV Cache、通信和 SLO 拆资源。prefill 更偏大矩阵和算力，decode 更偏逐 token、KV 读写和尾延迟；PD disaggregation 把两阶段放到不同 worker/pool，减少互相干扰并独立扩缩容，但必须付出 KV Cache 传输、路由、两套队列、显存布局和故障恢复的复杂度。MoE serving 还要处理 expert parallel、token dispatch、all-to-all、热门专家和负载均衡。

相关答案版：[answers/34_分布式推理_PD分离_KVCache传输与MoEServing_答案版.md](answers/34_分布式推理_PD分离_KVCache传输与MoEServing_答案版.md)

相邻章节：

- [24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md](24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md)：serving engine、prefill/decode、continuous batching、prefix cache、spec decoding。
- [25_GPU_CUDA_Triton与FlashAttention面试.md](25_GPU_CUDA_Triton与FlashAttention面试.md)：GPU 瓶颈、HBM、NCCL、all-to-all、Profiler。
- [17_大模型训练系统与分布式训练面试.md](17_大模型训练系统与分布式训练面试.md)：TP/PP/DP/EP、MoE 训练通信和分布式基本功。
- [18_DeepSeek_MoE_MLA与ReasoningModel面试.md](18_DeepSeek_MoE_MLA与ReasoningModel面试.md)：MoE、MLA、MTP 和 DeepSeek 系高频热点。
- [20_LLMOps模型网关与可观测性面试.md](20_LLMOps模型网关与可观测性面试.md)：模型网关、路由、trace、SLO、灰度和回滚。

## 1. 为什么需要分布式推理

单机单卡推理会遇到几类边界：

- 模型权重放不下：需要 tensor parallel / pipeline parallel / quantization。
- KV Cache 放不下：长上下文、高并发、多轮对话会线性放大 KV。
- 延迟不稳定：长 prompt prefill 会阻塞短请求 decode。
- QPS 上不去：需要多副本、动态 batch 和路由。
- 多租户隔离：不同客户、模型、SLO 要拆资源池。
- MoE 模型：专家可能跨设备，token dispatch 和 all-to-all 成为瓶颈。

面试句：

> 分布式推理不是“卡越多越快”。多卡会引入通信、同步和调度开销，只有当单卡显存、算力、KV Cache 或 SLO 已经成为瓶颈时，拆分才有意义。

## 2. Prefill 和 Decode 的资源差异

| 阶段 | 做什么 | 典型瓶颈 | 主要指标 |
| --- | --- | --- | --- |
| Prefill | 处理完整 prompt，生成 prompt 的 KV Cache | 长输入、矩阵乘、attention、排队 | TTFT |
| Decode | 逐 token 生成，读写历史 KV Cache | KV 读写、HBM、batch 调度、尾延迟 | TPOT / ITL |

prefill 的特点：

- 一次处理很多输入 token。
- 大矩阵乘规模更大，GPU 利用率较高。
- 长 prompt、RAG、文档问答会让 prefill 很重。
- 适合较大的 TP 或更强算力资源。

decode 的特点：

- 每一步新增一个 token。
- 单步计算小，但要读历史 KV Cache。
- 输出越长、并发越高，decode 压力越大。
- 更关注稳定 TPOT、流式体验和尾延迟。

这也是 PD 分离的出发点：两个阶段的资源画像不同，却被传统 aggregated serving 放在同一批 GPU 上调度。

## 3. Aggregated vs Disaggregated Serving

Aggregated serving：

```text
同一组 GPU 同时跑 prefill 和 decode
```

优点：

- 架构简单。
- 没有跨 worker KV Cache 传输。
- 适合小规模、短 prompt、流量简单场景。

缺点：

- 长 prefill 可能干扰 decode，TPOT/P99 抖动。
- prefill 和 decode 的资源比例被绑在一起。
- 很难分别针对 TTFT 和 TPOT 扩缩容。

Disaggregated serving / PD 分离：

```text
prefill worker/pool 负责 prompt -> KV Cache
decode worker/pool 负责 KV Cache -> output tokens
```

优点：

- prefill/decode 独立扩缩容。
- 降低两阶段调度干扰。
- 可以给 prefill 和 decode 配不同 GPU、TP、batch 策略。
- 更适合长 prompt、混合流量和严格 SLO。

代价：

- KV Cache 要传输。
- 需要 gateway/proxy/router 协调。
- 两套队列和监控。
- KV layout、page、TP size 不一致会复杂。
- 故障恢复和取消请求更难。

## 4. PD 分离架构

常见组件：

```text
Client
-> Gateway / Proxy / Router
-> Prefill Queue
-> Prefill Workers
-> KV Transfer / Connector
-> Decode Queue
-> Decode Workers
-> Streamer
-> Metrics / Trace / Autoscaling
```

一次请求流程：

```text
1. gateway 接收请求，选择 prefill worker 和 decode worker
2. prefill worker 做 prompt prefill，生成 KV Cache
3. decode worker 预留 KV slots / pages
4. KV connector 把 KV Cache 从 prefill 传到 decode
5. decode worker 跳过 prompt prefill，直接进入 decode
6. token stream 返回客户端
7. 请求结束后清理 prefill/decode 两侧资源
```

面试要点：

- prefill 生成的不只是首 token，更重要是 prompt 的 KV Cache。
- decode 必须拿到 KV Cache，才能继续生成。
- gateway 要知道请求状态，避免 prefill 成功但 decode 失败时泄漏资源。

## 5. KV Cache 传输为什么难

KV Cache 大小：

```text
KV bytes ≈ B * S * L * H_kv * D * 2 * bytes_per_element
```

PD 分离后，prefill 结束要把这些 KV 传给 decode。难点：

- KV 很大，网络传输可能吃掉分离收益。
- 跨节点需要 RDMA/NVLink/IB/UCX/NIXL/Mooncake 等高效传输。
- prefill 和 decode 的 TP size 可能不同，KV layout 不一致。
- PagedAttention 把 KV 分页，传输要处理 page/block 映射。
- decode 侧要预分配 KV slots，避免到了才发现放不下。
- 请求取消、超时、失败时要清理两侧资源。

SGLang 文档里提到过一个典型问题：prefill 和 decode TP 不同时，KV head slices 的布局不同，需要 staging buffer 把碎片化 KV 聚合成连续大块再传输，然后在 decode 侧 scatter 到正确 pages。

面试句：

> PD 分离的瓶颈常常不是算，而是 KV handoff。KV 传输慢、layout 转换慢、预分配失败或清理不及时，都会把收益吃掉。

## 6. KV Transfer Backend 怎么讲

常见关键词：

- NIXL：NVIDIA Inference Xfer Library，vLLM/Ray Serve 文档里常见。
- UCX / libfabric / EFA：网络传输后端。
- RDMA：绕过 CPU 拷贝、降低延迟和 CPU 开销。
- Mooncake：KVCache-centric disaggregated architecture，强调 KV Cache pool 和高效传输。
- MPI / UCX：TensorRT-LLM disaggregated serving 中常见。
- Connector：vLLM 里把 prefill/decode 连接起来的抽象。

面试不需要背每个命令，重点说：

```text
传输路径要低延迟、高带宽、少拷贝；
KV block/page 元数据要对齐；
prefill 和 decode 要有握手、预分配、状态同步和清理。
```

## 7. TP / PP / DP / EP 在推理里怎么选

Tensor Parallel，TP：

- 切单层矩阵。
- 解决单卡放不下或单层算力不够。
- 每层可能有 all-reduce/all-gather 通信。
- batch 小时通信开销可能明显。

Pipeline Parallel，PP：

- 按层切模型。
- 解决模型太深、权重放不下。
- 有 pipeline bubble，在线小 batch 时不一定划算。

Data Parallel / Replica：

- 多副本服务不同请求。
- 最简单扩 QPS。
- 每个副本需要完整模型或完整 TP group。

Expert Parallel，EP：

- MoE 模型把 experts 分到不同 GPU。
- token 根据 router 发到 experts。
- 核心通信是 all-to-all。
- 热门专家和负载不均会造成尾延迟。

面试决策：

```text
模型单卡放不下 -> TP/PP/量化
QPS 不够 -> 多副本 / DP
单层太大 -> TP
层数太多 -> PP
MoE experts 多 -> EP
decode 延迟高 -> 优先看 KV/调度/replica，不要盲目加 TP
```

## 8. Prefill 和 Decode 可以用不同并行策略吗

可以，而且这是 PD 分离的价值之一。

例如：

- prefill 用更大 TP，吃长 prompt 的大矩阵计算。
- decode 用更小 TP 或更多 replicas，减少通信、提升并发和 TPOT 稳定性。
- prefill 用高算力 GPU，decode 用更适合长时间 KV 驻留的资源。
- 长 prompt 任务给更多 prefill capacity，短问答任务走 aggregated path。

风险：

- TP size 不同导致 KV layout 不同。
- KV 传输要做 gather/scatter。
- 调度器要知道两边容量。
- profiling 和 autoscaling 更复杂。

## 9. Chunked Prefill、PD 分离、Prefix Cache 区别

| 技术 | 解决什么 | 代价 |
| --- | --- | --- |
| Chunked Prefill | 长 prompt 不要长时间独占 GPU | 调度更复杂，prefill 被切碎 |
| PD Disaggregation | prefill/decode 资源独立，减少互相干扰 | KV transfer、两套队列、架构复杂 |
| Prefix Cache | 相同前缀复用 KV，减少重复 prefill | 命中率和 cache 管理 |

三者可以组合：

```text
prefix cache 命中 -> 少做 prefill
长 prompt 未命中 -> chunked prefill
prefill/decode 干扰严重 -> PD 分离
```

面试别混：

> Chunked prefill 是把 prefill 切小；PD 分离是把 prefill 和 decode 放到不同资源池；prefix cache 是复用已有 KV。

## 10. 路由、Admission Control 和 Autoscaling

PD 分离后，路由器不再只是选一个模型副本，而是要选：

- prefill worker。
- decode worker。
- KV transfer path。
- 是否走 prefix cache。
- 是否走 aggregated fallback。
- 是否按请求长度、输出预算、租户、SLO 分流。

Admission control 要看：

- prefill queue depth。
- decode queue depth。
- decode KV capacity。
- 预计 prompt tokens 和 max_new_tokens。
- 当前 TTFT/TPOT SLO。
- 网络/KV transfer 带宽。

扩缩容：

```text
prefill backlog 高 / TTFT 高 -> 增加 prefill capacity
decode backlog 高 / TPOT 高 -> 增加 decode capacity
KV transfer 高 -> 调整 placement / connector / 是否本地化
短请求多 -> 可能不适合远程 prefill
长 prompt 多 -> 更适合 PD 分离
```

## 11. 监控指标

端到端：

- TTFT、TPOT/ITL、E2E latency。
- tokens/s、QPS、goodput。
- P50/P95/P99。
- timeout / cancel / error rate。

prefill：

- prefill queue depth。
- prefill tokens/s。
- prefill batch size。
- prefill GPU utilization。
- prefix cache hit rate。

decode：

- decode queue depth。
- running requests。
- KV cache utilization。
- TPOT / output tokens/s。
- decode GPU utilization / HBM bandwidth。

KV transfer：

- transfer latency。
- transfer bytes。
- transfer bandwidth。
- pending KV requests。
- prealloc failure。
- connector error rate。

资源：

- GPU memory。
- KV pages / blocks used。
- NCCL/UCX/RDMA errors。
- network bandwidth。
- CPU proxy overhead。

## 12. 什么时候 PD 分离可能不划算

- prompt 很短，prefill 本来不重。
- 输出很短，decode 也不重。
- 网络带宽不足，KV transfer 成为瓶颈。
- prefill/decode 放太远，跨机房或跨可用区传 KV。
- cache 命中率很高，重复 prefill 已被 prefix cache 解决。
- 流量很小，单池调度足够。
- 系统团队无法承担复杂度和故障恢复成本。
- 多租户安全要求不允许跨池共享 KV。

面试句：

> PD 分离适合 prefill/decode 干扰明显、长 prompt 或混合流量重、SLO 严格且网络足够好的场景。短请求、低流量或网络慢时，aggregated serving 可能更稳。

## 13. MoE Serving 为什么难

MoE 推理流程：

```text
token hidden state
-> router 选 top-k experts
-> dispatch tokens to experts
-> expert FFN compute
-> combine outputs
```

难点：

- expert parallel 需要 all-to-all token dispatch。
- 不同 expert 热度不均，热门 expert 形成尾延迟。
- token 数随 batch 和 routing 动态变化，kernel shape 不稳定。
- 跨节点 expert 通信开销大。
- MoE 参数总量大，权重放置和缓存复杂。
- prefill 阶段 token 多，expert dispatch 压力更大。
- decode 阶段 token 少，通信和调度开销占比可能更高。

优化方向：

- load balancing loss / routing 约束。
- expert replication：复制热门专家。
- expert placement：把常一起访问的专家放近。
- token batching / grouped GEMM。
- all-to-all overlap。
- 分离 prefill/decode 的 expert 策略。
- 监控 per-expert load 和 tail latency。

## 14. 多卡推理常见故障

TPOT 突然变高：

- decode KV Cache 过大。
- TP 通信开销高。
- network/RDMA/NCCL 抖动。
- MoE hot expert。
- PD KV transfer 堵塞。

TTFT 突然变高：

- prefill queue backlog。
- 长 prompt 占用 prefill worker。
- prefix cache miss。
- prefill worker 不足。
- gateway 路由错误。

OOM：

- decode KV capacity 不足。
- prealloc KV slots 太乐观。
- max_model_len / max_num_seqs 配置过高。
- draft/LoRA/quantization/PD 双侧资源叠加。

请求卡住：

- prefill 成功但 KV transfer 未完成。
- decode 等 remote KV。
- cleanup 失败导致资源泄漏。
- proxy 状态机丢事件。

## 15. 项目 8 分钟讲法

```text
背景：
  在线 LLM serving 有长 prompt + 长输出混合流量，TTFT 和 TPOT 都有 SLO。

baseline：
  aggregated vLLM/SGLang/TensorRT-LLM，记录 TTFT、TPOT、P95/P99、KV 占用。

问题：
  长 prefill 干扰 decode，decode 尾延迟高；或者 prefill capacity 和 decode capacity 比例不匹配。

方案：
  引入 PD disaggregation：
  - gateway 做请求路由
  - prefill pool 处理 prompt
  - KV connector 传输 KV Cache
  - decode pool 负责流式输出
  - 按 prompt/output 长度和 SLO 动态选择 aggregated / disaggregated

指标：
  prefill queue、decode queue、KV transfer latency、TTFT、TPOT、goodput、OOM、error rate。

取舍：
  长 prompt 收益明显，短 prompt 不一定；网络和 KV layout 是关键瓶颈。

上线：
  灰度、fallback 到 aggregated、超时清理、资源泄漏监控、SLO 告警。
```

## 16. 高频追问快答

**Q1：PD 分离为什么出现？**  
因为 prefill 和 decode 的资源画像不同，混在同一 GPU 池里会互相干扰，并且不能独立扩缩容。

**Q2：PD 分离的最大代价是什么？**  
KV Cache 传输和系统复杂度。传输慢会抵消收益，状态机和清理也更难。

**Q3：TTFT 高一定加 prefill worker 吗？**  
不一定。还要看 gateway 排队、tokenizer、RAG、prefix cache miss、长请求阻塞和 KV transfer。

**Q4：TPOT 高一定加 decode worker 吗？**  
不一定。可能是 KV Cache 太长、HBM 带宽、TP 通信、MoE all-to-all、spec decoding 接受率低或输出过长。

**Q5：MoE serving 为什么尾延迟高？**  
因为专家负载动态且不均，热门专家、all-to-all、跨节点 dispatch 和小 batch shape 都会造成尾延迟。

## 17. 面试前背诵版

分布式推理要围绕 SLO、显存、KV Cache 和通信讲。Prefill 处理 prompt，偏大矩阵和算力，主要影响 TTFT；decode 逐 token 生成，偏 KV Cache 读写和调度，主要影响 TPOT。传统 aggregated serving 把两阶段放一起，简单但会互相干扰；PD disaggregation 把 prefill/decode 分到不同 worker/pool，能独立扩缩容和优化并行策略，但必须传 KV Cache，并处理路由、两套队列、KV layout、预分配、清理和故障恢复。KV transfer 是核心风险，尤其是长上下文和跨节点。TP 切矩阵，PP 切层，DP/replica 扩 QPS，EP 切 MoE experts；MoE serving 难在 token dispatch、all-to-all、热门专家和负载均衡。线上要看 TTFT、TPOT、P95/P99、prefill/decode queue、KV transfer latency、KV capacity、GPU/HBM、network、OOM 和 goodput。PD 分离适合长 prompt、混合流量、两阶段干扰明显和 SLO 严格的场景；短请求、低流量或网络差时可能不划算。

## 参考资料

- vLLM Disaggregated Prefilling：[https://docs.vllm.ai/en/latest/features/disagg_prefill/](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- SGLang Prefill-Decode Disaggregation：[https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md)
- TensorRT-LLM Disaggregated Serving：[https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- TensorRT-LLM disagg-serving feature doc：[https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md)
- Ray Serve Prefill/decode disaggregation：[https://docs.ray.io/en/latest/serve/llm/user-guides/prefill-decode.html](https://docs.ray.io/en/latest/serve/llm/user-guides/prefill-decode.html)
- NVIDIA Dynamo Disaggregation：[https://docs.nvidia.com/dynamo/dev/backends/sg-lang/disaggregation](https://docs.nvidia.com/dynamo/dev/backends/sg-lang/disaggregation)
- DistServe 论文：[https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)
- Mooncake：[https://kvcache-ai.github.io/Mooncake/](https://kvcache-ai.github.io/Mooncake/)
- PyTorch + vLLM Disaggregated Inference：[https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/)
