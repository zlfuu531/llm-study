# 答案版 20：GPU Systems、CUDA / Triton 与 FlashAttention

对应题号：421-440。建议先读 [25_GPU_CUDA_Triton与FlashAttention面试.md](../25_GPU_CUDA_Triton与FlashAttention面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 421. 大模型 GPU 性能瓶颈一般怎么分类？

30 秒版：

主要分 compute-bound、memory-bound、launch/CPU-bound 和 communication-bound。先看算力、HBM 带宽、kernel timeline、CPU 喂数和 NCCL 通信，不要一上来就猜“加 batch”或“写 CUDA”。

2 分钟版：

Compute-bound 是计算单元忙，典型是大 matmul、tensor core 利用高。Memory-bound 是数据搬运慢，常见于 LayerNorm、softmax、decode KV Cache 读取、小算子链。Launch/CPU-bound 是小 kernel 太多或 Python/CPU 喂不动，timeline 里 GPU 有空洞。Communication-bound 是多卡训练/推理里 NCCL all-reduce、all-gather、all-to-all 等占比高。

排查顺序：

1. 看端到端 tokens/s、step time、TTFT/TPOT。
2. 用 profiler 看 CPU/GPU timeline。
3. 看 kernel 是算力高还是带宽高。
4. 多卡看 NCCL 和 overlap。
5. 再决定 fusion、FlashAttention、Triton、batch、并行策略或通信优化。

## 422. GPU 内存层次怎么影响性能？

30 秒版：

GPU 有 register、shared memory/SRAM、L2、HBM 等层次。HBM 容量大但相对慢，片上内存快但小。很多大模型优化的本质是让数据在片上多复用，少读写 HBM。

2 分钟版：

Register 最快但每线程很少；shared memory 是 block 内可共享的片上缓存；L2 是全局 cache；HBM 是显存，容量大、带宽高，但相对片上内存仍慢。CPU memory 和磁盘更慢。

标准 attention、LayerNorm、softmax、sampling 这类算子如果反复把中间张量写回 HBM，就容易 memory-bound。FlashAttention 通过 tiling 把 Q/K/V block 放到片上计算，避免写完整 attention matrix。Kernel fusion 也是类似思路：多个小算子合并，减少中间写回。

面试句：

> GPU 快的前提是数据访问模式也快。大模型系统优化常常不是减少数学公式，而是减少 HBM IO 和同步。

## 423. compute-bound 和 memory-bound 怎么判断？

30 秒版：

看算术强度、SM/tensor core 利用率和 memory bandwidth。算力接近峰值、带宽没满偏 compute-bound；带宽高、SM 等数据、小算子多偏 memory-bound。

2 分钟版：

Roofline 思路是：

```text
Arithmetic intensity = FLOPs / bytes moved
performance <= min(peak compute, bandwidth * arithmetic_intensity)
```

大矩阵乘 FLOPs 多、数据复用高，容易更接近 compute-bound。Elementwise、reduction、LayerNorm、softmax、decode 读 KV 这类算子每搬一点数据做的计算不多，容易 memory-bound。

实际要用 profiler 看：

- SM utilization。
- Tensor core utilization。
- HBM bandwidth。
- kernel 时间。
- warp stall reason。

优化方向不同：compute-bound 看 dtype、tensor core、matmul shape；memory-bound 看 fusion、tiling、减少中间张量、量化和内存访问合并。

## 424. CUDA kernel 的执行模型怎么讲？

30 秒版：

CUDA kernel 启动一个 grid，grid 里有很多 block，每个 block 里有多个 thread，thread 按 warp 调度。block 内可以共享 shared memory 和同步，block 之间通常不能直接同步。

2 分钟版：

执行层级是：

```text
Grid -> Blocks/CTAs -> Warps -> Threads
```

GPU 有很多 SM，block 被分配到 SM 上执行。一个 warp 通常 32 个线程一起调度，所以同一个 warp 内最好走相同分支，并访问连续地址。Block 内线程可以用 shared memory 共享数据，比如 matmul 和 attention 里的 tile。

面试补充：

- thread/block 划分决定并行粒度。
- shared memory 用来复用数据，但容量有限。
- register 太多会降低活跃 warp 数。
- coalesced memory access 对带宽很重要。
- kernel launch 本身有开销，小算子太多会拖慢。

## 425. 什么是 coalesced access，为什么重要？

30 秒版：

Coalesced access 是相邻线程访问连续内存地址，让 GPU 能把多次访问合并成更少的内存事务。否则会浪费 HBM 带宽，memory-bound 算子更慢。

2 分钟版：

GPU 是 SIMT 模型，一个 warp 里的线程一起执行。如果 32 个线程访问连续地址，硬件可以合并加载，带宽利用率高。如果每个线程跳着访问、访问不对齐或随机访问，就会产生更多 memory transaction，实际带宽下降。

大模型里的例子：

- Tensor layout 不合适导致非连续访问。
- 转置或 gather/scatter 导致访存不规则。
- KV Cache 访问如果布局差，decode 会更慢。
- 自定义 kernel 里 offsets 设计不合理会浪费带宽。

回答时可以说：

> 对 memory-bound kernel，访存模式往往比 FLOPs 更关键。优化先看是否 coalesced、是否有重复读写、是否能 tile 到 shared memory。

## 426. occupancy 越高越好吗？

30 秒版：

不是。Occupancy 高说明 SM 上活跃 warp 多，有助于隐藏延迟，但最终性能还要看计算、带宽、register、shared memory 和实际吞吐。盲目追 occupancy 可能没收益。

2 分钟版：

Occupancy 是活跃 warps 与理论最大 warps 的比例。它能帮助隐藏内存延迟，因为一个 warp 等数据时，SM 可以调度另一个 warp。但如果 kernel 已经 compute-bound，或者 memory bandwidth 已经打满，继续提高 occupancy 不一定提升性能。

另外，降低寄存器使用提高 occupancy，可能导致 spilling 到 local memory，反而变慢。增加 block 数也可能加剧 shared memory 压力。

更好的面试表达：

> Occupancy 是诊断指标，不是优化目标。优化目标是实际吞吐和延迟。要结合 SM utilization、memory bandwidth、warp stall、register pressure 和 roofline 一起看。

## 427. Kernel fusion 为什么能加速？

30 秒版：

Kernel fusion 把多个小算子合成一个 kernel，减少 launch overhead 和中间结果写回 HBM。它对 memory-bound、小算子链、elementwise/reduction 很有效。

2 分钟版：

未融合时，每个算子都要从 HBM 读输入、写输出，下一个 kernel 再读回来。融合后可以在寄存器或 shared memory 中完成多个操作，最后只写一次输出。

收益：

- 少启动 kernel。
- 少写中间张量。
- cache locality 更好。
- Python/PyTorch eager overhead 更低。

风险：

- fused kernel 更复杂。
- register pressure 可能上升。
- 太多逻辑融合可能降低 occupancy。
- 动态 shape 下编译和维护成本更高。

常见 fused ops：RMSNorm、LayerNorm、bias+GELU、SwiGLU、RoPE、softmax、sampling、quant/dequant。

## 428. FlashAttention 为什么快？

30 秒版：

FlashAttention 快是因为它是 IO-aware exact attention：用 tiling 和 online softmax 避免把完整 `N x N` attention matrix 写回 HBM，减少显存读写，而不是改 attention 公式。

2 分钟版：

标准 attention 会计算：

```text
S = QK^T
P = softmax(S)
O = PV
```

`S` 和 `P` 都是 `seq_len x seq_len`，长序列时非常大。如果完整写入 HBM，会产生大量 IO。FlashAttention 把 Q/K/V 分块，把 block 搬到片上 SRAM/shared memory 里计算，并用 online softmax 维护 running max 和 normalizer，使分块结果和全量 softmax 等价。

重点：

- 它是 exact attention，不是近似 attention。
- 它主要减少 HBM IO。
- 长序列和训练/prefill 场景收益明显。
- 它和 PagedAttention 不同，后者管 serving KV Cache。

## 429. FlashAttention 的 online softmax 直觉是什么？

30 秒版：

分块时不能一次看到所有 logits，所以要维护 running max 和 running normalizer。新 block 来时更新最大值，并把旧累积结果按新尺度重标定，最后得到和全量 softmax 等价的结果。

2 分钟版：

普通 softmax 为了数值稳定会减全局最大值：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

FlashAttention 分块计算 `QK^T`，每次只看到一部分 logits。它维护当前最大值 `m`、分母 `l` 和输出累积 `o`。当新 block 的最大值更大时，会把旧的分母和输出乘上尺度因子，换到新的最大值基准下，再合并新 block 的贡献。

面试不用背完整推导，但要说清：

> online softmax 让 attention 可以分块计算，同时保持数值稳定和 exact 结果，这是 FlashAttention 能避免写完整 attention matrix 的关键。

## 430. FlashAttention 和 PagedAttention 怎么区分？

30 秒版：

FlashAttention 优化单次 attention 计算的 IO，减少 `N x N` 中间矩阵写回；PagedAttention 优化在线推理的 KV Cache 管理，用 block/page 减少显存碎片。一个管计算，一个管缓存管理。

2 分钟版：

FlashAttention 发生在 attention kernel 内，训练、prefill 和长序列 attention 都会受益。它不改变 attention 结果，只改变计算组织方式。

PagedAttention 发生在 serving 系统里，面对很多请求动态到达、动态增长和结束。它把 KV Cache 分成 block/page，用映射表维护逻辑顺序，减少预留浪费和碎片，提高并发。

一句话：

> FlashAttention 是 IO-aware attention kernel；PagedAttention 是 KV Cache memory manager。

## 431. Triton 是什么，适合解决什么问题？

30 秒版：

Triton 是 Python-like 的 GPU kernel DSL，适合快速写 block-level fused kernel，比如 RMSNorm、softmax、RoPE、sampling、量化/反量化和自定义 attention。

2 分钟版：

Triton 让模型工程师不用写完整 CUDA C++，也能控制 program/block、tile、mask、load/store 和 num_warps。一个 Triton program 类似处理一个 block 的数据，开发者用 offsets 和 mask 处理当前 tile。

适合场景：

- PyTorch eager 里小算子很多。
- 需要 fused kernel 减少 HBM IO。
- 想快速实验自定义算子。
- 量化、RoPE、RMSNorm、sampling、attention 有特殊需求。

限制：

- 极致性能有时仍需要 CUDA/CUTLASS。
- 动态 shape 和复杂控制流会增加难度。
- 仍要懂内存访问、register、occupancy 和 profiler。

## 432. Triton kernel 里的 program_id、block、mask 怎么理解？

30 秒版：

`program_id` 表示当前 Triton program 负责哪一块数据，block 是一次处理的 tile 大小，mask 用来处理边界，避免越界 load/store。

2 分钟版：

Triton kernel 通常把数据切成很多 block。每个 program instance 处理一个 block。`tl.program_id(0)` 拿到当前 block 编号，然后计算 offsets：

```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n
```

`tl.load(x + offsets, mask=mask)` 表示只加载合法位置，越界位置不访问。`BLOCK` 决定 tile 大小，会影响并行度、寄存器使用、内存访问和 occupancy。

面试加分：

> Triton 写起来像 Python，但本质上还是 GPU kernel。性能取决于 tile、访存合并、register pressure、num_warps 和实际硬件。

## 433. PyTorch 2 的 torch.compile 为什么可能加速？

30 秒版：

`torch.compile` 捕获计算图，减少 Python overhead，并通过 Inductor 做图优化和算子融合，后端可能生成 Triton/C++ kernel。shape 稳定、算子组合固定时收益更明显。

2 分钟版：

PyTorch eager 每个 op 都由 Python 调度，容易产生很多小 kernel 和中间张量。`torch.compile` 会尽量捕获图，把多个 op 交给编译器优化。Inductor 可以做 fusion、layout 优化和代码生成，减少 launch overhead 和 HBM IO。

不一定加速的原因：

- graph break 太多。
- 动态 shape 太多。
- batch 太小，编译收益覆盖不了开销。
- 自定义 Python 逻辑或 unsupported op 阻断编译。
- 编译时间太长，不适合短生命周期任务。

面试表达：

> `torch.compile` 是先让编译器吃到更大的图，再做 fusion 和代码生成；它不是所有场景都快，要看 graph break、shape 稳定性和热路径是否足够长。

## 434. GPU 利用率低怎么排查？

30 秒版：

先看 timeline：CPU 是否喂不动、kernel 是否碎、batch/seq 是否太小、是否频繁同步、是否等 NCCL/IO、是否动态 shape 导致编译或调度开销。

2 分钟版：

排查链路：

1. 用 Nsight Systems 或 PyTorch Profiler 看 GPU timeline 是否有空洞。
2. 如果空洞前是 CPU op，看 tokenizer、dataloader、Python、JSON、日志和网络。
3. 如果是很多小 kernel，看 fusion、torch.compile、CUDA Graph。
4. 如果 batch 太小，看 batching、packing 或请求合并。
5. 如果有同步，看 `.item()`、CPU-GPU copy、blocking op。
6. 多卡看 NCCL 是否在等待。
7. 推理看是否 queue、prefill/decode 调度不合理。

不要只看 `nvidia-smi` 的瞬时 GPU util，它太粗。

## 435. 单个 kernel 慢怎么排查？

30 秒版：

用 Nsight Compute 看它是 memory-bound 还是 compute-bound，再看访存是否 coalesced、shared memory bank conflict、register pressure、occupancy、warp divergence、tensor core 是否用上、tile/block size 是否合理。

2 分钟版：

先定位哪个 kernel 慢，再看：

- Memory throughput 是否接近上限。
- SM/tensor core utilization 是否高。
- Warp stall reason 是 memory、barrier、dependency 还是 execution。
- Global load/store 是否合并。
- Shared memory 是否有 bank conflict。
- Register 使用是否导致 occupancy 低或 spilling。
- 分支是否导致 warp divergence。
- Matmul 是否走 tensor core。
- Tile size 是否导致数据复用不足。

对应优化：

- Memory-bound：fusion、tiling、coalesced、减少中间写回。
- Compute-bound：tensor core、dtype、matmul shape。
- Launch-bound：融合小 kernel、CUDA Graph。
- Register 太多：拆 kernel 或调 tile。

## 436. NCCL 和 all-reduce / all-gather / all-to-all 分别是什么？

30 秒版：

NCCL 是 NVIDIA 多 GPU 通信库。All-reduce 常用于梯度同步和 TP 结果合并，all-gather 常用于收集分片参数/激活，all-to-all 常用于 MoE expert parallel 的 token 分发。

2 分钟版：

常见 collective：

| Collective | 作用 | 大模型场景 |
| --- | --- | --- |
| all-reduce | 所有卡求和并拿到结果 | DDP 梯度、TP 中间结果 |
| all-gather | 收集所有卡的数据 | FSDP 参数聚合、序列/张量并行 |
| reduce-scatter | reduce 后每卡拿一片 | ZeRO/FSDP |
| all-to-all | 每卡发不同数据给其他卡 | MoE expert parallel |
| broadcast | 一卡发给多卡 | 初始化或同步状态 |

面试加分：

> 不同并行策略对应不同通信模式。通信慢要结合 timeline 看 collective 类型、占比、overlap、拓扑和负载均衡。

## 437. 多卡训练/推理通信慢怎么排查？

30 秒版：

看 NCCL 在 timeline 里的占比和是否与计算 overlap，再查 collective 类型、bucket size、micro-batch、并行策略、NVLink/PCIe/跨机网络、RDMA/IB 和 MoE 负载均衡。

2 分钟版：

排查步骤：

1. 看 step time 中 NCCL 占比。
2. 看通信是否和 backward/forward overlap。
3. 看是 all-reduce、all-gather、reduce-scatter 还是 all-to-all 慢。
4. 看 bucket size 是否导致通信太碎或太晚。
5. 看并行策略是否合理：DP、TP、PP、FSDP、MoE。
6. 看 GPU 拓扑和跨机网络。
7. 看某些 GPU 是否负载不均，尤其 MoE。
8. 看 batch/micro-batch 是否太小，计算不足以掩盖通信。

回答模板：

> 多卡慢不是一句“网络问题”。我会先用 timeline 确认 NCCL 占比，再结合 collective 类型、overlap、拓扑和并行策略定位。

## 438. 量化为什么不一定提速？

30 秒版：

量化一定能省存储，但不一定提速。速度取决于硬件是否支持低比特计算、kernel 是否高效、解量化开销、batch/seq shape、memory-bound 还是 compute-bound。

2 分钟版：

INT8/INT4/FP8 能减少权重、激活或 KV Cache 的内存占用和带宽压力。如果瓶颈是 HBM bandwidth，量化可能明显加速；如果瓶颈是 CPU、通信、kernel launch 或没有低比特高效 kernel，收益可能很小。

常见坑：

- 权重量化后计算前要 dequant，抵消收益。
- kernel 不支持某 shape。
- 小 batch 下 overhead 更明显。
- 精度下降导致要用更大模型或更多重试。
- KV Cache 量化可能影响长上下文质量。

上线要同时看 latency、tokens/s、显存、P95/P99、质量和稳定性。

## 439. 自定义 Triton/CUDA kernel 项目怎么讲？

30 秒版：

按“定位热点 -> 判断瓶颈 -> 设计 kernel -> 验证正确性 -> profiler 对比 -> 上线风险”讲。不要一上来说自己写了 kernel，要说为什么现有算子不够。

2 分钟版：

项目讲法：

1. 背景：哪个模型/服务慢，目标是 step time、tokens/s 还是 TTFT/TPOT。
2. 定位：用 profiler 发现哪个 op 或 kernel 是热点。
3. 分类：memory-bound、compute-bound、launch-bound 还是 communication-bound。
4. 方案：Triton fused RMSNorm / RoPE / sampling / quant-dequant 等。
5. 正确性：和 PyTorch baseline 比较误差，覆盖不同 shape、dtype、边界。
6. 性能：比较 kernel time、end-to-end latency、tokens/s、显存、P95。
7. 风险：数值稳定、动态 shape、编译缓存、硬件兼容、fallback。

示例句：

> 我们先用 PyTorch Profiler 定位到多个 elementwise 小算子和中间张量写回，判断是 memory/launch-bound。然后用 Triton 写 fused kernel，把 RMSNorm 和后续 elementwise 合并，减少 HBM 读写。上线前用不同 batch/seq/dtype 做误差和性能回归。

## 440. GPU Systems 面试前最后怎么复习？

30 秒版：

按五条线复习：内存层次和 roofline、CUDA 执行模型、FlashAttention、Triton/fusion、profiler/NCCL 排查。每条准备一个 30 秒解释和一个项目例子。

2 分钟版：

最后三天不要试图补完整 CUDA。优先背：

1. GPU 内存层次：register/shared/L2/HBM，为什么 HBM IO 是瓶颈。
2. Roofline：compute-bound vs memory-bound 怎么判断。
3. CUDA 模型：grid/block/warp/thread、coalesced、occupancy、divergence。
4. FlashAttention：IO-aware exact attention，tiling + online softmax。
5. Triton：program_id、block、mask、tl.load/store、fused kernel。
6. Profiler：PyTorch Profiler、Nsight Systems、Nsight Compute 分别看什么。
7. NCCL：all-reduce、all-gather、all-to-all 和通信排查。

面试背诵句：

> 我不会把 GPU 优化当成玄学，先用 profiler 定位瓶颈，再判断算力、带宽、launch、CPU 还是通信，最后选择 fusion、FlashAttention、Triton、torch.compile、batch 或并行策略。

## 本组题的复习顺序

1. 先背 421-423：瓶颈分类、内存层次、roofline。
2. 再背 424-427：CUDA 执行模型、访存、occupancy、fusion。
3. 然后背 428-433：FlashAttention、Triton、torch.compile。
4. 最后背 434-440：GPU profiler、NCCL、量化、项目讲法。

## 延伸阅读

- NVIDIA CUDA C++ Programming Guide：[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- NVIDIA CUDA C++ Best Practices Guide：[https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Triton 官方文档：[https://triton-lang.org/main/](https://triton-lang.org/main/)
- FlashAttention 论文：[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- FlashAttention-2 论文：[https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- PyTorch Profiler 文档：[https://pytorch.org/docs/stable/profiler.html](https://pytorch.org/docs/stable/profiler.html)
- Nsight Systems：[https://docs.nvidia.com/nsight-systems/UserGuide/](https://docs.nvidia.com/nsight-systems/UserGuide/)
- Nsight Compute：[https://docs.nvidia.com/nsight-compute/](https://docs.nvidia.com/nsight-compute/)
- NCCL User Guide：[https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
