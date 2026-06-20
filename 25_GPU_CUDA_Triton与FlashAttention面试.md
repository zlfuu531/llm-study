# GPU Systems、CUDA / Triton 与 FlashAttention 面试

这一章面向 AI Infra、推理优化、训练系统、模型服务平台、端侧/云端部署和偏系统的大模型岗位。很多 2025-2026 面经里会问到“FlashAttention 为什么快”“CUDA kernel 优化怎么做”“Triton 写过吗”“GPU 利用率低怎么排查”。你不需要两个月内变成 CUDA 专家，但要能把 GPU 性能问题讲清楚。

本章目标：

- 能解释 GPU 为什么适合大模型矩阵计算。
- 能区分 compute-bound、memory-bound、communication-bound。
- 能讲清 HBM、SRAM/shared memory、register、coalesced access、occupancy、warp divergence。
- 能把 FlashAttention 说成 IO-aware attention，而不是“用了更快的 attention”。
- 能解释 Triton 的 program/block 思路，以及为什么它适合写 fused kernel。
- 能给出 GPU 利用率低、kernel 慢、显存带宽打满、多卡通信慢的排查链路。

如果面试官把“量化为什么不一定提速”追到 INT4 kernel、dequant、metadata、FP8 Tensor Core 或 KV Cache 量化，配合 [37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md](37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md) 一起看。本章讲硬件性能语言，量化章讲低比特方案和部署取舍。

如果面试官把“speculative decoding 为什么有时不提速”追到 batch、GPU 利用率、验证并行、draft 开销和显存压力，配合 [38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md](38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md) 一起看。

如果面试官追 PD 分离里的 KV Cache transfer、RDMA/NIXL、NCCL/UCX、MoE all-to-all 或多卡 P99 排查，配合 [39_分布式推理_PD分离_KVCache传输与MoEServing面试.md](39_分布式推理_PD分离_KVCache传输与MoEServing面试.md) 一起看。

## 一句话总览

大模型系统性能不是“GPU 越贵越快”这么简单。模型训练和推理通常卡在四类瓶颈：

| 瓶颈 | 直觉 | 典型现象 | 常见优化 |
| --- | --- | --- | --- |
| Compute-bound | 算不完 | SM busy 高，tensor core 利用高 | FP8/BF16、更高效 matmul、tensor core、减少 FLOPs |
| Memory-bound | 数据搬不动 | HBM bandwidth 高，SM 等数据 | kernel 融合、tiling、减少读写、量化、cache locality |
| Launch/CPU-bound | kernel 太碎或 CPU 喂不动 | GPU 空洞多，小 kernel 很多 | fused ops、CUDA Graph、减少同步、batch |
| Communication-bound | 多卡等通信 | NCCL 时间高，GPU 间等待 | 并行策略、overlap、拓扑、bucket、网络排查 |

面试时你可以先说：

> 我会先判断瓶颈是算力、显存带宽、CPU/kernel launch 还是多卡通信。FlashAttention、Triton fused kernel 和量化很多时候不是“魔法加速”，而是在减少 HBM 读写、减少中间张量和提高硬件利用率。

## GPU 内存层次

理解 GPU 性能，先理解数据在哪里。

```text
Register / Tensor Core operand
  fastest, per thread, very small

Shared Memory / SRAM
  on-chip, per block / CTA, fast, manually managed

L2 Cache
  on-chip cache, shared by SMs

HBM / Global Memory
  off-chip, large, high bandwidth but相对慢

CPU Memory / Storage
  更慢，训练推理中要避免频繁同步和拷贝
```

大模型里很多算子慢，不是因为 FLOPs 算不过来，而是因为中间结果太大，反复从 HBM 读写。标准 attention 如果显式写出 `S = QK^T` 和 `P = softmax(S)`，中间矩阵是 `N x N`，长序列时 HBM 访问非常重。

### 面试必须会说的几个词

| 词 | 口语解释 | 面试怎么用 |
| --- | --- | --- |
| HBM | GPU 大显存，带宽高但仍比片上内存慢 | memory-bound 时主要看它 |
| Shared memory | 片上可编程缓存 | tiling 把数据搬进来复用 |
| Register | 每个 thread 的最快存储 | 太多会降低 occupancy |
| Coalesced access | 相邻线程访问连续地址 | 不合并访问会浪费带宽 |
| Occupancy | SM 上活跃 warp/block 的程度 | 低不一定慢，高也不一定快 |
| Warp divergence | 同一个 warp 内分支不一致 | 会串行执行不同分支 |
| Bank conflict | shared memory 访问冲突 | 会让片上访问变慢 |

## Roofline 思维

Roofline 不是必须画图，但面试要会用它的思路。

核心概念：

```text
Arithmetic intensity = FLOPs / bytes moved
```

- 算术强度低：每搬一点数据只做很少计算，容易 memory-bound。
- 算术强度高：搬一次数据能复用很多次，可能 compute-bound。

性能上限粗略看：

```text
attainable performance <= min(peak compute, memory bandwidth * arithmetic intensity)
```

大模型常见判断：

- 大矩阵乘：通常更容易利用 Tensor Core，偏 compute-bound 或混合瓶颈。
- LayerNorm、RMSNorm、softmax、小 batch decode：常常 memory-bound 或 launch-bound。
- attention 长序列：朴素实现会被 `N x N` 中间矩阵读写拖住。
- decode 阶段：每步算得少，但读 KV Cache 多，容易 memory bandwidth-bound。

面试表达：

> 我不会先猜哪个优化一定有效，而是先看算术强度、HBM bandwidth、SM utilization 和 kernel 时间。如果算子 memory-bound，堆更多 FLOPs 没用，应该减少读写和中间张量；如果 compute-bound，才重点看 tensor core、数据类型和 matmul shape。

## CUDA Kernel 基础

你不一定要手写完整 CUDA，但要能说出 kernel 的执行模型。

```text
Grid
  -> Blocks / CTAs
      -> Warps
          -> Threads
```

- 一个 CUDA kernel 启动很多 block。
- 每个 block 分配到某个 SM 上执行。
- block 内线程可以用 shared memory 和同步。
- warp 通常是 32 个线程一起调度。

### CUDA kernel 优化常见方向

1. 合理划分 block/thread，让并行度足够。
2. 尽量让全局内存访问 coalesced。
3. 用 tiling 把数据搬到 shared memory 复用。
4. 减少 HBM 读写和中间张量。
5. 减少分支发散。
6. 控制 register 使用，避免 occupancy 太低。
7. 使用向量化 load/store。
8. 用 fused kernel 减少 launch 和中间写回。
9. 对 matmul 使用 Tensor Core 友好的 shape 和 dtype。
10. 用 profiler 先定位，再优化。

### Occupancy 的常见误区

Occupancy 高不等于一定快。它只是说明 SM 上有多少活跃 warp，可以帮助隐藏内存延迟。但如果 kernel 本身 memory-bound，或者寄存器/共享内存使用很重，盲目追 occupancy 可能没有收益。

更好的说法：

> Occupancy 是诊断指标，不是最终目标。最终看的是吞吐、延迟、SM 利用、memory bandwidth 和是否达到 roofline 附近。

## Kernel Fusion

Kernel fusion 是把多个小算子合成一个 kernel。它常见于 LayerNorm、RMSNorm、bias + activation、dropout、softmax、elementwise ops、quant/dequant 等。

未融合：

```text
x -> kernel1 -> 写 HBM
  -> kernel2 -> 写 HBM
  -> kernel3 -> 写 HBM
```

融合后：

```text
x -> fused kernel -> 写一次 HBM
```

收益：

- 减少 kernel launch overhead。
- 减少中间张量写回 HBM。
- 提高 cache locality。
- 对小算子和 memory-bound 算子很有效。

风险：

- kernel 更复杂。
- register pressure 可能上升。
- fused 太多可能降低 occupancy。
- shape 多变时编译和维护成本增加。

面试表达：

> Fusion 的本质不是把代码写少，而是减少中间结果在 HBM 来回搬运。对 memory-bound 小算子收益大，但要用 profiler 验证 register、occupancy 和实际带宽。

## FlashAttention 为什么快

标准 attention：

```text
S = Q K^T
P = softmax(S)
O = P V
```

其中 `S` 和 `P` 都是 `seq_len x seq_len`，长序列时非常大。如果把它们完整写入 HBM，再读回来做 softmax 和乘 V，会产生大量 HBM IO。

FlashAttention 的核心是 IO-aware exact attention：

- 把 Q、K、V 分块。
- 在片上 SRAM/shared memory 中计算局部 attention。
- 用 online softmax 维护数值稳定的归一化统计。
- 不把完整 `N x N` attention matrix 写回 HBM。
- 前向和反向都重新组织计算，减少 HBM 读写。

它不是近似 attention，不是稀疏 attention，也不是改了注意力公式。输出在数学上仍然是 exact attention，只是执行方式更省 IO。

### Online Softmax 直觉

普通 softmax：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

分块时不能一次看到全部 `x`，所以要维护：

- 当前最大值 `m`。
- 当前归一化分母 `l`。
- 当前输出累积 `o`。

当新 block 来时，用新的最大值更新旧分母和输出，使结果等价于全量 softmax。

面试不用推完整公式，但要会说：

> FlashAttention 能分块做 softmax，是因为它维护了 running max 和 running normalizer，保证数值稳定，并把不同 block 的贡献正确合并。

### FlashAttention 和 PagedAttention 再对比一次

| 技术 | 解决对象 | 核心收益 |
| --- | --- | --- |
| FlashAttention | 单次 attention 计算的 IO | 少写 `N x N` 中间矩阵，提升训练/长 prompt prefill |
| PagedAttention | 在线 serving 的 KV Cache 管理 | 减少 KV 显存碎片，提高并发 |

## FlashAttention-2 / 3 怎么讲

面试中一般不要求背实现细节，但你要知道后续版本的方向：

- FlashAttention-2：进一步改善 work partitioning，减少非矩阵乘部分开销，提高并行性和硬件利用率。
- FlashAttention-3：面向 Hopper 等新硬件进一步利用异步、Tensor Core 和 FP8 能力。

一句话：

> FlashAttention 系列的主线一直是围绕 GPU 硬件特性重排 attention 计算，让更多时间花在高效矩阵计算上，减少 HBM IO、同步和非 matmul 开销。

## Triton 怎么讲

Triton 是一个面向 GPU kernel 的 Python-like DSL。它让你不用写完整 CUDA C++，也能表达 block-level 的并行计算和内存访问。

Triton 的核心概念：

- `@triton.jit`：JIT 编译 kernel。
- program instance：类似一个 block/CTA 级任务。
- `tl.program_id(axis)`：拿到当前 program 的编号。
- block pointers / offsets：计算当前 program 负责的数据范围。
- mask：处理边界。
- `tl.load` / `tl.store`：显式加载和写回。
- `num_warps`、`BLOCK_SIZE`：调并行粒度和 tile size。

极简伪代码：

```python
@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(x + offsets, mask=mask)
    b = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, a + b, mask=mask)
```

为什么大模型岗会问 Triton：

- PyTorch eager 里很多小算子会产生多个 kernel。
- 写 fused kernel 可以减少 HBM IO 和 launch overhead。
- Triton 比 CUDA C++ 更易写、更易实验。
- 量化、RMSNorm、rotary embedding、sampling、custom attention 都常有自定义 kernel 需求。

面试表达：

> Triton 不是替代所有 CUDA，而是让模型工程师能更快写出 block-level fused kernel。优化时仍然要理解内存访问、tile、mask、register、occupancy 和 profiler。

## PyTorch 2 / torch.compile / Inductor

现在面试也会问“为什么 PyTorch 2 能加速”。可以这样讲：

- `torch.compile` 捕获图，减少 Python overhead。
- Inductor 做图级优化和算子融合。
- 后端可能生成 Triton/C++ kernel。
- 对 shape 稳定、算子组合固定的模型收益更好。
- 对动态 shape、控制流、频繁 graph break 的代码收益会下降。

常见排查：

- graph break 太多。
- dynamic shape 太多。
- 小 batch 下 kernel launch overhead 仍然明显。
- 自定义 op 或不支持的 Python 逻辑阻断编译。
- 编译时间和运行收益不匹配。

## Profiler 排查链路

不要靠感觉优化 GPU。面试要能讲一套 profiler 链路。

### 先分层

1. 端到端：请求延迟、tokens/s、step time、TTFT/TPOT。
2. CPU：tokenizer、dataloader、Python、JSON、日志、网络。
3. GPU：kernel timeline、SM utilization、memory bandwidth、tensor core utilization。
4. 多卡：NCCL 时间、通信 overlap、拓扑和网络。

### 工具怎么分

| 工具 | 看什么 |
| --- | --- |
| PyTorch Profiler | Python/PyTorch op、CPU/GPU 时间、算子级热点 |
| Nsight Systems | 全局 timeline、CPU-GPU 空洞、kernel launch、NCCL、同步 |
| Nsight Compute | 单个 kernel 的深度指标，如 occupancy、memory throughput、warp stall |
| nvidia-smi / DCGM | 粗粒度 GPU util、显存、功耗、温度 |

### GPU 利用率低怎么排查

```text
GPU util 低
  -> CPU/dataloader/tokenizer 是否喂不动
  -> kernel launch 是否很多小碎片
  -> batch/seq 是否太小
  -> 是否有频繁 CPU-GPU sync
  -> 是否在等 NCCL/网络/IO
  -> 是否 shape 动态导致 compile/cache miss
```

### Kernel 慢怎么排查

```text
单个 kernel 慢
  -> memory-bound 还是 compute-bound
  -> global memory access 是否合并
  -> shared memory bank conflict
  -> register pressure / occupancy
  -> warp divergence
  -> tensor core 是否用上
  -> tile/block size 是否合理
```

## NCCL 和多卡通信

训练和推理多卡都会遇到通信。

常见 collective：

| Collective | 作用 | 大模型场景 |
| --- | --- | --- |
| all-reduce | 多卡求和并广播结果 | DDP 梯度同步、TP 中间结果 |
| all-gather | 收集所有卡的数据 | FSDP 参数聚合、序列/张量并行 |
| reduce-scatter | reduce 后分片 | ZeRO/FSDP 梯度/状态切分 |
| all-to-all | 每卡发不同数据给其他卡 | MoE expert parallel |
| broadcast | 一卡发给多卡 | 初始化参数、同步状态 |

通信慢排查：

- 是否通信占 step time 比例过高。
- NCCL 是否和计算 overlap。
- bucket size 是否合理。
- GPU 拓扑：NVLink、PCIe、跨机网络。
- 网卡/RDMA/IB 是否异常。
- MoE all-to-all 是否被负载不均放大。
- 小 batch / micro-batch 是否导致通信频繁但计算太少。

面试表达：

> 多卡慢不能只看 GPU 利用率，要看 timeline 里计算和 NCCL 是否交错。如果大量时间在等 all-reduce/all-gather/all-to-all，就要从并行策略、bucket、overlap、拓扑和负载均衡排查。

## 大模型里常见 Fused Kernel

| Kernel | 为什么常融合 | 面试点 |
| --- | --- | --- |
| RMSNorm / LayerNorm | reduction + elementwise，memory-bound | 减少读写，控制数值稳定 |
| Bias + GELU / SwiGLU | elementwise 链条 | 减少中间张量 |
| RoPE | 对 Q/K 做旋转 | 可和 Q/K projection 或 attention 前处理融合 |
| Softmax | reduction + exp + normalize | 数值稳定、warp/block reduction |
| Sampling | top-k/top-p/temperature | 输出阶段小算子多，避免 CPU 往返 |
| Quant/Dequant | 低比特转换 | scale、zero point、group size、kernel 支持 |
| Attention | QK、softmax、PV | FlashAttention/Triton custom attention |

## 常见面试误区

1. 把 FlashAttention 说成近似 attention。
2. 把 PagedAttention 和 FlashAttention 混在一起。
3. 说 GPU 利用率低就“加 batch”，但不拆 CPU、launch、通信、shape。
4. 认为 occupancy 越高越好。
5. 认为 kernel fusion 一定更快。
6. 认为量化一定提速。
7. 只看 `nvidia-smi`，不看 profiler timeline。
8. 多卡慢只说“网络不好”，不讲 collective、overlap 和拓扑。

## 项目里怎么讲 GPU 优化

8 分钟讲稿：

1. 背景：训练/推理哪里慢，业务目标是什么。
2. 指标：step time、tokens/s、TTFT/TPOT、GPU util、HBM bandwidth、NCCL time。
3. 定位：用 PyTorch Profiler / Nsight Systems 找热点。
4. 分类：CPU-bound、launch-bound、memory-bound、compute-bound、communication-bound。
5. 方案：fusion、FlashAttention、torch.compile、Triton kernel、batch/shape 调整、并行策略、通信 overlap。
6. 验证：分桶压测，对比 P50/P95、tokens/s、显存、质量。
7. 风险：数值误差、动态 shape、编译时间、kernel 维护成本、硬件兼容。
8. 复盘：如果重做，会先补 profiler 自动化和回归 benchmark。

项目表达示例：

> 我们不是一上来就写 kernel，而是先用 profiler 看 timeline。发现 GPU 中间有很多空洞，小算子 launch 很碎，同时 RMSNorm、RoPE 和 sampling 都偏 memory-bound。第一步用 torch.compile 和已有 fused ops 减少 Python overhead；第二步对固定 shape 的热路径写 Triton fused kernel，减少中间张量写 HBM；第三步对长 prompt 开 FlashAttention，降低 attention IO。最后按 input/output token 分桶压测，确认 tokens/s 提升，同时 P95 和数值误差在可接受范围内。

## 高频快答

### GPU memory-bound 怎么判断？

看 memory bandwidth 接近上限、SM/tensor core 利用不高、算子主要是 elementwise/reduction/load-store，优化方向是减少 HBM 读写、fusion、tiling、量化和更好的内存访问。

### FlashAttention 为什么不是近似 attention？

它没有改 attention 公式，而是用 tiling 和 online softmax 重新组织计算，避免完整 `N x N` 矩阵写回 HBM。

### Triton 适合写什么？

适合写 block-level fused kernel，比如 RMSNorm、softmax、RoPE、sampling、量化/反量化、自定义 attention。复杂极致优化仍可能需要 CUDA/CUTLASS。

### 为什么 GPU 利用率低？

可能是 CPU 喂不动、kernel 太碎、batch 太小、频繁同步、shape 动态导致编译/调度开销、多卡通信等待或 IO 阻塞。

### NCCL 慢怎么说？

先看 timeline 中 collective 占比，再看 all-reduce/all-gather/all-to-all 类型、overlap、bucket、拓扑、网络和负载均衡。

## 面试背诵版

GPU 优化要先分瓶颈：compute-bound 看 tensor core 和 FLOPs，memory-bound 看 HBM 读写和算术强度，launch/CPU-bound 看小 kernel 和 Python/CPU 喂数，多卡 communication-bound 看 NCCL collective、overlap 和拓扑。FlashAttention 是 IO-aware exact attention，通过 tiling 和 online softmax 避免把完整 `N x N` attention matrix 写回 HBM；PagedAttention 则是 serving 中 KV Cache 的 block/page 管理，两者不是一回事。Triton 是 Python-like GPU kernel DSL，适合快速写 fused kernel，但仍要理解 tile、mask、coalesced access、register、occupancy 和 profiler。优化时不要只看 `nvidia-smi`，要用 PyTorch Profiler、Nsight Systems 和 Nsight Compute 分层定位，再验证 tokens/s、P95、显存和数值正确性。

## 延伸阅读

- NVIDIA CUDA C++ Programming Guide：[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- NVIDIA CUDA C++ Best Practices Guide：[https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Triton 官方文档：[https://triton-lang.org/main/](https://triton-lang.org/main/)
- Triton tutorials：[https://triton-lang.org/main/getting-started/tutorials/](https://triton-lang.org/main/getting-started/tutorials/)
- FlashAttention 论文：[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- FlashAttention-2 论文：[https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- FlashAttention GitHub：[https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- PyTorch Profiler 文档：[https://pytorch.org/docs/stable/profiler.html](https://pytorch.org/docs/stable/profiler.html)
- NVIDIA Nsight Systems User Guide：[https://docs.nvidia.com/nsight-systems/UserGuide/](https://docs.nvidia.com/nsight-systems/UserGuide/)
- NVIDIA Nsight Compute Documentation：[https://docs.nvidia.com/nsight-compute/](https://docs.nvidia.com/nsight-compute/)
- NVIDIA NCCL User Guide：[https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
