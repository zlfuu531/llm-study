# 模型量化、低比特推理、GPTQ/AWQ/SmoothQuant 面试

这一章面向大模型算法、推理部署、AI Infra、端侧 AI、模型压缩和微调工程岗位。量化题看起来像“省显存技巧”，但面试真正想看的是：你是否能把低比特表示、误差来源、校准数据、kernel 支持、KV Cache、评估和线上排查讲成一条完整链路。

如果时间很紧，先背这句：

> 量化是用更低精度表示权重、激活或 KV Cache，主要收益来自减少显存占用和内存带宽压力；但是否提速取决于量化粒度、反量化开销、硬件指令、kernel、batch/seq shape 和质量约束。GPTQ/AWQ 多用于 weight-only 低比特推理，SmoothQuant 面向 W8A8，QLoRA 用 4-bit 冻结基座加 LoRA 训练，FP8 更依赖硬件和混合精度生态。

相关答案版：[answers/32_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant_答案版.md](answers/32_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant_答案版.md)

相邻章节：

- [14_端侧小模型与模型压缩面试.md](14_端侧小模型与模型压缩面试.md)：蒸馏、剪枝、GGUF、本地部署和端侧评估。
- [24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md](24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md)：vLLM、TensorRT-LLM、KV Cache、batching、TTFT/TPOT。
- [25_GPU_CUDA_Triton与FlashAttention面试.md](25_GPU_CUDA_Triton与FlashAttention面试.md)：GPU 内存带宽、Tensor Core、kernel fusion、Profiler。
- [34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md](34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md)：bitsandbytes、QLoRA、PEFT、加载量化模型。
- [17_大模型训练系统与分布式训练面试.md](17_大模型训练系统与分布式训练面试.md)：FP16/BF16/FP8、训练显存、分布式训练和混合精度。

## 1. 为什么大模型需要量化

LLM 的推理成本主要卡在两类资源：

- 权重显存：模型参数越大，加载需要的显存越多。
- 内存带宽：decode 阶段每生成一个 token，很多层都要读取权重，batch 小时常常是 memory-bound。
- KV Cache：长上下文和高并发时，历史 K/V 占用会随 batch、层数、上下文长度线性增长。
- 端侧约束：手机、PC、边缘设备有内存、功耗、发热和离线隐私要求。
- 成本约束：同一张卡能放更大模型或更多请求，直接影响线上成本。

参数显存粗估：

```text
P = 参数量
FP32 权重 ≈ 4P bytes
FP16/BF16 权重 ≈ 2P bytes
INT8 权重 ≈ 1P bytes
INT4 权重 ≈ 0.5P bytes
```

例如 7B 模型只看权重：

```text
FP16: 7e9 * 2 bytes ≈ 14 GB
INT8: 7e9 * 1 byte ≈ 7 GB
INT4: 7e9 * 0.5 byte ≈ 3.5 GB
```

但真实占用还要加：

- scale / zero point / group metadata。
- embedding、lm head、norm 等可能不量化或保留高精度。
- runtime buffer、activation、CUDA graph、KV Cache。
- LoRA adapter、tokenizer、服务框架额外内存。

面试别说“INT4 就是 FP16 的四分之一显存”后停住，要补一句：这是权重主体的理论值，实际还要看量化粒度和运行时开销。

## 2. 量化基本公式

量化的目标是把连续值 `x` 映射到有限整数集合 `q`，推理时再近似还原为 `x_hat`。

### 对称量化

对称量化把 0 对齐到整数 0：

```text
q = clip(round(x / s), q_min, q_max)
x_hat = s * q
```

`b` bit 有符号整数时，常见范围近似为：

```text
q_min = -2^(b-1)
q_max =  2^(b-1) - 1
s ≈ max(|x|) / q_max
```

直觉：用一个 scale 把浮点数缩放到整数网格。对称量化简单，硬件友好，适合权重分布较对称的情况。

### 非对称量化

非对称量化引入 zero point：

```text
q = clip(round(x / s + z), q_min, q_max)
x_hat = s * (q - z)
```

常见 scale / zero point：

```text
s = (x_max - x_min) / (q_max - q_min)
z = round(q_min - x_min / s)
```

直觉：非对称量化能更好覆盖非零中心分布，但多了 zero point，kernel 实现和计算可能更复杂。

### 量化误差

误差来自：

```text
e = x - x_hat
```

影响误差的因素：

- bit 数越低，网格越粗。
- outlier 越大，scale 被撑大，普通值的分辨率越低。
- 粒度越粗，多个通道共享同一个 scale，误差越大。
- 激活分布随输入变化，通常比权重更难量化。

## 3. PTQ 和 QAT 怎么选

PTQ，Post-Training Quantization，训练后量化：

- 不重新训练或只用少量校准数据。
- 成本低，适合已有模型快速部署。
- GPTQ、AWQ、SmoothQuant 常作为 PTQ 方案讨论。
- 风险是低比特下质量可能下降，校准集不匹配会更明显。

QAT，Quantization-Aware Training，量化感知训练：

- 训练或微调时模拟量化误差。
- 更可能保住低比特质量。
- 成本高，需要训练数据、算力和训练稳定性经验。
- 端侧极低比特、质量要求高、已有 PTQ 不够时更常考虑。

面试决策：

```text
已有大模型快速上线 -> 先 PTQ
INT8/W8A8 或 INT4 weight-only 能满足质量 -> 不急着 QAT
质量明显下降且业务收益足够 -> 再考虑 QAT / LoRA repair / 蒸馏
```

## 4. Weight-only、W8A8、KV Cache 量化

### Weight-only

只量化权重，激活仍用 FP16/BF16 或其他高精度：

- 常见于 GPTQ、AWQ、GGUF Q4/Q5/Q8。
- 显著降低模型权重显存。
- batch 小、decode memory-bound 时可能提速。
- 每次矩阵乘前或乘中需要反量化或专用 kernel。
- 激活和 KV Cache 不一定省。

### W8A8

权重和激活都用 8-bit：

- 目标是让 GEMM 真正跑低精度。
- 对硬件、kernel 和激活 outlier 处理要求更高。
- SmoothQuant 常被问作 W8A8 的代表思路。

### KV Cache 量化

量化推理阶段缓存的 key/value：

```text
KV cache memory ≈ batch * seq_len * layers * kv_heads * head_dim * 2(K,V) * bytes
```

收益：

- 长上下文和高并发时显存下降明显。
- 能提高可承载 token 数或并发。
- 对 decode 吞吐可能有帮助。

风险：

- 注意力分数受 K 精度影响，V 精度影响上下文聚合。
- 长上下文、数学、代码、结构化输出可能更敏感。
- 需要分桶评估，不要只看短问答。

## 5. per-tensor、per-channel、group-wise

量化粒度决定多少元素共享一个 scale。

| 粒度 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| per-tensor | 整个张量一个 scale | metadata 少、实现简单 | outlier 会伤害所有值 |
| per-channel | 每个输出/输入通道一个 scale | 更准，适合通道分布差异大 | scale 更多，kernel 更复杂 |
| group-wise | 每组若干元素一个 scale | LLM 低比特常用折中 | group size 需要调，metadata 增加 |

group size 取舍：

- group 越小：误差通常越低，scale 越多，metadata 和访存开销越大。
- group 越大：metadata 少，速度可能更好，但质量风险更大。

面试句：

> LLM 4-bit weight-only 常用 group-wise，因为每个通道/矩阵块的分布差异很大，per-tensor 太粗，per-channel 又可能带来更多元数据和 kernel 复杂度。

## 6. LLM 里的 outlier 为什么重要

量化最怕 outlier。假设某一组值大多数在 `[-1, 1]`，但有一个值是 `20`，如果用同一个 scale：

```text
s ≈ 20 / 7    # 以 4-bit signed 粗略看 q_max=7
```

普通值会被映射到很少的整数档位，分辨率很差。LLM 激活中常有明显 outlier channel，尤其是某些层、某些 token、某些通道会异常大。

解决思路大概分三类：

- 更细粒度：per-channel / group-wise，减少 outlier 影响范围。
- 保护敏感通道：AWQ 根据激活统计保护重要权重通道。
- 平滑激活 outlier：SmoothQuant 把激活难量化的问题迁移到权重上。

## 7. GPTQ

GPTQ 是常见 weight-only PTQ 方法，核心是：逐层量化权重，并用近似二阶信息补偿量化误差。

直觉版：

```text
普通量化：把 W 直接 round 到低比特。
GPTQ：量化一部分权重后，估计它对层输出的影响，再调整剩余权重来补偿误差。
```

为什么说二阶？

对线性层 `Y = XW`，量化权重 `W -> W_hat` 后，希望层输出误差小：

```text
min || XW - XW_hat ||^2
```

这里输入激活 `X` 的相关性会影响不同权重量化误差对输出的影响。GPTQ 用近似 Hessian / inverse Hessian 信息做误差补偿。

优点：

- 适合 3/4-bit weight-only。
- 常见预量化模型多，生态成熟。
- 在 GPU 推理里常能显著省权重显存。

局限：

- 需要校准数据。
- 校准数据不匹配可能损失质量。
- 量化过程相对复杂。
- 是否提速依赖 kernel，如 Marlin / exllama / vLLM 支持等。

一句话答案：

> GPTQ 不是简单 round，而是用校准激活估计权重量化对输出的影响，用近似二阶误差补偿逐步量化权重，目标是在 3/4-bit 下尽量保住层输出。

## 8. AWQ

AWQ，Activation-aware Weight Quantization，也是 weight-only PTQ。它的核心发现是：不是所有权重同样重要，少量与大激活相关的 salient channels 对输出影响很大。

直觉版：

```text
哪些权重重要，不只看权重大小，还要看它们乘到的激活有多大。
保护这些敏感通道，整体 4-bit 质量会好很多。
```

AWQ 的做法可以概括为：

- 用校准数据收集激活统计。
- 找到对输出更敏感的通道。
- 通过等价缩放保护这些通道，降低量化误差。
- 仍保持硬件友好的 weight-only 低比特格式。

优点：

- 4-bit 场景常见。
- 对 instruction-tuned、代码、数学、多模态模型更友好这一点经常被提到。
- 不依赖反向传播或重构，泛化性通常较好。

局限：

- 仍需要校准数据。
- 不是所有 runtime 都同等优化。
- 质量和速度依赖模型结构、group size、kernel、硬件。

一句话答案：

> AWQ 用激活统计识别重要权重通道，通过保护少量 salient channels 来减少低比特 weight-only 量化误差，重点是 activation-aware，而不是只看权重本身。

## 9. SmoothQuant

SmoothQuant 解决的是 W8A8 里的激活 outlier 问题。

问题：

- 权重通常相对静态，比较容易离线量化。
- 激活随输入变化，outlier 明显，直接 INT8 激活量化容易掉质量。

SmoothQuant 的核心思想：

```text
Y = XW
  = (X / s) * (sW)
```

更准确地说，对通道做等价缩放，把激活上的量化困难迁移到权重上。因为权重离线可处理，激活变平滑后更容易 W8A8。

`alpha` 控制迁移程度：

```text
alpha 越大：更多难度从 activation 迁移到 weight
alpha 太大：权重量化压力变大
alpha 太小：激活 outlier 仍然明显
```

优点：

- training-free PTQ 思路。
- 面向 W8A8，更接近硬件高效 GEMM。
- 对大模型服务端 INT8 推理很重要。

局限：

- 需要校准激活统计。
- alpha、粒度、kernel 支持会影响质量和速度。
- 不是 INT4 weight-only 的同一类问题。

一句话答案：

> SmoothQuant 通过等价缩放把激活 outlier 的量化难度转移到权重，让激活更平滑，从而支持 W8A8 的高效 INT8 推理。

## 10. GPTQ、AWQ、SmoothQuant 对比

| 方法 | 典型目标 | 关键思想 | 常见场景 |
| --- | --- | --- | --- |
| GPTQ | 3/4-bit weight-only | 近似二阶误差补偿 | GPU 低显存推理 |
| AWQ | 4-bit weight-only | 激活感知，保护敏感通道 | 指令模型、代码/数学、多模态、端侧 |
| SmoothQuant | W8A8 | 平滑激活 outlier，把难度迁移到权重 | 服务端 INT8 高效 GEMM |

面试不要说“谁绝对更好”。更稳的说法：

> GPTQ 和 AWQ 常用于低比特 weight-only，重点省权重显存；SmoothQuant 面向权重和激活都 8-bit 的推理，重点解决激活 outlier。选哪个要看模型、硬件、runtime、质量指标和校准数据。

## 11. bitsandbytes、NF4 和 QLoRA

bitsandbytes 在 Hugging Face 生态里常用于：

- 8-bit / 4-bit 加载模型。
- QLoRA 微调。
- 低显存实验和 adapter 训练。

QLoRA 的典型结构：

```text
base model: 4-bit quantized, frozen
LoRA adapter: trainable, higher precision
gradient: backprop through quantized base into LoRA params
```

三个关键词：

- NF4：NormalFloat 4-bit，针对近似正态分布权重设计。
- double quantization：再量化量化常数，降低 scale/metadata 开销。
- paged optimizer：缓解训练时 optimizer state 的显存峰值。

常见误区：

- QLoRA 不是“直接训练 INT4 全参数模型”。
- 量化基座通常冻结，训练的是 LoRA adapter。
- compute dtype 常用 FP16/BF16，4-bit 更多是存储和加载层面的省显存。
- adapter merge 后是否再量化、如何导出，要结合部署 runtime。

一句话答案：

> QLoRA 用 4-bit 量化冻结基座模型节省显存，把梯度传到 LoRA adapter 上训练；NF4 适合权重分布，double quant 降低量化常数开销，paged optimizer 缓解显存峰值。

## 12. FP8 和 INT8/INT4 有什么区别

INT8/INT4 是整数低比特，通常需要 scale / zero point 来映射浮点范围。FP8 是 8-bit 浮点格式，保留指数和尾数，动态范围表达方式不同。

常见 FP8 格式：

- E4M3：指数 4 bit，尾数 3 bit，精度更高，动态范围较小。
- E5M2：指数 5 bit，尾数 2 bit，动态范围更大，精度较低。

面试要点：

- FP8 常和 Hopper / Ada / Blackwell 等硬件、Transformer Engine、Tensor Core、混合精度训练/推理绑定。
- FP8 不是“万能比 INT8 好”，它的收益依赖硬件、scaling recipe、kernel 和框架支持。
- 训练里 FP8 要关注 amax、scaling、延迟缩放、溢出和梯度稳定性。
- 推理里 FP8 可用于权重、激活或 KV Cache，但格式和硬件支持要具体看 runtime。

一句话答案：

> INT8/INT4 更像用整数网格近似浮点，FP8 仍是浮点格式，有指数和尾数；FP8 的优势来自硬件 Tensor Core 和混合精度生态，但必须配合 scaling 和框架支持。

## 13. KV Cache 量化怎么讲

KV Cache 是 decode 阶段的显存大头之一。对普通 MHA：

```text
KV cache ≈ B * S * L * H_kv * D * 2 * bytes
```

其中：

- `B`：batch 或并发序列数。
- `S`：缓存 token 数。
- `L`：层数。
- `H_kv`：KV head 数，GQA/MQA 会降低它。
- `D`：head_dim。
- `2`：K 和 V。

量化后如果从 FP16 变 FP8，理论上 bytes 从 2 降到 1，KV Cache 主体约减半。实际还要看 scale、block、page 管理和 runtime buffer。

质量风险：

- K 影响 attention score，误差可能改变关注位置。
- V 影响取回内容，误差可能影响答案细节。
- 长上下文比短上下文更敏感。
- 代码、数学、表格、JSON、检索问答可能比闲聊更敏感。

评估方法：

- 不只看 PPL。
- 分桶看短问答、长上下文、RAG、数学、代码、结构化输出。
- 看 TTFT、TPOT、tokens/s、max concurrency、OOM 率。
- 看 P95/P99，而不是只看平均。

## 14. GGUF / llama.cpp

GGUF 常和 llama.cpp 本地推理、CPU/端侧/PC 部署一起出现。

你可以这样讲：

> GGUF 更像模型文件和张量元数据格式，llama.cpp 提供转换、量化和推理运行时。Q4/Q5/Q8 等表示不同量化方案，适合本地部署和端侧试验，但质量、速度和内存取决于量化类型、CPU/GPU 后端、上下文长度、线程数和模型结构。

常见用途：

- 本地问答、离线知识库、隐私场景。
- PC/CPU/Mac/移动端推理。
- 快速比较 Q4、Q5、Q8 的质量和速度。

面试提醒：

- GGUF 不是一种算法本身，它包含格式和量化张量信息。
- Q4 不等于一定比 Q8 划算，要看质量和延迟。
- CPU 上瓶颈常是内存带宽、线程调度和 cache locality。
- 端侧评估要看功耗、发热、首 token 延迟、连续对话稳定性。

## 15. 为什么 INT4 不一定比 FP16 快

这是量化面试最常见高压追问。

INT4 省显存，但提速不一定，因为：

- 反量化开销：低比特权重要解包、乘 scale，可能抵消收益。
- kernel 支持：没有高效 INT4 GEMM / dequant-fused kernel 时，速度可能不理想。
- 访存 vs 计算：memory-bound 时省带宽收益大，compute-bound 时收益小。
- batch shape：小 batch decode 和大 batch prefill 的瓶颈不同。
- 硬件指令：不同 GPU 代际、CPU、NPU 对 INT4/FP8 支持差异大。
- group metadata：group-wise scale/zero point 也要读取。
- 非量化部分：embedding、norm、attention、KV Cache、采样后处理仍有开销。
- 质量补偿：如果要多采样、rerank、fallback，端到端成本可能上升。

面试句：

> 量化先保证省显存，不天然保证提速。真正提速要看低比特矩阵乘是不是被高效 kernel 吃掉，以及端到端瓶颈是不是权重带宽。

## 16. 量化后怎么评估

离线评估：

- PPL / CE：看语言建模整体退化。
- 通用 benchmark：MMLU、GSM8K、HumanEval 等按岗位选择。
- 私有 eval：业务任务最重要。
- 格式有效率：JSON、工具调用、代码 patch 是否可执行。
- 长上下文：needle、RAG、多文档问答、lost-in-the-middle。
- 安全：拒答、越权、敏感信息泄露。

系统评估：

- 权重显存、峰值显存、KV Cache 占用。
- TTFT、TPOT、tokens/s。
- max batch / max concurrency。
- P50/P95/P99 延迟。
- OOM 率、超时率、fallback 率。
- cost per request / cost per solved task。

ablation 建议：

```text
FP16 baseline
-> INT8 / W8A8
-> INT4 GPTQ
-> INT4 AWQ
-> KV cache FP8
-> 不同 group size / calibration data
```

不要只报“显存省了 50%”。更像项目的说法：

> 我们以 FP16 为 baseline，比较 AWQ/GPTQ/FP8 KV Cache 在私有 eval、PPL、结构化输出有效率、P95 TPOT、峰值显存和成本上的变化，最后选择质量下降可控且吞吐收益最大的配置。

## 17. 量化后质量下降怎么排查

按这个顺序查：

- 校准数据是否匹配：是否覆盖真实 prompt、语言、领域、长度、格式。
- 量化粒度是否过粗：group size 太大、per-tensor outlier 太强。
- 哪些模块被量化：lm head、embedding、attention output、MoE router、norm 是否敏感。
- 激活 outlier 是否处理：W8A8 场景考虑 SmoothQuant / per-channel scaling。
- KV Cache 是否量化：长上下文退化先关掉 KV quant 对比。
- dtype 是否混乱：compute dtype、load dtype、adapter dtype 是否一致。
- tokenizer/chat template 是否变了：很多“量化退化”其实是输入协议变了。
- decoding config 是否变了：temperature、top_p、max_new_tokens、stop token 必须一致。
- 是否需要混合精度保留：对敏感层保持 FP16/BF16。
- 是否用 LoRA repair / 蒸馏 / QAT：PTQ 不够再考虑训练补偿。

排查口诀：

```text
先固定模型输入和解码配置
再对比 FP16 baseline
再逐个开关 weight quant / activation quant / KV quant
最后按任务分桶看退化来源
```

## 18. 项目 8 分钟讲法

如果简历里有量化或低成本部署项目，可以按这条线讲：

```text
背景：
  线上模型 FP16 显存高，单卡并发低，成本高。

约束：
  质量下降不超过 X%，P95 TPOT 降到 Y ms 内，支持 Z 并发。

方案：
  比较 FP16、INT8、GPTQ-4bit、AWQ-4bit、FP8 KV Cache。
  校准数据来自真实业务 prompt，覆盖长文本、代码、结构化输出和边界样本。

评估：
  私有 eval + PPL + 格式有效率 + RAG faithfulness + 延迟/吞吐/显存。

结果：
  选择 AWQ-4bit + FP8 KV Cache 或 INT8 W8A8，显存下降，吞吐提升，质量可控。

上线：
  灰度、监控、fallback、badcase 回流、模型和量化配置版本化。

复盘：
  哪些任务最敏感，哪些层保留高精度，哪些指标不能只看平均。
```

如果没做过真实项目，也可以讲“你会怎么做”，但要把评估和回滚说清楚。

## 19. 高频追问快答

**Q1：量化为什么能省显存？**  
因为用更少 bit 存权重/激活/KV Cache。FP16 每参数 2 bytes，INT4 主体每参数 0.5 bytes，但实际要加 scale、zero point、metadata 和 runtime buffer。

**Q2：PTQ 和 QAT 区别？**  
PTQ 训练后量化，快但低比特质量有风险；QAT 训练时模拟量化误差，质量可能更好但成本高。

**Q3：GPTQ 和 AWQ 最大区别？**  
GPTQ 强调近似二阶误差补偿；AWQ 强调用激活统计保护敏感权重通道。两者多是 weight-only 低比特。

**Q4：SmoothQuant 解决什么？**  
解决激活 outlier 导致 W8A8 难量化的问题，用等价缩放把难度从 activation 迁移到 weight。

**Q5：KV Cache 量化和权重量化一样吗？**  
不一样。GPTQ/AWQ 主要量化权重；KV Cache 量化是推理时缓存 K/V，收益随上下文和并发增长，质量风险集中在长上下文和精细推理。

**Q6：量化后为什么数学/代码变差？**  
数学/代码对小概率 token、长依赖和精确格式更敏感。需要分桶 eval，检查 calibration、group size、敏感层、KV quant 和 decoding config。

**Q7：什么时候选 FP8？**  
当硬件、框架和 kernel 支持 FP8，并且想在训练或推理中利用混合精度加速。它不是纯模型文件压缩技巧，强依赖生态。

**Q8：GGUF 适合什么？**  
适合 llama.cpp 本地/端侧/CPU/PC 推理。看重离线、隐私、低成本和易部署，但要实际测质量、速度、功耗和上下文长度。

## 20. 面试前背诵版

模型量化的核心是用低精度表示权重、激活或 KV Cache，降低显存和带宽压力。基本公式是 `q=round(x/s)`、`x_hat=s*q`，非对称量化再加 zero point。LLM 低比特难点在 outlier、校准数据和 kernel 支持。GPTQ 是 weight-only PTQ，用近似二阶信息补偿权重量化误差；AWQ 用激活统计识别并保护敏感通道；SmoothQuant 通过等价缩放把激活 outlier 的量化难度迁移到权重，支持 W8A8。QLoRA 是 4-bit 冻结基座加 LoRA adapter 训练，关键词是 NF4、double quant 和 paged optimizer。FP8 是浮点低精度，依赖硬件、scaling 和混合精度框架。INT4 不一定比 FP16 快，因为反量化、metadata、kernel、硬件和端到端瓶颈都会影响速度。量化方案必须用 FP16 baseline 对照，分桶评估质量、长上下文、代码/数学、结构化输出、TTFT/TPOT、显存、并发和成本。

## 参考资料

- Hugging Face Transformers Quantization：[https://huggingface.co/docs/transformers/en/main_classes/quantization](https://huggingface.co/docs/transformers/en/main_classes/quantization)
- Hugging Face GPTQ：[https://huggingface.co/docs/transformers/en/quantization/gptq](https://huggingface.co/docs/transformers/en/quantization/gptq)
- Hugging Face AWQ：[https://huggingface.co/docs/transformers/en/quantization/awq](https://huggingface.co/docs/transformers/en/quantization/awq)
- vLLM Quantization：[https://docs.vllm.ai/en/latest/features/quantization/](https://docs.vllm.ai/en/latest/features/quantization/)
- GPTQ 论文：[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- AWQ 论文：[https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- SmoothQuant 论文：[https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)
- QLoRA 论文：[https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- NVIDIA Transformer Engine FP8 Primer：[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- llama.cpp quantize README：[https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
