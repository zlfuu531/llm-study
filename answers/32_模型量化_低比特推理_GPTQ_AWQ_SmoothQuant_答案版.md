# 答案版 32：模型量化、低比特推理、GPTQ/AWQ/SmoothQuant

对应题目：`03_高频题单100题.md` 的 661-680。

用法：先把每题 30 秒版背顺，再用 2 分钟版补公式、工程取舍和项目排查。量化题最怕答成“INT4 省显存”，要主动讲误差、校准、粒度、kernel、KV Cache、评估和回滚。

## 661. 模型量化为什么能省显存和带宽？

30 秒版：

量化用更少 bit 表示权重、激活或 KV Cache。FP16 权重每参数 2 bytes，INT8 约 1 byte，INT4 主体约 0.5 byte，所以能降低模型加载显存和权重读取带宽。

2 分钟版：

粗估公式：

```text
FP16/BF16 权重 ≈ 2P bytes
INT8 权重 ≈ 1P bytes
INT4 权重 ≈ 0.5P bytes
```

但真实部署要加 scale、zero point、group metadata、runtime buffer、KV Cache 和未量化模块。量化能不能提速还要看端到端瓶颈：decode 小 batch 常受权重带宽影响，低比特更可能有效；prefill 大 batch 可能更偏计算和 attention，收益不一定同等明显。

## 662. 对称量化和非对称量化公式是什么？

30 秒版：

对称量化是 `q=clip(round(x/s))`，还原 `x_hat=s*q`；非对称量化多一个 zero point，`q=clip(round(x/s+z))`，还原 `x_hat=s*(q-z)`。

2 分钟版：

对称量化：

```text
q = clip(round(x / s), q_min, q_max)
x_hat = s * q
s ≈ max(|x|) / (2^(b-1)-1)
```

非对称量化：

```text
s = (x_max - x_min) / (q_max - q_min)
z = round(q_min - x_min / s)
q = clip(round(x / s + z), q_min, q_max)
x_hat = s * (q - z)
```

对称简单硬件友好；非对称能更好覆盖非零中心分布，但多了 zero point 和实现复杂度。

## 663. PTQ 和 QAT 怎么选？

30 秒版：

PTQ 是训练后量化，成本低、上线快；QAT 是训练时模拟量化误差，质量可能更好但成本高。已有大模型部署一般先试 PTQ，不够再考虑 QAT、LoRA repair 或蒸馏。

2 分钟版：

PTQ 适合已有 FP16/BF16 模型快速压缩，用少量校准数据估计 scale 或敏感通道。GPTQ、AWQ、SmoothQuant 常按 PTQ 思路讲。QAT 适合极低比特、端侧强约束或 PTQ 掉点严重场景，但要训练数据、算力和稳定性经验。面试决策句：先用 PTQ 做 baseline 和 ablation，只有业务收益覆盖训练成本时才做 QAT。

## 664. Weight-only、W8A8、KV Cache 量化有什么区别？

30 秒版：

Weight-only 只量化权重，GPTQ/AWQ 常见；W8A8 同时量化权重和激活，SmoothQuant 常见；KV Cache 量化量的是推理缓存 K/V，主要解决长上下文和高并发显存。

2 分钟版：

Weight-only 主要省模型权重显存，激活和 KV Cache 仍可能是 FP16/BF16。W8A8 希望 GEMM 真正跑 INT8 权重和 INT8 激活，但要处理激活 outlier 和 kernel 支持。KV Cache 量化收益随 `batch * seq_len * layers * kv_heads * head_dim` 增长，长上下文场景尤其重要，但可能影响注意力位置和上下文细节。

## 665. per-tensor、per-channel、group-wise 量化怎么取舍？

30 秒版：

per-tensor 最简单但误差大；per-channel 更准但 scale 更多；group-wise 是 LLM 低比特常用折中，按一组权重共享 scale。

2 分钟版：

outlier 会把共享 scale 撑大，所以粒度越粗，普通值分辨率越差。group size 越小，误差通常越低，但 scale/metadata 更多，kernel 和访存开销更高；group size 越大，元数据少但质量风险增加。LLM INT4 常用 group-wise，是精度、元数据和硬件效率之间的折中。

## 666. LLM 量化里 outlier 为什么重要？

30 秒版：

outlier 会决定 scale，导致大量普通值被压到很少的整数档位，量化误差变大。LLM 激活 outlier 尤其明显，所以 W8A8 比单纯权重量化更难。

2 分钟版：

如果一组值大多在 `[-1,1]`，但有一个值是 `20`，scale 会被 20 撑大，普通值的低比特分辨率很差。解决思路包括更细粒度量化、保护敏感通道、SmoothQuant 平滑激活 outlier、敏感层保留高精度。面试要说：outlier 不是小瑕疵，它直接决定低比特质量。

## 667. GPTQ 的核心思想是什么？

30 秒版：

GPTQ 是 weight-only PTQ，用校准激活和近似二阶信息估计权重量化对层输出的影响，在逐步量化时补偿误差，常用于 3/4-bit 推理。

2 分钟版：

对线性层 `Y=XW`，量化后希望 `XW_hat` 尽量接近 `XW`。GPTQ 不是简单 round 权重，而是用输入激活近似 Hessian 信息，量化一部分权重后调整剩余权重来补偿输出误差。优点是低比特精度较好、生态成熟；局限是需要校准数据、量化过程复杂、是否提速依赖 kernel。

## 668. AWQ 的核心思想是什么？

30 秒版：

AWQ 是 activation-aware weight quantization，用校准激活找出重要权重通道，通过保护少量 salient channels 降低 4-bit weight-only 量化误差。

2 分钟版：

AWQ 的关键是“重要性不只看权重大小，还要看激活”。如果某个通道激活大，对输出影响大，量化误差也更致命。AWQ 用激活统计做等价缩放，保护敏感通道，同时保持硬件友好的低比特权重格式。它常用于指令模型、代码/数学、多模态和端侧部署讨论。

## 669. SmoothQuant 的核心思想是什么？

30 秒版：

SmoothQuant 通过等价缩放把激活 outlier 的量化难度迁移到权重，让激活更平滑，从而支持 W8A8 INT8 推理。

2 分钟版：

对线性层：

```text
Y = XW = (X / s) * (sW)
```

激活难在线上动态量化，权重可以离线处理。SmoothQuant 用通道缩放减少激活 outlier，同时把难度转给权重。`alpha` 控制迁移比例，太小激活仍难量化，太大权重量化压力变大。

## 670. GPTQ、AWQ、SmoothQuant 怎么对比？

30 秒版：

GPTQ 和 AWQ 都常用于低比特 weight-only；GPTQ 强调近似二阶误差补偿，AWQ 强调激活感知保护敏感通道；SmoothQuant 面向 W8A8，重点处理激活 outlier。

2 分钟版：

对比表：

| 方法 | 目标 | 关键词 |
| --- | --- | --- |
| GPTQ | 3/4-bit weight-only | Hessian、误差补偿、校准 |
| AWQ | 4-bit weight-only | activation-aware、salient channels |
| SmoothQuant | W8A8 | activation outlier、等价缩放 |

不要说谁绝对最好。正确选择取决于模型、硬件、runtime、校准数据、质量目标和延迟成本。

## 671. bitsandbytes / QLoRA 的 NF4、double quant、compute dtype 怎么讲？

30 秒版：

QLoRA 用 4-bit 量化冻结基座模型，把梯度传到 LoRA adapter 训练。NF4 是适合近似正态权重的 4-bit 格式，double quant 再量化量化常数，compute dtype 是实际计算用的 FP16/BF16 等精度。

2 分钟版：

QLoRA 不是训练全量 INT4 参数，而是 base model 4-bit 存储、冻结，LoRA A/B 矩阵训练。double quant 降低 scale 等量化常数的额外开销，paged optimizer 缓解显存峰值。compute dtype 很关键，因为低比特权重通常会在计算中反量化或混合精度计算，质量和速度不只由 storage bit 决定。

## 672. FP8 和 INT8/INT4 有什么区别？

30 秒版：

INT8/INT4 是整数低比特，靠 scale/zero point 表示浮点范围；FP8 仍是浮点，有指数和尾数，如 E4M3、E5M2，更依赖硬件 Tensor Core 和混合精度框架。

2 分钟版：

E4M3 精度更高但动态范围较小，E5M2 动态范围更大但尾数更少。FP8 常用于训练或推理混合精度，需要 scaling、amax 管理和框架支持。INT4 更常见于 weight-only 模型压缩。面试要强调：FP8 不是模型文件变小这么简单，它是硬件和软件栈配合的低精度计算方案。

## 673. KV Cache 量化有什么收益和风险？

30 秒版：

收益是长上下文和高并发时 KV Cache 显存下降，能支持更多 token 或并发；风险是 K/V 误差会影响注意力和上下文聚合，长上下文、代码、数学、结构化输出更敏感。

2 分钟版：

公式：

```text
KV cache ≈ B * S * L * H_kv * D * 2 * bytes
```

从 FP16 到 FP8，主体 bytes 大约从 2 到 1。评估不能只看短问答，要看长上下文 needle、RAG、多文档、数学、代码、JSON、工具调用，再结合 TTFT、TPOT、tokens/s、P95/P99、OOM 率。

## 674. GGUF / llama.cpp 适合什么场景？

30 秒版：

GGUF / llama.cpp 适合本地、端侧、CPU/PC、隐私离线和低成本部署。Q4/Q5/Q8 等量化格式能压缩模型，但质量和速度要实际测。

2 分钟版：

GGUF 更像模型文件和张量元数据格式，llama.cpp 提供转换、量化和推理 runtime。它适合个人 PC、本地知识库、边缘设备和端侧 demo。面试别把 GGUF 说成单一算法；要补充线程数、CPU/GPU 后端、上下文长度、量化类型、功耗和发热都会影响最终体验。

## 675. 为什么 INT4 不一定比 FP16 快？

30 秒版：

INT4 省显存，但可能有解包、反量化、scale 读取和 kernel 开销。如果硬件或 runtime 没有高效低比特 kernel，或者瓶颈不在权重带宽，INT4 不一定更快。

2 分钟版：

影响因素包括：dequant 是否 fusion、group metadata 访存、Tensor Core 是否支持、batch/seq shape、prefill/decode 瓶颈差异、非量化模块开销、KV Cache 是否仍是瓶颈。端到端还要考虑质量下降后是否需要多采样、rerank 或 fallback。面试结论：量化先保证省显存，提速要靠硬件和 kernel 把低比特优势兑现。

## 676. 量化后质量下降怎么评估？

30 秒版：

用 FP16 baseline 对照，分桶看 PPL、私有 eval、数学、代码、长上下文、RAG、结构化输出和安全，再看延迟、吞吐、显存、P95/P99 和成本。

2 分钟版：

评估矩阵：

```text
质量：PPL / benchmark / 私有 eval / 格式有效率 / badcase
系统：显存 / TTFT / TPOT / tokens/s / 并发 / OOM / 成本
分桶：短问答 / 长上下文 / 代码 / 数学 / RAG / JSON / 工具调用
```

不要只报平均分。量化常常是某些桶明显退化，比如长上下文、代码格式或少数领域术语。

## 677. 量化后数学/代码/长上下文变差怎么排查？

30 秒版：

先固定 tokenizer、chat template 和解码参数，对比 FP16 baseline；再查校准数据、group size、敏感层、KV Cache 量化、compute dtype 和是否需要混合精度保留。

2 分钟版：

排查顺序：

```text
固定输入协议和 generation config
-> 关掉 KV cache quant 对比
-> 换真实校准数据
-> 调 group size / bitwidth
-> 保留 lm head / embedding / router / attention output 高精度
-> 分桶看数学、代码、长文、JSON
-> 必要时 LoRA repair / 蒸馏 / QAT
```

很多退化不是量化本身，而是模型版本、模板、stop token 或采样参数变了。

## 678. 量化部署项目 8 分钟怎么讲？

30 秒版：

按“背景约束 -> baseline -> 量化方案对比 -> 校准数据 -> 质量和系统评估 -> 上线灰度 -> fallback 和复盘”讲，不要只说用了 AWQ。

2 分钟版：

示例结构：

```text
FP16 模型单卡并发低、成本高。
比较 INT8、GPTQ-4bit、AWQ-4bit、FP8 KV Cache。
校准集来自真实 prompt，覆盖长文本、代码、JSON 和业务术语。
评估私有任务成功率、PPL、格式有效率、P95 TPOT、峰值显存。
最终选择质量下降可控且吞吐收益最大的方案。
上线做灰度、监控、fallback、badcase 回流和量化配置版本化。
```

重点是指标和取舍，不是工具名。

## 679. 量化相关手撕/公式题怎么准备？

30 秒版：

会写对称/非对称量化公式、反量化、per-group scale、KV Cache 显存估算，并能手写一个简化 quantize-dequantize 函数。

2 分钟版：

准备这些：

```python
def fake_quant(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale
```

追问点包括：`scale=0` 怎么处理、按 channel 怎么 broadcast、group-wise 怎么 reshape、为什么 STE 用于 QAT、KV Cache 公式里为什么有 `2`、GQA/MQA 怎么降低 `H_kv`。

## 680. 模型量化面试前最后怎么复习？

30 秒版：

背公式、背 GPTQ/AWQ/SmoothQuant/QLoRA/FP8/KV Cache 的一句话区别，再准备一个端到端项目讲法和“INT4 为什么不一定快”的回答。

2 分钟版：

最后清单：

- 公式：对称、非对称、KV Cache 显存。
- 分类：weight-only、W8A8、KV Cache、FP8。
- 方法：GPTQ 二阶补偿，AWQ 激活感知，SmoothQuant 平滑 outlier，QLoRA 冻结 4-bit 基座训练 LoRA。
- 工程：kernel、dequant、metadata、batch shape、硬件支持。
- 评估：FP16 baseline、私有 eval、长上下文、数学/代码、结构化输出、P95/P99、成本。
- 项目：校准数据、ablation、灰度、fallback、版本化。
