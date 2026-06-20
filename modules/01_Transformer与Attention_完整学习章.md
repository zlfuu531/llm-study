# 完整学习章 01：Transformer 与 Attention

## 你学完要能做到什么

这一章是大模型八股的地基。学完后你要能做到：

- 从 `x: [B, T, C]` 一路讲到 MHA 输出。
- 解释 Self-Attention 公式里的每一项。
- 说明为什么要除以 `sqrt(d_k)`。
- 讲清 causal mask、RoPE、KV Cache。
- 区分 MHA、MQA、GQA、MLA。
- 手写 Self-Attention、MHA、RoPE、RMSNorm 的简化版。

一句话记忆：

> Transformer 的核心是用 Attention 做 token 间信息交互，用 FFN 做逐 token 非线性变换，用残差和归一化保证深层训练稳定。

## 1. Transformer 总体结构

主流大语言模型大多是 Decoder-only Transformer。一个典型 block 包含：

```text
x
-> RMSNorm / LayerNorm
-> Causal Self-Attention
-> Residual
-> RMSNorm / LayerNorm
-> FFN / SwiGLU / MoE
-> Residual
```

整体流程：

1. 文本被 tokenizer 切成 token id。
2. token id 经过 embedding 变成向量。
3. 多层 Transformer block 反复更新 token 表示。
4. 最后一层 hidden state 经过 LM head，输出下一个 token 的 logits。

**面试答案：Transformer 为什么适合大模型？**  
Transformer 能并行处理序列，训练效率比 RNN 高；Self-Attention 能直接建模长距离依赖；结构简单可扩展，堆层数、hidden size 和数据量后效果很好。

## 2. Encoder-only、Decoder-only、Encoder-Decoder

| 结构 | 代表模型 | 注意力方式 | 适合任务 |
| --- | --- | --- | --- |
| Encoder-only | BERT | 双向 attention | 分类、匹配、抽取 |
| Decoder-only | GPT、LLaMA、Qwen、DeepSeek | causal attention | 生成、对话、代码 |
| Encoder-Decoder | T5、BART | encoder 双向 + decoder causal/cross | 翻译、摘要、seq2seq |

**面试答案：GPT 为什么多用 Decoder-only？**  
GPT 的训练目标是根据前文预测下一个 token。Decoder-only 通过 causal mask 保证只能看左侧上下文，训练目标和推理方式统一，结构简单，适合大规模扩展。

## 3. 先把张量和矩阵乘法讲清楚

Transformer 里最重要的是 shape。你只要把 shape 讲清楚，很多公式就不抽象了。

常用符号：

| 符号 | 含义 |
| --- | --- |
| `B` | batch size，一次输入多少条样本 |
| `T` | sequence length，每条样本多少个 token |
| `C` / `d_model` | hidden size，每个 token 的向量维度 |
| `H` | attention head 数 |
| `D` / `d_head` | 每个 head 的维度，通常 `D = C / H` |
| `Vocab` | 词表大小 |

输入 token id：

```text
input_ids: [B, T]
```

经过 embedding 表：

```text
E: [Vocab, C]
x = E[input_ids]
x: [B, T, C]
```

每个 token id 会被查表成一个 `C` 维向量。Transformer 后续所有层，基本都在处理 `x: [B, T, C]` 这种三维张量。

### 线性层到底在算什么

一个线性层：

```text
y = x W + b
```

如果：

```text
x: [B, T, C_in]
W: [C_in, C_out]
b: [C_out]
```

那么：

```text
y: [B, T, C_out]
```

直觉：对 batch 里的每个样本、每个 token，都把它的 `C_in` 维向量投影成 `C_out` 维向量。`B` 和 `T` 不变，变的是最后一维。

在 PyTorch 的 `nn.Linear(C_in, C_out)` 里，权重通常存成：

```text
weight: [C_out, C_in]
```

但数学表达习惯写成 `x W`，你只要知道最后一维从 `C_in` 变成 `C_out` 即可。

### 为什么 Transformer 喜欢把计算写成矩阵乘法

因为矩阵乘法可以一次性并行处理所有 token：

```text
x: [B, T, C]
W_q: [C, C]
Q = x W_q
Q: [B, T, C]
```

这比 RNN 一个 token 一个 token 顺序处理更适合 GPU。

## 4. 从 token 到 logits 的完整计算链

以 Decoder-only LLM 为例，一次前向计算可以写成：

```text
input_ids: [B, T]
token embedding -> x: [B, T, C]
position information -> x: [B, T, C]
N 个 Transformer blocks -> h: [B, T, C]
LM Head -> logits: [B, T, Vocab]
```

LM Head 通常是一个线性层：

```text
logits = h W_vocab
W_vocab: [C, Vocab]
logits: [B, T, Vocab]
```

`logits[b, t, :]` 表示第 `b` 条样本第 `t` 个位置，对词表里每个 token 的未归一化分数。

训练时做 next token prediction：

```text
输入位置:  x_0, x_1, ..., x_{T-2}
预测目标:  x_1, x_2, ..., x_{T-1}
```

也就是 logits 左移/labels 右移后计算交叉熵：

```text
loss = CrossEntropy(logits[:, :-1, :], labels[:, 1:])
```

**面试答案：LLM 最后一层输出是什么？**  
最后输出的是 `[B, T, Vocab]` 的 logits，每个位置都有一个词表大小的分数向量。训练时每个位置预测下一个 token，推理时通常取最后一个位置的 logits 来采样下一个 token。

## 5. Self-Attention 的直觉

把每个 token 想成一个正在查资料的人：

- Query：我现在想找什么。
- Key：我能被别人怎样匹配到。
- Value：我真正提供什么信息。

每个 token 拿自己的 Query 去和所有 token 的 Key 算相似度，得到注意力权重，再把所有 token 的 Value 按权重加权求和。

公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

输入 shape：

```text
X: [B, T, C]
Q/K/V: [B, T, D]
score = QK^T: [B, T, T]
output: [B, T, D]
```

**面试答案：Q/K/V 为什么分开？**  
同一个 token 在不同角色下需要不同表示。Q 表示“我要找什么”，K 表示“我怎么被匹配”，V 表示“我被找到后提供什么内容”。分开投影能提升表达能力。

## 6. Self-Attention 的逐步计算公式

先看单头 attention。输入：

```text
X: [B, T, C]
```

三组线性投影：

```text
Q = X W_q
K = X W_k
V = X W_v
```

如果单头输出维度是 `D`：

```text
W_q: [C, D]
W_k: [C, D]
W_v: [C, D]

Q: [B, T, D]
K: [B, T, D]
V: [B, T, D]
```

### 第一步：算注意力分数

对每个样本，拿 Q 和 K 做矩阵乘法：

```text
S = Q K^T
```

shape：

```text
Q:     [B, T, D]
K^T:   [B, D, T]
S:     [B, T, T]
```

`S[b, i, j]` 表示第 `b` 条样本中，第 `i` 个 token 对第 `j` 个 token 的关注分数。

展开写就是：

```text
S[i, j] = q_i · k_j = sum_{m=1}^{D} q_i[m] * k_j[m]
```

### 第二步：缩放

```text
S_scaled = S / sqrt(D)
```

### 第三步：加 mask

Decoder-only 使用 causal mask：

```text
S_masked[i, j] = -inf, if j > i
```

表示第 `i` 个位置不能看未来的第 `j` 个位置。

### 第四步：softmax 得到权重

对每一行做 softmax：

```text
A[i, j] = exp(S_masked[i, j]) / sum_k exp(S_masked[i, k])
```

shape：

```text
A: [B, T, T]
```

每一行加起来等于 1，表示当前位置对所有可见 token 的注意力分布。

### 第五步：加权求和 Value

```text
O = A V
```

shape：

```text
A: [B, T, T]
V: [B, T, D]
O: [B, T, D]
```

展开：

```text
o_i = sum_j A[i, j] * v_j
```

意思是：第 `i` 个 token 的新表示，是它能看到的所有 token 的 value 的加权和。

## 7. 为什么除以 `sqrt(d_k)`

如果 Q/K 每个维度均值为 0、方差为 1，那么点积是很多维相乘再相加。维度越大，点积值的方差越大。

不缩放会导致：

- attention score 数值过大。
- softmax 输出过于尖锐，接近 one-hot。
- 梯度变小，训练不稳定。

除以 `sqrt(d_k)` 是为了把点积尺度拉回稳定范围。

**面试答案：为什么不是除以 `d_k`？**  
因为点积标准差随 `sqrt(d_k)` 增长，缩放标准差应该除以 `sqrt(d_k)`。

## 8. causal mask

Decoder-only 模型预测第 i 个 token 时不能看未来 token。score 矩阵 `[T, T]` 中：

- 行：当前 query 位置。
- 列：可看的 key/value 位置。

第 i 行只能看 `0...i` 列，不能看 `i+1...T-1` 列。

实现：

```python
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
score = score.masked_fill(mask, float("-inf"))
attn = torch.softmax(score, dim=-1)
```

**常见追问：mask 在 softmax 前还是后？**  
在 softmax 前。把未来位置设成 `-inf` 后，softmax 概率自然变成 0。如果 softmax 后再乘 0，还要重新归一化。

## 9. Multi-Head Attention

MHA 把 hidden size 拆成多个 head，在不同子空间里做 attention。

shape：

```text
x:     [B, T, C]
qkv:   [B, T, 3C]
q/k/v: [B, H, T, D]
score: [B, H, T, T]
out:   [B, T, C]
```

其中 `D = C / H`。

### MHA 的完整计算公式

通常实现里会一次性投影出 Q/K/V：

```text
QKV = X W_qkv
```

如果：

```text
X:     [B, T, C]
W_qkv: [C, 3C]
```

则：

```text
QKV: [B, T, 3C]
```

然后 reshape：

```text
QKV -> [B, T, 3, H, D]
Q, K, V -> [B, T, H, D]
transpose -> [B, H, T, D]
```

对每个 head 独立做 attention：

```text
head_i = Attention(Q_i, K_i, V_i)
head_i: [B, T, D]
```

拼接所有 head：

```text
Concat(head_1, ..., head_H): [B, T, H*D] = [B, T, C]
```

最后再过一个输出投影：

```text
O = Concat(heads) W_o
W_o: [C, C]
O: [B, T, C]
```

### MHA 参数量速算

如果 hidden size 是 `C`：

```text
W_q: [C, C]
W_k: [C, C]
W_v: [C, C]
W_o: [C, C]
```

忽略 bias，MHA 参数量约为：

```text
4 * C * C
```

注意：head 数 H 改变时，只要总 hidden size C 不变，标准 MHA 的投影参数量大致不变；变的是每个 head 的维度 `D = C/H`。

### MHA 计算复杂度

主要两部分：

1. QKV 和输出投影：

```text
O(B * T * C^2)
```

2. attention score 和加权求和：

```text
O(B * H * T^2 * D) = O(B * T^2 * C)
```

所以长上下文时，`T^2` 是 attention 的核心瓶颈。

**面试答案：为什么要多头？**  
单头只能在一个子空间里学习依赖，多头能并行学习不同类型的关系，比如局部关系、长距离引用、语法关系或格式关系。最后把不同 head 的信息拼接起来。

**追问：head 越多越好吗？**  
不是。head 多时每个 head 的维度变小，单头表达能力下降；还会增加调度和实现成本。head 数要和 hidden size、模型规模匹配。

## 10. MHA、MQA、GQA、MLA

它们都在处理一个问题：

> Attention 需要历史 K/V，但 KV Cache 太占显存。

| 方法 | K/V 设计 | 优点 | 缺点 |
| --- | --- | --- | --- |
| MHA | 每个 Q head 有自己的 K/V | 表达强 | KV Cache 大 |
| MQA | 所有 Q head 共享一组 K/V | 显存低、推理快 | 表达可能下降 |
| GQA | Q head 分组共享 K/V | 折中效果和成本 | 比 MQA 复杂 |
| MLA | K/V 信息压缩到 latent 空间 | 更省 KV Cache | 实现和理解更复杂 |

**面试答案：GQA 为什么能降低 KV Cache？**  
KV Cache 和 `kv_heads` 成正比。GQA 让多个 query head 共享一组 K/V，减少 `kv_heads`，所以缓存显存下降。

**面试答案：MLA 为什么最近常被问？**  
MLA 和 DeepSeek 系模型相关。它关注通过低维 latent 压缩 K/V 信息，降低长上下文和高并发推理中的 KV Cache 成本。

## 11. RoPE

RoPE 是 Rotary Position Embedding，旋转位置编码。

如果面试官继续追 RoPE scaling、Position Interpolation、NTK-aware、YaRN、LongRoPE、Ring Attention 或长上下文评测，跳到进阶专题：[../41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md](../41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md)。

它不是把位置向量加到 embedding 上，而是在 attention 中对 Q/K 的每两个维度做旋转。旋转角度与 token 的 position 有关。

记忆：

- 作用在 Q/K。
- 不作用在 V。
- 每两个维度一组旋转。
- Q/K 内积后体现相对位置信息。

**面试答案：RoPE 为什么作用在 Q/K 上？**  
位置关系影响的是“当前位置应该关注哪些位置”，也就是 attention score。score 来自 Q/K 内积，所以 RoPE 作用在 Q/K 上。V 是被聚合的内容，不负责决定关注谁。

**面试答案：RoPE 为什么能表示相对位置？**  
位置 m 的 Q 和位置 n 的 K 分别旋转后做内积，结果可以写成和 `m-n` 有关的形式。因此虽然旋转角度使用绝对位置，attention score 中体现的是相对位置差。

## 12. KV Cache

自回归生成有两个阶段：

- Prefill：一次性处理 prompt，计算所有 token 的 K/V。
- Decode：每步生成一个新 token，只计算新 token 的 Q/K/V，并复用历史 K/V。

缓存的是每一层历史 token 的 K 和 V。

为什么不缓存 Q：

- 历史 Q 不再用于当前步生成。
- 当前步只需要当前 token 的 Q 去 attend 历史 K/V。

显存估算：

```text
KV cache ≈ batch * seq_len * layers * kv_heads * head_dim * 2 * bytes
```

`2` 表示 K 和 V。

**面试答案：KV Cache 是什么？**  
KV Cache 是用显存换计算。它缓存历史 token 的 K/V，避免 decode 阶段重复计算历史 token，从而加速生成，但显存会随 batch、上下文长度、层数和 kv_heads 增长。

## 13. FFN / SwiGLU 的计算

Attention 负责 token 之间的信息交互，FFN 负责对每个 token 自己的表示做非线性变换。

普通 FFN：

```text
FFN(x) = W_2 * activation(W_1 x + b_1) + b_2
```

shape：

```text
x:      [B, T, C]
W_1:    [C, C_ffn]
hidden: [B, T, C_ffn]
W_2:    [C_ffn, C]
out:    [B, T, C]
```

通常 `C_ffn` 比 `C` 大，比如 4 倍左右。

SwiGLU：

```text
SwiGLU(x) = W_down( SiLU(W_gate x) * W_up x )
```

shape：

```text
W_gate: [C, C_ffn]
W_up:   [C, C_ffn]
W_down: [C_ffn, C]
```

直觉：

- `W_up x` 产生候选内容。
- `SiLU(W_gate x)` 产生门控。
- 二者逐元素相乘，决定哪些内容通过。

**面试答案：Attention 和 FFN 分别做什么？**  
Attention 做 token 间信息混合，FFN 做每个 token 内部的非线性特征变换。Attention 让 token 看上下文，FFN 提升表示能力。

## 14. 残差连接与归一化的计算

Transformer block 不是直接：

```text
x -> attention -> ffn
```

而是带残差：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

PreNorm 形式：

```text
y = x + Sublayer(Norm(x))
```

PostNorm 形式：

```text
y = Norm(x + Sublayer(x))
```

残差连接的作用：

- 保留原始信息。
- 缓解梯度消失。
- 让深层网络更容易训练。

归一化的作用：

- 稳定激活分布。
- 稳定梯度。
- 提升深层训练稳定性。

## 15. LayerNorm、RMSNorm、PreNorm、PostNorm

LayerNorm：

```text
(x - mean) / sqrt(var + eps)
```

RMSNorm：

```text
x / sqrt(mean(x^2) + eps)
```

区别：

- LayerNorm 减均值并除标准差。
- RMSNorm 不减均值，只按均方根缩放。
- RMSNorm 更简单，在现代 LLM 中常见。

PreNorm / PostNorm：

```text
PreNorm:  x + Sublayer(Norm(x))
PostNorm: Norm(x + Sublayer(x))
```

**面试答案：为什么大模型常用 PreNorm？**  
PreNorm 的梯度路径更稳定，深层 Transformer 更容易训练。

## 16. LM Head 与训练 loss

Transformer block 输出：

```text
h: [B, T, C]
```

LM Head：

```text
logits = h W_vocab
W_vocab: [C, Vocab]
logits: [B, T, Vocab]
```

对每个位置，logits 是对整个词表的打分。

训练时预测下一个 token：

```text
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
loss = CrossEntropy(shift_logits, shift_labels)
```

交叉熵：

```text
CE = - log p_true
```

其中：

```text
p = softmax(logits)
```

推理时通常只取最后一个位置：

```text
next_logits = logits[:, -1, :]
next_token = sample(next_logits)
```

然后把 `next_token` 拼回输入，继续下一步 decode。

**面试答案：训练和推理有什么不同？**  
训练时可以并行计算所有位置的 next token loss；推理时必须自回归逐 token 生成，每一步依赖上一步生成的 token。

## 17. SwiGLU 与 MoE

SwiGLU：

```text
SwiGLU(x) = W_down( SiLU(W_gate x) * W_up x )
```

直觉：一路生成候选特征，一路生成门控，门控控制哪些信息通过。

MoE：

- 多个专家网络。
- 每个 token 只路由到 top-k 个专家。
- 总参数量大，但单 token 激活参数量有限。

**面试答案：MoE 的核心是什么？**  
MoE 用稀疏激活扩大模型容量。它让模型拥有很多专家，但每个 token 只走少数专家。难点是路由负载均衡、通信开销和部署复杂度。

## 18. 本章高频问答

### Q1：Self-Attention 的公式是什么？

```text
Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
```

先算 Q/K 相似度，再 softmax 成权重，最后对 V 加权求和。

### Q2：Self-Attention 复杂度是多少？

时间和空间都含有 `O(T^2)`，因为每个 token 都要和所有 token 计算 attention score。

### Q2.5：一个 Decoder-only block 的计算顺序是什么？

常见 PreNorm 结构是：`x -> Norm -> Attention -> residual add -> Norm -> FFN/SwiGLU -> residual add`。Attention 负责跨 token 交互，FFN 负责逐 token 非线性变换。

### Q3：GQA 和 MQA 区别？

MQA 是所有 query head 共享一组 K/V；GQA 是多个 query head 分组共享 K/V。GQA 在效果和 KV Cache 成本之间折中。

### Q4：RoPE 和 ALiBi 区别？

RoPE 是旋转 Q/K；ALiBi 是给 attention score 加距离相关的线性 bias。RoPE 在现代开源 LLM 中更常见，ALiBi 简单且外推性较好。

### Q5：长上下文为什么贵？

Prefill 阶段 attention 近似 `O(T^2)`，decode 阶段 KV Cache 随 T 线性增长。上下文越长，计算、显存和延迟都增加。

### Q6：训练时 logits 和 labels 怎么对齐？

模型第 `t` 个位置预测第 `t+1` 个 token，所以训练时通常用 `logits[:, :-1, :]` 对齐 `labels[:, 1:]` 计算交叉熵。

## 19. 本章手撕代码清单

必须能写：

1. Self-Attention。
2. Multi-Head Attention。
3. causal mask。
4. RoPE。
5. RMSNorm。
6. KV Cache 简化版。

写 MHA 时一定边写边说：

```text
x: [B,T,C] -> qkv: [B,T,3C] -> q/k/v: [B,H,T,D]
score: [B,H,T,T] -> out: [B,T,C]
```

## 20. 面试前背诵版

Transformer 从 `input_ids: [B,T]` 开始，先查 embedding 得到 `x: [B,T,C]`，经过多层 block 后得到 `h: [B,T,C]`，再经 LM Head 得到 `logits: [B,T,Vocab]`。Self-Attention 用 Q/K 算相关性，用 V 汇总信息，公式是 `softmax(QK^T/sqrt(d_k))V`。缩放是为了防止维度大时点积过大导致 softmax 饱和。MHA 把 `C` 拆成 `H` 个 head，每个 head 维度 `D=C/H`，attention score 是 `[B,H,T,T]`。Decoder-only 模型用 causal mask 防止看未来，训练时并行预测下一个 token，推理时逐 token decode。现代 LLM 常用 RoPE 把位置信息注入 Q/K，用 KV Cache 加速 decode，用 GQA/MLA 降低 KV Cache 成本，用 RMSNorm、SwiGLU、MoE 提升训练稳定性和模型容量。
