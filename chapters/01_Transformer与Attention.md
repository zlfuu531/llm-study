# Transformer 与 Attention

## 面试目标

这一章要达到的状态：

- 能从输入 shape 讲到 Self-Attention 输出 shape。
- 能解释为什么 attention score 要除以 `sqrt(d_k)`。
- 能区分 MHA、MQA、GQA、MLA。
- 能讲清 RoPE、KV Cache、RMSNorm、SwiGLU、MoE 的作用。
- 能手写简化版 Self-Attention / MHA。

更深入的通俗版推导见：[../deepdives/01_Attention_RoPE_KVCache_深挖.md](../deepdives/01_Attention_RoPE_KVCache_深挖.md)

## Transformer 总体结构

大模型主流是 Decoder-only Transformer。核心模块通常是：

1. Token embedding
2. 位置编码或位置旋转
3. 多层 Transformer block
4. 每个 block 包含 Attention、Norm、FFN
5. LM head 输出下一个 token 概率

面试表达：

> Transformer 的核心是用 Self-Attention 建模序列中 token 之间的依赖关系，再用 FFN 做逐 token 的非线性变换。大语言模型多采用 Decoder-only 结构，因为它天然适合 causal language modeling，即根据前文预测下一个 token。

## Self-Attention

输入：

- `X: [B, T, C]`
- `B` 是 batch size
- `T` 是序列长度
- `C` 是 hidden size

线性投影：

- `Q = XW_q`
- `K = XW_k`
- `V = XW_v`

注意力：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

shape：

- `Q: [B, T, D]`
- `K: [B, T, D]`
- `QK^T: [B, T, T]`
- `softmax(...)V: [B, T, D]`

为什么除以 `sqrt(d_k)`：

- 如果不缩放，`QK^T` 的方差会随维度增大而变大。
- softmax 输入过大时容易饱和，梯度变小。
- 缩放能让分布更稳定，训练更容易。

## Causal Mask

自回归模型不能看到未来 token，所以要 mask 掉当前位置之后的信息。

常见追问：

- mask 是上三角还是下三角？
- mask 用 `-inf` 还是 0？
- mask 在 softmax 前还是后？

合格回答：

> causal mask 通常在 attention score softmax 前加到未来位置上，把未来位置设为一个很小的数，例如 `-inf`，这样 softmax 后概率接近 0。mask 的方向取决于 score 矩阵的定义，但本质是第 i 个 token 只能 attend 到 `<= i` 的位置。

## Multi-Head Attention

MHA 的思想是把 hidden size 拆成多个 head，让不同 head 学不同关系。

shape：

- 输入 `x: [B, T, C]`
- head 数 `H`
- 每个 head 维度 `D = C / H`
- `q/k/v: [B, H, T, D]`
- score: `[B, H, T, T]`
- concat 后回到 `[B, T, C]`

为什么需要多头：

- 单头 attention 表达能力有限。
- 多头能在不同子空间学习不同依赖，比如局部关系、长程关系、语义关系。
- 但 head 不是越多越好，head_dim 太小会影响单头表达。

## MHA / MQA / GQA / MLA

### MHA

每个 head 都有自己的 Q/K/V。

优点：

- 表达能力强。

缺点：

- KV Cache 显存大。

### MQA

多个 query head 共享一组 K/V。

优点：

- 大幅减少 KV Cache。
- 推理更快。

缺点：

- 表达能力可能下降。

### GQA

把 query head 分组，每组共享 K/V。

优点：

- 在 MHA 和 MQA 之间折中。
- 兼顾效果和推理成本。

面试表达：

> GQA 可以看成 MHA 到 MQA 的折中。Query 仍然有多个 head，但 K/V 按组共享，所以 KV Cache 比 MHA 小，同时比 MQA 保留更多表达能力。

### MLA

MLA 常和 DeepSeek 系模型一起被问。核心思路是把 K/V 相关信息压缩到低维 latent 空间，减少 KV Cache 和推理显存压力。

面试注意：

- 不要只背名字。
- 要强调它解决的是长上下文和推理阶段 KV Cache 成本。
- 能说出和 MQA/GQA 的共同目标：降低 K/V 存储和读取成本。

## RoPE

RoPE，即 Rotary Position Embedding。它把位置信息通过旋转矩阵作用到 Q/K 上，让 attention score 能感知相对位置。

为什么常被问：

- LLaMA、Qwen、DeepSeek 等模型都广泛使用。
- 和长上下文扩展强相关。
- 很适合手撕简化实现。

合格回答：

> RoPE 不是把位置向量加到 token embedding 上，而是在 attention 里对 Q/K 的每两个维度做旋转。旋转角度和 token 位置有关，因此 Q 和 K 做内积时会自然带上相对位置信息。它的好处是适合自回归模型，能较好支持相对位置建模，也方便做一定的长度外推改造。

常见追问：

- RoPE 作用在 Q/K 还是 V？
- 为什么内积后包含相对位置信息？
- RoPE 外推为什么会退化？
- NTK scaling、YaRN 这类方法在解决什么？

## KV Cache

自回归解码包括两个阶段：

- Prefill：一次性处理 prompt，计算所有历史 token 的 K/V。
- Decode：每步只输入新 token，计算新 token 的 Q/K/V，并和缓存中的历史 K/V 做 attention。

为什么需要：

- 不缓存时，每生成一个 token 都要重复计算所有历史 token 的 K/V。
- 缓存后，历史 K/V 复用，decode 阶段更快。

代价：

- 显存占用随 batch、层数、序列长度、K/V head 数、head_dim、dtype 增长。
- 长上下文和高并发时 KV Cache 成为瓶颈。

面试公式口径：

```text
KV cache memory ≈ batch * seq_len * layers * kv_heads * head_dim * 2 * bytes
```

其中 `2` 表示 K 和 V。

## RMSNorm / LayerNorm

LayerNorm：

- 减均值，再除以标准差。

RMSNorm：

- 不减均值，只按均方根缩放。

RMSNorm 优点：

- 计算更简单。
- 训练稳定。
- 大模型中常见。

面试表达：

> RMSNorm 可以看成 LayerNorm 的简化版，它去掉了中心化，只保留按均方根归一化。这样计算更省，同时在大模型中经验效果很好。

## SwiGLU

传统 FFN：

```text
FFN(x) = W2 activation(W1 x)
```

SwiGLU：

```text
SwiGLU(x) = (Swish(W_gate x) * W_up x) W_down
```

直觉：

- 一路产生候选特征。
- 一路产生门控。
- 门控决定哪些信息通过。

## MoE

MoE，即 Mixture of Experts。每个 token 只路由到少数专家 FFN。

优点：

- 增加总参数量。
- 每个 token 的激活参数量相对较小。
- 适合扩大模型容量。

挑战：

- 路由负载均衡。
- 通信开销。
- 训练稳定性。
- 推理部署复杂。

面试表达：

> MoE 的核心是稀疏激活：模型有很多专家，但每个 token 只走 top-k 个专家。这样可以在不线性增加单 token 计算量的情况下提升模型容量。难点是路由均衡和分布式通信。

## 高频问题

1. Transformer 为什么比 RNN 更适合大规模训练？
2. Self-Attention 的复杂度是多少？
3. 为什么 attention score 要除以 `sqrt(d_k)`？
4. causal mask 怎么实现？
5. MHA 和 GQA 的区别是什么？
6. KV Cache 为什么能加速推理？
7. KV Cache 显存怎么估算？
8. RoPE 为什么能表示相对位置？
9. LayerNorm 和 RMSNorm 有什么区别？
10. SwiGLU 比普通 FFN 多了什么？
11. MoE 为什么能提升参数规模但不等比例增加计算量？

## 手撕优先级

1. Self-Attention
2. MHA
3. causal mask
4. RoPE
5. RMSNorm
6. KV Cache 简化版

## 延伸阅读

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- 袁朝发 Self-Attention 手写: https://yuanchaofa.com/hands-on-code/from-self-attention-to-multi-head-self-attention
- TorchLeet: https://github.com/Exorust/TorchLeet
- LLM-Agent-Interview-Guide: https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide
