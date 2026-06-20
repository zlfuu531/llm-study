# 深挖 01：Attention、RoPE 与 KV Cache

## 这一章解决什么问题

很多人会背：

```text
Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
```

但面试官继续问：

- 为什么要除以 `sqrt(d_k)`？
- causal mask 到底 mask 谁？
- RoPE 为什么作用在 Q/K 上？
- KV Cache 缓存的到底是什么？
- MQA/GQA/MLA 为什么能省显存？

如果要继续准备 RoPE 外推、Position Interpolation、NTK-aware、YaRN、LongRoPE、Ring Attention、StreamingLLM 和长上下文 eval，跳到进阶专题：[../41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md](../41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md)。

如果这些答不清，说明只是记公式，没有真正理解大模型推理。

## 1. Attention 的直觉

一句话：

> Attention 是让每个 token 根据“我现在需要什么”，去整段序列里找“谁对我有用”，再把有用信息加权汇总回来。

可以把它想成查资料：

- Query：我现在想查什么问题。
- Key：每份资料的标签。
- Value：资料真正的内容。

第 i 个 token 先生成自己的 Query，再和所有 token 的 Key 做相似度。相似度越高，说明这个 token 对当前位置越重要。softmax 把相似度变成权重，然后对 Value 加权求和。

## 2. Q、K、V 为什么要分开

同一个 token 在不同角色下需要不同表示：

- 作为 Query：我要找什么。
- 作为 Key：我能被别人怎么找到。
- 作为 Value：我被找到后能提供什么信息。

如果 Q/K/V 都共用一个向量，表达能力会被限制。分开投影后，模型可以学习“查找空间”和“内容空间”的不同表示。

面试表达：

> Q/K/V 分开不是数学上必须，而是表达能力上的需要。Q 和 K 负责计算相关性，V 负责提供被聚合的信息。分开投影后，模型可以在不同子空间里学习匹配关系和内容表示。

## 3. 为什么除以 `sqrt(d_k)`

假设 Q 和 K 的每个维度均值为 0、方差为 1。

两个向量点积：

```text
q · k = q1*k1 + q2*k2 + ... + qd*kd
```

如果各维近似独立，点积方差会随维度 `d` 增长。维度越大，`QK^T` 的值越容易变得很大。

softmax 对大数很敏感：

- 输入差距很大时，softmax 会接近 one-hot。
- 概率过于尖锐，梯度变小。
- 训练不稳定。

除以 `sqrt(d_k)` 后，点积的尺度被拉回相对稳定范围。

面试表达：

> 缩放不是为了让数值更好看，而是为了控制点积分布的方差。如果不除以 `sqrt(d_k)`，维度大时 attention score 容易过大，softmax 饱和，梯度变差。

## 4. causal mask 到底在 mask 什么

Decoder-only 模型做 next token prediction 时，第 i 个位置不能看到 i 之后的 token。

score 矩阵形状：

```text
[T, T]
行：当前 query 位置
列：被看的 key/value 位置
```

第 i 行只能看第 `0...i` 列，不能看 `i+1...T-1` 列。

实现时通常在 softmax 前把未来位置加上 `-inf`：

```text
masked_score = score + mask
attn = softmax(masked_score)
```

为什么在 softmax 前：

- softmax 前设为 `-inf`，概率会变成 0。
- 如果 softmax 后再乘 0，概率和不再自然归一，还要重新归一化。

常见误区：

- 把 mask 方向写反。
- mask 用 0 而不是 `-inf`，导致未来 token 仍然参与 softmax。
- mask 维度没有 broadcast 到 `[B, H, T, T]`。

## 5. Multi-Head Attention 为什么有效

单头 attention 只能在一个子空间里做匹配。多头 attention 把 hidden size 拆成多个 head，让不同 head 学不同关系。

比如：

- 一个 head 关注主谓关系。
- 一个 head 关注局部相邻词。
- 一个 head 关注长距离实体引用。
- 一个 head 关注格式或标点。

当然，这只是直觉，不是每个 head 都能被清晰解释。

面试表达：

> MHA 的价值在于让模型在多个子空间并行建模依赖关系。不同 head 可以学习不同类型的 token 交互，最后 concat 回完整 hidden 表示。

## 6. RoPE 为什么作用在 Q/K 上

attention score 来自：

```text
QK^T
```

位置关系影响的是“当前位置应该关注哪些位置”。也就是影响 Query 和 Key 的匹配，而不是 Value 的内容本身。

所以 RoPE 作用在 Q/K 上，让 score 在做内积时带上位置信息。

直觉：

- token 内容表示：我是什么。
- 位置旋转：我在哪。
- Q/K 内积：我在这个位置上，和另一个位置的 token 是否相关。

## 7. RoPE 如何表示相对位置

RoPE 对向量的每两个维度做二维旋转。

位置为 `m` 的 token 旋转角度和 `m` 有关：

```text
rotate(x, m)
```

当位置 `m` 的 Q 和位置 `n` 的 K 做内积时，结果会和 `m-n` 有关。也就是说，绝对位置旋转之后，内积里自然出现相对位置信息。

不用在面试里完整推导矩阵，但要说清：

- RoPE 是乘法/旋转式位置编码，不是加法位置编码。
- 它作用在 Q/K。
- 它让 attention score 感知相对位置。

## 8. KV Cache 缓存的是什么

自回归生成时，每一步只多一个新 token。

没有 cache：

```text
第 1 步：算 token 1 的 K/V
第 2 步：重新算 token 1,2 的 K/V
第 3 步：重新算 token 1,2,3 的 K/V
```

有 cache：

```text
第 1 步：算 token 1 的 K/V，存起来
第 2 步：只算 token 2 的 K/V，和 token 1 的 cache 拼起来
第 3 步：只算 token 3 的 K/V，和历史 cache 拼起来
```

缓存的是每一层、每个历史 token 的 K 和 V。

为什么不缓存 Q：

- 每一步只需要当前新 token 的 Q。
- 历史 token 的 Q 不再被用来生成当前 token。
- 当前 Q 会去 attend 历史 K/V。

## 9. KV Cache 为什么省计算但费显存

省计算：

- 不再重复计算历史 token 的 K/V。

费显存：

- 每层都要存历史 token 的 K/V。
- batch 越大、序列越长、层数越多，cache 越大。

粗略估算：

```text
KV cache ≈ batch * seq_len * layers * kv_heads * head_dim * 2 * bytes
```

`2` 表示 K 和 V。

面试表达：

> KV Cache 是用显存换计算。它让 decode 阶段不用重复计算历史 token 的 K/V，但会让显存随上下文长度和并发请求增长。

## 10. MHA / MQA / GQA / MLA 的共同问题

它们都在处理一个核心矛盾：

> Attention 需要历史 K/V，但历史 K/V 太占显存。

MHA：

- 每个 query head 都有自己的 K/V。
- 效果好，但 KV Cache 大。

MQA：

- 所有 query head 共享一组 K/V。
- KV Cache 小，但表达能力可能损失。

GQA：

- 多个 query head 分组共享 K/V。
- 在效果和显存之间折中。

MLA：

- 把 K/V 相关信息压缩到低维 latent 表示。
- 目标是进一步降低 KV Cache，同时保持效果。

## 11. 为什么 MLA 最近常被问

DeepSeek-V2/V3/R1 相关讨论让 MLA 变成高频热点。

面试不需要完整复现 DeepSeek 论文，但要知道：

- MLA 关注 KV Cache 压缩。
- 它不是简单共享 K/V，而是低秩/latent 压缩思路。
- 它和 MoE 一起服务于“更强但更经济”的模型设计。

可以这样答：

> MHA 的 KV Cache 随 head 数增长很快，GQA/MQA 通过共享 K/V 降低缓存。MLA 更进一步，把 K/V 信息压缩到 latent 空间，在推理时减少需要缓存和读取的数据。它本质上是为长上下文和高并发推理降低显存压力。

## 12. 面试官追问路线

如果面试官问 Self-Attention，常见追问链：

1. 公式是什么？
2. Q/K/V shape 是什么？
3. 为什么除以 `sqrt(d_k)`？
4. mask 在哪一步加？
5. MHA 怎么 reshape？
6. KV Cache 缓存什么？
7. GQA 为什么省显存？
8. RoPE 为什么作用在 Q/K？
9. 长上下文下 attention 有什么问题？
10. FlashAttention / PagedAttention 分别解决什么？

## 13. 你应该准备的手撕代码

必须会：

- Self-Attention
- Multi-Head Attention
- causal mask
- RoPE
- RMSNorm
- KV Cache 简化版

写代码时一定同步说 shape。面试官很多时候不是看你背 API，而是看你有没有真实实现过。

## 参考来源

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- DeepSeek-V2: https://arxiv.org/html/2405.04434v2
- FlashAttention: https://arxiv.org/abs/2205.14135
- 袁朝发 Self-Attention 手写: https://yuanchaofa.com/hands-on-code/from-self-attention-to-multi-head-self-attention
- TorchLeet: https://github.com/Exorust/TorchLeet
- LLM-Agent-Interview-Guide: https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide
