# 答案版 01：Transformer 与模型结构

这一份对应 `03_高频题单100题.md` 的第 1-20 题。每个答案都按“先给面试可说版本，再给追问展开”的方式写。

## 1. Transformer 的 Encoder-only、Decoder-only、Encoder-Decoder 有什么区别？

**面试答案：**  
Encoder-only 只做双向编码，代表是 BERT，适合理解类任务，比如分类、匹配、抽取。Decoder-only 只做自回归生成，代表是 GPT、LLaMA、Qwen、DeepSeek，适合根据前文预测下一个 token。Encoder-Decoder 同时有编码器和解码器，代表是 T5、BART，适合输入到输出的转换任务，比如翻译、摘要。

**深入理解：**  
区别的关键在 attention mask。Encoder-only 通常能双向看上下文，所以不适合直接做严格的自回归生成。Decoder-only 用 causal mask，只能看左边，非常适合 next token prediction。Encoder-Decoder 的 encoder 负责理解输入，decoder 负责生成输出，decoder 既有自注意力，也有 cross-attention 去看 encoder 输出。

**常见追问：**  
为什么现在 LLM 主流是 Decoder-only？因为训练目标简单、扩展性强、推理模式统一，能把问答、代码、翻译、摘要都包装成“根据上下文继续生成”。

## 2. GPT 为什么多用 Decoder-only？

**面试答案：**  
GPT 的目标是自回归语言建模，也就是根据前文预测下一个 token。Decoder-only 结构天然适合这个目标：训练时用 causal mask 防止看到未来，推理时逐 token 生成。它结构简单、预训练和推理形式一致，也更适合大规模扩展。

**深入理解：**  
Encoder-Decoder 在 seq2seq 任务里很自然，但通用对话和指令跟随可以全部写成 prompt + completion。Decoder-only 不需要单独编码输入和解码输出，所有信息都在同一个上下文窗口中处理，工程上更统一。

## 3. Self-Attention 的公式是什么？

**面试答案：**

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Q 表示当前位置想找什么，K 表示每个位置能被怎样匹配，V 表示每个位置真正提供的信息。先用 Q 和 K 算相似度，再 softmax 成权重，最后对 V 加权求和。

**Shape 版本：**

```text
X: [B, T, C]
Q/K/V: [B, T, D]
score = QK^T: [B, T, T]
output: [B, T, D]
```

**一句话直觉：**  
每个 token 都拿自己的 query 去问整段序列：“谁对我有用？”然后把有用 token 的 value 汇总回来。

## 4. 为什么 attention score 要除以 `sqrt(d_k)`？

**面试答案：**  
因为 Q 和 K 的点积会随着维度增大而变大。如果不缩放，softmax 的输入可能过大，输出会接近 one-hot，梯度变小，训练不稳定。除以 `sqrt(d_k)` 可以控制点积分布的方差，让 softmax 更稳定。

**更深入一点：**  
假设 Q/K 每个维度均值为 0、方差为 1，点积是 d 个随机变量相加，方差大约随 d 增长。除以 `sqrt(d)` 后，方差被拉回稳定量级。

**追问回答：**  
为什么不是除以 `d_k`？因为标准差随 `sqrt(d_k)` 增长，缩放标准差应该除以 `sqrt(d_k)`。

## 5. causal mask 怎么实现？

**面试答案：**  
causal mask 用来防止当前位置看到未来 token。实现时通常在 attention score 的未来位置加上 `-inf`，再做 softmax，这样未来位置的概率就接近 0。

```text
score: [B, H, T, T]
mask:  [1, 1, T, T]
masked_score = score + mask
attn = softmax(masked_score, dim=-1)
```

**常见错误：**

- mask 方向写反。
- 在 softmax 后 mask，导致概率不归一。
- 用 0 mask 未来位置，未来 token 仍然会参与 softmax。

## 6. Self-Attention 的时间复杂度和空间复杂度是多少？

**面试答案：**  
Self-Attention 对序列长度 T 的复杂度是 `O(T^2 * D)`，因为每个 token 都要和所有 token 计算相似度。attention score 矩阵大小是 `[T, T]`，所以空间复杂度里也有 `O(T^2)`。

**工程意义：**  
长上下文贵，主要就贵在 attention 随序列长度平方增长。虽然 FlashAttention 能减少显存读写和中间矩阵存储，但标准全量 attention 的计算量本质仍和 `T^2` 相关。

## 7. MHA 为什么要分多个 head？

**面试答案：**  
MHA 把 hidden size 切成多个 head，让模型在不同子空间里并行学习不同关系。一个 head 可能更关注局部依赖，一个 head 可能关注长程实体引用，最后把多个 head 的结果拼回去。

**追问：head 越多越好吗？**  
不是。head 数变多时，每个 head 的维度会变小，单头表达能力下降；同时计算和工程实现也有成本。一般是模型规模、hidden size 和经验效果共同决定。

## 8. MHA、MQA、GQA、MLA 的区别是什么？

**面试答案：**

- MHA：每个 query head 都有自己的 K/V，表达能力强，但 KV Cache 大。
- MQA：多个 query head 共享一组 K/V，KV Cache 小，推理快，但效果可能下降。
- GQA：query head 分组，每组共享 K/V，是 MHA 和 MQA 的折中。
- MLA：把 K/V 相关信息压缩到低维 latent 空间，进一步降低 KV Cache 压力，常和 DeepSeek 系模型一起被问。

**一句话总结：**  
它们都在解决同一个矛盾：attention 需要历史 K/V，但 KV Cache 太占显存。

## 9. GQA 为什么能降低 KV Cache？

**面试答案：**  
KV Cache 存的是每层历史 token 的 K 和 V。MHA 中每个 head 都有 K/V，而 GQA 让多个 query head 共享一组 K/V，所以 K/V head 数减少，缓存量也减少。

**估算口径：**

```text
KV cache ≈ batch * seq_len * layers * kv_heads * head_dim * 2 * bytes
```

GQA 降低的是 `kv_heads`。

## 10. RoPE 的核心思想是什么？

**面试答案：**  
RoPE 是旋转位置编码。它不是把位置向量加到 embedding 上，而是在 attention 里对 Q/K 的每两个维度做旋转，旋转角度和 token 位置有关。这样 Q/K 内积时就能自然体现相对位置信息。

**为什么适合 LLM：**  
RoPE 对自回归模型友好，能比较自然地建模相对位置，也方便做长上下文扩展，所以 LLaMA、Qwen、DeepSeek 等模型都常用。

## 11. RoPE 作用在 Q/K 还是 V？

**面试答案：**  
RoPE 作用在 Q 和 K 上，不作用在 V 上。因为位置信息影响的是“当前位置应该关注哪些位置”，这个关系通过 Q/K 的 attention score 决定；V 是被聚合的内容本身。

**追问：为什么不是加到 embedding？**  
加到 embedding 是绝对位置编码的思路，RoPE 是在 Q/K 匹配时引入位置旋转，让相对位置通过内积体现出来。

## 12. RoPE 为什么能表示相对位置？

**面试答案：**  
RoPE 对位置 m 的 Q 和位置 n 的 K 分别做旋转。旋转矩阵有一个性质：两个旋转后向量的内积可以写成和 `m-n` 相关的形式，所以 attention score 里自然包含相对位置信息。

**面试说法：**  
不一定要完整推矩阵，但要说清：RoPE 用绝对位置控制旋转角度，Q/K 内积后出现相对位置差。

## 13. ALiBi 和 RoPE 有什么区别？

**面试答案：**  
RoPE 是对 Q/K 做旋转，把位置信息融入 attention score。ALiBi 是给 attention score 加一个和距离相关的线性 bias，距离越远惩罚越大。RoPE 是旋转式编码，ALiBi 是注意力偏置。

**取舍：**  
ALiBi 更简单，长度外推能力不错；RoPE 在现代开源 LLM 中更常见，表达能力和实践效果都很好，但长上下文外推通常需要额外 scaling 技巧。

## 14. KV Cache 是什么？

**面试答案：**  
KV Cache 是自回归生成时缓存历史 token 在每一层的 Key 和 Value。生成新 token 时，只需要计算新 token 的 Q/K/V，然后用新 Q 去 attend 历史 K/V，不用重复计算历史 token 的 K/V。

**为什么不缓存 Q：**  
历史 token 的 Q 不再用于生成当前 token，当前步只需要当前 token 的 Q。

## 15. KV Cache 显存如何估算？

**面试答案：**

```text
KV cache ≈ batch * seq_len * layers * kv_heads * head_dim * 2 * bytes
```

这里 `2` 是 K 和 V。batch 越大、上下文越长、层数越多、kv_heads 越多，KV Cache 越大。

**追问：怎么降低？**

- 用 MQA/GQA/MLA 减少 K/V head 或压缩 K/V。
- 限制上下文长度。
- KV Cache 量化。
- PagedAttention 管理缓存碎片。
- Prefix cache 复用公共前缀。

## 16. prefill 和 decode 有什么区别？

**面试答案：**  
prefill 是处理 prompt 阶段，一次性计算输入序列的 hidden states 和 KV Cache；decode 是逐 token 生成阶段，每步只生成一个新 token，并复用历史 KV Cache。

**工程区别：**  
prefill 更偏计算密集，因为可以并行处理 prompt；decode 更偏内存带宽和调度瓶颈，因为每步都要读历史 KV Cache，而且生成是串行的。

## 17. LayerNorm 和 RMSNorm 的区别是什么？

**面试答案：**  
LayerNorm 会减均值再除以标准差，RMSNorm 不减均值，只按均方根做缩放。RMSNorm 计算更简单，在大模型中常见，能保持训练稳定并降低一点计算开销。

```text
LayerNorm: (x - mean) / sqrt(var + eps)
RMSNorm:   x / sqrt(mean(x^2) + eps)
```

## 18. PreNorm 和 PostNorm 的区别是什么？

**面试答案：**  
PreNorm 是先 Norm 再进 Attention/FFN，PostNorm 是先过子层再 Norm。PreNorm 通常训练更稳定，尤其适合很深的 Transformer；PostNorm 在原始 Transformer 中使用，但深层训练可能更难。

**结构：**

```text
PreNorm:  x + Sublayer(Norm(x))
PostNorm: Norm(x + Sublayer(x))
```

## 19. SwiGLU 相比普通 FFN 有什么不同？

**面试答案：**  
普通 FFN 是一次上投影、激活、下投影。SwiGLU 引入门控分支，一路生成候选特征，一路用 Swish/SiLU 生成 gate，然后逐元素相乘。它能增强非线性表达能力，是现代 LLM 常见 FFN 变体。

```text
SwiGLU(x) = W_down( SiLU(W_gate x) * W_up x )
```

## 20. MoE 的核心思想是什么？

**面试答案：**  
MoE，即 Mixture of Experts，用多个专家网络替代普通 FFN。每个 token 只路由到 top-k 个专家，所以模型总参数量很大，但每个 token 激活的参数量有限。

**优点：**

- 增加模型容量。
- 单 token 计算量不按总参数线性增长。
- 适合扩大模型规模。

**难点：**

- 路由负载均衡。
- 专家并行通信。
- 训练稳定性。
- 推理部署复杂。

**面试一句话：**  
MoE 是用稀疏激活换模型容量，核心难点是路由和工程效率。

