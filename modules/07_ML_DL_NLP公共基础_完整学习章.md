# 完整学习章 07：ML / DL / NLP 公共基础

## 你学完要能做到什么

大模型面试不只问 LLM。面试官经常会从一个大模型问题追到机器学习和深度学习基础，比如：

- 为什么用交叉熵？
- Adam 和 SGD 有什么区别？
- 过拟合怎么办？
- AUC 怎么理解？
- BN、LN、RMSNorm 区别？
- BERT 和 GPT 区别？
- embedding、word2vec、BPE 到底在解决什么？

这一章不是从零学 ML，而是给你准备“被追问时能站住”的答案。

如果你想集中准备公式和项目排查，先跳到 [28_ML_DL数学基础_损失函数优化器与指标面试.md](../28_ML_DL数学基础_损失函数优化器与指标面试.md)。那里把 CE/MLE、logits、BCE、PPL、loss mask、AUC/PR-AUC、校准、AdamW、InfoNCE 和 loss/metric 不一致排查整理成独立专题，并配有 [answers/23_ML_DL数学基础_损失函数优化器指标_答案版.md](../answers/23_ML_DL数学基础_损失函数优化器指标_答案版.md)。

## 1. 监督学习三件事

监督学习可以拆成：

1. 模型：把输入映射到输出。
2. 损失函数：衡量预测和标签的差距。
3. 优化器：调整参数让损失下降。

面试表达：

> 训练模型本质是在给定数据和损失函数下，用优化算法寻找一组参数，让模型在训练集和未来数据上都有较好表现。

## 2. 交叉熵

分类任务常用交叉熵：

```text
CE = - sum y_i log p_i
```

如果标签是 one-hot，就等价于：

```text
CE = - log p_true
```

直觉：

- 正确类别概率越高，loss 越小。
- 正确类别概率越低，loss 越大。

**面试答案：为什么分类用交叉熵？**  
交叉熵来自最大似然估计。分类模型输出类别概率，训练目标是最大化真实标签的概率；取负 log 后就是交叉熵。它对错误且自信的预测惩罚很大，梯度性质也适合分类。

**和 MSE 区别：**  
MSE 更适合回归。分类中 softmax + CE 梯度更直接，收敛通常更好。

### softmax + CE 的梯度为什么好

对 logits `z` 做 softmax 得到概率 `p`，one-hot 标签是 `y`：

```text
p_i = exp(z_i) / sum_j exp(z_j)
CE = - sum_i y_i log p_i
```

合在一起求梯度会得到非常干净的形式：

```text
dL / dz_i = p_i - y_i
```

直觉：

- 正确类：如果 `p_true` 太小，梯度是负的，会把正确类 logit 往上推。
- 错误类：如果 `p_wrong` 太大，梯度是正的，会把错误类 logit 往下压。

**面试答案：为什么 PyTorch CrossEntropyLoss 输入 logits 而不是 softmax 后概率？**  
因为实现里会把 `log_softmax` 和 `negative log likelihood` 合在一起，数值更稳定。如果你先手动 softmax 再传进去，不仅重复计算，还可能导致梯度和数值稳定性变差。

### Label Smoothing

one-hot 标签太硬：

```text
y_true = [0, 0, 1, 0]
```

label smoothing 会把一部分概率分给其他类别：

```text
y_smooth = (1 - eps) * y + eps / C
```

作用：

- 减少模型过度自信。
- 提升泛化。
- 对噪声标签更稳。

风险：

- 如果任务需要极强置信度，可能影响校准和边界样本。
- 生成任务里乱用可能影响输出分布。

### Perplexity

语言模型常用困惑度：

```text
PPL = exp(平均 token cross entropy)
```

直觉：

- PPL 越低，模型对下一个 token 越不“困惑”。
- PPL 可以衡量语言建模能力，但不等价于真实对话体验。

**面试答案：PPL 低是不是模型一定好？**  
不一定。PPL 衡量 next-token 预测，不直接衡量指令遵循、事实性、安全性、推理能力和项目效果。SFT、RAG、Agent 还要看任务成功率、人工偏好、faithfulness 和业务指标。

## 3. Softmax

公式：

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

作用：

- 把 logits 转成概率分布。
- 保持相对大小关系。
- 所有类别概率和为 1。

数值稳定：

```text
softmax(z) = softmax(z - max(z))
```

**面试答案：为什么 softmax 前可以减 max？**  
softmax 对整体平移不变，所有 logits 同时减去同一个常数，概率不变。减 max 可以防止 exp 溢出。

## 4. 梯度下降、SGD、Adam

Batch GD：

- 用全量数据算梯度。
- 稳定但慢。

SGD：

- 每次用一个样本或小 batch。
- 噪声大但泛化可能好。

Adam：

- 使用一阶动量和二阶动量。
- 自动调节每个参数学习率。
- 收敛快，深度学习常用。

**面试答案：Adam 和 SGD 区别？**  
SGD 更新简单，依赖全局学习率；Adam 维护梯度一阶矩和二阶矩，对不同参数自适应调整步长，通常收敛更快。但 SGD 在某些场景泛化更好，Adam 也更需要注意学习率和 weight decay。

Adam 公式：

```text
g_t = grad(theta_t)
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

解释：

- `m_t` 是一阶动量，像带惯性的平均梯度。
- `v_t` 是二阶动量，估计梯度平方大小。
- bias correction 是因为训练初期 `m/v` 从 0 开始，会偏小。

### AdamW 和 weight decay

普通 L2 正则是把 `lambda * w` 加进梯度；AdamW 是把 weight decay 从 Adam 的自适应梯度里解耦出来：

```text
theta = theta - lr * AdamUpdate(theta)
theta = theta - lr * weight_decay * theta
```

**面试答案：AdamW 为什么常用于大模型？**  
AdamW 把权重衰减和自适应梯度更新解耦，训练更稳定，也更符合“直接衰减权重”的直觉。大模型训练和微调里，AdamW 是非常常见的默认优化器。

### 梯度爆炸、梯度消失和梯度裁剪

梯度消失：

- 深层网络反向传播时梯度越来越小。
- 早期层学得慢。
- RNN、sigmoid/tanh 深层网络更明显。

梯度爆炸：

- 梯度范数突然很大。
- loss 变 NaN 或训练发散。

常见处理：

- 残差连接。
- 归一化层。
- 合理初始化。
- 学习率 warmup。
- gradient clipping。
- 使用 ReLU/GELU/SwiGLU 等更稳的激活。

梯度裁剪：

```text
if ||g|| > threshold:
    g = g * threshold / ||g||
```

**面试答案：大模型训练为什么常用 gradient clipping？**  
大模型训练中偶发 batch、长序列或异常样本可能带来很大梯度。gradient clipping 能限制更新步长，防止训练突然发散，但它不能替代数据清洗和学习率调参。

## 5. 学习率

学习率太大：

- loss 震荡。
- 训练发散。

学习率太小：

- 收敛慢。
- 可能卡在不理想区域。

常见策略：

- warmup。
- cosine decay。
- step decay。
- linear decay。

**大模型为什么要 warmup？**  
训练初期参数和 optimizer 状态还不稳定，直接用大学习率可能发散。warmup 让学习率逐渐升高，训练更稳。

## 6. 过拟合和欠拟合

过拟合：

- 训练集好，验证集差。
- 模型记住训练数据，泛化差。

欠拟合：

- 训练集和验证集都差。
- 模型能力不足或训练不够。

过拟合解决：

- 更多数据。
- 数据增强。
- 正则化。
- dropout。
- early stopping。
- 减小模型。
- 清洗标签噪声。

**面试答案：大模型微调过拟合怎么办？**  
降低学习率和 epoch，增加/清洗数据，使用验证集 early stopping，降低 LoRA rank，加强正则，检查 loss mask 和重复样本，并做 bad case 人工评估。

## 7. L1 / L2 正则

L1：

```text
loss + lambda * |w|
```

倾向稀疏权重。

L2：

```text
loss + lambda * w^2
```

倾向小权重，常和 weight decay 相关。

**面试答案：L1 和 L2 区别？**  
L1 更容易产生稀疏解，适合特征选择；L2 会平滑地压小权重，减少模型复杂度，深度学习里更常见。

## 8. Dropout

训练时随机丢弃部分神经元，推理时使用完整网络。

直觉：

- 防止模型过度依赖某些特征。
- 类似训练多个子网络再集成。

**大模型里 dropout 为什么有时很小甚至不用？**  
大模型训练数据巨大，过拟合风险相对小；dropout 会影响训练效率和稳定性。很多现代 LLM 使用较小 dropout 或不用 dropout，但微调小数据时仍可能有帮助。

## 9. BatchNorm、LayerNorm、RMSNorm

BatchNorm：

- 沿 batch 统计均值方差。
- CV 中常见。
- 小 batch 或序列任务不稳定。

LayerNorm：

- 对单个样本的 hidden dimension 归一化。
- NLP/Transformer 常用。

RMSNorm：

- 不减均值，只按均方根缩放。
- 现代 LLM 常用。

**面试答案：为什么 Transformer 常用 LayerNorm 而不是 BatchNorm？**  
NLP 序列长度可变，batch 内样本差异大，训练和推理 batch 也可能不同。LayerNorm 对每个样本独立归一化，不依赖 batch 统计，更适合 Transformer。

## 10. 激活函数

Sigmoid：

- 输出 0-1。
- 容易梯度消失。

Tanh：

- 输出 -1 到 1。
- 也可能梯度消失。

ReLU：

- 简单高效。
- 负半轴为 0，可能 dead ReLU。

GELU：

- BERT/GPT 常见。
- 平滑，带概率门控直觉。

SiLU / Swish：

- `x * sigmoid(x)`。
- SwiGLU 中常见。

**面试答案：GELU 和 ReLU 区别？**  
ReLU 是硬截断，GELU 是平滑激活，会根据输入大小以概率式方式保留信息。Transformer 中 GELU/SwiGLU 更常见。

## 11. AUC

AUC 可以理解为：

> 随机抽一个正样本和一个负样本，模型给正样本打分高于负样本的概率。

优点：

- 与阈值无关。
- 适合排序型二分类。
- 对类别不均衡相对稳。

限制：

- 不反映具体阈值下 precision/recall。
- 不直接反映概率校准。

**面试答案：AUC 为什么适合排序？**  
AUC 只关心正负样本相对排序，不关心具体分数值，因此适合衡量模型把正样本排在负样本前面的能力。

## 12. Precision、Recall、F1

Precision：

```text
预测为正的里面有多少是真的正
```

Recall：

```text
真实为正的里面召回了多少
```

F1：

```text
2 * P * R / (P + R)
```

场景：

- 宁可少报也要准：看 precision。
- 宁可多报也别漏：看 recall。
- 需要平衡：看 F1。

## 13. NDCG / MRR

NDCG：

- 排序指标。
- 相关性高的结果排前面得分高。
- 用理想排序归一化。

MRR：

- 看第一个正确答案出现的位置。
- 第一个正确越靠前越好。

RAG 检索、搜索、推荐都可能问。

### 评估指标补充：PR-AUC、校准、BLEU、ROUGE

PR-AUC：

- 看 precision-recall 曲线下面积。
- 类别极不平衡时比 ROC-AUC 更敏感。
- 正样本很少的检测、风控、故障识别常用。

ROC-AUC vs PR-AUC：

| 指标 | 更关心什么 | 适合场景 |
| --- | --- | --- |
| ROC-AUC | 正负样本整体排序 | 类别不平衡不极端、排序能力 |
| PR-AUC | 正类检出质量 | 正样本稀少、误报/漏报敏感 |

概率校准：

- AUC 高不代表概率准。
- 模型可能排序很好，但输出 0.9 的样本真实命中率只有 0.6。
- 需要看 calibration curve、ECE、Brier score 等。

**面试答案：AUC 高但线上效果差可能为什么？**  
可能阈值没选好、概率未校准、训练/线上分布漂移、正样本定义变化、业务成本不对称，或者模型只会排序但无法满足 precision/recall 约束。

BLEU / ROUGE：

- BLEU 看生成文本和参考答案的 n-gram 精确匹配，机器翻译常见。
- ROUGE 看召回式重叠，摘要任务常见。

限制：

- 不能很好评价事实性、推理质量、帮助性和安全性。
- 对开放式问答和 RAG/Agent，只能作为辅助。

**面试答案：为什么 LLM 不只看 BLEU/ROUGE？**  
因为同一个问题可以有很多正确表达，n-gram 重叠低不代表答案差；反过来重叠高也可能事实错误。LLM 应用更需要人工评估、LLM-as-judge、任务成功率、faithfulness、引用正确率和业务指标。

## 14. Word2Vec

Word2Vec 让词变成向量。

CBOW：

- 用上下文预测中心词。

Skip-gram：

- 用中心词预测上下文。

直觉：

> 出现在相似上下文中的词，语义也相似。

**面试答案：Word2Vec 和 Transformer embedding 区别？**  
Word2Vec 是静态词向量，一个词通常一个向量；Transformer 里的 token 表示是上下文相关的，同一个词在不同句子里表示可以不同。

## 15. BPE / WordPiece / SentencePiece

目的：

- 避免纯词级词表太大。
- 避免字符级序列太长。
- 缓解 OOV。

BPE：

- 从字符开始，不断合并高频 pair。

WordPiece：

- 类似子词思想，合并标准和 likelihood 相关。

SentencePiece：

- 把文本当 unicode 序列，不依赖预分词，多语言友好。

**面试答案：为什么 LLM 用 subword tokenizer？**  
它在词表大小和序列长度之间折中，能处理未登录词、拼写变化、多语言和代码。

### BPE、WordPiece、SentencePiece 怎么区分

| 方法 | 常见模型 | 核心思路 | 面试抓手 |
| --- | --- | --- | --- |
| BPE | GPT 系常见 | 从字符/字节开始合并高频 pair | 简单高效，常用于 byte-level BPE |
| WordPiece | BERT 系常见 | 选择能提升似然的子词 | tokenization 时常用最长匹配 |
| SentencePiece | T5/LLaMA 等常见 | 把原始文本当序列处理，不强依赖空格预分词 | 多语言、中文、无空格语言更方便 |

词表大小 trade-off：

- 词表太小：序列变长，推理更慢，上下文占用多。
- 词表太大：embedding/LM head 参数变多，低频 token 学不好。

中文场景：

- 字、词、subword 都可能出现。
- SentencePiece 这类不依赖空格的方案更自然。
- 关键是 tokenizer 要和预训练模型一致，不能随便换。

**面试答案：为什么 tokenizer 不能随便改？**  
embedding 和 LM head 都和 token id 绑定。换 tokenizer 会改变输入切分和词表 id，原模型学到的表示不再对应，除非重新训练或做复杂适配。

## 16. RNN / LSTM / Transformer

RNN：

- 顺序处理。
- 难并行。
- 长距离依赖弱。

LSTM：

- 用门控缓解长期依赖问题。
- 仍然顺序计算。

Transformer：

- attention 直接连接任意位置。
- 训练可并行。
- 扩展性强。

**面试答案：Transformer 为什么替代 RNN？**  
Transformer 并行效率高，长距离依赖建模更直接，堆规模效果好。RNN 序列依赖强，训练慢，长程信息难保留。

## 17. BERT 和 GPT

BERT：

- Encoder-only。
- 双向上下文。
- Masked LM。
- 适合理解任务。

GPT：

- Decoder-only。
- causal mask。
- next token prediction。
- 适合生成任务。

**面试答案：BERT 为什么不适合直接生成？**  
BERT 训练时能双向看上下文，不符合自回归生成时只能看左侧的约束。可以改造，但天然不如 Decoder-only 适合生成。

### MLM、CLM、Prefix LM、Seq2Seq

MLM：

- Masked Language Modeling。
- 随机 mask 一些 token，让模型预测。
- BERT 代表。

CLM：

- Causal Language Modeling。
- 只能看左侧，预测下一个 token。
- GPT 代表。

Prefix LM：

- prefix 部分可以双向看。
- 生成部分只能看 prefix 和左侧已生成。
- 适合“给定上下文再生成”的折中形式。

Seq2Seq：

- encoder 读输入。
- decoder 自回归生成输出。
- T5、BART 代表。

**面试答案：Encoder-only、Decoder-only、Encoder-Decoder 怎么选？**  
理解/分类/检索表征任务，Encoder-only 很合适；开放生成和对话，Decoder-only 更主流；输入输出结构差异大、翻译/摘要等，Encoder-Decoder 很自然。现代大模型应用里 Decoder-only 因统一生成接口和扩展性成为主流。

## 18. 为什么 embedding 要乘 `sqrt(d_model)`

原始 Transformer 中会对 embedding 乘 `sqrt(d_model)`，让 embedding 尺度和 position encoding 更匹配，避免位置编码相对过强。

现代 LLM 实现不一定都这么做，但面试知道这个来源即可。

## 19. 常见追问快速答案

### Q1：交叉熵和 KL 散度关系？

```text
H(p, q) = H(p) + KL(p || q)
```

当真实分布 p 固定时，最小化交叉熵等价于最小化 KL。

### Q2：weight decay 和 L2 正则完全一样吗？

普通 SGD 下很接近；AdamW 把 weight decay 从梯度更新中解耦，实践中更稳定。

### Q3：类别不平衡怎么办？

重采样、类别权重、focal loss、阈值调整、看 PR-AUC/F1 而不是只看 accuracy。

### Q4：为什么 accuracy 不够？

类别不平衡时 accuracy 可能虚高。例如 99% 都是负样本，全预测负也有 99% accuracy，但模型没用。

### Q5：为什么要验证集？

训练集用于拟合参数，验证集用于调参和早停，测试集用于最终无偏评估。

### Q6：CrossEntropyLoss 里还要不要手动 softmax？

不要。常见框架里的交叉熵实现通常接收 logits，内部做 log-softmax 和 NLL，数值更稳定。

### Q7：PPL、BLEU、ROUGE、人工评估怎么选？

预训练语言建模看 PPL；翻译/摘要可以参考 BLEU/ROUGE；开放问答、RAG、Agent 要看人工评估、任务成功率、faithfulness、引用正确率和业务指标。

### Q8：AUC 和 PR-AUC 怎么选？

一般排序二分类可以看 ROC-AUC；正样本极少且更关心正类检出质量时，看 PR-AUC 更有意义。

### Q9：AdamW 的 weight decay 要不要作用在所有参数上？

通常不对 bias、LayerNorm/RMSNorm 的 scale 参数做 weight decay，只对主要权重矩阵做。这样能避免破坏归一化和偏置项的作用。

### Q10：tokenizer 词表越大越好吗？

不是。词表大能缩短序列，但会增加 embedding/LM head 参数和低频 token 学习难度。词表小参数少，但序列更长、推理成本更高。

### Q11：为什么 decoder-only LLM 也能做理解任务？

因为理解任务可以转成生成任务，例如“判断情感：正/负”。Decoder-only 通过 prompt 把分类、抽取、问答都统一成 next-token prediction。

## 20. 面试前背诵版

机器学习训练就是模型、损失和优化。分类常用交叉熵，因为它等价于最大化真实类别概率，softmax + CE 的梯度是 `p-y`。PyTorch 这类框架的 CrossEntropyLoss 通常输入 logits，不要先手动 softmax。Adam 用一阶和二阶动量自适应更新，AdamW 把 weight decay 和自适应梯度解耦；大模型训练常配 warmup 和 gradient clipping。过拟合可以用数据、正则、dropout、early stopping 和降低模型容量处理。Transformer 常用 LayerNorm/RMSNorm，因为它不依赖 batch 统计。AUC 衡量正样本排在负样本前面的概率，PR-AUC 更适合正样本稀少场景。BERT 是双向 Encoder，适合理解；GPT 是 causal Decoder，适合生成。LLM 用 subword tokenizer 是为了在词表大小、序列长度和 OOV 之间折中，tokenizer 不能随便换。
