# ML / DL 数学基础、损失函数、优化器与指标面试

这一章解决一个很实际的问题：你投的是大模型岗位，但面试官经常从 Transformer、SFT、RAG、Embedding 或项目指标，一路追到最基础的机器学习和深度学习数学。

你不需要把统计学习方法从头背一遍，但必须能把下面这些问题讲清楚：

- 交叉熵为什么等价于最大似然？
- 为什么 `CrossEntropyLoss` 接收 logits，而不是 softmax 后的概率？
- Adam、AdamW、SGD 的区别是什么？
- AUC、PR-AUC、F1、NDCG 分别适合什么场景？
- PPL 低为什么不代表模型一定好？
- loss 下降但线上效果不涨，怎么排查？
- LLM 训练里的 token loss、loss mask、packing、label smoothing、长度偏置怎么讲？

面试里最稳的表达方式是：**先说公式，再说直觉，再说适用场景，最后说坑和项目排查**。

## 1. 一句话框架：模型、损失、优化、指标

监督学习可以拆成四件事：

```text
输入 x
-> 模型 f_theta(x)
-> 损失 L(f_theta(x), y)
-> 优化器更新 theta
-> 指标评估业务目标
```

这四件事不能混：

| 概念 | 回答的核心问题 | 例子 |
| --- | --- | --- |
| 模型 | 怎么从输入得到输出 | Transformer、MLP、reranker |
| 损失 | 训练时优化什么 | CE、BCE、DPO、InfoNCE |
| 优化器 | 怎么更新参数 | SGD、Adam、AdamW |
| 指标 | 最终评价什么 | AUC、F1、NDCG、任务成功率 |

面试关键点：

- **loss 是训练信号，不一定等于业务指标。**
- **metric 是评估目标，不一定可导。**
- **优化器只负责沿着 loss 的梯度更新参数，不保证业务最优。**

30 秒答案：

> 训练可以看成模型、损失、优化器和指标四层。模型输出 logits 或分数，损失把预测和标签变成可导目标，优化器根据梯度更新参数，指标再从业务角度评价效果。很多问题出在 loss 和 metric 不一致，比如交叉熵下降但召回率、任务成功率或用户满意度没有提升。

## 2. Logits、概率和为什么不要乱 softmax

分类模型最后通常输出 logits：

```text
z = W h + b
```

logits 不是概率，可以是任意实数。softmax 把 logits 转成概率：

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

softmax 有平移不变性：

```text
softmax(z) = softmax(z - max(z))
```

所以实现里常减去最大值，避免 `exp` 溢出。

为什么很多 loss 接收 logits：

- 数值稳定：内部可以融合 `log_softmax` 和 NLL。
- 梯度稳定：避免手动 softmax 后再 log 造成极小概率下溢。
- 语义清晰：训练阶段用 logits，展示或阈值决策时才转概率。

30 秒答案：

> logits 是模型未归一化的分数，softmax 后才是概率。PyTorch 的 `CrossEntropyLoss` 这类实现通常直接吃 logits，因为内部会做稳定的 log-softmax 和 NLL，避免手动 softmax 带来的数值不稳定和重复计算。面试里要强调：训练传 logits，解释概率或做阈值决策时再转概率。

常见坑：

- `CrossEntropyLoss` 前手动加 softmax。
- 多标签任务错用 softmax CE。
- 二分类时搞混 `BCEWithLogitsLoss` 和 `BCELoss`。
- 指标计算时把 logits 当概率直接解释。

## 3. 交叉熵：从最大似然到梯度

多分类交叉熵：

```text
CE(y, p) = - sum_i y_i log p_i
```

如果标签是 one-hot：

```text
CE = - log p_true
```

### 3.1 为什么分类常用交叉熵

分类模型建模的是条件概率：

```text
p_theta(y | x)
```

最大似然训练是让真实标签概率尽可能大：

```text
max sum log p_theta(y | x)
```

把最大化变成最小化，就是负对数似然：

```text
min - sum log p_theta(y | x)
```

one-hot 分类下，这就是交叉熵。

30 秒答案：

> 分类用交叉熵，是因为它来自最大似然估计。模型输出每个类别的概率，训练时希望真实类别概率最大，取负 log 后就是交叉熵。它会强烈惩罚“错得很自信”的预测，而且 softmax 和 CE 合起来梯度很简洁，适合分类优化。

### 3.2 softmax + CE 的梯度

设：

```text
p_i = softmax(z)_i
L = - sum_i y_i log p_i
```

对 logit 求导：

```text
dL / dz_i = p_i - y_i
```

直觉：

- 正确类：如果 `p_true < 1`，`p_true - 1` 为负，梯度下降会把正确类 logit 往上推。
- 错误类：如果 `p_wrong > 0`，梯度为正，梯度下降会把错误类 logit 往下压。

面试加分点：

> softmax 和 CE 单独求导看起来复杂，但组合后梯度是 `p - y`。这也是分类里它比 MSE 更自然的原因之一。

### 3.3 交叉熵和 KL 散度

交叉熵：

```text
H(p, q) = - sum_x p(x) log q(x)
```

KL 散度：

```text
KL(p || q) = sum_x p(x) log(p(x) / q(x))
          = H(p, q) - H(p)
```

如果真实分布 `p` 固定，最小化交叉熵等价于最小化 `KL(p || q)`。

面试表达：

> 交叉熵衡量用模型分布 q 去编码真实分布 p 的代价。真实分布固定时，H(p) 是常数，所以最小化交叉熵等价于让模型分布接近真实分布。

## 4. LLM 里的 token loss、loss mask 和 PPL

自回归语言模型训练目标：

```text
p(x_1, ..., x_T) = product_t p(x_t | x_<t)
```

训练 loss：

```text
L = - sum_t log p_theta(x_t | x_<t)
```

实际常用平均 token loss：

```text
L_avg = - (1 / N) sum_{t in valid_tokens} log p_theta(x_t | x_<t)
```

这里的 `valid_tokens` 很重要。

### 4.1 loss mask

SFT 里常见两种 mask：

| 训练方式 | 哪些 token 算 loss | 风险 |
| --- | --- | --- |
| 全文 loss | system/user/assistant 都算 | 模型学会复述用户和 system |
| answer-only loss | 只算 assistant answer | 更符合指令微调目标 |

常见场景：

- padding token 不算 loss。
- prompt token 不算 loss。
- assistant answer 算 loss。
- 多轮对话要确保 mask 和 chat template 对齐。

30 秒答案：

> LLM 的训练 loss 本质是 next-token 交叉熵，但不是所有 token 都应该算。SFT 常用 answer-only loss，只让模型学习 assistant 回复，不让它学习复述用户输入。padding、被截断的无效 token、某些 system/user token 都要通过 loss mask 排除，否则 loss 数值和训练目标都会偏。

### 4.2 packing 为什么容易出 bug

packing 是把多条短样本拼进一个长序列，提高训练效率。

风险：

- 样本之间 attention mask 没隔开，发生跨样本泄漏。
- loss mask 错位，prompt token 被算进 loss。
- position id 处理不一致。
- 统计平均 loss 时分母用错。

面试表达：

> packing 提升吞吐，但一定要检查 attention mask、loss mask、position id 和样本边界。否则模型可能看到不该看的 token，或者 loss 看起来下降但训练目标已经错了。

### 4.3 PPL

困惑度：

```text
PPL = exp(平均 token cross entropy)
```

PPL 低说明模型对下一个 token 更不困惑，但不等价于真实应用好。

PPL 不覆盖：

- 指令遵循。
- 多步推理。
- 事实性。
- 安全性。
- 工具调用成功率。
- RAG groundedness。
- 用户偏好。

30 秒答案：

> PPL 是平均 token 交叉熵的指数形式，衡量语言建模能力。PPL 低不代表模型一定好，因为它主要看 next-token prediction，不直接衡量指令遵循、事实性、推理、安全和业务任务成功率。大模型项目里 PPL 只能作为基础参考，还要配合私有 eval、人工偏好和线上指标。

## 5. BCE、CE、MSE：别把任务类型搞混

| Loss | 适用任务 | 输出层 | 典型输入 |
| --- | --- | --- | --- |
| CrossEntropyLoss | 单标签多分类 | softmax | logits `[B, C]` + class id |
| BCEWithLogitsLoss | 二分类或多标签 | sigmoid | logits `[B]` 或 `[B, C]` + 0/1 labels |
| MSELoss | 回归 | 无固定要求 | continuous value |

### 5.1 BCEWithLogitsLoss

二分类 BCE：

```text
BCE = - y log p - (1 - y) log(1 - p)
```

`BCEWithLogitsLoss` 把 sigmoid 和 BCE 合在一起，数值更稳定。

适合：

- 二分类。
- 多标签分类，每个标签独立为 0/1。

不适合：

- 单标签多分类。单标签多分类应该用 softmax CE。

### 5.2 CE vs BCE

单标签多分类：

```text
一张图只能属于 cat/dog/bird 中一个
-> CrossEntropyLoss
```

多标签分类：

```text
一篇文章可以同时属于 finance、risk、policy
-> BCEWithLogitsLoss
```

30 秒答案：

> CE 用在互斥类别，多分类时 softmax 后所有类别概率和为 1。BCE 用在二分类或多标签，每个标签独立做 sigmoid。MSE 更适合回归。面试里要先判断标签是否互斥，再选 loss。

### 5.3 CE vs MSE

分类里 MSE 的问题：

- 和最大似然不如 CE 直接。
- softmax 饱和时梯度可能不理想。
- 对概率分布的建模不如 CE 自然。

但 MSE 不是没用：

- 回归任务。
- 某些蒸馏或 embedding 对齐。
- value/reward 预测。

## 6. Label smoothing、类别不均衡和 Focal Loss

### 6.1 Label smoothing

one-hot：

```text
y = [0, 0, 1, 0]
```

平滑后：

```text
y_smooth = (1 - eps) * y + eps / C
```

作用：

- 减少过度自信。
- 对噪声标签更稳。
- 有时提升泛化。

风险：

- 可能降低极强判别边界。
- 可能影响概率校准。
- 生成任务中过强 smoothing 可能让模型输出变软。

### 6.2 类别不均衡

类别不均衡时，accuracy 往往骗人。

例子：

```text
10000 个样本，只有 50 个正例
模型全预测负例，accuracy = 99.5%
但召回率 = 0
```

处理方式：

- class weight。
- over-sampling / under-sampling。
- focal loss。
- 调整 threshold。
- 用 PR-AUC、Recall@K、F1 等指标。

### 6.3 Focal Loss

Focal Loss 常见形式：

```text
FL = - alpha * (1 - p_t)^gamma * log(p_t)
```

直觉：

- 简单样本 `p_t` 高，权重变小。
- 困难样本 `p_t` 低，权重更大。

面试表达：

> 类别极不均衡时，CE 容易被大量简单负样本主导。Focal Loss 通过降低简单样本权重，让训练更关注难例和少数类。但它不是万能的，仍要看业务指标、阈值和数据采样。

## 7. 对比学习和 InfoNCE

Embedding、reranker、多模态 CLIP 里常见对比学习。

核心思想：

```text
正样本相似度更高
负样本相似度更低
```

InfoNCE 常见形式：

```text
L_i = - log exp(sim(q_i, d_i+) / tau)
          / sum_j exp(sim(q_i, d_j) / tau)
```

其中：

- `q_i` 是 query。
- `d_i+` 是正样本。
- `d_j` 包含正样本和负样本。
- `tau` 是 temperature。

面试追问：

- in-batch negatives 怎么来？
- hard negative 为什么重要？
- temperature 太大太小有什么影响？
- false negative 怎么处理？

30 秒答案：

> InfoNCE 本质也是一个 softmax 分类目标：给定 query，从一批候选里把正样本分出来。它会拉近正样本、推远负样本，常用于 embedding、CLIP 和检索模型训练。难点在负样本质量，in-batch negative 高效但可能有 false negative，hard negative 有帮助但太难或标错会伤模型。

## 8. 优化器：SGD、Momentum、Adam、AdamW

### 8.1 SGD

基本更新：

```text
theta_t = theta_{t-1} - lr * g_t
```

优点：

- 简单。
- 泛化有时好。
- 状态少，省显存。

缺点：

- 收敛慢。
- 对学习率敏感。
- 不同参数共享同一个全局步长。

### 8.2 Momentum

```text
v_t = beta * v_{t-1} + g_t
theta_t = theta_{t-1} - lr * v_t
```

直觉：

- 梯度方向稳定时加速。
- 梯度来回震荡时平滑。

### 8.3 Adam

Adam 维护一阶和二阶动量：

```text
g_t = grad(theta_t)
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

解释：

- `m_t`：梯度方向的指数滑动平均。
- `v_t`：梯度平方的指数滑动平均。
- `m_hat / v_hat`：修正初期从 0 开始的偏差。
- 分母大时步子小，分母小时步子大。

30 秒答案：

> SGD 用当前梯度直接更新，Adam 同时维护一阶动量和二阶动量，对每个参数自适应调整步长，所以深度学习里通常收敛更快、更稳。但 Adam 状态更占显存，也不是必然泛化更好。大模型训练和微调里 AdamW 更常见。

### 8.4 AdamW

普通 L2 正则把 `lambda * theta` 加进梯度。AdamW 把 weight decay 从 Adam 的自适应梯度里解耦：

```text
theta = theta - lr * AdamUpdate(theta)
theta = theta - lr * weight_decay * theta
```

更准确写法里，衰减项和梯度自适应项是两条路径。

为什么重要：

- Adam 的自适应缩放会改变 L2 正则效果。
- 解耦后 weight decay 更像直接衰减权重。
- 大模型训练、SFT、LoRA 微调里很常见。

30 秒答案：

> AdamW 的核心是 decoupled weight decay。普通 Adam 加 L2 时，正则项也会进入 Adam 的自适应缩放，效果不等同于直接衰减权重。AdamW 把权重衰减从梯度更新里解耦出来，训练更稳定，也更容易调参，所以大模型里常作为默认优化器。

### 8.5 哪些参数通常不做 weight decay

常见不衰减：

- bias。
- LayerNorm / RMSNorm 的 scale 参数。
- embedding 有时也单独处理。

原因：

- 这些参数不是主要的矩阵权重。
- 对 norm 参数做 decay 可能破坏归一化层稳定性。

面试表达：

> 实践中常对线性层权重做 decay，对 bias 和 norm 参数不做。因为 bias 和 norm scale 的作用不同，直接衰减可能影响稳定性。具体还要看框架默认配置和实验结果。

## 9. 学习率、warmup、cosine、batch 和梯度裁剪

### 9.1 学习率

学习率太大：

- loss 震荡。
- 发散。
- 出现 NaN。

学习率太小：

- 收敛慢。
- 卡在较差区域。
- 训练成本高。

### 9.2 warmup

大模型常用 warmup：

```text
前若干 step 学习率从 0 线性升到目标值
```

原因：

- 初期参数和 optimizer state 不稳定。
- 大 batch、大模型直接上大学习率容易发散。
- warmup 给 Adam 的动量估计一个稳定过程。

### 9.3 cosine decay

```text
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

作用：

- 前期保持较大学习率探索。
- 后期逐渐变小，稳定收敛。

### 9.4 batch size 和 gradient accumulation

global batch：

```text
global_batch = micro_batch * data_parallel_size * grad_accum_steps
```

gradient accumulation：

- 每个 micro batch 反向传播累积梯度。
- 累积若干次后再 optimizer step。
- 用时间换显存。

风险：

- 学习率和 batch size 要匹配。
- 梯度累积改变 optimizer step 频率。
- logging loss 要区分 micro step 和 update step。

### 9.5 梯度消失、爆炸和 clipping

梯度裁剪：

```text
if ||g|| > threshold:
    g = g * threshold / ||g||
```

作用：

- 限制更新步长。
- 防止偶发 batch 导致训练发散。

不能解决：

- 数据脏。
- 学习率过大。
- loss mask 错。
- 混合精度溢出。

30 秒答案：

> 大模型训练常用 warmup、cosine decay 和 gradient clipping。warmup 让初期训练更稳，cosine 让后期逐步收敛，clipping 限制异常梯度导致的大更新。但如果 loss mask、数据或学习率错了，clipping 只能缓解，不能根治。

## 10. 指标：Accuracy、Precision、Recall、F1

二分类混淆矩阵：

| | 预测正 | 预测负 |
| --- | --- | --- |
| 真实正 | TP | FN |
| 真实负 | FP | TN |

公式：

```text
Accuracy = (TP + TN) / (TP + FP + FN + TN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

怎么选：

- 更怕误报：看 precision。
- 更怕漏报：看 recall。
- 正负样本均衡且错误代价接近：accuracy 可以参考。
- 正负不均衡：accuracy 往往不够。
- 想平衡 precision 和 recall：F1。

面试例子：

| 场景 | 更关注 |
| --- | --- |
| 风险拦截 | recall，不能漏太多 |
| 自动封禁 | precision，不能误伤用户 |
| 医疗初筛 | recall |
| 高置信自动回复 | precision |
| 客服召回候选 | recall@K |

30 秒答案：

> Precision 看预测为正的样本里有多少是真的，Recall 看真实正例里有多少被找到了。F1 是二者调和平均。类别不均衡时 accuracy 容易骗人，要结合业务代价选 precision、recall、F1 或 PR-AUC。

## 11. ROC-AUC、PR-AUC、阈值和排序指标

### 11.1 ROC-AUC

ROC：

- x 轴：FPR
- y 轴：TPR，也就是 Recall

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

AUC 的概率解释：

> 随机抽一个正样本和一个负样本，模型给正样本打分高于负样本的概率。

适合：

- 评价排序能力。
- 阈值尚未固定。
- 正负样本不是极端不均衡时。

### 11.2 PR-AUC

PR 曲线：

- x 轴：Recall
- y 轴：Precision

适合：

- 正例稀少。
- 更关心正类检出质量。

30 秒答案：

> ROC-AUC 衡量正样本排在负样本前面的能力，可以理解成随机正负样本对里正样本得分更高的概率。PR-AUC 更关注正类，在正例稀少时更敏感。AUC 是排序指标，不直接告诉你阈值怎么选，最终上线还要根据业务成本选 threshold。

### 11.3 阈值

模型输出分数后，阈值决定最终分类：

```text
score >= threshold -> positive
score < threshold -> negative
```

阈值升高：

- precision 通常升高。
- recall 通常降低。

阈值降低：

- recall 通常升高。
- precision 通常降低。

上线阈值怎么选：

- 根据误报和漏报成本。
- 根据人工审核容量。
- 根据业务 SLA。
- 根据验证集和线上 A/B。

### 11.4 NDCG、MRR、Recall@K

检索和排序常用：

```text
Recall@K = 前 K 个结果是否召回相关文档
MRR = 1 / 第一个相关结果的排名
NDCG = 考虑相关性等级和排序位置的增益
```

RAG 里常见：

- Recall@K：gold evidence 是否被召回。
- NDCG/MRR：相关证据是否排在前面。
- Faithfulness：答案是否被证据支持。
- Citation accuracy：引用是否准确。

## 12. 概率校准：模型自信不等于真的准

校准问题：

```text
模型说 0.8 概率的样本，真实正确率是否接近 80%
```

如果模型经常说 0.9，但实际只有 0.6，就过度自信。

常见方法：

- reliability diagram。
- ECE，Expected Calibration Error。
- Brier score。
- temperature scaling。

temperature scaling：

```text
p = softmax(z / T)
```

`T > 1`：

- 分布变平。
- 置信度降低。

`T < 1`：

- 分布变尖。
- 置信度升高。

面试表达：

> 校准关注概率是否可信，不只是排序是否正确。AUC 高说明排序好，但不代表 0.8 的概率真的有 80% 正确率。温度缩放可以在验证集上调整 logits 的尺度，改善置信度校准，但不会改变样本排序。

LLM 项目里的校准：

- 分类器/Reranker 置信度。
- 是否自动执行高风险工具。
- 是否进入人工审核。
- RAG 答案置信度提示。
- 多模型路由阈值。

## 13. loss 下降但线上效果不涨，怎么排查

这是非常高频的项目追问。

排查顺序：

1. **数据是否一致**：训练集、验证集、线上分布是否漂移。
2. **label 是否可靠**：脏标签、偏好标注冲突、自动标注噪声。
3. **loss 是否对齐业务**：CE 降了，但 F1、NDCG、任务成功率没涨。
4. **mask 是否正确**：padding、prompt、answer、packing 边界。
5. **指标是否选错**：正负不均衡只看 accuracy。
6. **验证集是否污染**：训练数据泄漏或 benchmark 污染。
7. **阈值是否更新**：模型分数分布变了，但阈值没重调。
8. **线上链路是否一致**：template、tokenizer、retrieval、prompt、模型版本不同。
9. **bad case 是否集中**：长尾、特定人群、特定 query 类型。

30 秒答案：

> 我不会只看 loss。我会先确认训练和线上数据分布、label 质量和 loss mask，再看 loss 是否真的对应业务指标。比如 CE 下降可能只是常见样本更准，但少数类 recall 没变；reranker loss 下降也可能 NDCG 没涨。最后要看阈值、线上链路一致性和 bad case 分布。

## 14. 这章和大模型八股怎么连起来

| 大模型问题 | 会追到的基础 |
| --- | --- |
| SFT 为什么只算 answer loss | token CE、loss mask |
| DPO 为什么看 logprob 差 | sequence logprob、KL、sigmoid loss |
| RAG reranker 怎么训练 | CE、BCE、InfoNCE、pairwise/listwise |
| 模型选型为什么不能只看 PPL | PPL 和任务指标不一致 |
| Agent eval 怎么设计 | metric 和业务成功标准 |
| safety classifier 怎么调阈值 | precision/recall、校准、阈值成本 |
| embedding 为什么要 hard negative | 对比学习、InfoNCE |
| 训练 NaN 怎么排查 | 学习率、梯度爆炸、混合精度、脏数据 |

面试时你可以这样串：

> 大模型训练底层还是概率建模和梯度优化。预训练和 SFT 主要是 token-level CE；偏好优化会进一步比较 sequence logprob；Embedding 和 reranker 会用对比学习、CE 或 pairwise loss；上线评价则不能只看 loss，要看任务成功率、faithfulness、NDCG、F1、成本和延迟。

## 15. 高频追问清单

1. 交叉熵为什么来自最大似然？
2. softmax + CE 的梯度为什么是 `p - y`？
3. 为什么 `CrossEntropyLoss` 接收 logits？
4. BCE、CE、MSE 怎么选？
5. 多分类和多标签有什么区别？
6. label smoothing 有什么好处和风险？
7. PPL 低为什么不等于大模型好？
8. SFT 的 loss mask 怎么做？
9. packing 会带来哪些训练 bug？
10. Adam、SGD、AdamW 怎么区分？
11. AdamW 为什么叫 decoupled weight decay？
12. 哪些参数通常不做 weight decay？
13. warmup 和 cosine decay 解决什么？
14. gradient clipping 能解决训练发散吗？
15. AUC 的概率解释是什么？
16. PR-AUC 什么时候比 ROC-AUC 更重要？
17. F1、precision、recall 怎么按业务选择？
18. 什么是概率校准？
19. temperature scaling 改变排序吗？
20. loss 降了但线上指标没涨怎么排查？

## 16. 8 分钟项目讲法模板

如果面试官让你讲一个“训练/评估/分类/rerank/安全识别”项目，可以按这个结构：

1. **任务和业务代价**：正负样本是什么，误报和漏报哪个更贵。
2. **数据和标签**：数据来源、清洗、标注、一致性、类别分布。
3. **模型输出**：logits、概率、embedding 相似度或排序分数。
4. **loss 选择**：CE/BCE/InfoNCE/pairwise，为什么适合这个任务。
5. **优化细节**：AdamW、学习率、warmup、batch、weight decay、clipping。
6. **指标选择**：AUC、PR-AUC、F1、NDCG、Recall@K、业务指标。
7. **阈值和校准**：验证集调阈值，校准置信度，控制人工审核量。
8. **bad case 和上线**：错误类型、分布漂移、回归评估、监控和灰度。

最后收一句：

> 这个项目里我没有只追训练 loss，而是把 loss、验证指标、阈值、业务成本和线上监控连起来看。

## 17. 推荐阅读

- PyTorch CrossEntropyLoss：[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- PyTorch BCEWithLogitsLoss：[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- PyTorch AdamW：[https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- Adam 论文：[https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- AdamW 论文：[https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- scikit-learn 模型评估指标：[https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
- 现代神经网络校准论文：[https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)
- Hugging Face Tokenizers Course：[https://huggingface.co/learn/llm-course/chapter6/5](https://huggingface.co/learn/llm-course/chapter6/5)

## 18. 本章复习顺序

第一遍只抓主线：

1. CE、softmax、logits。
2. LLM token loss、loss mask、PPL。
3. Adam、AdamW、warmup、clipping。
4. Precision、Recall、F1、AUC、PR-AUC。
5. 校准和 loss/metric 不一致排查。

第二遍开始背题：

- 先背 481-488：损失函数。
- 再背 489-491：指标和校准。
- 再背 492-497：优化器和训练稳定性。
- 最后背 498-500：对比学习、项目排查和项目讲法。
