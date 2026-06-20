# 答案版 23：ML / DL 数学基础、损失函数、优化器与指标

对应题号：481-500。建议先读 [28_ML_DL数学基础_损失函数优化器与指标面试.md](../28_ML_DL数学基础_损失函数优化器与指标面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 481. 交叉熵为什么可以从最大似然推出来？

30 秒版：

交叉熵来自最大似然。分类模型输出 `p_theta(y|x)`，训练时希望真实标签的概率尽可能大，也就是最大化 `sum log p_theta(y|x)`。把最大化变成最小化，就是 `-sum log p_theta(y|x)`，one-hot 标签下正好等价于交叉熵。

2 分钟版：

分类任务里，模型不是只输出一个类别，而是在建模条件概率分布：

```text
p_theta(y | x)
```

最大似然估计的目标是让训练集中真实标签出现的概率最大：

```text
max_theta sum_i log p_theta(y_i | x_i)
```

优化时习惯最小化 loss，所以取负号：

```text
min_theta - sum_i log p_theta(y_i | x_i)
```

如果标签是 one-hot，交叉熵是：

```text
CE = - sum_c y_c log p_c
```

只有真实类别那一项 `y_c = 1`，所以：

```text
CE = - log p_true
```

也就是负对数似然。

面试补一句：

> 所以 CE 不是凭空选的 loss，它和概率建模、最大似然是一致的。

容易踩坑：

- 只说“分类都用 CE”，但说不出为什么。
- 不区分 one-hot 标签和 soft label。
- 忘记 CE 和 KL 的关系：真实分布固定时，最小化 CE 等价于最小化 KL。

## 482. 为什么 PyTorch CrossEntropyLoss 输入 logits，而不是 softmax 后的概率？

30 秒版：

因为它内部会把 `log_softmax` 和 NLL 合在一起做，数值更稳定。直接传 logits 可以避免手动 softmax 后概率太小再取 log 导致下溢，也避免重复计算。训练时传 logits，解释概率或算阈值时再 softmax。

2 分钟版：

logits 是未归一化分数：

```text
z = W h + b
```

softmax 后才是概率：

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵需要：

```text
- log p_true
```

如果先手动 softmax，再 log，极端情况下 `p_true` 可能非常小，数值不稳定。框架实现通常会做稳定的 `log_softmax`：

```text
log_softmax(z_i) = z_i - logsumexp(z)
```

`logsumexp` 可以用减最大值技巧避免 `exp` 溢出。

项目里怎么说：

> 我训练时会保留 logits 给 loss，评估时再根据任务需要转 probability。二分类或多标签则用 `BCEWithLogitsLoss`，同样是把 sigmoid 和 BCE 融合起来。

容易踩坑：

- `CrossEntropyLoss(softmax(logits), labels)`。
- `BCEWithLogitsLoss(sigmoid(logits), labels)`。
- 混淆 logits、概率和最终 hard label。

## 483. softmax + cross entropy 的梯度为什么是 `p - y`？

30 秒版：

设 `p=softmax(z)`，`L=-sum y_i log p_i`，对 logits 求导后可以化简成 `dL/dz_i = p_i - y_i`。直觉是：正确类概率低时梯度会把正确 logit 往上推，错误类概率高时梯度会把错误 logit 往下压。

2 分钟版：

softmax：

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

CE：

```text
L = - sum_i y_i log p_i
```

两者组合求导，结果非常简洁：

```text
dL / dz_i = p_i - y_i
```

如果是 one-hot 标签：

- 正确类：`y_i = 1`，梯度 `p_i - 1`，通常为负。梯度下降会增加这个 logit。
- 错误类：`y_i = 0`，梯度 `p_i`，为正。梯度下降会降低这些 logits。

这个结果说明 CE 的梯度直接反映“预测概率和标签分布的差距”。

面试加分：

> 这也是分类里 softmax + CE 比 softmax + MSE 更自然的原因之一，梯度信号更直接。

注意：

- 如果标签是 soft label，`y` 不是 one-hot，但梯度仍是 `p - y`。
- 多标签 BCE 不是 softmax，而是每个标签独立 sigmoid。

## 484. CE、BCE、MSE 分别适合什么任务？

30 秒版：

CE 用于单标签多分类，类别互斥；BCE 用于二分类或多标签，每个标签独立为 0/1；MSE 用于回归，预测连续值。先判断标签是否互斥，再选 loss。

2 分钟版：

对比表：

| Loss | 任务 | 输出解释 |
| --- | --- | --- |
| CrossEntropyLoss | 单标签多分类 | softmax 后所有类别概率和为 1 |
| BCEWithLogitsLoss | 二分类 / 多标签 | 每个标签独立 sigmoid |
| MSELoss | 回归 | 连续值误差 |

例子：

- 图片只能是猫、狗、鸟之一：CE。
- 一篇新闻可以同时是金融、政策、风险：BCE。
- 预测房价、评分、reward value：MSE。

为什么不要乱用：

- 多标签任务用 CE，会强迫标签互斥。
- 单标签多分类用 BCE，类别间竞争关系弱。
- 分类任务用 MSE，概率建模和梯度性质通常不如 CE。

LLM 相关：

- next-token prediction：CE。
- safety 多标签分类：BCE。
- reward model 分数回归：可能 MSE 或 pairwise loss。
- embedding 检索：常用 InfoNCE、triplet 或 pairwise ranking loss。

## 485. Label smoothing 有什么好处和风险？

30 秒版：

Label smoothing 是把 one-hot 标签变软，比如真实类从 1 变成 `1-eps`，其他类分到 `eps/C`。好处是减少过度自信、缓解噪声标签、提升泛化。风险是可能降低边界判别能力，影响校准或生成质量。

2 分钟版：

one-hot：

```text
[0, 0, 1, 0]
```

label smoothing：

```text
y_smooth = (1 - eps) * y + eps / C
```

直觉：

- 不让模型把真实类别概率推到绝对 1。
- 给模型留一点不确定性。
- 对错误标注更鲁棒。

适合：

- 标签有噪声。
- 模型过度自信。
- 分类泛化不佳。

风险：

- 如果任务需要强置信边界，可能伤性能。
- 可能让模型概率变得过平。
- 生成模型里过强 smoothing 可能影响 next-token 分布。

面试表达：

> 我会把它当正则化手段，而不是默认必开。是否使用要看验证集、校准、业务指标和 bad case。

## 486. PPL 低是不是模型一定好？

30 秒版：

不一定。PPL 是平均 token 交叉熵的指数形式，衡量模型对下一个 token 的困惑程度。它能反映语言建模能力，但不直接衡量指令遵循、事实性、推理、安全、工具调用和业务任务成功率。

2 分钟版：

公式：

```text
PPL = exp(平均 token CE)
```

PPL 低说明模型更容易预测验证文本里的下一个 token。

但大模型好不好还要看：

- instruction following。
- factuality。
- reasoning。
- safety。
- tool calling。
- RAG faithfulness。
- user preference。
- latency 和 cost。

例子：

一个模型在通用语料上 PPL 很低，但可能不会遵循企业客服格式；另一个模型 PPL 一般，但经过 SFT 和偏好优化后更适合业务。

面试结论：

> PPL 是基础指标，不是最终指标。预训练阶段可以看 PPL，SFT/RAG/Agent 项目还要看私有 eval、人工偏好、任务成功率和线上指标。

## 487. LLM SFT 里的 loss mask、padding、packing 要注意什么？

30 秒版：

SFT 的 token loss 不是所有 token 都算。padding 不算，很多场景 user/system prompt 不算，只算 assistant answer。packing 能提高吞吐，但要处理 attention mask、loss mask、position id 和样本边界，否则可能跨样本泄漏或 loss 错位。

2 分钟版：

自回归 LM 的 loss：

```text
L = - sum_t log p(x_t | x_<t)
```

实际训练常是：

```text
L = - (1 / N_valid) sum_{t in valid_tokens} log p(x_t | x_<t)
```

`valid_tokens` 由 loss mask 决定。

常见规则：

- padding token 不算。
- prompt token 通常不算。
- assistant answer 算。
- 被截断或无效 token 不算。

packing 风险：

- attention mask 没隔开，样本 A 看到样本 B。
- loss mask 错位，把 user prompt 算进 loss。
- position id 不符合训练假设。
- 平均 loss 分母错误。

项目排查：

> 如果 SFT loss 很好但模型学会复述用户，或回答格式异常，我会优先查 chat template 和 loss mask。

## 488. 类别不均衡时怎么选 loss 和指标？

30 秒版：

类别不均衡时 accuracy 容易骗人，要看 precision、recall、F1、PR-AUC、Recall@K 等。loss 上可以用 class weight、重采样或 focal loss；上线还要根据误报和漏报成本调 threshold。

2 分钟版：

极端例子：

```text
10000 个样本，50 个正例
全预测负例，accuracy = 99.5%
但 recall = 0
```

所以要先问业务代价：

- 漏掉正例代价高：提高 recall。
- 误报代价高：提高 precision。
- 两者都重要：看 F1。
- 正例稀少：PR-AUC 比 ROC-AUC 更敏感。

loss 侧：

- class weight：提高少数类权重。
- over-sampling / under-sampling。
- focal loss：降低简单样本权重。
- hard negative mining。

上线侧：

- 用验证集调 threshold。
- 看不同阈值下的 precision/recall trade-off。
- 结合人工审核容量或业务成本。

面试表达：

> 不均衡问题不是只改 loss，还要改指标、采样、阈值和 bad case 分析。

## 489. AUC、PR-AUC、Precision、Recall、F1 怎么解释？

30 秒版：

Precision 是预测为正里有多少真是正，Recall 是真实正例里找回了多少，F1 是二者调和平均。ROC-AUC 衡量排序能力，可以理解成随机正负样本对里正样本得分更高的概率。PR-AUC 在正例稀少时更有参考价值。

2 分钟版：

公式：

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2PR / (P + R)
```

ROC：

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

ROC-AUC 概率解释：

> 随机抽一个正样本和负样本，模型给正样本打分更高的概率。

PR 曲线：

- x 轴：Recall。
- y 轴：Precision。

什么时候看什么：

- 风险识别：可能重 recall。
- 自动拦截/封禁：重 precision。
- 正例很少：重 PR-AUC。
- 排序能力：AUC、NDCG、MRR。

项目表达：

> 我会根据业务代价选主指标，不会只报 accuracy。

## 490. 阈值指标和排序指标有什么区别？

30 秒版：

AUC、NDCG、MRR 这类排序指标看分数排序好不好，不依赖固定阈值；precision、recall、F1 通常依赖阈值。AUC 高不代表某个阈值下业务效果好，上线前还要根据误报漏报成本调 threshold。

2 分钟版：

排序指标回答：

```text
模型能不能把好结果排在前面
```

例如：

- ROC-AUC：正样本是否排在负样本前。
- NDCG：相关结果是否排在更靠前位置。
- MRR：第一个相关结果出现得早不早。
- Recall@K：前 K 个是否召回目标。

阈值指标回答：

```text
分数超过某个 threshold 后，最终决策效果如何
```

例如：

- precision。
- recall。
- F1。
- false positive rate。
- manual review volume。

上线流程：

1. 先用排序指标确认模型区分能力。
2. 再在验证集选 threshold。
3. 看阈值下的业务代价。
4. 灰度观察线上分布和回归。

面试加分：

> 温度缩放能改善概率校准，但不改变排序；调阈值会改变 precision/recall，但不改变模型本身排序能力。

## 491. 什么是概率校准？temperature scaling 有什么用？

30 秒版：

校准看模型置信度是否可信。比如模型预测 0.8 的样本，真实正确率是否接近 80%。Temperature scaling 用 `softmax(z/T)` 调整分布尖锐程度，常在验证集上学习 T，改善置信度校准，但一般不改变排序。

2 分钟版：

校准问题：

```text
confidence = 0.8
实际正确率是否约等于 0.8
```

过度自信：

- 模型经常输出 0.9。
- 实际正确率只有 0.6。

欠自信：

- 模型输出 0.6。
- 实际正确率 0.8。

temperature scaling：

```text
p = softmax(z / T)
```

`T > 1`：

- logits 变小。
- 分布变平。
- 置信度降低。

`T < 1`：

- 分布变尖。
- 置信度升高。

项目用途：

- 高风险自动工具调用。
- 安全分类器阈值。
- RAG 答案置信度。
- 多模型路由。
- 人工审核分流。

注意：

> 校准不是提升模型排序能力，而是让分数更可信。

## 492. Adam 和 SGD 有什么区别？

30 秒版：

SGD 用当前梯度和全局学习率更新，简单、省状态，但收敛可能慢。Adam 维护一阶动量和二阶动量，对每个参数自适应调整步长，通常收敛更快、更稳，但状态更占显存，泛化也不一定总比 SGD 好。

2 分钟版：

SGD：

```text
theta = theta - lr * g
```

Adam：

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

区别：

| 维度 | SGD | Adam |
| --- | --- | --- |
| 更新 | 当前梯度 | 一阶 + 二阶动量 |
| 学习率 | 全局为主 | 参数自适应 |
| 收敛 | 可能慢 | 通常快 |
| 状态 | 少 | 多 |
| 大模型 | 少见作为默认 | AdamW 常见 |

面试表达：

> Adam 的二阶动量不是严格 Hessian，而是梯度平方的滑动平均，用来调节每个参数的有效步长。

## 493. AdamW 和 Adam + L2 正则有什么区别？

30 秒版：

AdamW 的核心是 decoupled weight decay。Adam + L2 会把正则项加进梯度，再被 Adam 的自适应缩放处理；AdamW 把权重衰减从梯度更新中解耦，直接对权重做衰减，更符合 weight decay 的本意，也更容易调参。

2 分钟版：

普通 L2：

```text
g = grad(loss) + lambda * theta
```

这在 SGD 里和 weight decay 接近，但在 Adam 里，`lambda * theta` 会进入自适应分母，被不同参数的 `v_t` 缩放，效果不再是简单衰减权重。

AdamW：

```text
theta = theta - lr * AdamUpdate(grad)
theta = theta - lr * weight_decay * theta
```

也就是把优化 loss 的梯度更新和衰减权重分开。

为什么大模型常用：

- 训练稳定。
- 调参直观。
- AdamW 是很多预训练、SFT、LoRA 微调的默认选择。

补充：

> 实践中常对 linear weight 做 decay，对 bias、LayerNorm/RMSNorm 参数不做 decay。

## 494. warmup 和 cosine decay 分别解决什么问题？

30 秒版：

warmup 解决训练初期不稳定的问题，让学习率从小逐渐升高，避免一开始大步更新导致发散。cosine decay 是后期逐渐降低学习率，让训练从探索转向稳定收敛。

2 分钟版：

warmup：

```text
lr 从 0 或很小值线性升到 lr_max
```

原因：

- 初期 optimizer state 还不稳定。
- 大模型和大 batch 对学习率敏感。
- 直接大学习率可能 loss 爆炸。

cosine decay：

```text
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

直觉：

- 前期学习率较大，快速学习。
- 后期学习率变小，稳定收敛。

面试表达：

> warmup 是为了起步稳，decay 是为了收口稳。它们经常和 AdamW、gradient clipping、mixed precision 一起构成大模型训练的稳定性组合。

## 495. 梯度消失、梯度爆炸和 gradient clipping 怎么讲？

30 秒版：

梯度消失是反向传播到前面层时梯度越来越小，前层学不动；梯度爆炸是梯度范数异常大，导致 loss 震荡、NaN 或发散。gradient clipping 会把梯度范数限制在阈值内，防止偶发大更新，但不能替代数据清洗、学习率调参和 mask 检查。

2 分钟版：

梯度消失原因：

- 深层链式求导连乘。
- sigmoid/tanh 饱和。
- 初始化或归一化不合适。

缓解：

- residual connection。
- LayerNorm/RMSNorm。
- ReLU/GELU/SwiGLU。
- 合理初始化。

梯度爆炸表现：

- grad norm 突增。
- loss 变 NaN。
- 参数更新过大。

gradient clipping：

```text
if ||g|| > c:
    g = g * c / ||g||
```

大模型训练里的排查：

- learning rate。
- batch 脏数据。
- mixed precision overflow。
- loss mask 错。
- reward 或 label 异常。

面试结论：

> clipping 是安全带，不是方向盘。它能限制更新幅度，但根因还要从数据、loss、学习率和数值精度排查。

## 496. 哪些参数通常不做 weight decay？为什么？

30 秒版：

通常不对 bias、LayerNorm/RMSNorm 的 scale 参数做 weight decay，有时 embedding 也单独处理。因为这些参数不是主要的矩阵权重，衰减它们可能破坏归一化层稳定性或带来不必要约束。

2 分钟版：

常见分组：

```text
decay:
  Linear weight
  Attention/MLP projection weight

no_decay:
  bias
  LayerNorm/RMSNorm weight
  sometimes embedding
```

原因：

- bias 只是平移项，衰减意义不大。
- norm scale 控制归一化后的尺度，衰减可能影响稳定性。
- embedding 是否 decay 要看模型和实验设置。

面试表达：

> 我会按参数名或模块类型分 optimizer param groups，确保 decay 和 no_decay 分开，并核对框架默认行为。

项目坑：

- LoRA 参数和 base 参数分组不清。
- 冻结参数仍被传入 optimizer。
- weight decay 作用到 norm 参数导致微调不稳定。

## 497. batch size、gradient accumulation 和学习率有什么关系？

30 秒版：

global batch 等于 `micro_batch * data_parallel_size * grad_accum_steps`。gradient accumulation 用多次小 batch 累积梯度后再更新，能省显存但改变 optimizer step 频率。batch 变大时通常要重新调学习率、warmup 和训练步数。

2 分钟版：

公式：

```text
global_batch = micro_batch_per_gpu * dp_size * grad_accum_steps
```

gradient accumulation：

1. 前向和反向多个 micro batch。
2. 梯度累积。
3. 达到累积步数后 optimizer step。
4. 再清梯度。

好处：

- 显存有限时模拟大 batch。
- 提升梯度估计稳定性。

风险：

- optimizer step 数变少。
- scheduler 如果按 step 计算，学习率曲线会变。
- logging 时 micro loss 和 update loss 容易混。
- batch 过大可能泛化变差或需要更大学习率。

面试表达：

> 改 batch 不是只改一个参数，要一起看 global batch、有效 token 数、学习率、warmup、总 update steps 和显存。

## 498. InfoNCE / 对比学习怎么用于 embedding 或 reranker？

30 秒版：

InfoNCE 可以看成在一批候选里把正样本分出来的 softmax 分类目标。它拉近 query 和正样本，推远负样本，常用于 embedding、CLIP、检索模型。关键是负样本质量，hard negative 有用，但 false negative 会伤模型。

2 分钟版：

常见形式：

```text
L_i = - log exp(sim(q_i, d_i+) / tau)
          / sum_j exp(sim(q_i, d_j) / tau)
```

含义：

- query 和正样本相似度越高，loss 越小。
- query 和负样本相似度越高，loss 越大。
- temperature `tau` 控制分布尖锐程度。

负样本：

- in-batch negative：同 batch 其他样本当负例，高效。
- hard negative：相似但不相关的负例，提升判别力。
- false negative：看似负例但其实相关，会污染训练。

RAG 项目表达：

> Embedding 模型重召回，reranker 重排序。对比学习训练 embedding 时，我会特别关注 hard negative 构造、false negative 过滤和 Recall@K/NDCG 指标。

## 499. loss 下降但线上指标没涨，怎么排查？

30 秒版：

我会先查 loss 和业务指标是否一致，再查数据分布、标签质量、loss mask、验证集污染、阈值、线上链路一致性和 bad case。loss 降低只能说明训练目标变好，不保证 F1、NDCG、任务成功率或用户满意度变好。

2 分钟版：

排查顺序：

1. 数据分布：训练、验证、线上是否一致。
2. 标签质量：是否有脏标签、冲突偏好、自动标注噪声。
3. loss 目标：CE 降了是否真的对应业务指标。
4. mask 和 template：SFT 是否只算 answer loss。
5. 指标选择：类别不均衡时 accuracy 是否误导。
6. 验证集污染：是否和训练集重复。
7. 阈值：模型分数分布变了，threshold 是否重调。
8. 线上链路：tokenizer、prompt、RAG、模型版本是否一致。
9. bad case：是否只改善简单样本，难例没变。

面试表达：

> 我会把 loss 当训练信号，把线上指标当最终目标。两者不一致时，要从数据、目标、指标、阈值和链路五个层面排查。

## 500. 数学基础 / loss / 指标项目怎么讲 8 分钟？

30 秒版：

按“任务和业务代价、数据标签、模型输出、loss 选择、优化策略、指标和阈值、bad case、上线监控”讲。重点不是炫公式，而是证明你知道 loss、metric 和业务目标怎么对齐。

8 分钟结构：

1. 背景：

> 这个项目是做 X，正例表示 Y，误报代价是 A，漏报代价是 B。

2. 数据：

> 数据来自日志/标注/规则，做了去重、清洗、类别分布统计和训练验证切分。

3. 模型输出：

> 模型输出 logits/相似度/排序分数，再根据任务转成概率或排名。

4. loss：

> 如果是单标签多分类用 CE，多标签或二分类用 BCEWithLogits，检索 embedding 用 InfoNCE，排序可用 pairwise/listwise。

5. 优化：

> 用 AdamW，设置 warmup、cosine decay、gradient clipping，并分组处理 weight decay。

6. 指标：

> 不只看 loss。正例稀少时看 PR-AUC、F1、Recall@K；排序任务看 NDCG/MRR；上线看业务成功率和人工审核量。

7. 阈值和校准：

> 在验证集调 threshold，高风险场景还要做校准和人工兜底。

8. bad case：

> 错误集中在某些长尾 query/类别/人群，我做了 hard negative、补标、阈值分桶或数据增强。

9. 收尾：

> 最终我把训练 loss、验证指标、线上指标和 bad case 闭环起来，而不是只看一个离线 loss。

反问准备：

- 团队更关注离线指标还是线上 A/B？
- 误报和漏报哪个成本更高？
- 标注数据是否稳定，是否有主动学习或难例挖掘？

## 本组题的复习顺序

1. 先背 481-487：CE、logits、LLM token loss。
2. 再背 488-491：不均衡、AUC、阈值和校准。
3. 再背 492-497：AdamW、warmup、clipping、batch。
4. 最后背 498-500：InfoNCE、loss/metric 排查、项目讲法。

## 延伸阅读

- PyTorch CrossEntropyLoss：[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- PyTorch BCEWithLogitsLoss：[https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- PyTorch AdamW：[https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- Adam：[https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- AdamW：[https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- scikit-learn 模型评估：[https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
- 现代神经网络校准：[https://arxiv.org/abs/1706.04599](https://arxiv.org/abs/1706.04599)
