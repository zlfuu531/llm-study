# 深挖 02：LoRA、DPO、ORPO 与 GRPO

## 这一章解决什么问题

训练和对齐常被问得很细：

- LoRA 为什么低秩就够？
- LoRA 的 A/B 矩阵 shape 是什么？
- LoRA+ 为什么要给 A/B 不同学习率？
- QLoRA 到底量化了什么？
- DPO 为什么不用 Reward Model？
- DPO 里的 reference model 有什么用？
- ORPO 为什么不需要 reference model？
- GRPO 和 PPO 的差别在哪里？

这章把这些问题讲成一条逻辑线。

## 1. 为什么需要参数高效微调

全量微调一个大模型意味着：

- 所有参数都要参与训练。
- optimizer state 占大量显存。
- 每个任务保存一份完整模型成本很高。
- 小数据场景容易过拟合。

参数高效微调的思路：

> 冻结大部分基座参数，只训练很少一部分新增参数，让模型适配新任务。

## 2. LoRA 的直觉

假设原始线性层：

```text
y = Wx
```

全量微调会直接更新 `W`。

LoRA 认为：任务适配时，`W` 的变化量 `ΔW` 不一定需要满秩，可以用两个小矩阵近似：

```text
ΔW = B A
```

于是：

```text
y = Wx + scale * B(Ax)
```

其中：

- `W` 冻结。
- `A` 和 `B` 可训练。
- rank `r` 很小。

## 3. LoRA 的 shape

假设原线性层：

```text
W: [out_dim, in_dim]
x: [batch, in_dim]
```

LoRA：

```text
A: [r, in_dim]
B: [out_dim, r]
```

参数量：

```text
原始 W 参数量 = out_dim * in_dim
LoRA 参数量 = r * in_dim + out_dim * r
```

当 `r` 很小时，参数量大幅降低。

## 4. 为什么低秩可能有效

直觉解释：

- 预训练模型已经有强大的通用能力。
- 下游任务通常不需要重学全部知识。
- 只需要在少数方向上调整模型行为。
- 这些调整可以用低维子空间表达。

面试表达：

> LoRA 的假设是下游任务对权重的有效更新具有低秩结构。基座模型已经学到通用表示，微调更多是在少数方向上调整行为，因此不一定需要更新完整权重矩阵。

## 5. LoRA 常加在哪里

常见：

- attention 的 `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- FFN 的 `up_proj`
- `down_proj`
- `gate_proj`

怎么选：

- 只加 Q/V：参数更少，常见默认。
- attention + FFN 都加：效果可能更好，成本更高。
- 任务复杂或数据充足时可以加更多层。

## 6. rank 和 alpha

rank `r`：

- 控制低秩更新的容量。
- 越大参数越多，表达更强。
- 太小可能欠拟合，太大可能过拟合或浪费。

alpha：

- 控制 LoRA 更新的缩放。
- 常见 scale 是 `alpha / r`。

面试表达：

> rank 决定 LoRA 分支有多大表达能力，alpha 决定这个分支对原模型输出的影响强度。实际选择要看任务复杂度、数据量和验证集效果。

## 7. LoRA+：A/B 为什么要不同学习率

LoRA+ 的核心不是换结构，而是换优化器参数组。

普通 LoRA：

```text
lr_A = lr_B
```

LoRA+：

```text
lr_A = base_lr
lr_B = ratio * base_lr
ratio = lr_B / lr_A
```

直觉：

- A、B 的角色不同，梯度动态也不同。
- 原始 LoRA 用同一个学习率，在大宽度模型上可能让特征学习不充分。
- LoRA+ 通过 `ratio` 让 B 矩阵通常学得更快，同时保持 LoRA 的推理结构不变。

面试表达：

> LoRA+ 是 LoRA 的 optimizer 改进，不是新的 adapter 结构。训练时给 A、B 两组参数不同学习率，常见是 `lr_B > lr_A`；推理时仍然是普通 LoRA 分支，可以 merge。

调参注意：

- `ratio` 不是越大越好，要和 base learning rate 一起调。
- 任务越难、越需要学习新特征时，可以尝试更大的 ratio。
- 如果 loss 抖动、格式退化或过拟合，先降低 base lr 或 ratio。

## 8. QLoRA 到底省在哪里

QLoRA 的关键：

- 冻结的基座模型用低比特量化存储，例如 4-bit。
- LoRA 参数仍然以较高精度训练。
- 反向传播只更新 LoRA 参数。

误区：

- 不是把所有训练都变成 4-bit。
- 不是 LoRA 本身一定 4-bit。
- 省显存主要来自基座权重量化和优化器状态减少。

面试表达：

> QLoRA 是在量化的冻结基座模型上训练 LoRA 参数。它让大模型微调能在更小显存上进行，但效果仍要通过验证集确认，不能默认无损。

## 9. 对齐到底在对齐什么

SFT 让模型学会回答，但不保证：

- 回答是否有帮助。
- 是否诚实。
- 是否安全。
- 是否符合人类偏好。
- 是否在多个可能答案中选择更好的那个。

偏好对齐就是让模型在多个可能回答之间，更倾向人类认为好的回答。

## 10. RLHF 的逻辑

RLHF 流程：

1. SFT 模型生成多个回答。
2. 人类标注偏好。
3. 用偏好数据训练 Reward Model。
4. 用 PPO 优化策略模型，让它得到更高 reward。

难点：

- Reward Model 可能被钻空子。
- PPO 训练复杂，不稳定。
- 标注成本高。
- 需要控制模型不要偏离太远。

## 11. DPO 的直觉

DPO 不显式训练 Reward Model。

它直接使用偏好对：

- chosen：人类更喜欢的回答。
- rejected：人类不喜欢的回答。

目标：

> 让当前模型相对 reference model 更偏向 chosen，而不是 rejected。

核心形式：

```text
pi_logratio = logp_policy(chosen) - logp_policy(rejected)
ref_logratio = logp_ref(chosen) - logp_ref(rejected)
loss = -log sigmoid(beta * (pi_logratio - ref_logratio))
```

直觉：

- 如果 policy 比 reference 更能区分 chosen/rejected，loss 小。
- 如果 policy 没有偏向 chosen，loss 大。

## 12. DPO 为什么需要 reference model

如果只让 chosen 概率越来越高，模型可能：

- 偏离原模型太远。
- 语言质量下降。
- 学到偏好数据中的噪声。
- 过度优化某种回答风格。

reference model 提供锚点：

> 我希望你比原模型更偏好 chosen，但不要无限制地偏离原模型。

面试表达：

> DPO 里的 reference model 类似约束基准。它让优化关注 policy 相对 reference 的偏好变化，避免模型为了迎合偏好数据而过度漂移。

## 13. beta 的作用

beta 控制偏好优化强度。

- beta 大：更强地拉开 chosen/rejected。
- beta 小：更新更温和。

可以类比温度或约束强度，但不要说成完全等价。

## 14. ORPO 的直觉

ORPO 也是用 `(prompt, chosen, rejected)`，但它不保留 reference model。它把两件事合在一个目标里：

- SFT：让模型继续学习 chosen answer。
- Odds ratio preference：让 chosen 的 odds 高于 rejected。

简化公式：

```text
L_ORPO = L_SFT + lambda * L_OR
L_OR = -log sigmoid(log odds_chosen - log odds_rejected)
odds(y|x) = P(y|x) / (1 - P(y|x))
```

一句话：

> ORPO 是 reference-free 的单阶段偏好优化：chosen 既是监督答案，又在 odds ratio 项里相对 rejected 被拉高。

面试里要补一句风险：没有 reference model 不代表没有约束，`lambda`、学习率、数据质量和长度归一化会变得更关键。

## 15. PPO、DPO、ORPO、GRPO 怎么区分

PPO：

- 强化学习方法。
- 需要 reward。
- 通常有 value/critic。
- 训练复杂。

DPO：

- 直接用偏好对。
- 不显式训练 reward model。
- 通常需要 reference model。
- 更像监督式偏好优化。

ORPO：

- 直接用偏好对。
- 不需要 reward model。
- 不需要 reference model。
- 用 SFT + odds ratio 单阶段优化。

GRPO：

- 对同一个 prompt 采样一组回答。
- 用组内相对奖励估计优势。
- 减少对 value model 的依赖。
- 常在 DeepSeek-R1 / reasoning model 语境下被问。

## 16. GRPO 的直觉

同一个题目生成多个答案。

比如：

```text
answer A: reward 0.8
answer B: reward 0.3
answer C: reward 0.6
```

不一定需要一个单独 value model 来判断绝对好坏，可以看组内相对表现：

- 比组平均好，就鼓励。
- 比组平均差，就压低。

面试表达：

> GRPO 的核心是 group-relative advantage。对同一个 prompt 采样多个回答，用组内相对奖励来估计哪些回答更值得强化，从而降低传统 PPO 中 value model 的成本和复杂度。

## 17. 微调项目怎么讲才像做过

不要说：

> 我用了 LoRA 微调模型，效果还可以。

要说：

> 我先判断 prompt/RAG 是否能解决，确认需要行为适配后，把业务数据清洗成 instruction-response 格式，并只对 answer 部分计算 loss。训练上冻结基座模型，用 LoRA 加在 attention 的 q/v projection 上，rank 根据验证集效果选择。评估时除了自动指标，还人工检查幻觉、格式遵循和 bad case。最后如果部署，还要考虑 LoRA merge、量化和回滚。

## 18. 高频追问

1. LoRA 为什么用低秩矩阵？
2. LoRA 的参数量怎么计算？
3. LoRA+ 为什么要给 A/B 不同学习率？
4. LoRA+ 和调大 rank、alpha 有什么区别？
5. LoRA 加 Q/V 和加所有 projection 有什么区别？
6. QLoRA 量化了哪些参数？
7. SFT 后为什么还要对齐？
8. RLHF 的 Reward Model 怎么训练？
9. PPO 为什么需要 KL？
10. DPO 为什么不需要显式 Reward Model？
11. DPO 的 reference model 有什么用？
12. ORPO 和 DPO 有什么区别？
13. GRPO 为什么适合 reasoning model 讨论？

## 19. 手撕代码优先级

- LoRA linear layer
- loss mask
- LoRA+ optimizer param groups
- DPO loss
- ORPO loss
- preference pair batch 组织

## 参考来源

- LoRA: https://arxiv.org/abs/2106.09685
- LoRA+: https://arxiv.org/abs/2402.12354
- DPO: https://arxiv.org/abs/2305.18290
- ORPO: https://arxiv.org/abs/2403.07691
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- Hugging Face fine-tuning docs: https://huggingface.co/docs/transformers/en/training
- TorchLeet: https://github.com/Exorust/TorchLeet
- Datawhale Happy-LLM: https://github.com/datawhalechina/happy-llm
