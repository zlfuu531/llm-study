# DeepSeek、MoE、MLA 与 Reasoning Model 面试

这一章面向大模型算法、Post-training、AI Infra、推理部署、模型评测和热点追问。DeepSeek 系列在 2025-2026 面试里高频，不是因为“背一个模型名字”，而是因为它把几条主线绑在一起：MoE 扩容量、MLA 降 KV Cache、GRPO 做 reasoning RL、蒸馏迁移推理能力、MTP/长思考提高推理效率和能力。

你需要能回答：

- DeepSeek-V3 和 DeepSeek-R1 的定位区别。
- R1-Zero 和 R1 的训练流程差异。
- MoE 为什么能扩大模型容量，难点是什么。
- MLA 和 MHA/MQA/GQA 的区别。
- GRPO 相比 PPO/DPO 省在哪里。
- Reasoning Model 为什么需要长思考、RL、蒸馏和可验证 reward。
- 面试官追问“DeepSeek 为什么便宜”时怎么讲得不玄学。

如果面试官继续追问 RLVR、verifier、ORM/PRM、Best-of-N、Self-Consistency、test-time scaling 或 reasoning 上线控成本，直接跳到 [29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md](29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md) 和 [answers/24_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling_答案版.md](answers/24_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling_答案版.md)。

如果面试官把 MTP 继续追到 speculative decoding、draft model、EAGLE、Medusa、accept rate 和低延迟 serving，跳到独立专题：[38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md](38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md)。

如果面试官把 MoE 继续追到 expert parallel、all-to-all、hot experts、MoE serving 和 P99 排查，跳到独立专题：[39_分布式推理_PD分离_KVCache传输与MoEServing面试.md](39_分布式推理_PD分离_KVCache传输与MoEServing面试.md)。

## 一句话总览

```text
DeepSeek-V3：高效 MoE 基座/通用模型
  -> MLA 降低 KV Cache
  -> DeepSeekMoE 扩大容量但控制每 token 计算
  -> MTP 辅助训练和推理效率讨论

DeepSeek-R1：面向推理能力的 reasoning model
  -> R1-Zero 探索纯 RL 激发推理
  -> R1 加 cold-start 和多阶段训练提升可读性、稳定性和通用性
  -> reasoning distillation 把强推理能力迁移到小模型
```

一句话背诵：

> DeepSeek 高频是因为它同时对应架构、训练和工程效率。V3 更像高效 MoE 通用模型，重点是 MLA、MoE、MTP 和训练效率；R1 更像 reasoning model，重点是用 RL 激发长链推理，再通过 cold-start、SFT、RL 和蒸馏提升可读性和稳定性。面试不要只说“便宜”，要拆成稀疏激活、KV Cache 压缩、训练策略、推理成本和评测闭环。

## V3 和 R1 怎么区分

| 模型 | 重点 | 面试表达 |
| --- | --- | --- |
| DeepSeek-V3 | 高效 MoE 通用模型 | 架构和系统效率：MoE、MLA、MTP、训练成本 |
| DeepSeek-R1 | 推理模型 | Post-training 和 reasoning RL：R1-Zero、cold-start、多阶段 RL、蒸馏 |

V3 更适合回答：

- 为什么 MoE 可以降低每 token 计算。
- MLA 怎么降低 KV Cache。
- MTP 有什么作用。
- 大模型训练如何追求性价比。

R1 更适合回答：

- RL 怎么激发推理能力。
- R1-Zero 和 R1 的区别。
- GRPO 为什么常和 R1 一起被问。
- reasoning distillation 为什么有效。
- 长思考的收益和风险。

面试口语：

> V3 和 R1 不是同一个层面的问题。V3 主要看高效模型架构和训练系统，R1 主要看 reasoning post-training。V3 的关键词是 MoE、MLA、MTP 和成本效率；R1 的关键词是 RL、cold-start、多阶段训练、可验证 reward 和蒸馏。

## DeepSeekMoE：为什么要稀疏专家

Dense FFN：

```text
每个 token 都经过同一套 FFN 参数
总参数量大 -> 每 token 计算也大
```

MoE：

```text
有很多 experts
每个 token 只路由到 top-k 个 experts
总参数量大，但每 token 激活参数量有限
```

简化公式：

```text
y = shared_expert(x) + sum_{i in TopK(router(x))} gate_i(x) * expert_i(x)
```

为什么有用：

- 扩大总参数量，提高模型容量。
- 每 token 只用部分 experts，控制计算量。
- 不同 experts 可以学到不同模式。
- 适合在训练和推理成本受限时扩模型。

难点：

- 路由负载不均。
- expert collapse：少数专家被过度使用。
- all-to-all 通信重。
- token dispatch/combine 有工程开销。
- expert capacity 溢出可能丢 token 或降低质量。
- 推理部署比 dense 模型复杂。

面试口语：

> MoE 的核心不是“参数少”，而是“总参数多、激活参数少”。每个 token 只经过 top-k 个专家，所以每 token 计算受控。难点是路由、负载均衡和 all-to-all 通信，如果很多 token 挤到少数专家，系统效率和模型质量都会出问题。

## 路由和负载均衡怎么讲

MoE 路由流程：

```text
token hidden state -> router logits -> softmax scores
-> top-k experts -> expert computation -> weighted combine
```

关键概念：

- router：给每个 token 分配 experts。
- top-k：每个 token 选择几个 experts。
- gate weight：不同 expert 输出的权重。
- shared expert：所有 token 都经过的共享专家，保留通用能力。
- routed expert：由 router 动态选择的专家。
- capacity：每个 expert 能接收的 token 上限。

负载均衡为什么重要：

- 训练时负载不均会导致某些 GPU 空转、某些 GPU 拥堵。
- 推理时热门专家会成为尾延迟瓶颈。
- 少数 experts 过热会导致专家专业化不足或 collapse。

常见控制：

- load balancing loss。
- expert-level / device-level balance。
- 限制 capacity。
- router 正则或约束。
- 工程上做 token dispatch 优化。

DeepSeek 相关讨论里还会提到 auxiliary-loss-free load balancing：核心是希望减少负载均衡辅助损失对主任务目标的干扰，同时仍然让专家负载更均衡。面试不用死背实现细节，重点说清“负载均衡不能以明显损害主任务为代价”。

## MLA：为什么降 KV Cache

MHA 的 KV Cache：

```text
每层、每个 token、每个 KV head 都要缓存 K 和 V
KV cache ~= layers * seq_len * kv_heads * head_dim * 2 * dtype_bytes
```

MQA：

- 所有 query heads 共享一组 K/V。
- KV Cache 最小，但表达能力可能损失。

GQA：

- 多个 query heads 共享一组 K/V。
- 在效果和显存之间折中。

MLA：

- 用 latent 表示压缩 K/V 相关信息。
- 推理时缓存更紧凑的 latent。
- 目标是降低 KV Cache 显存和读带宽，同时保持多头表达能力。

面试口语：

> MLA 主要解决推理阶段 KV Cache 显存和带宽压力。MHA 是每个 head 缓存自己的 K/V，GQA 是分组共享 K/V，MLA 更进一步，把 K/V 相关信息压缩到低维 latent 表示。它和 MoE 一起服务于“能力更强但成本更低”的模型设计。

常见追问：

### MLA 是不是等于低秩分解？

可以用低秩/latent 压缩来帮助理解，但不要把 MLA 简化成一句“低秩分解”。面试表达应强调：它面向 attention KV Cache 的压缩和恢复，不只是普通矩阵分解。

### MLA 和 LoRA 有什么关系？

都能看到低秩思想，但用途不同。LoRA 是参数高效微调，冻结原权重、训练低秩增量；MLA 是 attention 结构设计，目标是压缩推理时 K/V 表示。

## MTP：Multi-Token Prediction 怎么讲

普通 next-token prediction：

```text
给定 x_1...x_t，预测 x_{t+1}
```

MTP 思路：

```text
不仅预测下一个 token，也辅助预测后面多个 token
```

可能收益：

- 给模型更强的未来 token 训练信号。
- 让表示更关注后续连续预测。
- 在某些推理加速或 speculative decoding 讨论中有联系。

注意：

- MTP 不是简单把生成一次变成多个 token 就万事大吉。
- 训练目标、推理接受率、质量和实现复杂度都要评估。
- 面试里更重要的是知道它属于“提高训练信号和推理效率潜力”的方向。

面试口语：

> MTP 可以理解为在 next-token prediction 之外加入多 token 预测信号，让模型学习更远的未来 token。它可能提升训练效率和对连续生成的建模，也和推理加速讨论有关。但是否真正加速要看推理机制和接受率，不能只看名字。

MTP 和 speculative decoding 的详细关系见：[38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md](38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md)。

## GRPO：为什么和 R1 一起高频

PPO 通常需要：

- policy model。
- reference model。
- reward model。
- value/critic model。
- KL 约束。

GRPO 思路：

```text
对同一个 prompt 采样一组回答
每个回答得到 reward
用组内均值/方差做相对优势
好于组平均的回答被鼓励，差于组平均的回答被抑制
```

简化 advantage：

```text
A_i = (r_i - mean(r_group)) / std(r_group)
```

为什么省：

- 不显式训练 value model。
- 用组内相对比较估计优势。
- 适合可验证 reward 的数学、代码、逻辑题。

和 DPO 区别：

- DPO：离线偏好对，chosen/rejected。
- GRPO：在线采样多回答，用 reward 做组内相对优化。

和 PPO 区别：

- PPO 依赖 value/critic 估计 advantage。
- GRPO 用同 prompt 的一组样本做相对 advantage，减少 value model 依赖。

面试口语：

> GRPO 的关键是 group relative。对同一道题采样多个回答，用 reward 比较组内好坏，构造相对优势。相比 PPO，它减少 value model 依赖；相比 DPO，它更像在线 RL，适合数学、代码这类可验证 reward 的 reasoning 任务。

## R1-Zero 和 R1

R1-Zero：

- 强调从 base model 出发，用 RL 激发推理能力。
- 展现出长推理、自我反思等行为。
- 问题是可读性、语言混杂和稳定性可能不够好。

R1：

- 引入 cold-start 数据。
- 多阶段训练：SFT、RL、拒绝采样、再训练等组合。
- 目标是保留推理能力，同时改善可读性、通用能力和对齐。

面试口语：

> R1-Zero 的看点是不用传统大量 SFT，也能通过 RL 激发推理行为；但它会有可读性和稳定性问题。R1 在此基础上加入 cold-start 和多阶段训练，让推理过程更可读、语言更稳定，也兼顾通用能力。

不要说：

- “R1-Zero 就是没有数据训练。”
- “R1 只是蒸馏模型。”
- “GRPO 等于 R1。”

更稳的说法：

> R1-Zero 是探索 RL 能否直接激发 reasoning；R1 是把这个能力工程化和产品化，加入冷启动、监督数据、多阶段 RL 和蒸馏，让效果、可读性和稳定性更好。

## Reasoning Model 的训练闭环

Reasoning model 通常围绕可验证任务构造闭环。

```text
base model
-> cold-start SFT / reasoning traces
-> RL with verifiable reward
-> rejection sampling
-> SFT on high-quality reasoning data
-> RL for helpfulness / safety / reasoning
-> distillation to smaller models
-> private eval and bad case loop
```

可验证 reward：

- 数学最终答案是否正确。
- 代码是否通过单元测试。
- 工具调用结果是否满足目标。
- 格式是否符合要求。

难点：

- reward hacking。
- 答案对但过程乱。
- 过程看似合理但不忠实。
- 长思考导致延迟和成本上升。
- 简单题过度思考。
- 安全和隐私风险。

面试口语：

> 推理模型的关键不是让模型“多说过程”，而是让模型在可验证任务上学会搜索、检查和修正。数学和代码适合 RL，因为最终答案或测试能给 reward。但要防 reward hacking 和长思考成本，所以还要做格式约束、长度控制、评估和蒸馏。

## Reasoning Distillation

Reasoning distillation 是把强模型的推理能力迁移给小模型。

常见数据：

- question + reasoning trace + answer。
- question + concise reasoning + answer。
- question + multiple sampled solutions + verified answer。
- code problem + solution explanation + tests。

收益：

- 小模型能学到更强的解题模式。
- 成本低于直接对小模型做大规模 RL。
- 适合端侧、小模型和垂直领域。

风险：

- teacher 的错误会被放大。
- 过程可能不忠实。
- 过长 trace 增加输出成本。
- 小模型可能只模仿格式，不真正会推理。
- 训练数据如果只包含长链，简单问题会过度思考。

面试口语：

> 推理蒸馏不是越长越好。真正有用的是高质量、可验证、适合 student 能力边界的推理数据。要过滤错解、控制长度、保留拒答和边界样本，并用数学、代码、业务私有 eval 验证是否真的提升。

## 长思考和 Test-Time Compute

Reasoning model 常见现象：

- 输出更长的推理过程。
- 尝试多步规划。
- 自我检查。
- 对难题可能多采样或搜索。

收益：

- 数学、代码、多步规划更强。
- 能在推理时投入更多计算。
- 与 verifier、self-consistency、best-of-N 结合。

成本：

- 延迟变高。
- 输出 token 成本变高。
- 简单问题可能过度推理。
- 长上下文和 KV Cache 压力增加。
- 推理过程可能泄露敏感中间信息。

工程控制：

- simple/hard query 路由。
- max reasoning tokens。
- budget forcing。
- verifier 早停。
- 分难度动态采样。
- 蒸馏成短推理。

面试口语：

> Reasoning model 本质上把一部分能力提升转移到 test-time compute。难题可以多思考、多采样和验证，简单题则要快速回答。工程上要做预算控制和难度路由，否则质量提升会被延迟和成本吞掉。

## DeepSeek 为什么“便宜”要怎么讲

不要只说“因为国产模型便宜”或“因为开源”。更稳的拆法：

1. 架构效率：MoE 稀疏激活，总参数大但每 token 激活有限。
2. 推理显存：MLA 降低 KV Cache 压力。
3. 训练效率：工程优化、并行训练、混合精度等系统能力。
4. Post-training：GRPO 减少 value model 依赖，适合可验证任务。
5. 蒸馏：把强 reasoning 能力迁移到更小模型。
6. 服务策略：模型路由、缓存、批处理和限流也会影响实际成本。

面试口语：

> 便宜不是单点技术，而是模型架构、训练方法和系统工程叠加。MoE 控制每 token 计算，MLA 控制 KV Cache，GRPO 降低 reasoning RL 复杂度，蒸馏让小模型继承强模型能力，再配合推理服务优化，整体成本才会下来。

## 面试常见高压问题

### Q1：DeepSeek-R1 最大亮点是什么？

不是只说开源或便宜。可以说：它展示了 RL 对 reasoning 能力的激发，尤其是可验证任务上的长思考、自我检查和多步推理。同时通过 R1-Zero 到 R1 的流程说明，纯 RL 有潜力但也有可读性和稳定性问题，需要 cold-start、多阶段训练和蒸馏来工程化。

### Q2：R1 和普通 SFT 模型有什么区别？

普通 SFT 更像学习“如何按指令回答”，R1 这类 reasoning model 更强调复杂任务的搜索、验证和自我修正。训练上不只是 SFT，还会引入可验证 reward、RL、多阶段数据生成和推理蒸馏。

### Q3：MoE 为什么推理不一定快？

MoE 每 token 激活参数少，但有 router、token dispatch、expert computation、combine 和 all-to-all 通信。负载不均会导致尾延迟，专家分布跨设备时通信可能成为瓶颈。所以 MoE 不是天然快，必须结合 expert parallel 和调度优化。

### Q4：MLA 和 GQA 怎么区分？

GQA 是把多个 query heads 分组共享 K/V，减少 KV head 数。MLA 是把 K/V 相关信息压缩到 latent 表示，目标是进一步降低 KV Cache 和带宽压力。两者都服务于推理效率，但机制不同。

### Q5：Reasoning distillation 会不会泄露错误思路？

会有风险。teacher 的错误过程、幻觉或过长 trace 都可能被 student 学到。所以要做验证过滤、去重、长度控制、人工抽检和私有 eval。对数学代码可以用最终答案或测试过滤，对业务任务要用 reference answer、rubric 和安全检查。

## 面试前背诵版

DeepSeek 面试要拆成 V3 和 R1。V3 重点是高效架构：MoE 用稀疏激活扩大容量但控制每 token 计算，MLA 用 latent 压缩降低 KV Cache，MTP 增强多 token 预测信号；R1 重点是 reasoning post-training：R1-Zero 用 RL 激发推理，R1 加 cold-start 和多阶段训练提升可读性与稳定性，GRPO 用组内相对 advantage 减少 value model 依赖，蒸馏把推理能力迁移到小模型。风险包括路由负载不均、all-to-all 通信、长思考成本、reward hacking、过程不忠实和简单题过度推理。

## 本轮参考来源

- DeepSeek-V3 Technical Report：https://arxiv.org/abs/2412.19437
- DeepSeek-R1 论文：https://arxiv.org/abs/2501.12948
- DeepSeek-V2 论文：https://arxiv.org/abs/2405.04434
- DeepSeekMoE 论文：https://arxiv.org/abs/2401.06066
- DeepSeekMath / GRPO 论文：https://arxiv.org/abs/2402.03300
- Switch Transformers 论文：https://arxiv.org/abs/2101.03961
- Chain-of-Thought Prompting 论文：https://arxiv.org/abs/2201.11903
- Self-Consistency 论文：https://arxiv.org/abs/2203.11171
- Scaling LLM Test-Time Compute 论文：https://arxiv.org/abs/2408.03314
- 本地外部资料中的 `EasyOffer` DeepSeek/MoE 面经、`LLM-HandCoding-Interview` MLA/GRPO、`TorchLeet` GRPO/MoE 题目
