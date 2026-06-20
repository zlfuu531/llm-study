# 答案版 13：DeepSeek、MoE、MLA 与 Reasoning Model

对应 `03_高频题单100题.md` 的 281-300 题。

## 281. DeepSeek-V3 和 DeepSeek-R1 的定位有什么区别？

V3 更偏高效 MoE 通用模型，重点是架构和系统效率，比如 MoE、MLA、MTP、训练成本。R1 更偏 reasoning model，重点是 post-training 和 reasoning RL，比如 R1-Zero、cold-start、多阶段 RL 和推理蒸馏。面试要说：V3 看架构效率，R1 看推理能力训练。

## 282. DeepSeek 为什么在 2025-2026 面试里高频？

因为它把多个高频主题连在一起：MoE 扩容量但控制每 token 计算，MLA 降低 KV Cache，GRPO 降低 reasoning RL 对 value model 的依赖，R1 展示 RL 激发推理能力，蒸馏让小模型继承推理能力。这些都对应能力、显存、成本和评测的核心矛盾。

## 283. MoE 的核心公式和直觉是什么？

直觉是“总参数多、激活参数少”。每个 token 经过 router 选择 top-k experts，输出可以写成 `y = shared_expert(x) + sum gate_i * expert_i(x)`。它不是减少总参数，而是让每个 token 只激活少数专家，从而扩大容量但控制计算。

## 284. MoE 为什么不一定推理更快？

MoE 虽然每 token 激活参数少，但多了 router、token dispatch、expert compute、combine 和 all-to-all 通信。负载不均时少数专家会成为瓶颈，跨设备 expert parallel 会带来通信开销。所以 MoE 是否快取决于路由、负载均衡、专家部署和推理调度。

## 285. MoE 负载均衡为什么重要？

如果大量 token 路由到少数 experts，会造成专家过载、GPU 利用率不均、尾延迟升高，还可能导致专家专业化不足或 collapse。训练时常用 load balancing loss、capacity、router 约束；工程上要优化 token dispatch 和 expert parallel。

## 286. shared expert 和 routed expert 怎么理解？

shared expert 是所有 token 都经过的共享专家，用来保留通用能力；routed expert 是 router 动态选择的专家，用来增加稀疏容量和专业化。这样可以兼顾通用表达和专家分工，避免完全依赖路由专家导致通用能力不稳。

## 287. MLA 解决什么问题？

MLA 主要解决推理阶段 KV Cache 显存和带宽压力。MHA 每个 head 都缓存 K/V，GQA 是分组共享 K/V，MLA 则把 K/V 相关信息压缩到 latent 表示，推理时缓存更紧凑的 latent，从而降低长上下文和高并发成本。

## 288. MLA 和 MQA/GQA 有什么区别？

MQA 是所有 query heads 共享一组 K/V，GQA 是多个 query heads 分组共享 K/V。MLA 不是简单减少 KV head 数，而是用 latent 压缩 K/V 信息。三者都降低 KV Cache，但 MLA 更强调低维 latent 表示和恢复。

## 289. MLA 和 LoRA 有什么关系和区别？

两者都能用低秩/latent 思想帮助理解，但用途不同。LoRA 是参数高效微调，冻结原权重、训练低秩增量；MLA 是 attention 架构设计，目标是压缩推理时 K/V 表示，降低 KV Cache 显存和带宽。

## 290. MTP 是什么，有什么作用？

MTP 即 Multi-Token Prediction，不只预测下一个 token，也辅助预测后续多个 token。它能提供更远的未来 token 训练信号，可能帮助连续生成建模，也和推理效率讨论有关。但是否加速要看推理机制、接受率和质量评估，不能只看名字。

## 291. GRPO 为什么和 R1 一起被问？

R1 类 reasoning model 常用可验证 reward 做 RL。GRPO 对同一 prompt 采样一组回答，用组内 reward 均值和方差构造相对优势，减少对 value model 的依赖。它适合数学、代码这类答案可验证的推理任务，所以经常和 R1 一起被问。

## 292. GRPO、PPO、DPO 怎么区分？

PPO 是在线 RL，通常需要 reward model 和 value/critic；DPO 是离线偏好优化，用 chosen/rejected 直接训练；GRPO 也是更接近在线 RL，但用同一 prompt 的一组回答做相对比较，构造 group advantage，减少 value model 依赖。

## 293. R1-Zero 和 R1 有什么区别？

R1-Zero 强调从 base model 出发用 RL 激发推理能力，会出现长思考和自我反思，但可读性、语言稳定性和通用性可能不足。R1 在此基础上加入 cold-start 和多阶段训练，提升可读性、稳定性、安全性和通用能力。

## 294. cold-start 数据在 R1 里有什么作用？

cold-start 数据给模型一个更好的初始推理格式和可读性基础，避免纯 RL 早期输出混乱、语言混杂或格式不稳定。它不是替代 RL，而是让后续 RL 在更稳的起点上强化推理能力。

## 295. Reasoning Model 和普通指令模型有什么区别？

普通指令模型更关注直接回答、通用对话和格式遵循；Reasoning Model 更强调数学、代码、多步规划、自我检查和复杂推理。训练上通常会用 reasoning traces、可验证 reward、RL、rejection sampling 和推理蒸馏。

## 296. 可验证 reward 为什么适合 reasoning RL？

数学最终答案、代码单元测试、工具任务成功条件都能给相对明确的 reward，减少纯人工偏好标注成本。它能鼓励模型探索和自我修正。但也有 reward hacking 风险，比如答案碰巧对、过程错误、格式投机，所以要配合过程检查和 eval。

## 297. Reasoning distillation 有什么收益和风险？

收益是把强模型的解题步骤、搜索策略和验证能力迁移到小模型，成本低于直接对小模型做大规模 RL。风险是 teacher 错误会被放大，过程可能不忠实，过长 trace 增加成本，小模型可能只学格式不学能力。要做验证过滤、长度控制和私有 eval。

## 298. 长思考为什么能提升能力，又为什么有成本？

长思考让模型在推理时投入更多 test-time compute，可以多步拆解、自我检查、多采样和验证，提升数学、代码和规划任务。但它会增加输出 token、延迟、KV Cache 压力和成本，简单题还可能过度思考，所以要做难度路由、预算控制和早停。

## 299. 怎么回答“DeepSeek 为什么便宜”？

不要只说开源或价格低。要拆成架构、训练和系统：MoE 稀疏激活控制每 token 计算，MLA 降低 KV Cache，GRPO 降低 reasoning RL 的 value model 依赖，蒸馏把强能力迁移到小模型，再配合推理服务里的批处理、缓存、路由和限流，整体成本才会下降。

## 300. DeepSeek / Reasoning Model 项目 8 分钟深挖版讲什么？

按“问题 -> 架构 -> 训练 -> 推理 -> 评测 -> 风险”讲。比如先说明目标是提升数学/代码/业务推理；架构上解释 MoE/MLA 如何控制成本；训练上解释 SFT、RL、GRPO、可验证 reward、蒸馏；推理上讲长思考预算和路由；评测上用数学、代码、业务私有 eval；最后补充 reward hacking、长思考成本和安全风险。
