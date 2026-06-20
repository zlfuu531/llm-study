# Reasoning Post-training、RLVR、Verifier 与 Test-Time Scaling 面试

这一章专门补 2025-2026 大模型面试里非常容易被追问的一条线：

```text
Reasoning Model
-> RLVR / 可验证奖励
-> Verifier / Reward Model
-> ORM / PRM
-> Best-of-N / Self-Consistency / Search
-> Test-Time Scaling
-> 成本、延迟、评估和安全
```

已有的 [18_DeepSeek_MoE_MLA与ReasoningModel面试.md](18_DeepSeek_MoE_MLA与ReasoningModel面试.md) 更偏 DeepSeek/R1/V3 总览，本章更偏“推理能力怎么训练、怎么验证、怎么在推理时花计算、怎么上线控成本”。

如果被继续追问 Best-of-N、Self-Consistency、pass@k 背后的 temperature/top-p、多采样成本和生成参数，配合 [36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md](36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md) 看。

你学完要能回答：

- RLVR 和 RLHF 有什么区别？
- 可验证 reward、reward model、LLM-as-judge 怎么区分？
- ORM 和 PRM 的区别是什么？
- verifier 是训练时用，还是推理时用？
- Best-of-N、Self-Consistency、majority vote、tree search 怎么区分？
- Test-time scaling 为什么有效，又为什么不一定划算？
- 为什么简单题不能无脑长思考？
- 线上 reasoning model 怎么做预算、路由、早停和回归评估？

## 1. 一句话总览

Reasoning post-training 的核心不是“让模型多写几行思考”，而是让模型在复杂任务里学会：

1. 搜索多个候选思路。
2. 检查中间步骤或最终答案。
3. 修正错误路径。
4. 在预算内选择更可靠的答案。

可以把训练和推理拆成两条线：

```text
Train-time:
base/instruct model
-> reasoning data / cold start
-> RLVR / GRPO / verifier reward
-> rejection sampling / distillation
-> eval and bad case loop

Test-time:
prompt
-> difficulty routing
-> sample / think / search
-> verifier / checker / vote
-> early stop / budget control
-> final answer
```

30 秒答案：

> Reasoning model 的关键是把训练时的可验证奖励和推理时的额外计算结合起来。训练侧用数学答案、代码测试、规则校验或 verifier 给 reward；推理侧可以通过长思考、多采样、投票、verifier rerank 或搜索提升正确率。但这会增加延迟和成本，所以工程上必须做难度路由、预算控制、早停和回归评估。

## 2. RLVR 是什么

RLVR：Reinforcement Learning with Verifiable Rewards，可以理解成“用可自动验证的 reward 做强化学习”。

典型任务：

| 任务 | 可验证 reward |
| --- | --- |
| 数学题 | 最终答案是否等于标准答案 |
| 代码题 | 是否通过单元测试 |
| 工具调用 | 调用结果是否满足目标 |
| 结构化抽取 | JSON/schema/字段值是否正确 |
| 规则任务 | 格式、约束、计算结果是否满足规则 |

和普通 RLHF 的区别：

| 维度 | RLHF | RLVR |
| --- | --- | --- |
| reward 来源 | 人类偏好 / Reward Model | 自动 checker / 可验证结果 / verifier |
| 适合任务 | 开放式 helpfulness、偏好、风格 | 数学、代码、工具、规则任务 |
| 优点 | 覆盖主观偏好 | 标注成本低、反馈稳定、可规模化 |
| 风险 | RM 偏差、reward hacking | checker 漏洞、答案投机、过程不忠实 |

面试表达：

> RLVR 适合结果能自动验证的任务，比如数学最终答案、代码测试、工具执行结果。它比纯人工偏好更容易规模化，但 reward 只覆盖可验证部分，模型可能学会钻 checker 漏洞，所以还要配合格式约束、过程检查、bad case 和安全评估。

## 3. 可验证 reward、Reward Model、LLM-as-Judge 怎么区分

这三个东西经常被混在一起，面试要先拆开：

| 名称 | 本质 | 优点 | 风险 |
| --- | --- | --- | --- |
| 可验证 reward | 规则/程序/测试给分 | 稳定、便宜、可复现 | 覆盖面有限，易被钻规则 |
| Reward Model | 训练出的打分模型 | 可泛化到偏好和开放回答 | 会有偏差和 reward hacking |
| LLM-as-Judge | 用 LLM 打分或比较 | 快速覆盖主观维度 | judge 偏见、位置偏差、提示敏感 |

例子：

```text
数学题:
  final answer exact match -> 可验证 reward
  解题步骤好坏 -> PRM / judge

代码题:
  unit tests pass -> 可验证 reward
  patch 是否简洁安全 -> judge / human review

客服回答:
  JSON 格式合法 -> 可验证 reward
  是否有帮助、语气是否好 -> judge / human preference
```

30 秒答案：

> 可验证 reward 是程序化检查，Reward Model 是训练出的评分模型，LLM-as-Judge 是用大模型当评委。数学和代码更适合可验证 reward，开放式问答常需要 RM 或 judge。上线时不能只信 judge，要固定 judge 版本、rubric、采样参数，并做人类抽检和回归集。

## 4. ORM 和 PRM

ORM：Outcome Reward Model，只看最终结果或完整答案。

PRM：Process Reward Model，评价中间推理步骤。

| 维度 | ORM | PRM |
| --- | --- | --- |
| 评分对象 | 最终答案 / 完整解法 | 每一步推理 |
| 标注成本 | 较低 | 高，需要 step-level label |
| 适合 | Best-of-N、最终 rerank | 搜索、早停、过程监督 |
| 风险 | 答案对但过程错 | 过程看似对但不忠实，标注难 |

ORM 例子：

```text
回答 A: 最终答案正确 -> high reward
回答 B: 最终答案错误 -> low reward
```

PRM 例子：

```text
step1: 正确拆题 -> +1
step2: 公式写错 -> -1
step3: 后续结果受污染 -> lower reward
```

30 秒答案：

> ORM 看最终答案，PRM 看中间步骤。ORM 便宜，适合最终答案可验证或 Best-of-N rerank；PRM 更细，能指导搜索和发现中间错误，但标注成本高，也有过程不忠实问题。数学推理里 PRM 常被问，是因为它能帮助模型不只撞答案，还能学会更可靠的解题过程。

## 5. Verifier 是什么

Verifier 泛指“判断一个候选答案好不好”的模块，可以是：

- 程序 checker。
- 单元测试。
- Reward Model。
- Process Reward Model。
- LLM-as-Judge。
- 规则 + 模型的组合。

Verifier 可以在两个阶段用：

```text
Train-time:
samples -> verifier/reward -> RL update / rejection sampling / distillation

Test-time:
sample N answers -> verifier score -> choose best / vote / search / early stop
```

常见结构：

```text
prompt
-> proposer/model generates candidates
-> verifier scores candidates
-> selector chooses final answer
```

30 秒答案：

> Verifier 是候选答案的检查器或评分器。训练时它可以提供 reward，帮助做 RL、拒绝采样和蒸馏；推理时它可以从多个候选里选最可靠的。数学可以用答案 checker，代码可以用测试，开放任务可能用 reward model 或 LLM judge。关键风险是 verifier 本身可能错、慢、被钻漏洞。

## 6. Best-of-N、Self-Consistency、Majority Vote、Tree Search

### Best-of-N

```text
同一问题采样 N 个答案
-> verifier / reward model 给每个答案打分
-> 选最高分
```

适合：

- verifier 比 generator 更可靠。
- 候选答案多样性有价值。
- 允许增加 N 倍推理成本。

### Self-Consistency

```text
同一问题采样多条推理路径
-> 提取最终答案
-> 选出现最多或最一致的答案
```

适合：

- 数学、常识、多步推理。
- 最终答案容易归一化。

### Majority Vote

Self-Consistency 的常见形式，更强调最终答案投票。

问题：

- 错误答案也可能多数。
- 需要答案归一化。
- N 大时成本高。

### Tree Search

```text
把推理过程当成搜索树
节点 = 中间状态 / step
边 = 下一步候选
verifier/PRM = 节点评分
```

适合：

- 中间状态可评价。
- 问题难，需要回溯和探索。

30 秒答案：

> Best-of-N 是多采样后用 verifier 选最高分；Self-Consistency 是多条推理路径投票选一致答案；Tree Search 是把推理步骤当搜索树，用 PRM 或 verifier 指导展开。它们都属于 test-time compute，用更多推理成本换更高正确率，但收益取决于任务难度、候选多样性和 verifier 质量。

## 7. Test-Time Scaling 为什么有效

Test-time scaling 指推理时投入更多计算来提升答案质量。

常见方式：

- 更长思考。
- 多采样。
- Self-Consistency。
- Best-of-N。
- verifier rerank。
- tree search。
- tool execution and feedback。

为什么有效：

1. 模型第一次生成可能走错路径，多采样能覆盖更多可能解法。
2. verifier 可以从候选里筛掉明显错误。
3. 长思考给模型更多分解、检查和修正机会。
4. 对数学/代码这种有明确正确性的任务收益更明显。

为什么不一定有效：

- 任务太简单，多花计算浪费。
- 任务太难，采样很多也找不到正确路径。
- verifier 不可靠，选错答案。
- 候选缺乏多样性，N 再大也相似。
- 成本和延迟超过业务收益。

30 秒答案：

> Test-time scaling 的本质是把一部分能力提升放到推理阶段。难题可以通过长思考、多采样、投票或 verifier rerank 提升正确率；但收益不是线性的，简单题会浪费，极难题可能没收益，工程上要按难度动态分配预算。

## 8. Compute-Optimal：怎么花推理预算

固定预算下，不要每个问题都用同样 N。

更好的策略：

```text
输入问题
-> 难度/不确定性估计
-> 简单题快速回答
-> 中等题少量多采样或 verifier
-> 难题更多采样/search/tool
-> 达到置信度或预算上限就停止
```

预算维度：

- max reasoning tokens。
- sample count `N`。
- verifier calls。
- tool calls。
- wall-clock latency。
- cost per request。

早停条件：

- 多个样本答案一致。
- verifier 分数超过阈值。
- 单元测试通过。
- 预算达到上限。
- 用户 SLA 不允许继续。

面试表达：

> Compute-optimal 不是“越想越久越好”，而是把推理预算分配给最可能受益的问题。简单题直接答，难题再多采样、用 verifier 或工具。线上要把质量收益、P95 延迟、token 成本和用户场景一起优化。

## 9. GRPO 在 RLVR 里怎么理解

GRPO 常和 reasoning RL 一起被问，因为它适合“同一 prompt 多采样 + reward 比较”的训练方式。

简化流程：

```text
for each prompt:
    sample G responses
    compute reward for each response
    normalize rewards inside group
    use relative advantage to update policy
```

组内相对优势：

```text
A_i = (r_i - mean(r_group)) / std(r_group)
```

为什么适合 RLVR：

- 同一题的多个回答可以用 checker / tests 打分。
- 组内比较减少对 value model 的依赖。
- 数学和代码的 reward 更明确。

和 PPO/DPO 的区别：

| 方法 | 数据/反馈 | 是否在线采样 | 关键点 |
| --- | --- | --- | --- |
| PPO | reward model / reward | 是 | policy ratio、critic/value、KL |
| DPO | chosen/rejected | 否，偏离线 | 直接偏好优化，不显式 RM |
| GRPO | 同 prompt 多回答 reward | 是 | group relative advantage |
| RLVR | 可验证 reward 的 RL 范式 | 常在线 | reward 来自 checker/tests/verifier |

30 秒答案：

> GRPO 可以看成 reasoning RL 里一种更省 value model 的优化方法。它对同一 prompt 采样一组回答，用可验证 reward 打分，再用组内相对优势更新模型。RLVR 是更大的范式，强调 reward 可验证；GRPO 是其中常用的优化算法之一。

## 10. Rejection Sampling 和 Reasoning Distillation

Rejection sampling：

```text
model samples many solutions
-> checker/verifier filters high-quality ones
-> keep good traces
-> SFT / distill / build preference data
```

优点：

- 不一定马上做 RL。
- 可以用 verifier 过滤高质量数据。
- 适合构建 reasoning SFT 数据。

风险：

- 只保留成功样本，覆盖不了失败边界。
- verifier 偏差会进入训练集。
- 长推理 trace 可能让学生模型过度思考。

Distillation：

```text
strong reasoning model -> generate verified traces
smaller/student model -> learn answer style and reasoning pattern
```

面试表达：

> 拒绝采样是用模型自己生成候选，再用 checker 或 verifier 过滤好样本。它可以给后续 SFT 或蒸馏提供数据。推理蒸馏则把强模型的解题模式迁移给小模型，但必须过滤错误、控制长度，并保留简单题的短答能力。

## 11. Process Supervision vs Outcome Supervision

Outcome supervision：

```text
只看最终答案对不对
```

Process supervision：

```text
看每一步推理是否合理
```

过程监督的优势：

- 更早发现错误步骤。
- 可用于搜索和早停。
- 对复杂多步问题更细粒度。

过程监督的成本：

- 需要 step-level 标注。
- 标准不一定统一。
- 模型过程不一定忠实于真实内部推理。
- 可能学会写“看起来合理”的过程。

面试表达：

> Outcome supervision 成本低，但只能告诉模型最终结果好坏；process supervision 更细，可以帮助定位中间错误和训练 PRM，但标注成本高，也不保证过程忠实。实际系统常把最终 checker、PRM、judge 和人工抽检结合起来。

## 12. Reward Hacking 和 Verifier Hacking

Reasoning RL 最大风险之一是模型学会“钻奖励漏洞”。

常见形式：

- 数学答案格式投机。
- 代码硬编码测试。
- 生成看似严谨但错误的过程。
- 利用 judge 偏好写冗长答案。
- 迎合 verifier 的模式而不真正解决问题。
- 工具调用任务里通过不合理副作用让检查通过。

治理：

- hidden tests。
- 多 verifier / 多 judge。
- human audit。
- adversarial eval。
- format 和 semantic 双校验。
- held-out task 和分布外测试。
- 监控 reward 与真实指标偏离。

30 秒答案：

> RLVR 不是没有 reward hacking。只要 reward 有漏洞，模型就可能利用它。数学可能格式投机，代码可能过拟合 public tests，judge 可能偏好冗长答案。所以要用隐藏测试、多评委、人工抽检、对抗评估和线上 bad case 回流来治理。

## 13. 答案对但过程错怎么办

这在 reasoning model 面试里非常常见。

情况：

1. 最终答案正确，但中间步骤有错误。
2. 过程看起来合理，但其实模型是猜到答案。
3. 过程冗长，包含无关推理。
4. 过程泄露敏感信息或内部策略。

处理：

- 训练侧：PRM、过程过滤、短推理蒸馏、错误 trace 删除。
- 评估侧：final answer + process quality 双指标。
- 产品侧：不直接展示原始 hidden chain-of-thought，只给简洁解释。
- 安全侧：过滤敏感中间信息。

面试表达：

> 对用户来说最终答案重要，但对训练和安全来说过程也重要。最终答案正确不代表过程可靠，所以数学/代码可以用最终 checker，再配合过程质量抽检或 PRM。产品上也不一定展示原始思维链，更常见的是输出可读摘要或关键依据。

## 14. 评估：pass@1、pass@k、cons@k、verifier accuracy

常用指标：

| 指标 | 含义 | 场景 |
| --- | --- | --- |
| pass@1 | 一次回答正确率 | 默认能力 |
| pass@k | k 个候选中至少一个正确 | 代码/数学多采样 |
| cons@k | k 次采样投票后的正确率 | Self-Consistency |
| verifier accuracy | verifier 判断候选好坏能力 | rerank/search |
| cost per solved problem | 每解出一道题的成本 | 上线收益 |
| latency P95/P99 | 延迟尾部 | 产品体验 |

pass@k 直觉：

```text
采样 k 个候选，只要有一个正确就算成功
```

但上线不能只看 pass@k：

- 用户通常只看到一个最终答案。
- 需要 selector/verifier 把正确候选选出来。
- k 越大成本越高。

面试表达：

> pass@k 衡量“能不能采到正确答案”，pass@1 衡量“一次能不能答对”。如果没有好的 verifier，pass@k 高也不代表最终输出好。reasoning 系统还要看 cons@k、verifier rerank、成本和 P95 延迟。

## 15. 数据怎么构造

Reasoning post-training 常见数据：

| 数据类型 | 作用 |
| --- | --- |
| cold-start reasoning traces | 给模型初始可读推理格式 |
| verifiable tasks | RLVR reward 来源 |
| rejection sampled traces | 高质量 SFT / distill 数据 |
| preference pairs | DPO/RM/偏好训练 |
| process labels | PRM / 过程监督 |
| hard negatives | 训练 verifier 和模型识别错误 |
| safety reasoning data | 教模型在高风险任务里推理拒答 |

构造原则：

- 可验证优先。
- 任务难度分层。
- 保留失败样本和边界样本。
- 去重和防污染。
- 控制 trace 长度。
- 区分简单题短答和难题长思考。

面试表达：

> Reasoning 数据不是只收长 CoT。要按难度、任务类型和验证方式组织：简单题保留短答，难题保留高质量推理，数学代码用 checker 过滤，开放任务用 rubric/judge/人工抽检，还要防数据污染和长思考滥用。

## 16. 线上怎么部署 reasoning model

不要把 reasoning model 当普通 chat model 直接全量替换。

推荐链路：

```text
request
-> intent and difficulty classifier
-> model/router chooses normal vs reasoning model
-> budget policy sets max tokens / N / verifier
-> generation/search
-> validation/verifier
-> final answer
-> trace/eval/cost logging
```

预算控制：

- 简单问题走普通模型。
- 中等问题走 short reasoning。
- 难题走 long reasoning / multi-sample。
- 高风险工具调用必须 verifier 或人工确认。

监控：

- 任务成功率。
- final answer accuracy。
- tool success。
- refusal correctness。
- reasoning token cost。
- P95/P99 latency。
- timeout / early stop rate。
- user feedback。

30 秒答案：

> Reasoning model 上线要做路由和预算控制。不是所有请求都长思考，简单题走普通模型，难题才分配更多 reasoning token、多采样或 verifier。监控上除了质量，还要看 reasoning token 成本、P95 延迟、早停率和 bad case 回流。

## 17. 面试高压问题

### Q1：RLVR 会取代 RLHF 吗？

不会。RLVR 适合可验证任务，RLHF/偏好数据适合开放式 helpfulness、语气、安全和主观偏好。真实 post-training 往往组合 SFT、RLVR、偏好优化、安全数据和评估闭环。

### Q2：Verifier 越强越好吗？

不是。Verifier 太慢会增加系统成本；如果偏差大，会把模型带偏；如果只覆盖部分规则，会引发 hacking。要看 verifier accuracy、coverage、latency 和和真实指标的相关性。

### Q3：为什么 test-time scaling 不是无脑增加 N？

收益随 N 递减，成本近似线性增加。简单题 N=1 就够，极难题 N 很大也未必采到正确解。更合理的是按难度动态分配预算，并设置早停。

### Q4：为什么隐藏 CoT？

原始思维链可能冗长、不忠实、泄露内部策略或敏感信息，也可能给攻击者线索。产品上通常输出可读摘要、关键依据或最终答案，而不是完整 raw CoT。

### Q5：代码题为什么适合 RLVR？

因为单元测试、编译、静态检查和隐藏测试能提供可验证 reward。风险是模型可能过拟合 public tests 或写硬编码，所以要用 hidden tests、多样化测试和代码审查。

## 18. 8 分钟项目讲法模板

如果你要讲一个 reasoning/RLVR/verifier 项目，按这个结构：

1. **任务**：数学、代码、工具调用还是业务推理，成功标准是什么。
2. **数据**：题目来源、难度分层、是否可验证、是否有污染风险。
3. **模型**：base/instruct/reasoning model，是否做 SFT/RL/蒸馏。
4. **Reward/Verifier**：checker、unit tests、ORM、PRM、judge 怎么设计。
5. **训练**：cold-start、rejection sampling、GRPO/RLVR、distillation。
6. **推理**：N 次采样、Self-Consistency、Best-of-N、早停和预算。
7. **评估**：pass@1、pass@k、cons@k、成本、P95、bad case。
8. **风险**：reward hacking、过程不忠实、过度思考、成本爆炸。
9. **上线**：路由、灰度、回滚、日志、隐私和人工兜底。

收尾句：

> 我们没有把 reasoning 当成“多输出几行 CoT”，而是把可验证 reward、候选搜索、verifier、预算控制和回归评估串成闭环。

## 19. 高频追问清单

1. RLVR 和 RLHF 有什么区别？
2. Verifiable reward、Reward Model、LLM-as-Judge 怎么区分？
3. ORM 和 PRM 有什么区别？
4. verifier 怎么训练和使用？
5. Best-of-N 和 Self-Consistency 有什么区别？
6. Test-time scaling 为什么有效？
7. 为什么 test-time compute 要按难度动态分配？
8. GRPO 在 RLVR 里怎么理解？
9. PPO、DPO、GRPO、RLVR 怎么放在同一张图里？
10. process supervision 和 outcome supervision 怎么选？
11. reward hacking 怎么发生？
12. 数学和代码的 reward 怎么设计？
13. 答案对但过程错怎么处理？
14. 为什么不直接展示 raw CoT？
15. rejection sampling 和 distillation 怎么结合？
16. pass@1、pass@k、cons@k 怎么解释？
17. reasoning 数据怎么构造？
18. 线上 reasoning model 怎么控成本？
19. 怎么避免简单题过度思考？
20. RLVR/verifier 项目怎么讲 8 分钟？

## 20. 推荐阅读

- OpenAI o1 / Learning to reason with LLMs：[https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/)
- DeepSeek-R1：[https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- DeepSeekMath / GRPO：[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- Let's Verify Step by Step：[https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)
- Training Verifiers to Solve Math Word Problems：[https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
- Self-Consistency：[https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- Scaling LLM Test-Time Compute：[https://arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)
- DAPO：[https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)

## 21. 本章复习顺序

第一遍：

1. RLVR、verifiable reward、verifier。
2. ORM、PRM、process vs outcome。
3. Best-of-N、Self-Consistency、test-time scaling。
4. GRPO、rejection sampling、distillation。
5. Reward hacking、成本控制和上线。

第二遍：

- 先背 501-505：概念区分。
- 再背 506-510：test-time scaling 和训练方法。
- 再背 511-517：风险、评估和数据。
- 最后背 518-520：上线和项目讲法。
