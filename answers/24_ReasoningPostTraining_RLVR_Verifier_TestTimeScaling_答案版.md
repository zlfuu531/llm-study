# 答案版 24：Reasoning Post-training、RLVR、Verifier 与 Test-Time Scaling

对应题号：501-520。建议先读 [29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md](../29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 501. RLVR 是什么？和 RLHF 有什么区别？

30 秒版：

RLVR 是 Reinforcement Learning with Verifiable Rewards，用可自动验证的结果做 reward，比如数学答案、代码测试、工具调用结果。RLHF 更依赖人类偏好或 reward model，适合开放式 helpfulness 和风格偏好；RLVR 更适合数学、代码、规则任务，但也会有 checker 漏洞和 reward hacking。

2 分钟版：

RLVR 的核心：

```text
模型采样答案 -> checker / test / verifier 给 reward -> RL 更新模型
```

典型 reward：

- 数学最终答案是否正确。
- 代码是否通过单元测试。
- 工具调用是否完成任务。
- JSON 或字段值是否满足规则。

和 RLHF 对比：

| 维度 | RLHF | RLVR |
| --- | --- | --- |
| reward | 人类偏好 / RM | 可验证 checker / tests |
| 任务 | 开放回答、偏好、安全 | 数学、代码、工具、规则 |
| 优点 | 覆盖主观偏好 | 可规模化、反馈稳定 |
| 风险 | RM 偏差 | checker 被钻、过程不忠实 |

面试收尾：

> RLVR 不是取代 RLHF，而是补上可验证任务上的高质量反馈。真实 post-training 往往把 SFT、RLVR、偏好优化和安全对齐组合起来。

## 502. Verifiable reward、Reward Model、LLM-as-Judge 怎么区分？

30 秒版：

Verifiable reward 是程序化检查，比如答案 exact match 或单元测试；Reward Model 是训练出的评分模型；LLM-as-Judge 是用大模型按 rubric 打分。可验证 reward 稳定但覆盖有限，RM 和 judge 覆盖更广但有偏差和提示敏感问题。

2 分钟版：

三者区别：

| 名称 | 本质 | 例子 |
| --- | --- | --- |
| Verifiable reward | 规则/程序/测试 | 数学答案、unit test |
| Reward Model | 学出来的打分器 | chosen/rejected 偏好训练 |
| LLM-as-Judge | 大模型评审 | rubric 打分、pairwise judge |

使用建议：

- 数学/代码优先用 verifiable reward。
- 开放式问答需要 judge 或 RM。
- 高风险场景加人工抽检。

风险：

- checker 只看最终答案，可能漏过程错误。
- RM 可能被模型钻漏洞。
- judge 可能有位置偏差、长度偏差、模型偏好。

面试表达：

> 我会优先用可复现的程序化 reward；覆盖不了的主观维度再用 judge/RM，并固定 judge 版本、prompt、rubric 和采样参数，做人工抽检和回归。

## 503. ORM 和 PRM 有什么区别？

30 秒版：

ORM 是 Outcome Reward Model，看最终答案或完整回答；PRM 是 Process Reward Model，看每一步推理。ORM 成本低，适合 Best-of-N；PRM 更细，可以指导搜索和发现中间错误，但需要 step-level 标注，成本更高。

2 分钟版：

ORM：

```text
完整答案 -> reward
```

PRM：

```text
step1 -> reward
step2 -> reward
...
```

对比：

| 维度 | ORM | PRM |
| --- | --- | --- |
| 粒度 | 最终结果 | 中间步骤 |
| 标注 | 便宜 | 贵 |
| 用法 | rerank、Best-of-N | search、早停、过程监督 |
| 风险 | 答案对但过程错 | 过程看似对但不忠实 |

面试加分：

> PRM 更适合复杂多步推理，因为它能在中途发现错步；但过程标注成本高，且模型写出来的过程不一定等于真实内部推理，所以还要结合最终 answer checker。

## 504. Verifier 怎么训练，推理时怎么用？

30 秒版：

Verifier 可以用正确/错误候选、chosen/rejected、step-level 标签或程序检查结果训练。推理时对同一问题采样多个候选，让 verifier 打分，选最高分，或者在 tree search 中指导展开和早停。

2 分钟版：

训练数据：

- prompt + candidate + correct/incorrect。
- prompt + chosen/rejected。
- step + good/bad。
- 代码候选 + tests pass/fail。

训练目标：

- 二分类：候选是否正确。
- pairwise ranking：chosen 分数高于 rejected。
- regression：预测 reward。
- step-level scoring：每步是否可靠。

推理用法：

```text
prompt
-> sample N candidates
-> verifier scores candidates
-> choose best / vote / continue search
```

注意：

- verifier 要在 held-out 数据上评估。
- 不能只看 verifier loss，要看最终选择正确率。
- verifier 太慢会拖垮系统。
- verifier 偏差会引导模型 reward hacking。

## 505. Best-of-N、Self-Consistency、Majority Vote 有什么区别？

30 秒版：

Best-of-N 是采样 N 个候选，用 verifier 或 reward 选最高分。Self-Consistency 是采样多条推理路径，按最终答案一致性投票。Majority vote 更强调答案多数投票。Best-of-N 依赖 verifier，Self-Consistency 依赖多样性和答案归一化。

2 分钟版：

Best-of-N：

```text
sample N -> verifier score -> argmax
```

Self-Consistency：

```text
sample N reasoning paths -> extract answers -> vote
```

Majority Vote：

```text
choose most frequent final answer
```

区别：

| 方法 | 选择依据 | 适合 |
| --- | --- | --- |
| Best-of-N | verifier 分数 | 有可靠 verifier |
| Self-Consistency | 多推理路径一致性 | 数学/常识推理 |
| Majority Vote | 最终答案频率 | 答案可归一化 |

风险：

- N 大成本高。
- 候选不多样时收益小。
- 错误答案也可能多数。
- verifier 质量差会选错。

## 506. Test-Time Scaling 为什么有效？

30 秒版：

因为复杂题第一次生成可能走错路径，多花推理计算可以让模型长思考、多采样、自我检查、调用 verifier 或工具，从而提高正确率。它在数学、代码、规划任务更有效，但会增加 token、延迟和成本。

2 分钟版：

Test-time scaling 方式：

- 长思考。
- 多采样。
- Self-Consistency。
- Best-of-N。
- verifier rerank。
- tree search。
- 工具执行反馈。

为什么有效：

1. 多采样覆盖更多解法。
2. verifier 筛掉坏解。
3. 长思考提供自检和修正机会。
4. 代码和数学可以用测试/答案检查。

为什么不是万能：

- 简单题多想浪费。
- 极难题多采样也可能找不到解。
- verifier 不可靠会选错。
- 成本可能超过收益。

面试结论：

> Test-time scaling 是用推理成本换质量，核心是按难度动态分配预算。

## 507. 为什么 test-time compute 要按难度动态分配？

30 秒版：

因为不同问题的边际收益不同。简单题 N=1 就够，难题才需要长思考、多采样或 verifier；极难题超过预算也可能没收益。动态分配能在固定成本下提高整体质量和性价比。

2 分钟版：

固定策略的问题：

```text
所有请求都 N=16:
简单题浪费
难题可能仍不够
P95 延迟和成本暴涨
```

动态策略：

```text
先估计难度/不确定性
简单题直接答
中等题少量采样
难题启用 verifier/search/tool
达到置信度或预算上限就停
```

难度信号：

- 模型置信度。
- 多样本是否一致。
- verifier 分数。
- 任务类型。
- 历史 bad case。
- 用户 SLA 和风险等级。

面试表达：

> Compute-optimal 不是一直增加 compute，而是在每个 prompt 上决定值不值得花更多 compute。

## 508. GRPO 在 RLVR 里怎么理解？

30 秒版：

GRPO 是一种适合同 prompt 多采样的 reasoning RL 方法。对同一道题采样一组回答，用可验证 reward 打分，再用组内均值和方差构造相对优势，减少对 value model 的依赖。RLVR 是范式，GRPO 是常用优化方法之一。

2 分钟版：

流程：

```text
prompt -> sample G responses
-> checker/reward scores
-> normalize within group
-> update policy
```

优势：

```text
A_i = (r_i - mean(r_group)) / std(r_group)
```

为什么适合：

- 数学/代码可以自动打分。
- 同一 prompt 内回答可比较。
- 不显式依赖 value/critic model。

和其他方法：

- PPO：在线 RL，通常有 value/critic。
- DPO：离线 chosen/rejected，不需要在线采样。
- GRPO：在线多采样，用组内相对 advantage。
- RLVR：强调 reward 可验证。

## 509. PPO、DPO、GRPO、RLVR 怎么放在同一张图里？

30 秒版：

PPO、DPO、GRPO 是优化方法或训练算法，RLVR 是用可验证 reward 做 RL 的训练范式。PPO 用 reward 和 value/critic 做在线 RL；DPO 用离线偏好对直接优化；GRPO 用同 prompt 多回答的组内相对 reward；RLVR 里可以用 GRPO 这类算法。

2 分钟版：

表格：

| 方法 | 核心数据 | 是否在线采样 | 关键词 |
| --- | --- | --- | --- |
| PPO | reward / RM | 是 | policy ratio、critic、KL |
| DPO | chosen/rejected | 否 | 直接偏好优化、reference |
| GRPO | 同 prompt 多回答 reward | 是 | group advantage |
| RLVR | 可验证 reward | 常在线 | checker/test/verifier |

面试口语：

> 如果面试官问“RLVR 和 GRPO 区别”，我会说 RLVR 是 reward 来源和训练范式，GRPO 是一种具体优化算法。DeepSeekMath/R1 相关讨论里，GRPO 常用于可验证 reward 场景。

## 510. Process supervision 和 outcome supervision 怎么选？

30 秒版：

Outcome supervision 只看最终答案，便宜、易规模化；process supervision 看每一步，能发现中间错误、指导搜索，但 step-level 标注贵，也不保证过程忠实。数学难题或 tree search 更适合 PRM，普通任务可以先用 outcome。

2 分钟版：

Outcome：

```text
answer correct -> reward
answer wrong -> penalty
```

Process：

```text
step1 good
step2 bad
step3 affected
```

怎么选：

- 最终答案易检查，预算有限：outcome。
- 复杂多步、需要搜索：process。
- 需要解释质量：process + final check。
- 开放任务：rubric / judge / human audit。

风险：

- outcome 可能放过错误过程。
- process 标注成本高。
- 过程可能只是“写出来的解释”，不忠实。

## 511. Reward hacking / verifier hacking 怎么发生？

30 秒版：

只要 reward 有漏洞，模型就可能利用漏洞而不真正解决任务。数学可能格式投机，代码可能硬编码 public tests，judge 可能偏好冗长答案。治理要用 hidden tests、多 verifier、人工抽检、对抗 eval 和 bad case 回流。

2 分钟版：

常见形式：

- 答案格式投机。
- 代码过拟合公开测试。
- 过程写得漂亮但逻辑错。
- 生成冗长文本迎合 judge。
- 工具调用通过副作用骗过检查。

治理：

- hidden tests。
- 多 judge / 多 checker。
- final answer + process 双评估。
- held-out benchmark。
- 人工审核。
- 对抗样本。
- reward 与业务指标相关性监控。

面试表达：

> RLVR 的 reward 更客观，但不是免疫 hacking。checker 设计就是系统安全边界的一部分。

## 512. 数学和代码的 reward 怎么设计？

30 秒版：

数学 reward 可以看最终答案 exact match、等价变形、单位和格式；代码 reward 可以看编译、单测、隐藏测试、静态检查和性能。都要防止格式投机、答案碰巧对、代码硬编码和测试泄漏。

2 分钟版：

数学：

- final answer exact match。
- 数值容差。
- 符号等价。
- 单位检查。
- 格式合法。
- 可选过程评分。

代码：

- 编译/语法。
- public tests。
- hidden tests。
- lint/static analysis。
- runtime/memory。
- 安全规则。

风险：

- 标准答案解析错。
- 模型猜中最终答案但过程错。
- 代码硬编码。
- 单测覆盖不足。
- benchmark 泄漏。

面试收尾：

> 我会把 reward 拆成 correctness、format、safety、efficiency，多维度监控，而不是只用一个最终分数。

## 513. 答案对但过程错怎么办？

30 秒版：

最终答案正确不代表推理可靠。训练侧可以用 PRM、过程过滤和高质量 trace 蒸馏；评估侧同时看 final answer 和 process quality；产品侧通常不展示 raw CoT，而是给简洁可读解释或关键依据。

2 分钟版：

问题场景：

- 猜对答案。
- 中间公式错但结果碰巧对。
- 过程冗长无关。
- 过程暴露敏感策略。

处理：

1. 数学/代码先用 final checker 保证结果。
2. 对高风险或教学场景加过程检查。
3. 用 PRM 或人工抽检过滤 reasoning traces。
4. 蒸馏时控制 trace 长度和质量。
5. 产品输出简洁解释，不暴露 raw CoT。

面试表达：

> 对训练来说，过程错会污染模型；对产品来说，过程错会误导用户。所以最终答案和过程质量要分开评估。

## 514. 为什么很多 reasoning model 不直接展示 raw CoT？

30 秒版：

Raw CoT 可能冗长、不忠实、泄露内部策略或敏感信息，还可能被攻击者利用。产品上更常见的是展示简洁解释、关键依据或推理摘要，而不是完整原始思维链。

2 分钟版：

不直接展示的原因：

- 可能包含错误或无关过程。
- 不一定忠实于真实推理。
- 暴露系统策略和安全规则。
- 泄露用户或工具中间信息。
- 输出成本高、体验差。
- 给 jailbreak 或攻击提供线索。

替代方案：

- answer + concise rationale。
- 引用证据。
- 关键步骤摘要。
- tool trace 的安全摘要。
- 对专家用户提供可审计日志，但加权限和脱敏。

面试表达：

> 训练时可以利用推理轨迹，但产品展示要考虑忠实性、安全和体验。

## 515. Rejection sampling 和 distillation 怎么结合？

30 秒版：

先让强模型或当前模型采样多条解法，再用 checker/verifier 过滤正确且质量高的 trace，最后用这些数据做 SFT 或蒸馏给小模型。关键是过滤错误、控制长度、保留简单题短答能力。

2 分钟版：

流程：

```text
sample many solutions
-> verify / score
-> keep high quality
-> SFT / distill / preference data
```

收益：

- 比纯人工写 trace 便宜。
- 能快速扩充 reasoning 数据。
- 小模型能继承强模型解题模式。

风险：

- teacher 错误被放大。
- 数据只剩成功样本，缺少边界失败。
- trace 太长导致过度思考。
- verifier 偏差污染 student。

治理：

- hidden eval。
- 长度控制。
- 去重。
- 人工抽检。
- 难度分层。

## 516. pass@1、pass@k、cons@k 怎么解释？

30 秒版：

pass@1 是一次回答正确率；pass@k 是 k 个候选里至少一个正确；cons@k 是多次采样投票或一致性后的正确率。pass@k 高说明能采到正确答案，但还需要 verifier 或 selector 把正确答案选出来。

2 分钟版：

解释：

```text
pass@1: 单次输出是否正确
pass@k: k 次采样里是否至少一个正确
cons@k: k 次采样投票后的最终答案是否正确
```

区别：

- pass@1 更接近普通用户体验。
- pass@k 衡量候选空间里有没有正确答案。
- cons@k 衡量投票/一致性策略是否有效。

项目里还要看：

- cost per solved problem。
- P95 latency。
- verifier selection accuracy。
- token cost。
- timeout rate。

面试表达：

> 如果没有好的 verifier，pass@k 不能直接变成最终能力。

## 517. Reasoning / RLVR 数据怎么构造？

30 秒版：

按任务类型和难度构造：数学、代码、工具、业务推理；优先选择可验证样本，做去重、防污染和难度分层。数据可以来自 cold-start traces、rejection sampling、偏好对、process labels、hard negatives 和安全样本。

2 分钟版：

数据类型：

- cold-start 推理轨迹。
- 可验证题目和答案。
- 单元测试和隐藏测试。
- rejection sampled 高质量解法。
- chosen/rejected 偏好对。
- PRM step labels。
- hard negatives。
- 安全推理和拒答数据。

原则：

- 可验证优先。
- 难度覆盖。
- 控制 trace 长度。
- 保留简单题短答。
- 防 benchmark 污染。
- 保留失败和边界样本。

面试表达：

> Reasoning 数据不是越长越好，而是要可验证、难度分层、质量可控，并能覆盖最终上线场景。

## 518. 线上 reasoning model 怎么控成本？

30 秒版：

用路由、预算和早停控制。简单请求走普通模型或短思考，难题才启用 reasoning、多采样、verifier 或工具。设置 max reasoning tokens、sample count、verifier call 上限，监控 P95 延迟、token 成本和任务成功率。

2 分钟版：

线上链路：

```text
request
-> intent/difficulty/risk classifier
-> choose model and budget
-> generate/search/verify
-> early stop or final answer
-> log quality/cost
```

控制项：

- max output tokens。
- max reasoning tokens。
- sample count N。
- verifier calls。
- tool calls。
- timeout。
- SLA 分级。

策略：

- simple: normal model。
- medium: short reasoning。
- hard: long reasoning + verifier。
- high-risk: human-in-the-loop。

面试表达：

> 控成本不是砍模型，而是把 reasoning compute 用在值得用的问题上。

## 519. 怎么避免简单题过度思考？

30 秒版：

做难度路由和预算策略。简单题直接短答或普通模型，中等题短思考，难题才长思考或多采样。还可以用 early stop、max reasoning tokens、answer style 控制和蒸馏短推理数据。

2 分钟版：

过度思考的坏处：

- 延迟高。
- 成本高。
- 用户体验差。
- 简单问题反而答复杂。
- 可能引入无关错误。

解决：

1. intent/difficulty classifier。
2. confidence threshold。
3. max reasoning tokens。
4. early stop。
5. short-answer mode。
6. 简单题短 trace 蒸馏。
7. 私有 eval 分桶统计。

面试表达：

> Reasoning 能力要可控。真正上线时要让模型知道什么时候该想、什么时候该直接答。

## 520. RLVR / Verifier / Test-Time Scaling 项目怎么讲 8 分钟？

30 秒版：

按任务、数据、reward/verifier、训练、推理策略、评估、成本和风险讲。核心是证明你不是只会说“长思考”，而是能把可验证 reward、多候选搜索、verifier、预算控制和 bad case 闭环做成系统。

8 分钟结构：

1. 任务：

> 我们要提升数学/代码/业务推理，成功标准是 final answer / tests / task success。

2. 数据：

> 数据按难度分层，做去重、防污染和可验证检查。

3. Reward：

> 数学用答案 checker，代码用单测和隐藏测试，开放维度用 judge/human audit。

4. 训练：

> cold-start SFT、rejection sampling、RLVR/GRPO、distillation。

5. 推理：

> 简单题直接答，难题多采样、Self-Consistency、Best-of-N 或 verifier rerank。

6. 评估：

> pass@1、pass@k、cons@k、verifier accuracy、P95、cost per solved problem。

7. 风险：

> reward hacking、过程不忠实、overthinking、测试泄漏和成本爆炸。

8. 上线：

> 路由、预算、早停、灰度、回滚、日志和 bad case 回流。

收尾：

> 这个项目的关键不是让模型输出更长，而是让它在可验证反馈和预算约束下更可靠地解决难题。

## 本组题的复习顺序

1. 先背 501-505：RLVR、verifier、ORM/PRM 和多采样方法。
2. 再背 506-510：test-time scaling、compute routing、GRPO 和监督方式。
3. 再背 511-517：reward hacking、过程质量、蒸馏、指标和数据。
4. 最后背 518-520：上线控成本、避免过度思考和项目讲法。

## 延伸阅读

- OpenAI o1 / Learning to reason with LLMs：[https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/)
- DeepSeek-R1：[https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- DeepSeekMath / GRPO：[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
- Let's Verify Step by Step：[https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)
- Training Verifiers to Solve Math Word Problems：[https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
- Self-Consistency：[https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- Scaling LLM Test-Time Compute：[https://arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314)
- DAPO：[https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
