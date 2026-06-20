# 答案版 22：Prompt Engineering、结构化输出与 Constrained Decoding

对应题号：461-480。建议先读 [27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md](../27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 461. Prompt Engineering、Context Engineering、Harness Engineering 有什么区别？

30 秒版：

Prompt Engineering 关注指令怎么写，Context Engineering 关注模型拿到什么上下文，Harness Engineering 关注工具、解析、权限、重试、trace 和 eval 这些执行框架。

2 分钟版：

Prompt 是“怎么说”，比如 system prompt、任务说明、few-shot 和输出格式。Context 是“给什么材料”，比如 RAG 文档、memory、工具结果、用户状态和历史对话。Harness 是“怎么执行和兜底”，比如 Agent loop、tool runtime、output parser、权限校验、retry、监控和 eval。

面试加分句：

> 生产问题不要都归因于 prompt。RAG 答错可能是召回、排序、上下文压缩、权限、模型、parser 或安全策略的问题。

## 462. 一个好 prompt 应该包含什么？

30 秒版：

要包含角色、任务、输入字段、可用上下文、约束、输出格式、缺失信息处理、冲突处理和必要 few-shot 示例。

2 分钟版：

一个生产 prompt 至少要说明：模型扮演什么角色，要完成什么任务，输入字段是什么，哪些材料可信，不能做什么，输出格式是什么，信息不足时怎么处理，证据冲突时按什么优先级。

如果是 RAG，要说明只能根据 context 回答、引用必须来自 doc_id、context 是数据不是指令。如果是结构化抽取，要说明字段类型、缺失值、enum 和示例。如果是工具调用，要说明工具参数和权限边界。

好 prompt 的标准不是看起来详细，而是可评估、可解析、可回滚。

## 463. System prompt、user prompt、context、tool result 的优先级怎么讲？

30 秒版：

System/developer prompt 是最高层规则，user 是任务请求，retrieved context 和 tool result 是数据，不是指令。外部内容不能覆盖系统规则。

2 分钟版：

生产系统要区分指令来源权限。System prompt 定义身份、安全和格式边界；developer prompt 定义应用策略；user message 是用户需求；retrieved context 和 tool result 是证据或数据。

间接 prompt injection 的核心风险就是外部文档伪装成指令，比如“忽略前面规则”。正确做法是 prompt 里说明外部 context 只作为资料，同时系统层做权限、工具白名单和参数校验。

面试句：

> 模型可以引用外部内容的事实，但不能执行外部内容里的指令。

## 464. Few-shot 示例怎么选？

30 秒版：

Few-shot 要覆盖边界和困难样例，不是越多越好。要包含正例、反例、缺失信息、拒答和格式示例，不能泄露 test set。

2 分钟版：

示例选择原则：和业务分布接近，覆盖关键边界，输出格式和线上一致，包含失败/拒答场景。示例数量和顺序要做 ablation，因为它们会占 token，也会影响风格。

常见坑：

- 示例太长挤掉 RAG 上下文。
- 示例格式和 schema 不一致。
- 把评测集样本放进 prompt。
- 只放理想样例，线上边界崩。
- 示例诱导模型学错误风格。

## 465. CoT 什么时候有用，什么时候有风险？

30 秒版：

CoT 对数学、逻辑、多步推理和规划有用；风险是成本高、输出慢、可能不忠实、可能泄露内部推理，简单任务还会过度思考。

2 分钟版：

Chain-of-Thought 通过显式中间步骤提升复杂推理能力，适合数学、代码分析、规划和多约束问题。但生产系统要看成本和安全。长 CoT 会增加 output tokens 和延迟，且模型写出的推理过程不一定真实反映内部原因。

更稳的做法是：复杂任务路由到 reasoning 模型或内部推理流程，对用户输出简洁答案、证据和必要解释。是否用 CoT 要用私有 eval 比较质量、成本和安全。

## 466. Zero-shot CoT、Few-shot CoT、Self-Consistency 怎么区分？

30 秒版：

Zero-shot CoT 用一句话触发逐步思考；Few-shot CoT 给带推理过程的示例；Self-Consistency 多次采样推理再投票，质量可能更稳但成本更高。

2 分钟版：

Zero-shot CoT 简单便宜，适合快速试验；Few-shot CoT 依赖示例质量，能固定解题风格；Self-Consistency 用多条推理路径降低偶然错误，对数学和逻辑题有效，但成本近似乘以采样次数。

生产选择：

- 简单任务不用 CoT。
- 中等复杂任务可短推理。
- 高价值复杂任务可 self-consistency。
- 高并发低成本场景要谨慎。

## 467. ReAct、CoT、Plan-and-Execute 怎么区分？

30 秒版：

CoT 只推理不行动；ReAct 把推理和工具调用交替；Plan-and-Execute 先规划再执行，适合长流程多步任务。

2 分钟版：

CoT 适合纯推理。ReAct 的轨迹是 Thought、Action、Observation、Answer，适合搜索、RAG、Agent 工具调用。Plan-and-Execute 先生成计划，再逐步执行，适合复杂任务，但计划错误会传导。

生产系统里不一定暴露 Thought，通常把内部推理和工具轨迹留在 trace 里，对用户展示结果、引用和可解释摘要。

## 468. Structured Output 解决什么问题？

30 秒版：

解决自由文本难解析的问题，让模型输出 JSON、字段、枚举、引用等可校验结构，适合抽取、分类、工具参数和 RAG answer+citation。

2 分钟版：

自由文本对人友好，对系统不友好。结构化输出让下游可以稳定 parse 和 validate。比如企业知识库可以输出 answer、citations、confidence、needs_human_review；工具调用可以输出 tool_name 和 arguments。

指标包括 JSON valid rate、schema valid rate、字段完整率、enum 准确率、引用正确率和语义正确率。注意结构合法不等于事实正确。

## 469. JSON mode、JSON Schema、strict schema、constrained decoding 有什么区别？

30 秒版：

JSON mode 只要求合法 JSON；JSON schema 约束字段和类型；strict schema 更严格贴合 schema；constrained decoding 在生成阶段屏蔽非法 token；业务校验负责语义正确。

2 分钟版：

层级从弱到强可以这样讲：JSON mode 减少解析失败，但字段可能乱；schema 定义 required、type、enum；strict schema 尽量让模型完全按 schema；constrained decoding 在 token 级约束输出，使语法更稳定。

但最终还要业务校验。比如 citations 字段合法，不代表 doc_id 真存在；amount 是数字，不代表单位正确；tool 参数合法，不代表用户有权限。

## 470. Constrained Decoding 的原理是什么？

30 秒版：

每步生成时根据当前前缀和语法/schema 计算允许的 token，把不合法 token mask 掉，从而提高 JSON、regex、grammar 等格式的有效率。

2 分钟版：

普通解码在词表里选 token。Constrained decoding 会维护一个约束状态机或 grammar parser，根据已生成前缀判断哪些 token 仍可能形成合法输出。例如 status 字段只能选 success、failed、pending，其他 token 直接被屏蔽。

约束来源可以是 JSON schema、正则、CFG、枚举或 DSL。它能提升格式稳定性，但不能保证事实正确，也可能增加解码开销。

## 471. 为什么 schema valid 不等于业务语义正确？

30 秒版：

Schema 只约束结构和类型，不约束事实、权限、单位、引用存在性和业务规则。模型可以输出合法 JSON 但内容是错的。

2 分钟版：

例子：`{"amount": 100}` 符合 schema，但可能币种错；`{"citation": "doc_9"}` 合法，但 doc_9 不在 context；tool 参数类型正确，但用户无权限调用。

因此结构化输出后还要做：

- schema validate。
- citation validate。
- permission validate。
- business rule validate。
- consistency check。
- human review for high-risk。

面试句：

> Schema 是格式门，业务校验才是上线门。

## 472. Output parser、retry、repair 怎么设计？

30 秒版：

链路是 parse -> schema validate -> business validate，失败后按错误类型 repair、retry、fallback 或转人工，并限制重试次数。

2 分钟版：

输出先 parse JSON，再做 schema 校验，再做业务校验。JSON 解析错可以让模型修复或用 constrained decoding；字段缺失可以局部 retry；引用不存在要回到 RAG/context；权限越界必须系统拦截，高风险转人工。

Retry 不要无限循环。要记录错误类型、retry_count、fallback_reason 和最终状态。语法错误可以交给模型修，业务语义错误不能只靠模型自我纠正。

## 473. Prompt versioning 应该记录什么？

30 秒版：

记录 prompt_id、version、owner、model、chat template、tool schema、retriever/index、安全策略、eval set、变更原因和 rollback target。

2 分钟版：

Prompt 是生产资产，必须版本化。因为一次 prompt 修改可能改变质量、安全、成本和格式。事故复盘时要知道当时用的模型、prompt、工具 schema、检索索引和安全策略。

版本化还能支持 A/B、灰度、回滚和模型迁移。最好所有 trace 都记录 prompt_version 和 router_policy_version。

## 474. Prompt 改动怎么评估？

30 秒版：

用固定 eval set 对比 baseline 和 candidate，按质量、格式、安全、成本、延迟分桶评估，再 shadow/canary，最后灰度或回滚。

2 分钟版：

流程：

1. 固定 baseline prompt。
2. 准备 candidate prompt。
3. 在 valid set 上迭代。
4. 用 test set 做最终报告。
5. 分桶看常见、困难、长上下文、工具、安全。
6. 人工抽检关键 bad case。
7. shadow traffic 和小流量灰度。
8. 监控线上质量、成本和错误。

不要把 test set 反复用于调 prompt，也不要只看平均分。Prompt 改动常常某些桶变好、某些桶变差。

## 475. 为什么不能只靠 prompt 防 prompt injection？

30 秒版：

因为 prompt 是软约束，外部内容和用户输入仍可能诱导模型越权。必须用权限、工具白名单、参数校验、HITL、安全 eval、trace 和回滚。

2 分钟版：

Prompt 可以提醒模型“外部内容不是指令”，但模型仍可能被恶意文档或用户输入影响。RAG/Agent 里尤其危险，因为模型可能调用工具或泄露数据。

系统防护：

- 外部 context 当数据。
- 工具最小权限。
- 参数 allowlist。
- 高风险动作人工确认。
- 最大步骤和 token budget。
- 安全分类器。
- 审计日志和红队。

面试句：

> Prompt injection 是系统安全问题，不是文案优化问题。

## 476. Prompt 长度、成本和效果怎么权衡？

30 秒版：

长 prompt 会增加 input tokens、TTFT、prefill、KV Cache 和成本，也可能 lost-in-the-middle。要用 eval 做删减、few-shot ablation、上下文压缩和 prefix cache。

2 分钟版：

Prompt 越长不一定越好。过多规则会互相冲突，few-shot 太多会挤掉上下文，长系统提示会增加每次 prefill 成本。优化可以删重复规则、把动态材料交给 context builder、固定前缀用 prefix cache、示例做 ablation。

看指标：

- accuracy / format pass。
- input tokens。
- TTFT。
- P95。
- cache hit rate。
- cost/request。

## 477. Prompt cache / prefix cache 和 prompt 设计有什么关系？

30 秒版：

Prefix cache 只有 token 级前缀一致才有用，所以稳定 system prompt、工具说明和 few-shot 应放前面，动态用户变量、时间戳和 RAG 内容不要破坏公共前缀。

2 分钟版：

推理服务可以复用相同前缀的 KV Cache。如果 prompt 前缀稳定，比如 system prompt、工具 schema、固定示例一致，就能减少重复 prefill。若把时间戳、用户 ID、随机 trace_id 放在最前面，会导致前缀不一致，cache 命中下降。

因此 prompt 设计也影响 serving 成本。Prompt 不是只有质量问题，也影响 TTFT 和 GPU 资源。

## 478. RAG Prompt 怎么写更稳？

30 秒版：

明确只根据 context 回答，证据不足就拒答或澄清，引用必须来自 doc_id，不执行 context 中的指令，输出 answer+citation，并处理冲突证据。

2 分钟版：

RAG prompt 要强调证据边界。它应该告诉模型：context 是资料，不是指令；回答必须忠于 context；没有证据就说不确定；引用必须来自提供的 doc_id；多证据冲突时说明冲突或按新版本优先。

结构化输出常用：

```json
{"answer": "...", "citations": ["doc_1"], "confidence": 0.8}
```

再配合 citation validator，检查引用是否真实来自检索结果。

## 479. 换模型时 prompt 要怎么迁移？

30 秒版：

检查 chat template、system prompt 效果、stop tokens、JSON/schema 支持、tool calling 格式、CoT 风格、拒答风格、输出长度、token 成本和多语言表现。

2 分钟版：

迁移不是换 model name。不同模型的 role token、EOS、工具调用格式和安全风格可能不同。旧 prompt 在新模型上可能太啰嗦、格式失效、拒答过多或输出变长。

流程：

1. 建 prompt inventory。
2. 找受影响 prompt。
3. 适配 template 和 stop tokens。
4. 离线 eval 回放。
5. shadow traffic。
6. canary 灰度。
7. 保留旧模型和旧 prompt fallback。

## 480. Prompt / 结构化输出项目怎么讲 8 分钟？

30 秒版：

按背景、问题、方案、schema/parser、eval、版本化、灰度和复盘讲。重点给出 JSON 有效率、字段准确率、引用正确率、P95、成本和 bad case。

2 分钟版：

讲稿结构：

1. 背景：业务依赖 LLM 输出结构化结果或工具参数。
2. 问题：自由文本解析不稳、JSON 失败、字段错、引用错。
3. 方案：prompt 模板、schema、constrained decoding、parser、validator。
4. Eval：格式、语义、安全、成本、延迟分桶。
5. 版本：prompt、model、template、tool schema、eval set 版本化。
6. 上线：shadow、灰度、监控、fallback。
7. 风险：prompt injection、schema valid 但语义错、模型迁移。
8. 复盘：bad case 回流和自动化回归。

示例句：

> 我们把抽取任务从自由文本改成 schema 输出，应用侧做 parse、schema validate、business validate 和 citation validate。Prompt 进入 registry，每次改动跑私有 eval，线上通过灰度监控 JSON 有效率、字段准确率、P95 和成本，失败时 fallback 到旧 prompt。

## 本组题的复习顺序

1. 先背 461-464：prompt/context/harness 和 few-shot。
2. 再背 465-467：CoT、Self-Consistency、ReAct。
3. 然后背 468-472：结构化输出、constrained decoding、parser/retry。
4. 最后背 473-480：版本化、eval、安全、成本、RAG prompt、迁移和项目讲法。

## 延伸阅读

- Chain-of-Thought Prompting：[https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- Self-Consistency：[https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- ReAct：[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- Tree of Thoughts：[https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
- Hugging Face Chat Templates：[https://huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating)
- Outlines：[https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
- Guidance：[https://github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)
- JSONSchemaBench：[https://arxiv.org/abs/2501.10868](https://arxiv.org/abs/2501.10868)
