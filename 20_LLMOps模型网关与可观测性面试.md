# LLMOps、模型网关与可观测性面试

这一章面向大模型应用开发、AI Infra、平台工程、RAG/Agent 系统设计和企业落地方向。面试官问 LLMOps，通常不是想听“我用了 LangSmith / Langfuse / LiteLLM”，而是想看你能不能把一个 demo 变成可上线、可观测、可回滚、可控成本的系统。

如果涉及 reasoning model 的线上成本、长思考预算、多采样、verifier 调用、早停和路由策略，配合 [29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md](29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md) 看。

如果涉及实时语音 Agent 的 ASR/LLM/TTS 串联、barge-in、streaming、低延迟、录音隐私和工具误触发，配合 [31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md](31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md) 看。

先记住一句话：

```text
LLMOps = 把模型调用、prompt、上下文、工具、评估、成本、安全和线上事故都纳入工程闭环。
```

一条生产链路可以这样讲：

```text
Client
  -> API Gateway
  -> Auth / Tenant / Quota
  -> Model Gateway
  -> Router / Rate Limit / Retry / Fallback / Budget
  -> Orchestrator
  -> Prompt / Context / Tool / Model
  -> Trace / Eval / Cost / Feedback
  -> Canary / Rollback / Bad-case Loop
```

## 1. 为什么 2025-2026 高频

早期大模型项目常见形态是“调一个模型 API + 写 prompt + 接一个向量库”。面试追问也偏 RAG、Agent、LoRA、推理指标。到了 2025-2026，很多团队真正要把 AI 功能接进业务系统，问题会变成：

- 多个模型供应商怎么统一调用？
- 一个请求花了多少钱、慢在哪里、错在哪里？
- prompt 改了以后怎么灰度和回滚？
- 用户说答错了，能不能回放当时的 retrieval、prompt、tool call 和模型输出？
- 一个租户刷爆 token 怎么限流？
- 模型不可用时是否可以 fallback？
- 日志里能不能记录用户原文？PII 怎么处理？
- Agent 调错工具造成损失，怎么审计和复盘？

所以 LLMOps 本质上不是“工具栈名词”，而是大模型系统上线后的治理能力。

## 2. LLMOps 和传统 MLOps 的区别

传统 MLOps 关注数据集、特征、训练、模型版本、部署、监控和漂移。LLMOps 也需要这些，但会多出几个 LLM 特有变量。

| 维度 | 传统 MLOps | LLMOps |
| --- | --- | --- |
| 核心产物 | 模型权重、特征管道、预测服务 | 模型 + prompt + context + tool + router + eval |
| 版本对象 | 数据、特征、模型、代码 | prompt 模板、模型版本、检索索引、embedding 模型、工具 schema、路由策略、安全策略 |
| 线上输入 | 结构化特征较多 | 自然语言、长上下文、多轮对话、工具返回 |
| 输出特征 | 分类/回归/排序较稳定 | 生成式输出，不确定性更强 |
| 评估方式 | 指标较清晰，如 AUC、F1、RMSE | 私有 eval、LLM-as-judge、人工抽检、bad case 分析 |
| 成本 | 机器资源为主 | token 成本、外部 API 成本、工具调用成本、长上下文成本 |
| 风险 | 数据漂移、模型退化 | 幻觉、越权、prompt injection、PII 泄露、工具误操作 |

面试可以这样答：

> LLMOps 和 MLOps 的共同点是都要做版本、部署、监控和回滚。区别是 LLM 应用的行为不仅由模型权重决定，还由 prompt、检索上下文、工具、路由和安全策略共同决定，所以 LLMOps 要把这些运行时组件也版本化、可观测、可评估、可回滚。

## 3. 模型网关需要做什么

模型网关不是简单把请求转发给 OpenAI-compatible API。它是业务系统和多模型服务之间的控制面。

```text
请求进入
  -> 鉴权和租户识别
  -> 请求规范化
  -> prompt 模板和版本选择
  -> 安全预检查
  -> token 预算估算
  -> 模型路由
  -> 限流、排队、熔断
  -> retry / fallback
  -> streaming 输出
  -> 日志、trace、成本归因
  -> eval 采样和反馈回流
```

常见职责：

| 职责 | 解决的问题 | 面试要点 |
| --- | --- | --- |
| 统一接口 | 不同模型 API 参数不同 | 统一 message、temperature、tools、stream、response format |
| 鉴权/租户 | 谁能用、用多少 | user、tenant、app、role、quota |
| 限流 | 防止打爆模型和预算 | QPS、tokens/min、concurrency、max context、max output |
| 路由 | 不同任务用不同模型 | 小模型/大模型、长上下文模型、代码模型、本地/云 |
| retry/fallback | 处理超时、429、5xx | 幂等性、超时预算、降级质量、记录原因 |
| prompt 版本 | prompt 可灰度可回滚 | prompt id、version、owner、eval gate |
| 安全过滤 | 过滤注入、敏感内容、越权工具 | pre-check、post-check、tool permission |
| 观测和审计 | 知道慢在哪里、错在哪里 | trace、span、tokens、latency、cost、error |
| 成本归因 | 谁花了多少钱 | tenant、app、feature、model、token |
| 灰度回滚 | 变更可控 | prompt/model/router/index/tool schema 都要能回滚 |

一句面试版：

> 模型网关的价值是把模型调用从“业务代码里散落的 SDK 调用”收敛成一个统一治理层，统一做鉴权、路由、限流、成本、安全、trace、灰度和回滚。

## 4. 为什么 LLM 限流不能只按 QPS

LLM 请求的压力不只取决于请求数量，还取决于 token 数、上下文长度和输出长度。

两个请求都是 1 次调用：

```text
请求 A：input 300 tokens，output 100 tokens
请求 B：input 80k tokens，output 4k tokens
```

对成本、显存、TTFT、TPOT、队列和供应商配额的压力完全不同。所以线上通常至少限制：

- requests per minute：防突发请求。
- input tokens per minute：控制 prefill 压力和外部 API 成本。
- output tokens per minute：控制 decode 压力和流式输出成本。
- concurrent requests：控制并发占用。
- max context length：防止超长 prompt 拖垮 TTFT。
- max generation tokens：防止无限生成和成本攻击。
- tool call steps：防止 Agent 循环调用工具。

一个更工程化的预算公式：

```text
estimated_cost =
  input_tokens * input_price_per_token
  + expected_output_tokens * output_price_per_token
  + retrieval_cost
  + tool_cost
```

限流策略可以按租户、用户、应用、接口、模型分别配置：

```text
tenant_budget > app_budget > user_budget > request_budget
```

面试高分点：限流不是只为了保护系统，也是为了保护用户预算和业务 SLO。

## 5. Model Router 怎么设计

模型路由解决“哪个请求用哪个模型”的问题。最简单的路由是写死模型名，稍复杂一点会基于任务、成本、延迟、安全和可用性动态选择。

如果面试继续追问候选模型怎么选、Qwen/Llama/DeepSeek 等开放权重模型怎么比较、chat template 和 tokenizer 怎么迁移，转到 [26_开源模型生态_模型选型与ChatTemplate面试.md](26_开源模型生态_模型选型与ChatTemplate面试.md)。

如果继续追问 prompt registry、结构化输出、schema 校验、constrained decoding、parser/retry 和 prompt 灰度回滚，转到 [27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md](27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md)。

常见路由维度：

| 维度 | 例子 |
| --- | --- |
| 任务类型 | 问答、总结、代码、抽取、翻译、工具调用 |
| 难度 | 简单 FAQ 用小模型，复杂推理用大模型 |
| 上下文长度 | 长文档任务走长上下文模型 |
| 延迟要求 | 实时客服优先低延迟模型，离线报告可用慢但强的模型 |
| 成本预算 | 免费用户用便宜模型，付费/关键任务用强模型 |
| 安全等级 | 高风险任务走更强安全策略或人工确认 |
| 供应商可用性 | 主模型 429/5xx 时切备用模型 |
| 历史 eval 分数 | 某类任务哪个模型效果更好就优先哪个 |

典型策略：

```text
if high_risk:
  use strong_model + safety_check + human_approval
elif long_context:
  use long_context_model
elif simple_task and low_budget:
  use small_model
elif code_task:
  use code_model
else:
  use default_model
```

不要只说“按成本路由”。真正上线要注意：

- fallback 后输出质量可能变化，要记录 model_from、model_to 和 reason。
- 不同模型的 tool calling、JSON schema、上下文窗口和拒答风格不同，不能无脑替换。
- 路由策略本身要版本化，否则事故复盘时不知道当时为什么选了这个模型。
- 需要离线 eval 和线上采样 eval 来验证路由收益。

## 6. Retry、Fallback、熔断和降级

这几个词面试常放在一起问，但含义不同。

| 机制 | 作用 | 风险 |
| --- | --- | --- |
| retry | 同一个模型/供应商再试一次 | 非幂等工具调用不能乱重试 |
| fallback | 切到备用模型、备用供应商或降级策略 | 质量、安全、格式可能变化 |
| circuit breaker | 某供应商错误率过高时短时间不再调用 | 误判会影响可用性 |
| graceful degradation | 降级为短答案、检索摘要、模板回复或人工转接 | 用户体验下降 |

错误分类很关键：

- 429：可能是限流或配额问题，可排队、退避、切备用。
- 5xx / network timeout：可能 retry 或 fallback。
- 4xx 参数错误：通常不能 retry，要修请求。
- prompt 太长：需要压缩、截断、换长上下文模型。
- JSON 解析失败：可以做 constrained decoding、repair prompt 或 schema 校验重试。
- 工具调用失败：要判断工具是否幂等，必要时人工确认。

重试要有总超时预算：

```text
total_timeout = user_slo
model_timeout + retry_backoff + fallback_timeout <= total_timeout
```

如果用户 SLO 是 3 秒，就不能模型调用 2.8 秒失败后再 fallback 5 秒。生产系统一般会记录：

```text
retry_count
fallback_from
fallback_to
fallback_reason
timeout_budget_ms
final_status
```

## 7. Prompt Registry 和版本管理

LLM 应用里，prompt 改一行可能比模型换版本影响还大。Prompt Registry 不是为了“保存 prompt 文本”，而是为了让 prompt 可审计、可灰度、可回滚、可评估。

一个 prompt 版本至少记录：

```text
prompt_id
prompt_version
owner
task_type
model_family
system_prompt
template_variables
tool_schema_version
retriever_version
index_version
safety_policy_version
eval_set_version
created_at
change_reason
```

上线流程可以这样讲：

```text
本地修改 prompt
  -> 离线 eval 过门槛
  -> 小流量灰度
  -> 监控质量、延迟、成本、安全
  -> 通过后放量
  -> 异常时回滚到上一 prompt 版本
```

面试高分点：prompt 版本不要只绑定模型，还要绑定工具 schema、检索索引和 eval 集。否则你以为 prompt 变好了，实际上可能是索引或工具返回变了。

## 8. Trace 里应该记录什么

Trace 的目的不是“多打点”，而是让 bad case 可以复盘。

面试里如果被问“trace 里应该记录什么”，不要只答请求耗时，要能把模型、prompt、检索、工具、成本和错误原因都串起来。

一个 RAG/Agent 请求至少拆成这些 span：

```text
request
  -> gateway.auth
  -> gateway.router
  -> safety.precheck
  -> retrieval.query_rewrite
  -> retrieval.search
  -> retrieval.rerank
  -> context.builder
  -> llm.call
  -> tool.call
  -> safety.postcheck
  -> eval.sample
  -> response.stream
```

关键字段：

| 字段 | 用途 |
| --- | --- |
| request_id / trace_id | 串起整条链路 |
| tenant_id / app_id / user_hash | 成本归因和租户排查，避免明文用户标识 |
| prompt_id / prompt_version | 复盘 prompt 变更 |
| model_provider / model_name / model_version | 复盘模型行为 |
| router_policy_version | 复盘为什么选这个模型 |
| input_tokens / output_tokens | 成本和性能分析 |
| TTFT / TPOT / total_latency | 性能定位 |
| retrieval_query / doc_ids / scores | RAG 复盘，不一定记录全文 |
| tool_name / tool_args_hash / tool_status | Agent 工具审计 |
| error_code / retry_count / fallback_reason | 可用性排查 |
| eval_score / feedback | 质量闭环 |

注意：trace 不等于把用户所有原文和工具输出裸写进日志。要有采样、脱敏、访问控制和保留期限。

## 9. OpenTelemetry GenAI 解决什么

OpenTelemetry 的 GenAI semantic conventions 试图给生成式 AI 调用定义统一的 telemetry 字段，例如 provider、operation、request model、response model、token usage、prompt/completion 事件等。它的价值是：不同框架和供应商的 trace 不再完全各记各的，后续可以统一接入监控系统。

面试不要背规范字段，重点讲三件事：

1. 统一语义：模型调用、token、延迟、错误、provider 等字段有统一命名。
2. 便于串链路：一次业务请求里的 RAG、tool call、LLM call 可以放到一条 trace。
3. 降低供应商绑定：Langfuse、Phoenix、LangSmith、MLflow 或自研平台都可以吸收类似概念。

安全注意：

- prompt 和 completion 可能包含隐私、密钥、业务数据，不应默认全量记录。
- 高风险工具的参数和返回要脱敏或只记录摘要/hash。
- 需要按租户、环境和数据等级控制采样率。

一句话：

> OpenTelemetry GenAI 的意义是把 LLM 调用变成可标准化观测的 span，而不是散落在业务日志里的字符串。

## 10. 可观测指标怎么分层

LLM 应用的监控不能只看接口 200 和 P95，要分质量、性能、成本、可靠性、安全五层。

### 质量指标

- task success rate：任务是否完成。
- answer correctness：答案是否正确。
- groundedness / faithfulness：回答是否基于证据。
- retrieval hit rate：召回是否命中 gold evidence。
- tool success rate：工具调用是否成功。
- format pass rate：JSON/schema/格式是否正确。
- human approval rate：人工审核通过率。
- thumbs up/down：用户反馈。

### 性能指标

- E2E latency：端到端耗时。
- TTFT：首 token 时间。
- TPOT：每 token 输出时间。
- P50/P95/P99：尾延迟。
- queue time：排队时间。
- retrieval latency、rerank latency、tool latency。
- streaming interrupted rate：流式输出中断率。

### 成本指标

- input tokens / output tokens。
- cost per request。
- cost per tenant / app / feature。
- cache hit rate。
- fallback cost。
- long-context request ratio。
- tool call cost。

### 可靠性指标

- 4xx / 5xx / timeout。
- provider availability。
- retry rate。
- fallback rate。
- circuit breaker open count。
- schema parse failure rate。
- rate-limit reject rate。

### 安全指标

- policy violation rate。
- prompt injection detected rate。
- PII detected rate。
- unauthorized tool call blocked rate。
- unsafe output blocked rate。
- data exfiltration risk alerts。

面试表达：

> 我会把监控拆成质量、性能、成本、可靠性和安全五类。只看延迟不够，因为 LLM 的线上问题经常是“答得快但错”“答得对但太贵”“能跑但越权”。

## 11. Cache 类型和风险

LLM 缓存不是只有 response cache。

| 缓存 | 作用 | 风险 |
| --- | --- | --- |
| response cache | 相同问题直接复用答案 | 个性化、权限和时效性问题 |
| embedding cache | 避免重复算 embedding | embedding 模型版本变化后要失效 |
| retrieval cache | 缓存召回结果 | 索引更新后可能 stale |
| rerank cache | 缓存排序结果 | query rewrite 或文档变化会影响 |
| prefix / prompt cache | 复用相同系统 prompt 或长前缀 | 前缀必须完全一致，且注意租户隔离 |
| context cache | 缓存构造好的上下文 | 权限和 freshness 风险 |
| tool result cache | 缓存工具返回 | 实时数据和副作用工具不能乱缓存 |

缓存 key 要包含：

```text
tenant_id / permission_scope
model_name
prompt_version
retriever_version
index_version
tool_schema_version
query_hash
```

面试提醒：企业知识库场景最怕缓存穿透权限边界。比如 A 用户查过某文档，B 用户问相似问题，不能因为命中 response cache 就把 A 有权限的内容返回给 B。

## 12. 灰度、A/B 和回滚

LLM 系统里的“版本”很多，所以灰度和回滚不能只盯模型。

可能需要灰度的对象：

- model provider / model version。
- prompt version。
- router policy。
- retriever / reranker。
- embedding model。
- vector index。
- chunking strategy。
- tool schema。
- safety policy。
- output parser。

灰度流程：

```text
离线 eval
  -> shadow traffic
  -> 1% canary
  -> 观察质量/延迟/成本/安全
  -> 逐步放量
  -> 触发阈值自动暂停或回滚
```

A/B 实验要注意：

- 用户随机要稳定，避免同一用户体验来回跳。
- 不能只看点击率，还要看人工质检、投诉率、成本和安全。
- 对话类产品要考虑 session 级别分桶，而不是每轮随机。
- 高风险动作不要直接 A/B，先 shadow 或人工审核。

回滚清单：

```text
prompt 回滚
model 回滚
router policy 回滚
retrieval index 回滚
embedding model 回滚
tool schema 回滚
safety policy 回滚
output parser 回滚
```

一句话：

> LLM 应用的回滚不是一个按钮回模型，而是要能回滚影响输出的整条运行时配置。

## 13. 线上 Eval 和反馈闭环

离线 eval 是上线前门槛，线上 eval 是上线后保险。

常见做法：

- 离线私有 eval：覆盖核心业务 query、bad case、安全样本、格式样本。
- shadow eval：新策略只旁路生成，不影响用户。
- canary eval：小流量真实用户验证。
- sampled judge：抽样用 LLM-as-judge 打分。
- human review：高风险和低置信样本人工质检。
- feedback loop：用户反馈、客服工单、低分样本进入错题库。

一个项目闭环：

```text
线上 trace
  -> 采样 bad case
  -> 标注错误类型
  -> 更新私有 eval
  -> 修改 prompt / retrieval / tool / model
  -> eval 过门槛
  -> 灰度上线
```

错误类型最好结构化：

| 类型 | 例子 | 可能修法 |
| --- | --- | --- |
| retrieval miss | 证据没召回 | 改 chunk、query rewrite、hybrid retrieval |
| context noise | 召回太多干扰 | rerank、压缩、排序 |
| hallucination | 没证据也编 | grounded prompt、拒答、post-check |
| tool error | 参数错或工具失败 | tool schema、参数校验、HITL |
| format error | JSON 不合法 | schema constrained、repair |
| safety error | 注入/越权/泄露 | 权限过滤、安全策略、审计 |

## 14. 日志、隐私和安全

LLMOps 很容易踩隐私坑，因为 prompt、retrieval 文档和工具返回都可能包含敏感信息。

原则：

- 默认不记录明文 PII、密钥、身份证、手机号、内部文档全文。
- user_id 做 hash 或 pseudonymization。
- prompt/completion 全量日志要按环境和租户开关控制。
- 高敏工具返回只记摘要、字段级脱敏或 hash。
- 日志访问要有权限控制和审计。
- retention policy 要明确，例如保留 7/30/90 天。
- 支持用户删除和合规导出时要能定位相关数据。
- eval 数据进入训练/微调前要重新做授权和脱敏。

面试可以说：

> 我会把日志分成可观测日志和训练数据两类。可观测日志用于排障，不等于自动进入训练集；进入数据闭环前要做授权、脱敏和质量筛选。

## 15. 成本治理

LLM 成本治理不只是“换便宜模型”。

常见手段：

- token budget：限制输入、输出、上下文、工具步数。
- model routing：简单任务用小模型，复杂任务用大模型。
- prompt 精简：删掉无用 few-shot 和重复系统提示。
- context compression：压缩长文档，但要评估信息损失。
- cache：response、embedding、retrieval、prefix。
- batch / async：离线任务批处理，非实时任务排队。
- eval gate：不要让低质量 prompt 上线增加返工和投诉。
- quota：按租户、用户、应用做预算。
- alert：cost/request、cost/day、long-context ratio 异常告警。

成本排查链路：

```text
总成本升高
  -> 是请求量涨、输入 token 涨、输出 token 涨，还是 fallback 涨？
  -> 是某租户、某功能、某 prompt 版本，还是某模型？
  -> 是长上下文、Agent 循环、cache miss，还是 retry 暴涨？
  -> 限制 max tokens、修 prompt、修路由、修 cache 或降级
```

面试高分点：成本要和质量一起看。不能为了省钱把关键场景答错，也不能为了追求强模型让所有简单 FAQ 都走最贵模型。

## 16. 事故复盘怎么讲

线上大模型事故常见类型：

- 模型供应商 5xx/429，服务不可用。
- 新 prompt 导致拒答率升高或幻觉变多。
- 检索索引更新导致召回错误。
- 工具 schema 改动导致 Agent 参数错。
- 安全策略过严导致大量误杀。
- 缓存没隔离租户导致权限泄露。
- Agent 循环调用工具导致成本暴涨。

复盘模板：

```text
1. 发现：哪个指标或反馈触发？
2. 定界：影响哪些租户、功能、模型、prompt 版本？
3. 止血：限流、熔断、回滚、关闭工具、人工转接？
4. 定因：trace 显示慢在检索、模型、工具还是网关？
5. 修复：修改 prompt、索引、工具 schema、路由或安全策略。
6. 验证：私有 eval + 线上小流量验证。
7. 防复发：告警、eval case、发布门禁、runbook。
```

面试官如果问“用户说答案错了你怎么查”，你可以这样答：

> 我先用 request_id 找到 trace，看当时的 prompt 版本、模型、retrieval docs、rerank 分数、tool call 和输出。然后判断是召回没命中、上下文有噪声、模型没遵循证据、工具返回错，还是安全策略影响。定位后把这个 case 加入私有 eval，再改对应环节并走灰度。

## 17. 项目 8 分钟讲法

可以按这个顺序讲一个 LLMOps/模型网关项目：

```text
背景：
业务里多个 AI 功能都在调用不同模型，SDK 分散在各服务里，成本、限流、prompt 版本和排障都不可控。

目标：
做一个统一模型网关，支持多模型路由、租户限流、prompt 版本管理、trace、成本统计、fallback 和灰度回滚。

架构：
Client -> API Gateway -> Auth/Tenant -> Model Gateway -> Router -> Model Provider
同时把 retrieval、tool call、LLM call、safety check 都接入 trace。

关键设计：
1. 按 request/token/concurrency 做多维限流。
2. 按任务类型、上下文长度、成本和安全等级做模型路由。
3. prompt、router、tool schema、retriever/index 都版本化。
4. trace 记录模型、tokens、latency、成本、fallback、retrieval docs、tool status。
5. 上线走 eval gate、小流量灰度和可回滚配置。

指标：
P95/P99、TTFT、TPOT、错误率、fallback rate、cost/request、cache hit rate、eval pass rate、安全违规率。

难点：
不同模型参数和 tool calling 行为不一致；日志里有隐私数据；fallback 会影响质量；cache 需要权限隔离。

结果：
调用链路统一了，bad case 可以复盘，prompt 变更可灰度回滚，成本能按租户和功能归因。
```

## 18. 高频追问快答

### 模型网关和 API Gateway 有什么区别？

API Gateway 更偏通用入口，做鉴权、路由、限流、负载均衡。模型网关更懂 LLM，额外处理 prompt 版本、模型路由、token 预算、流式输出、tool calling、LLM trace、成本和 eval。

### 为什么 trace 比普通日志重要？

普通日志往往只看到某个服务的局部信息。LLM bad case 需要把检索、rerank、context builder、prompt、模型、工具、安全过滤串起来看，trace 更适合还原整条调用链。

### fallback 会不会影响用户体验？

会。备用模型可能更慢、更弱、格式不同或安全策略不同。所以 fallback 要有场景白名单、超时预算、质量监控，并记录 fallback reason。

### Prompt 改动怎么上线？

prompt 改动要像代码一样走版本、评审、离线 eval、小流量灰度、监控和回滚。高风险场景不能直接全量。

### 怎么判断 LLMOps 做得好？

不是看接了多少工具，而是看 bad case 能不能复盘、变更能不能灰度回滚、成本能不能归因、线上质量能不能持续评估、安全事故能不能止血。

## 19. 面试前背诵版

LLMOps 是把大模型应用从 demo 做成生产系统的一套工程闭环。它和传统 MLOps 的区别在于，LLM 输出不仅由模型决定，还受 prompt、上下文、工具、路由、安全策略和 eval 影响，所以这些组件都要版本化、可观测、可评估、可回滚。模型网关是核心入口，负责统一模型接口、鉴权租户、token 限流、模型路由、retry/fallback、prompt 版本、安全过滤、trace、成本统计和灰度回滚。线上监控要分质量、性能、成本、可靠性和安全五类指标。bad case 复盘时用 request_id 找 trace，看 prompt 版本、模型、retrieval、tool call、tokens、latency、fallback 和输出，定位是召回、上下文、模型、工具还是安全问题，再把 case 回流到私有 eval。

## 本轮参考

- OpenTelemetry GenAI semantic conventions：[https://opentelemetry.io/docs/specs/semconv/gen-ai/](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- Langfuse 文档：[https://langfuse.com/docs](https://langfuse.com/docs)
- LangSmith 文档：[https://docs.smith.langchain.com/](https://docs.smith.langchain.com/)
- Arize Phoenix 文档：[https://arize.com/docs/phoenix](https://arize.com/docs/phoenix)
- LiteLLM 文档：[https://docs.litellm.ai/](https://docs.litellm.ai/)
- Helicone 文档：[https://docs.helicone.ai/](https://docs.helicone.ai/)
- Portkey AI Gateway 文档：[https://portkey.ai/docs](https://portkey.ai/docs)
- MLflow GenAI 文档：[https://mlflow.org/docs/latest/genai/](https://mlflow.org/docs/latest/genai/)
- OpenLLMetry 项目：[https://github.com/traceloop/openllmetry](https://github.com/traceloop/openllmetry)
