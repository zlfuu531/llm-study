# 答案版 15：LLMOps、模型网关与可观测性

对应题号：321-340。建议先读 [20_LLMOps模型网关与可观测性面试.md](../20_LLMOps模型网关与可观测性面试.md)，再用本文件做口语化复述。

## 321. LLMOps 和传统 MLOps 有什么不同？

一句话：LLMOps 不只管模型权重，还要管 prompt、上下文、工具、路由、评估、成本和安全。

传统 MLOps 主要关注数据、特征、训练、模型版本、部署、监控和漂移。LLMOps 也需要这些，但 LLM 应用的输出还强依赖 prompt 模板、检索结果、工具调用、模型路由和安全策略。比如同一个模型，换一个 prompt 或换一个检索索引，线上行为就可能完全不同。

面试可以答：共同点是都要版本、部署、监控和回滚；区别是 LLMOps 要把运行时组件也纳入工程治理，包括 prompt version、retriever/index version、tool schema、router policy、eval set 和 token 成本。

## 322. 模型网关需要做哪些事？

模型网关是业务系统和多个模型供应商之间的统一治理层。它通常负责鉴权、租户识别、请求规范化、prompt 版本、模型路由、token 预算、限流、retry/fallback、安全过滤、streaming、trace、成本统计、eval 采样和灰度回滚。

不要只说“转发请求”。高分答案要强调：模型网关把散落在业务代码里的 SDK 调用收敛起来，让模型调用可控、可观测、可审计、可回滚。

## 323. 为什么 LLM 限流不能只按 QPS？

因为 LLM 请求成本和压力主要由 token 决定，不只是请求次数。一个 200 token 输入的小请求和一个 80k token 输入、4k token 输出的长上下文请求，都是 1 QPS，但对 prefill、decode、显存、延迟和 API 成本的压力完全不同。

生产系统通常同时限制 requests/min、input tokens/min、output tokens/min、concurrent requests、max context length、max generation tokens 和 Agent tool steps。面试再补一句：限流既是保护系统，也是保护租户预算和服务 SLO。

## 324. 模型路由怎么设计？

模型路由可以按任务类型、难度、上下文长度、延迟要求、成本预算、安全等级、供应商可用性和历史 eval 分数来设计。

例子：简单 FAQ 用小模型；复杂推理用强模型；长文档问答走长上下文模型；高风险工具操作走强安全策略或人工确认；主供应商 429/5xx 时 fallback 到备用供应商。

关键风险：不同模型的 tool calling、JSON schema、上下文窗口和拒答风格可能不同，不能无脑 fallback。路由策略本身也要版本化，方便复盘。

## 325. fallback 和 retry 怎么设计？

retry 是同一模型或同一供应商再试；fallback 是切到备用模型、备用供应商或降级策略。设计时先分类错误：429、5xx、网络超时可以退避、排队或 fallback；4xx 参数错误通常不能 retry；prompt 太长要压缩或换长上下文模型；工具失败要判断工具是否幂等。

还要设置总超时预算。比如用户 SLO 是 3 秒，就不能主模型等 2.8 秒失败后再 fallback 5 秒。日志里要记录 retry_count、fallback_from、fallback_to、fallback_reason、timeout_budget 和 final_status。

## 326. prompt 版本管理要记录什么？

至少记录 prompt_id、prompt_version、owner、task_type、model_family、system prompt、模板变量、tool_schema_version、retriever_version、index_version、safety_policy_version、eval_set_version、创建时间和变更原因。

核心思想是：prompt 改动要像代码一样管理。上线前跑离线 eval，通过后小流量灰度，监控质量、延迟、成本和安全，异常时能回滚到上一版本。

## 327. trace 里应该记录哪些字段？

基础字段包括 request_id、trace_id、tenant_id、app_id、user_hash、prompt_id、prompt_version、model_provider、model_name、router_policy_version、input_tokens、output_tokens、TTFT、TPOT、total_latency、error_code、retry_count、fallback_reason、eval_score 和用户反馈。

RAG/Agent 还要记录 retrieval query、doc_ids、rerank scores、context builder 结果摘要、tool_name、tool_args_hash、tool_status 和 safety check 结果。注意不要默认全量记录用户原文和工具返回，要做脱敏、采样、权限控制和保留期限。

## 328. OpenTelemetry/GenAI telemetry 在 LLMOps 里解决什么？

它解决的是 LLM 调用可观测字段不统一的问题。OpenTelemetry GenAI 语义约定给模型 provider、operation、request/response model、token usage、prompt/completion events 等定义统一语义，方便把模型调用接入标准 tracing 和 monitoring 系统。

面试不要背字段表，讲价值：统一语义、串起 RAG/工具/模型调用链路、降低供应商绑定。再补安全点：prompt 和 completion 可能含敏感信息，不能默认全量采集。

## 329. LLM 服务要监控哪些指标？

分五类讲最清楚：

- 质量：task success、correctness、groundedness、retrieval hit、tool success、format pass、human approval、用户反馈。
- 性能：E2E latency、TTFT、TPOT、P95/P99、queue time、retrieval/rerank/tool latency。
- 成本：input/output tokens、cost/request、cost/tenant、cache hit、fallback cost、long-context ratio。
- 可靠性：4xx/5xx、timeout、retry rate、fallback rate、provider availability、schema parse failure。
- 安全：policy violation、prompt injection、PII detected、unauthorized tool blocked、unsafe output blocked。

一句话：只看延迟不够，因为 LLM 的问题可能是答得快但错、答得对但太贵，或者能跑但越权。

## 330. token 成本怎么治理？

先定位成本来源：请求量涨了、输入 token 涨了、输出 token 涨了、fallback 涨了、长上下文比例涨了，还是 Agent 工具循环了。再按租户、应用、功能、prompt 版本和模型拆账。

治理手段包括 token budget、max context、max output、模型路由、prompt 精简、上下文压缩、response/embedding/retrieval/prefix cache、离线任务批处理、quota 和成本告警。不要只说换小模型，成本要和质量、安全、延迟一起权衡。

## 331. cache 有哪些类型，风险是什么？

常见有 response cache、embedding cache、retrieval cache、rerank cache、prefix/prompt cache、context cache 和 tool result cache。

风险主要是 stale、权限泄露和个性化错误。企业知识库场景尤其要注意 tenant_id、permission_scope、prompt_version、model_name、retriever_version、index_version、tool_schema_version 都应该进入 cache key。不能因为 A 用户查过有权限的文档，就让 B 用户命中同一个回答。

## 332. 灰度上线和 A/B 怎么做？

流程可以讲：离线 eval 过门槛，shadow traffic 旁路验证，1% canary 小流量，观察质量、延迟、成本、安全指标，再逐步放量。异常触发阈值时自动暂停或回滚。

A/B 要注意用户分桶稳定，对话类产品按 session 或 user 分桶，不要每轮随机。指标不能只看点击率，还要看人工质检、投诉率、cost/request、安全违规率和长期留存。

## 333. 回滚要回滚哪些东西？

LLM 应用的回滚对象很多：prompt version、model version、router policy、retriever/reranker、embedding model、vector index、chunking strategy、tool schema、safety policy 和 output parser。

面试关键句：LLM 输出由模型、prompt、context、tool 和策略共同决定，所以回滚不是只回模型，而是回滚影响输出的整条运行时配置。

## 334. 线上 hallucination/答错事故怎么复盘？

先用 request_id 找 trace，看 prompt 版本、模型、retrieval docs、rerank 分数、context builder、tool call、输出和 safety post-check。然后分类：是召回没命中、上下文噪声太多、证据排序错、模型没遵循证据、prompt 引导错误，还是工具返回错。

定位后把样本加入私有 eval，修对应环节，例如改 chunk/query rewrite/rerank、加 grounded prompt、加拒答策略或改工具 schema。最后通过离线 eval 和小流量灰度验证。

## 335. 工具调用事故怎么复盘？

先看 tool trace：模型为什么选择这个工具、参数怎么生成、参数是否通过 schema 校验、权限是否通过、工具返回是什么、是否有副作用、是否重复调用。再判断问题来自 prompt、tool description、schema 设计、权限系统、工具实现还是模型推理。

止血手段包括关闭高风险工具、提高人工确认阈值、限制 tool steps、加参数校验、加幂等 key、回滚 tool schema 或 router policy。高风险工具一定要有 audit log 和 human-in-the-loop。

## 336. 日志和隐私怎么平衡？

原则是可观测不等于裸记录一切。prompt、completion、检索文档和工具返回都可能包含 PII、密钥和内部数据。应该做脱敏、采样、hash user_id、字段级访问控制、日志保留期限和审计。

还要区分排障日志和训练数据。排障日志用于复盘，不应该自动进入训练集；进入数据闭环前要重新做授权、脱敏、质量筛选和安全审查。

## 337. 如何做线上 eval 和 human feedback？

线上 eval 通常包括 sampled judge、人工抽检、用户反馈、客服工单、低置信样本回流和安全样本监控。高风险业务不能只靠 LLM-as-judge，要有人审和规则兜底。

闭环是：线上 trace 采样 bad case，标注错误类型，进入私有 eval，修改 prompt/retrieval/tool/model，离线 eval 过门槛后灰度上线。这样 eval 集会随着真实问题变强。

## 338. 多模型多供应商怎么做治理？

先用模型网关统一接口和鉴权，再做 provider/model abstraction，把 messages、tools、streaming、response format、timeout 和错误码统一。然后在路由层根据任务、成本、延迟、安全和可用性选择模型。

治理上要做供应商配额、SLA、成本、数据出境策略、fallback、兼容性测试和 eval 对比。不同供应商输出格式和 tool calling 行为可能不同，必须有适配层和回归测试。

## 339. LLMOps 项目怎么讲 8 分钟？

按背景、目标、架构、关键设计、指标、难点、结果讲。

背景：多个 AI 功能分散调用模型，成本、限流、prompt 版本和排障不可控。目标：建设统一模型网关，支持多模型路由、租户限流、prompt registry、trace、成本归因、fallback 和灰度回滚。架构：Client -> API Gateway -> Auth/Tenant -> Model Gateway -> Router -> Provider，并把 RAG、tool call、LLM call 接入 trace。

关键设计讲多维限流、模型路由、版本化、trace 字段、eval gate 和回滚。指标讲 P95/P99、TTFT、TPOT、错误率、fallback rate、cost/request、cache hit、eval pass、安全违规率。难点讲模型兼容、隐私日志、fallback 质量和权限缓存。

## 340. 模型网关和可观测性的核心原则是什么？

核心原则是：把不可控的模型调用变成可治理的工程系统。

具体展开：所有影响输出的组件都要版本化；所有线上请求都要能通过 trace 复盘；所有变更都要经过 eval、灰度和可回滚；所有成本都要能归因；所有高风险动作都要有权限、安全和审计；所有 bad case 都要能进入反馈闭环。

最后可以用一句收束：

> LLMOps 做得好，不是因为用了某个工具，而是因为系统在答错、变慢、变贵、越权或供应商失败时，都能定位、止血、复盘和持续改进。
