# 答案版 16：Agent 工程化、Tool Calling 与 MCP

对应题号：341-360。建议先读 [21_Agent工程化_ToolCalling与MCP面试.md](../21_Agent工程化_ToolCalling与MCP面试.md)，再用本文件做口语复述。

## 341. Agent 工程化和普通 Agent demo 有什么区别？

普通 demo 往往是 prompt 加几个工具，让模型自由决定怎么做。工程化 Agent 要把模型放进可控执行框架里，包括状态机、工具注册表、参数校验、权限系统、执行沙箱、trace、eval、失败恢复和灰度回滚。

面试可以说：demo 关注“能不能跑通一次”，工程化关注“能不能稳定、可复盘、可审计、可控成本地跑很多次”。

## 342. Agent Harness 负责什么？

Agent Harness 是包在模型外面的工程外壳，负责管理任务状态、上下文、工具列表、模型动作、参数校验、权限、工具执行、observation、终止条件和 trace。

核心观点：模型只负责提出动作建议，Harness 负责让动作被校验、授权、执行和记录。这样 Agent 不会完全依赖模型自觉。

## 343. Agent loop 怎么设计？

基本循环是：读状态，构造上下文，让模型选择动作，校验动作，执行工具，读取 observation，更新状态，判断是否结束。

关键不是循环代码，而是边界条件：可用工具要按权限裁剪；参数要 schema 校验；工具执行要有超时、重试和幂等；终止不能只靠模型说完成，要结合任务条件、工具结果和输出校验。

## 344. ReAct、Plan-and-Execute、状态图怎么选？

ReAct 适合短任务和探索式工具调用，但容易循环。Plan-and-Execute 适合长任务，先规划再执行，但初始计划错会传导。状态图适合企业流程、审批、订单、工单、代码修复等需要强约束的场景。

生产里常用“workflow 包 Agent”：固定流程用状态机保证边界，不确定节点交给模型决策。这样稳定性通常比完全自由 Agent 更好。

## 345. Tool Calling 的本质是什么？

Tool Calling 的本质是模型输出结构化动作意图：要调用哪个工具、参数是什么。真正执行工具、检查权限、处理错误、记录审计都在应用侧。

所以面试不要说“模型调用了函数”，更准确是“模型建议调用函数，系统校验后执行”。这也是为什么工具层必须做权限、schema 和业务校验。

## 346. 好的 tool schema 应该包含什么？

至少包含 tool name、description、parameters JSON schema、required 字段、enum 范围、additionalProperties、strict、返回结构、错误码、risk_level、permission_scope、timeout、retry_policy、idempotent、owner 和 version。

description 要写清什么时候用、什么时候不用。schema 不仅给模型看，也给系统做校验、审计、权限控制和 eval。

## 347. Structured Outputs / strict schema 解决什么问题？

它主要解决工具参数结构不稳定的问题，比如字段缺失、类型错误、枚举乱填、JSON 解析失败。严格 schema 可以让模型输出更贴合参数约束。

但它不能保证业务语义正确。比如 `order_id` 格式对了，不代表用户有权限退款，也不代表订单状态允许退款。所以应用侧仍要做业务校验和权限校验。

## 348. 流式 Tool Calling 怎么解析？

流式 tool call 的参数可能分多个 chunk 到达。系统应该按 tool_call_id 聚合 delta，等 arguments 完整后再 parse JSON、schema validate、permission check，然后执行工具。

不能看到半截参数就执行，尤其是支付、发邮件、删文件、改数据库这类有副作用工具。中途取消时还要明确工具是否已经执行，避免用户以为取消了但动作已经发生。

## 349. 工具很多时怎么做工具裁剪？

不要把所有工具都塞进上下文。可以按任务类型、用户权限、当前状态、风险等级和租户策略筛选；相似工具用 namespace 分组；工具生态很大时先做 tool retrieval，只暴露少量候选工具。

工具裁剪的目的有三个：减少上下文成本、降低误调用概率、缩小安全攻击面。

## 350. Agent 权限和沙箱怎么设计？

原则是最小权限。工具执行层必须检查权限，不能只靠 prompt 约束模型。读写工具分离，高风险动作先 dry-run，再 human approval，执行时用 idempotency key，最后写 audit log。

如果涉及文件、shell、数据库、支付、发消息等副作用工具，要有沙箱、超时、资源限制、回滚方案和人审阈值。

## 351. MCP、Tool Calling、Agent 框架怎么区分？

Tool Calling 是模型表达工具调用意图的方式；MCP 是模型应用连接外部工具、资源、提示模板的协议；Agent 框架负责规划、状态、记忆、工具编排和失败恢复。

可以组合起来：Agent 框架让模型通过 Tool Calling 生成动作，再通过 MCP server 调用外部工具或读取资源。

## 352. MCP 的 Host、Client、Server、Tools、Resources、Prompts 分别是什么？

Host 是模型应用宿主，比如 IDE、桌面助手、Agent 平台。Client 是 Host 内部和某个 MCP server 连接的组件。Server 是暴露能力的服务。

Tools 是可执行动作，Resources 是上下文数据，Prompts 是可复用提示模板或工作流模板。面试要把这些角色讲清，不要只说“MCP 是连接工具”。

## 353. MCP 有哪些安全风险？

主要风险包括 server 暴露过多资源、本地 server 权限过大、远程 server 鉴权不足、工具描述被当成可信指令、server 诱导 LLM sampling 泄露数据、工具执行没有用户确认、schema 版本变化导致调用错误。

高分答案：MCP 提供协议抽象，但不能自动解决安全。安全要靠 Host、Client、Server、工具执行层和用户授权流程共同完成。

## 354. Agent memory 怎么避免污染？

不要把所有聊天历史都当 memory。要区分 short-term、working memory、long-term memory 和 retrieval memory。写入长期记忆要有来源、时间、置信度、权限和可删除机制。

避免污染的关键是写入门槛、过期机制、纠错机制和权限隔离。临时推测不能写成长期事实，敏感信息不能默认写入长期记忆。

## 355. Multi-Agent 什么时候有用，什么时候不要用？

有用场景：任务天然有角色分工，需要审阅、辩论、交叉验证，或者单 Agent 上下文太复杂。比如代码修改可以有 implementer 和 reviewer，研究任务可以有 researcher 和 critic。

不要用的场景：简单问答、强实时任务、高风险工具操作、没有 eval 证明收益的任务。多 Agent 常常增加 token、延迟、状态和协调成本，必须用指标证明它值得。

## 356. Human-in-the-loop 怎么设计？

先定义触发条件：高风险工具、低置信度、权限不明确、金额或影响范围超过阈值、连续工具失败、涉及合规安全。触发后让 Agent 提出动作，系统摘要工具名、参数、证据、风险和影响范围，由人 approve、edit 或 reject。

人审后要把决策写回 state 和 audit log。人审界面不能只显示模型结论，要显示可验证证据和回滚方式。

## 357. Agent eval 要看哪些指标？

要看任务轨迹，不只看最终答案。指标包括 task success rate、tool selection accuracy、argument accuracy、step efficiency、recovery rate、safety violation rate、cost per task 和 user intervention rate。

评估集要覆盖正常任务、工具返回空、工具超时、参数缺失、权限不足、用户中途改需求、prompt injection、高风险动作和多轮状态延续。

## 358. 线上 Agent 工具调用错了怎么复盘？

用 request_id 回放 trace：当时的 state、prompt/model version、available tools、selected tool、arguments、permission decision、tool result、observation parsing、next action 和 final answer。

然后分类：工具选择错、参数错、权限漏检、工具返回错、observation 解析错、状态更新错，还是模型没遵循结果。定位后把 case 加入 eval，再修 schema、prompt、权限、状态机或工具实现。

## 359. 代码 Agent 和普通 Agent 有什么区别？

代码 Agent 更强调可验证反馈闭环。它需要 inspect repo、理解依赖、编辑文件、运行 tests/lint、读取失败日志、再 patch。普通客服 Agent 可能主要依赖用户反馈和工具返回，代码 Agent 有编译、测试、diff 这些强验证信号。

面试可以说：代码 Agent 是 tool use + state management + verifiable feedback loop 的典型场景，关键是最小化修改和用验证命令闭环。

## 360. Agent 工程化项目怎么讲 8 分钟？

按背景、目标、架构、关键设计、指标、难点、结果讲。

背景：要做能调用内部工具完成业务任务的 Agent，而不是聊天 demo。架构：Client -> Agent Harness -> State Manager -> Context Builder -> Tool Registry -> Permission Guard -> Tool Executor -> Trace/Eval。

关键设计：状态机管理任务阶段；工具注册 schema、风险等级、权限范围、超时和重试；模型只生成 tool call 意图；应用侧做参数和权限校验；高风险动作走 dry-run 和 human approval；trace 记录 state、tool、arguments、observation、cost 和错误。指标看任务成功率、工具准确率、参数准确率、平均步骤、安全违规率和成本。
