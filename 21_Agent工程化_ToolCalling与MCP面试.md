# Agent 工程化、Tool Calling 与 MCP 面试

这一章不是重新解释“Agent 是什么”，而是把 2025-2026 面试更爱追的工程化问题补全：状态机怎么设计、工具 schema 怎么写、流式 tool calling 怎么解析、MCP 到底是哪一层、多智能体什么时候有用、Agent 怎么评估、线上失败怎么恢复。

先背一句：

```text
生产级 Agent = LLM + 状态机 + 工具注册表 + 权限系统 + 参数校验 + 执行沙箱 + trace/eval + 失败恢复。
```

如果只靠 prompt 让模型自由发挥，它更像 demo；如果能把工具、状态、权限、评估和回滚都管住，才更像生产系统。

## 1. 为什么 Agent 面试从概念转向工程化

早期常问：

- Agent 和 Chatbot 区别是什么？
- ReAct 是什么？
- Function Calling 是什么？
- MCP 是什么？

现在更常见的追问是：

- Agent 为什么会死循环？
- 工具调用参数错了怎么修？
- 工具很多时怎么选择和裁剪？
- 高风险工具怎么做人审和回滚？
- MCP server 暴露文件系统会有什么风险？
- 多 Agent 是真的提升效果，还是只增加复杂度？
- 怎么评估一个 Agent，不只看最终答案？
- 线上用户说任务执行错了，怎么回放工具轨迹？

这说明面试官不满足于听名词，而是想知道你能不能把 Agent 做成一个可控系统。

## 2. Agent Harness 是什么

Harness 可以理解成包在大模型外面的工程外壳。它不负责“让模型变聪明”，而是负责让模型的行为更可控。

```text
User Goal
  -> Task Router
  -> State Manager
  -> Context Builder
  -> Planner / Policy
  -> Tool Selector
  -> Argument Validator
  -> Permission Guard
  -> Tool Executor / Sandbox
  -> Observation Parser
  -> State Update
  -> Stop / Replan / HITL
  -> Final Response
```

Prompt Engineering、Context Engineering、Harness Engineering 的区别：

| 层次 | 关注点 | 典型问题 |
| --- | --- | --- |
| Prompt Engineering | 怎么写指令 | system prompt、few-shot、格式要求 |
| Context Engineering | 给模型什么信息 | 历史、RAG、工具结果、memory、token budget |
| Harness Engineering | 怎么约束执行过程 | 状态、工具、权限、重试、trace、eval、回滚 |

面试表达：

> Agent Harness 的价值是把不稳定的模型决策放进可观测、可校验、可恢复的执行框架里。模型负责生成动作建议，系统负责校验、授权、执行、记录和终止。

## 3. Agent Loop 怎么设计

一个最小 Agent loop：

```text
while not done and step < max_steps:
    context = build_context(state, memory, observations)
    action = model_decide(context, available_tools)
    checked = validate(action, schema, permission, risk)
    observation = execute_or_reject(checked)
    state = update_state(state, action, observation)
    done = judge_done(state, observation)
```

每一步都要有边界：

- `available_tools`：按任务、租户、权限动态筛工具，不能把所有工具一次性暴露。
- `model_decide`：可以是 ReAct、Plan-and-Execute、状态图节点或策略模型。
- `validate`：检查工具名、参数类型、必填字段、枚举范围、权限和风险等级。
- `execute_or_reject`：处理超时、重试、幂等、熔断、人工确认和回滚。
- `update_state`：写入任务状态，而不是只堆对话历史。
- `judge_done`：不能只靠模型说“完成了”，要结合任务条件、工具结果和输出校验。

终止条件要显式设计：

```text
达到目标
达到最大步数
工具连续失败
用户缺少必要信息
权限不足
成本预算耗尽
触发高风险动作，转人工确认
```

## 4. ReAct、Plan-and-Execute、状态图怎么选

| 模式 | 适合场景 | 风险 |
| --- | --- | --- |
| ReAct | 短任务、探索式工具调用、信息检索 | 容易循环，对工具描述敏感 |
| Plan-and-Execute | 多步骤任务、先拆解再执行 | 初始计划错会传导，需要 replan |
| State Machine / State Graph | 企业流程、审批、工单、订单、代码修复 | 设计成本高，但可控性强 |
| Workflow + Agent | 固定流程里少量节点让模型决策 | 灵活性受限，但稳定性好 |
| Multi-Agent | 角色分工、审阅、辩论、复杂协作 | 通信成本和协调复杂度高 |

面试建议：

> 生产里我更倾向“workflow 包 Agent”，而不是完全开放的 autonomous agent。固定流程用状态机保证边界，少数不确定环节交给模型决策。

## 5. Tool Calling 的本质

Tool Calling / Function Calling 的本质不是模型真的执行函数，而是模型输出结构化动作意图：

```text
模型输出：我要调用 tool_name，参数是 arguments
应用侧：校验参数、检查权限、执行工具
工具返回：observation
模型继续：基于 observation 生成下一步或最终答案
```

工具 schema 要写清：

```text
name: 工具名，稳定且语义明确
description: 什么时候用，什么时候不用
parameters: JSON schema
required: 必填字段
enum: 可选值范围
additionalProperties: false
strict: true
returns: 返回结构和错误码
risk_level: low / medium / high
idempotent: true / false
```

好的 schema 不只是给模型看，也是给系统做校验、权限、审计和评估用。

## 6. Structured Outputs 和 strict schema 为什么重要

工具调用最常见问题是参数不合法、字段缺失、枚举值乱填、JSON 解析失败。Structured Outputs / strict schema 的作用是让模型输出尽量贴合约束，而不是“尽力而为”。

如果面试继续追问 JSON mode、JSON Schema、strict schema、constrained decoding、output parser、retry/repair 和 schema valid 为什么不等于业务正确，转到 [27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md](27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md)。

面试讲法：

> 如果模型调用工具只是生成一段自由文本，应用侧很难稳定解析。把工具参数定义成 JSON schema，并开启严格模式，可以提高参数结构稳定性。即使如此，应用侧仍然要做二次校验，因为 schema 只能约束格式，不能保证业务语义正确。

仍要注意：

- schema 太复杂会增加模型选择和生成难度。
- 工具太多会污染上下文，要做 tool retrieval 或按任务裁剪。
- `strict` 约束格式，不等于工具调用一定正确。
- 高风险工具不能只靠 schema，要加权限和人审。

## 7. 流式 Tool Calling 怎么解析

流式输出时，模型可能边生成边补工具参数。常见问题：

- 参数 JSON 被拆成多个 chunk。
- 多个 tool call 并行出现。
- partial arguments 暂时不是合法 JSON。
- 用户前端想实时展示“正在调用哪个工具”。
- 中途取消后工具是否已经执行不清楚。

工程处理方式：

```text
buffer tool_call_delta
  -> 按 tool_call_id 聚合
  -> 等 arguments 完整后 parse JSON
  -> schema validate
  -> permission check
  -> execute
  -> append observation
```

不要在参数还没完整时执行工具。尤其是支付、发邮件、删文件、下单、改数据库这类有副作用工具，必须等完整参数校验、人审或幂等 key 准备好后再执行。

## 8. 工具注册表和工具裁剪

工具越多，Agent 越不稳定。原因：

- 模型更难判断该用哪个工具。
- 工具描述互相重叠。
- 上下文被工具 schema 占满。
- 错误工具调用概率上升。
- 安全面扩大。

工具注册表至少记录：

```text
tool_id
tool_name
namespace
description
input_schema
output_schema
risk_level
permission_scope
timeout_ms
retry_policy
idempotent
owner
version
eval_cases
```

工具裁剪策略：

- 按任务类型筛选。
- 按用户权限筛选。
- 按当前状态筛选。
- 按风险等级筛选。
- 用 tool retrieval 先搜相关工具。
- 对相似工具做 namespace 分组，如 `crm.search_customer`、`billing.refund_order`。

## 9. 权限、沙箱和高风险动作

Agent 最大风险不是答错一句话，而是调错工具造成真实副作用。

高风险动作包括：

- 删除/覆盖文件。
- 修改数据库。
- 发邮件、发消息、发工单。
- 支付、退款、下单。
- 调用内部管理 API。
- 读取敏感文档。
- 执行 shell/code。

治理方式：

```text
least privilege
  -> permission scope
  -> read/write separation
  -> dry-run
  -> human approval
  -> idempotency key
  -> audit log
  -> rollback plan
```

面试里要强调：模型不能绕过权限。权限检查应该在工具执行层完成，而不是只在 prompt 里提醒模型“不要越权”。

## 10. MCP 到底是哪一层

MCP，即 Model Context Protocol。它不是另一个 ReAct，也不是 Function Calling 的同义词。它解决的是模型应用如何标准化连接外部工具、资源和提示模板。

核心结构：

```text
MCP Host
  -> MCP Client
      -> MCP Server
          -> Tools / Resources / Prompts
```

角色：

- Host：模型应用宿主，例如 IDE、桌面助手、Agent 平台。
- Client：Host 内部维护连接的组件，负责和某个 server 通信。
- Server：暴露能力的服务，可以本地 stdio，也可以远程 HTTP。

核心 primitives：

- Tools：可执行动作，例如查库、发请求、读写系统。
- Resources：上下文数据，例如文件、文档、数据库 schema。
- Prompts：可复用提示模板和工作流模板。

和 Tool Calling 的关系：

```text
Tool Calling: 模型如何表达“我要调用哪个工具和参数”
MCP: 工具/资源/提示词如何被发现、连接、授权和调用
Agent Framework: 如何规划、管理状态、编排工具、处理失败和评估
```

一句话：

> Agent 框架可以用 Tool Calling 让模型提出动作，再通过 MCP server 去执行或读取外部能力。

## 11. MCP 安全边界

MCP 的强大来自“连接外部系统”，风险也来自这里。

主要风险：

- server 暴露过多资源。
- tool 描述被当成可信指令，造成 tool poisoning。
- 本地 server 拥有文件系统或 shell 权限。
- 远程 server 鉴权不足。
- server 请求 LLM sampling 时泄露 prompt 或用户数据。
- 工具执行没有用户确认。
- client 缓存旧工具 schema，server 已经变更。

安全原则：

- 用户对数据访问和工具执行要有明确 consent。
- Host 不应无授权把用户数据发给 server。
- 工具描述、资源内容都可能是不可信输入。
- server 的权限要最小化。
- 高风险工具要二次确认和审计。
- 远程 server 要做鉴权、速率限制、超时和日志。

面试高分点：

> MCP 本身提供协议抽象，但不能替应用自动解决安全问题。真正的安全要在 Host、Client、Server、工具执行层和用户授权流程一起做。

## 12. Memory 怎么设计才不污染

Agent memory 可以分四类：

| 类型 | 内容 | 风险 |
| --- | --- | --- |
| Short-term | 当前对话窗口 | 超长、噪声、lost in the middle |
| Working memory | 当前任务状态 | 状态丢失、重复执行 |
| Long-term memory | 用户偏好、长期事实 | 错误记忆、隐私风险 |
| Retrieval memory | 可检索历史经验/案例 | 权限泄露、过期信息 |

写入原则：

- 不把临时推测写成长期事实。
- 不把敏感信息默认写入长期记忆。
- 写入要有来源、时间、置信度和可删除机制。
- 记忆检索要受用户/租户权限控制。
- 错误记忆要能纠正或失效。

面试表达：

> Memory 不是把聊天历史全塞回上下文，而是把任务状态、长期偏好和可检索经验分层管理。最重要的是写入门槛、权限隔离、过期机制和纠错。

## 13. 多智能体什么时候有用

Multi-Agent 常见模式：

- Supervisor：一个主控 Agent 分配任务给多个 worker。
- Debate：多个 Agent 给出观点，再由 judge 汇总。
- Role-based：研究员、执行者、审阅者、测试者分工。
- Critic/Reviewer：一个 Agent 生成，另一个检查。
- Planner/Executor：规划和执行分离。

适合：

- 任务天然需要角色分工。
- 需要审阅、批判或交叉验证。
- 单个 Agent 上下文太复杂。
- 代码/文档/研究类任务可以拆分。

不适合：

- 简单问答。
- 强实时低延迟任务。
- 工具副作用高、协调成本高的任务。
- 没有评估集，只想用多 Agent 显得高级。

面试提醒：

> 多 Agent 不一定更强，它常常用更多 token 和更复杂的状态换来更好的分工或审阅。要用任务成功率、成本、延迟和安全指标证明收益。

## 14. Human-in-the-loop 怎么设计

HITL 不是“人工兜底”四个字，而是一套触发和恢复机制。

触发条件：

- 高风险工具执行。
- 低置信度。
- 权限不明确。
- 金额/影响范围超过阈值。
- 连续工具失败。
- 输出涉及合规、安全、医疗、法律、金融。

HITL 流程：

```text
Agent proposes action
  -> system summarizes intent, parameters, risk
  -> human approve / edit / reject
  -> execute with audit log
  -> feed decision back to state
```

高分点：人审界面不能只显示模型结论，要显示工具名、参数、数据来源、影响范围和回滚方式。

## 15. Agent Eval 怎么做

Agent 评估要看轨迹，不只看最终答案。

指标：

| 指标 | 含义 |
| --- | --- |
| task success rate | 任务是否完成 |
| tool selection accuracy | 工具是否选对 |
| argument accuracy | 参数是否正确 |
| step efficiency | 步数是否过多 |
| recovery rate | 工具失败后能否恢复 |
| safety violation rate | 是否越权或误操作 |
| cost per task | token、工具、延迟成本 |
| user intervention rate | 需要人工介入比例 |

评估集要包含：

- 正常任务。
- 工具返回空。
- 工具超时。
- 参数缺失。
- 权限不足。
- 用户中途改需求。
- prompt injection。
- 高风险动作。
- 多轮状态延续。

轨迹判分：

```text
final answer correct?
tool sequence correct?
arguments correct?
state updates correct?
unnecessary steps?
unsafe action blocked?
```

## 16. 线上故障怎么复盘

常见故障：

- 工具选错。
- 参数错。
- 工具返回空但模型编答案。
- 工具超时后无限重试。
- 状态丢失导致重复执行。
- 多 Agent 相互冲突。
- MCP server 版本变更导致 schema 不兼容。
- 高风险工具误执行。

复盘链路：

```text
request_id
  -> state timeline
  -> prompt/model version
  -> available tools
  -> selected tool
  -> arguments
  -> permission decision
  -> tool result
  -> observation parsing
  -> next action
  -> final answer
```

定位后把 case 加入 eval，不要只临时改 prompt。

## 17. 代码 Agent 和普通 Agent 的区别

代码 Agent 更强调可验证闭环：

```text
issue/task
  -> inspect repo
  -> plan
  -> edit
  -> run tests/lint
  -> inspect failures
  -> patch again
  -> summarize evidence
```

关键能力：

- 能读 repo 结构和依赖。
- 能最小化修改。
- 能运行验证命令。
- 能根据失败日志定位。
- 能避免误改无关文件。
- 能处理长上下文和多文件状态。

和普通客服 Agent 不同，代码 Agent 的强项是有测试、编译、lint、diff 这些外部可验证信号。面试里可以把它归到“Agent + tool use + verifiable feedback loop”。

代码大模型、仓库级上下文、HumanEval/MBPP/SWE-bench、patch、测试闭环和代码执行沙箱的完整深挖见：[23_代码大模型_CodeAgent与SWEbench面试.md](23_代码大模型_CodeAgent与SWEbench面试.md)。

## 18. 项目 8 分钟讲法

```text
背景：
我们要做一个能调用内部工具完成业务任务的 Agent，而不是只聊天。任务包括查询、生成、修改和提交。

架构：
Client -> Agent Harness -> State Manager -> Context Builder
-> Tool Registry -> Permission Guard -> Tool Executor -> Trace/Eval

核心设计：
1. 用状态机管理任务阶段，避免纯 prompt 自由循环。
2. 工具统一注册 schema、风险等级、权限范围、超时和重试策略。
3. 模型只生成 tool call 意图，应用侧做 schema 校验和权限检查。
4. 高风险动作走 dry-run 和 human approval。
5. 线上 trace 记录 state、tool、arguments、observation、cost 和错误。
6. 用固定任务集评估 task success、tool accuracy、argument accuracy、step efficiency 和 safety violation。

难点：
工具很多时选择不稳定；工具参数和业务语义都要校验；状态容易丢；MCP server 和工具版本变更会影响线上行为；日志里有隐私和权限问题。

结果：
把原来的聊天式 demo 变成了可评估、可审计、可回放、可灰度的 Agent 执行系统。
```

## 19. 高频追问快答

### 为什么不要让 Agent 自由循环？

自由循环容易成本失控、重复调用工具、状态漂移和误执行。生产里要用最大步数、状态机、终止条件、工具白名单、预算和 trace 约束。

### 工具参数错怎么办？

先 schema 校验；缺字段就让用户补充或让模型修正；类型错直接拒绝；业务语义错要靠业务校验；高风险动作不能自动修完就执行，要人审。

### 工具很多怎么办？

不要全塞上下文。按任务、权限、状态和风险裁剪；相似工具 namespace 分组；必要时先做 tool retrieval，再把少量候选工具暴露给模型。

### MCP 和插件/普通 API 有什么区别？

普通 API 是具体系统接口；插件常是某个平台的一套扩展机制；MCP 是模型应用连接工具、资源和提示模板的开放协议层。MCP server 可以包装普通 API，但还要处理能力发现、schema、资源、授权和通信。

### Multi-Agent 怎么证明有效？

用 eval 对比单 Agent 和多 Agent：任务成功率是否提升，错误率是否下降，审阅是否抓到问题，同时成本、延迟和安全风险是否可接受。

## 20. 面试前背诵版

生产级 Agent 不是“LLM 加几个工具”，而是 LLM、状态机、工具注册表、参数校验、权限系统、执行沙箱、trace、eval 和失败恢复的组合。Tool Calling 的本质是模型输出结构化动作意图，真正执行、鉴权、重试和审计在应用侧。MCP 是模型应用连接外部工具、资源和提示模板的协议层，和 Function Calling 不是同一层。Agent 评估要看任务成功率、工具选择、参数准确率、步骤数、恢复能力、成本和安全违规率。线上 bad case 要用 request_id 回放状态、可用工具、tool call、arguments、permission、observation 和最终输出，再把 case 回流到 eval。

## 本轮参考

- Model Context Protocol Specification：[https://modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
- MCP Authorization Specification：[https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- OpenAI Function Calling / Tool Calling：[https://developers.openai.com/api/docs/guides/function-calling](https://developers.openai.com/api/docs/guides/function-calling)
- LangSmith / LangGraph Agent Deployment：[https://docs.langchain.com/langsmith/deployment](https://docs.langchain.com/langsmith/deployment)
- Microsoft AutoGen 文档：[https://microsoft.github.io/autogen/stable/](https://microsoft.github.io/autogen/stable/)
- ReAct 论文：[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- ToolBench 论文：[https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789)
- tau-bench 论文：[https://arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045)
- SWE-bench 论文：[https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
