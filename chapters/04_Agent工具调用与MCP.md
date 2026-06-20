# Agent、工具调用与 MCP

## 面试目标

这一章要达到的状态：

- 能讲清 Agent 和普通 Chatbot 的区别。
- 能解释 ReAct、Plan-and-Execute、工具调用、记忆、反思。
- 能说清 Function Calling / Tool Calling / MCP 的关系。
- 能回答 Agent 为什么不稳定，以及怎么评估。

更深入的通俗版讲解见：[../deepdives/03_RAG_Agent_MCP_深挖.md](../deepdives/03_RAG_Agent_MCP_深挖.md)

## Agent 是什么

普通 Chatbot：

- 输入问题。
- 模型直接回答。

Agent：

- 有目标。
- 能规划步骤。
- 能调用工具。
- 能观察环境反馈。
- 能更新状态或记忆。
- 可能多轮执行直到完成任务。

面试表达：

> Agent 可以理解为让 LLM 从“回答器”变成“任务执行器”。它不只是生成文本，还会基于目标做规划，调用外部工具，读取工具结果，再决定下一步动作。

## ReAct

ReAct = Reasoning + Acting。

典型循环：

1. Thought：思考下一步。
2. Action：选择工具和参数。
3. Observation：读取工具结果。
4. Repeat：继续推理或给最终答案。

优点：

- 推理过程和行动过程交替。
- 适合需要外部信息或工具的任务。

缺点：

- 容易循环。
- 工具选择可能错误。
- 对 prompt 和工具描述敏感。

## Plan-and-Execute

流程：

1. Planner 先拆解任务。
2. Executor 按步骤执行。
3. 必要时 re-plan。

适合：

- 长任务。
- 多步骤任务。
- 需要明确阶段的任务。

风险：

- 初始计划错误会传导。
- 执行中环境变化需要重规划。

## 工具调用

工具调用的本质：

- 模型输出结构化工具请求。
- 系统执行工具。
- 工具结果返回给模型。
- 模型继续生成。

工具 schema 需要：

- 工具名。
- 描述。
- 参数类型。
- 必填字段。
- 返回格式。
- 错误处理约定。

常见追问：

- 如何避免模型乱调用工具？
- 工具参数错了怎么办？
- 工具调用失败怎么恢复？
- 多工具选择怎么评估？

回答要点：

- 清晰 schema。
- 参数校验。
- 权限控制。
- 工具结果摘要。
- 失败重试和降级。
- 日志和可观测性。

## Function Calling / Tool Calling

Function Calling 是让模型按约定格式输出函数名和参数。

重点：

- 模型通常不真正执行函数。
- 执行由外部应用完成。
- 模型负责决定是否调用、调用哪个、参数是什么。

面试表达：

> Function Calling 不是让模型拥有函数能力，而是让模型以结构化方式告诉系统它想调用哪个函数。真正的函数执行、权限控制、错误处理和结果返回都在应用侧完成。

## MCP

MCP，即 Model Context Protocol。

直觉：

- 以前每个应用都要为每个模型/Agent 定制工具连接方式。
- MCP 试图提供统一协议，让模型应用更标准地连接外部工具、数据源和上下文。

面试表达：

> MCP 解决的是模型应用和外部工具/数据源之间的标准化连接问题。它和 Function Calling 不完全是同一层：Function Calling 更像模型输出工具调用意图的能力，而 MCP 更关注工具、资源、上下文如何以协议方式暴露给模型应用。

常见追问：

- MCP 和 API 有什么区别？
- MCP 和 Function Calling 什么关系？
- MCP 为什么最近常被问？
- 使用 MCP 有什么安全风险？

## Agent 记忆

类型：

- 短期记忆：当前对话上下文。
- 长期记忆：用户偏好、历史任务、知识。
- 工作记忆：当前任务状态。

难点：

- 记什么。
- 什么时候写入。
- 怎么检索。
- 怎么避免错误记忆污染。
- 隐私和权限。

## Agent 为什么不稳定

原因：

- 模型规划能力不稳定。
- 工具描述不清。
- 任务过长，状态丢失。
- 工具失败或返回噪声。
- 缺少终止条件。
- 没有评估和回放。

解决：

- 任务拆分。
- 工具 schema 清晰。
- 参数校验。
- 状态机约束。
- 设置最大步数。
- 失败重试和人工兜底。
- 记录 trace，做离线评估。

## Agent 评估

指标：

- 任务完成率。
- 工具调用正确率。
- 平均步骤数。
- 失败率。
- 成本和延迟。
- 安全违规率。
- 人工满意度。

评估方法：

- 固定任务集。
- 回放真实日志。
- 标注工具调用是否正确。
- 对比不同 prompt / model / tool schema。

更完整的 Agent 工程化、状态机、Tool Calling、流式工具解析、MCP 安全边界、多智能体、HITL 和 Agent eval 复习见：[../21_Agent工程化_ToolCalling与MCP面试.md](../21_Agent工程化_ToolCalling与MCP面试.md)。

## 项目讲法模板

> 我这个 Agent 项目不是只包了一层聊天接口，而是把任务拆成规划、工具调用、状态管理和结果生成几个部分。模型先根据用户目标生成计划，再按工具 schema 调用检索、数据库或业务 API。每次工具结果会进入 observation，系统会做参数校验、错误重试和最大步数限制。评估上我主要看任务完成率、工具调用正确率、平均步骤数、失败 case 和成本。

## 高频问题

1. Agent 和普通 Chatbot 区别是什么？
2. ReAct 的流程是什么？
3. Plan-and-Execute 适合什么场景？
4. Function Calling 的本质是什么？
5. MCP 解决什么问题？
6. 工具调用参数错误怎么办？
7. Agent 进入死循环怎么办？
8. Agent 记忆怎么设计？
9. 如何评估 Agent？
10. 多 Agent 有什么问题？

## 延伸阅读

- ReAct paper: https://arxiv.org/abs/2210.03629
- Model Context Protocol: https://modelcontextprotocol.io/
- Datawhale Hello-Agents: https://github.com/datawhalechina/hello-agents
- 小林面试笔记 AI: https://xiaolinnote.com/ai/
- 卡码笔记 2026 大模型面经: https://notes.kamacoder.com/interview/llm/
