# 开源模型生态、模型选型与 Chat Template 面试

这一章面向大模型算法、AI 应用开发、模型服务平台、RAG/Agent、推理部署和项目深挖面试。很多面试官不会只问“Transformer 怎么算”，还会问：

- 现在主流开源/开放权重模型体系有哪些？
- Qwen、Llama、DeepSeek、Mistral、Gemma、GLM 这类模型怎么选？
- Base、Instruct、Chat、Reasoning、Coder、VL、Embedding 模型有什么区别？
- 为什么 SFT 数据要统一 chat template？
- 为什么不能只看榜单选模型？
- 如果线上从一个模型迁到另一个模型，风险在哪里？

这一章不要背成“模型品牌清单”。模型版本会不断更新，面试真正考的是选型框架：你能不能把任务、数据、质量、成本、延迟、上下文、工具调用、安全、许可证和部署约束放在一起判断。

## 一句话总览

模型选型不是“哪个榜单第一就用哪个”，而是：

```text
任务定义
  -> 候选模型池
  -> 公开 benchmark 初筛
  -> 私有 eval 分桶验证
  -> 成本/延迟/上下文/安全/许可证/部署评估
  -> 灰度上线
  -> 线上 trace 和 bad case 回流
```

面试口语版：

> 我会先按任务和约束筛模型，而不是按热度。公开榜单只能做第一层过滤，最终要用私有 eval 和真实流量分桶验证质量、延迟、成本、安全和稳定性。模型上线后还要把 prompt、chat template、tokenizer、tool schema、采样参数和安全策略一起版本化。

## 开源/开放权重模型怎么分类

面试里可以按“能力形态”分类：

| 类型 | 解决什么 | 例子 |
| --- | --- | --- |
| Base model | 预训练底座，擅长续写和继续训练 | Llama Base、Qwen Base、Mistral Base |
| Instruct / Chat | 指令遵循、对话、多轮问答 | Qwen Instruct、Llama Instruct、Gemma IT |
| Reasoning | 数学、代码、复杂推理，可能有长思考 | DeepSeek-R1 类、Qwen reasoning 类 |
| Coder | 代码补全、生成、修复、仓库理解 | Qwen Coder、Code Llama、DeepSeek Coder |
| Math | 数学推理、竞赛题、符号推导 | Math 专项模型 |
| VL / Multimodal | 图像、OCR、视频、grounding | Qwen-VL、LLaVA 类、Gemma 多模态类 |
| Embedding | 文档向量化、召回 | BGE、E5、GTE、Qwen embedding |
| Reranker | 检索结果重排 | BGE reranker、cross-encoder reranker |
| Guard / Safety | 安全分类、拒答、内容审核 | Llama Guard 类、安全分类器 |

按“部署形态”分类：

- API 闭源模型：上手快、能力强、维护省，但成本、隐私、可控性受限。
- 开放权重模型：可私有化、可微调、可控成本，但需要部署、评测和运维能力。
- 端侧小模型：低延迟、隐私好、离线可用，但能力和上下文有限。
- 混合路由：简单任务小模型，复杂任务强模型，高风险任务强安全策略或人工确认。

## 主流模型家族怎么介绍

下面不是榜单排名，而是面试中常见的“生态定位”说法。具体版本要以官方文档和模型卡为准。

### Qwen

常见定位：

- 中文和多语言生态强。
- 通用、代码、数学、多模态、embedding/reranker 等覆盖较全。
- 国内应用和私有化部署中经常被评估。
- Hugging Face、ModelScope、vLLM 等生态支持较好。

面试怎么说：

> 如果业务偏中文、中文知识库、代码/数学/多模态和本地化生态，我会把 Qwen 系列放进候选池，但不会只凭通用榜单决定，仍然要做私有 eval、长上下文和工具调用评估。

### Llama

常见定位：

- Meta 开放权重生态影响力大。
- 社区微调、推理框架、工具链支持很广。
- 适合作为通用英文/多语言底座和生态基线。

面试怎么说：

> Llama 的优势是社区和生态广，很多工具、微调和 serving 方案会优先适配。选它时除了能力，还要看许可证、语言覆盖、业务私有 eval 和中文/领域表现。

### DeepSeek

常见定位：

- 在代码、数学、reasoning、MoE 和高效推理相关话题里很高频。
- V3/R1 带火了 MoE、MLA、GRPO、reasoning distillation 等追问。
- 适合准备推理模型、代码模型和高效架构相关面试。

面试怎么说：

> DeepSeek 相关模型我会重点关注 reasoning、代码、MoE/MLA 和推理成本。项目里选型要看是否真的需要长思考能力，以及长输出成本、延迟和安全策略是否可控。

### Mistral / Mixtral

常见定位：

- 欧洲开源/开放权重生态代表之一。
- Mistral 7B、Mixtral MoE 等经常作为小而强、MoE、滑动窗口/推理效率案例。
- 英文和通用场景常被纳入候选。

面试怎么说：

> Mistral/Mixtral 常被用来讨论小模型效率、MoE 和开放模型生态。选型时要看语言、许可证、部署支持和业务 eval。

### Gemma

常见定位：

- Google 开放模型生态。
- 适合讨论轻量模型、研究和安全/负责任 AI 文档。
- 可作为端侧、小模型和通用开放模型候选。

面试怎么说：

> Gemma 系列我会放在轻量开放模型候选里，重点看模型卡、许可证、部署端限制和业务质量。

### GLM / ChatGLM

常见定位：

- 国内大模型生态常见系列。
- 中文对话、私有化部署和企业应用里经常被提到。
- 面试里可能问 ChatGLM、GLM 架构、tokenizer、工具调用或国产生态。

面试怎么说：

> GLM/ChatGLM 可以作为中文企业应用和国产生态候选，但同样要通过私有 eval、上下文、工具调用、成本和部署约束来决定。

## Base、Instruct、Chat、Reasoning 怎么区分

### Base Model

Base model 是预训练底座，主要学语言建模。它擅长续写，不一定听指令。

适合：

- 继续预训练。
- 做领域适配。
- 作为 SFT / DPO / RL 的起点。

不适合直接用于生产对话，除非你自己做对齐和安全。

### Instruct / Chat Model

Instruct/Chat model 做过指令微调和对齐，更适合问答、对话、工具调用和生产应用。

适合：

- RAG 问答。
- Agent。
- 企业助手。
- 结构化输出。

注意：

- 要使用正确 chat template。
- 多轮对话要保留 role 和特殊 token。
- 安全策略和拒答行为可能已经内化。

### Reasoning Model

Reasoning model 强调复杂推理，可能通过 RL、蒸馏或长思考训练增强。

适合：

- 数学、代码、规划、复杂分析。
- 需要多步推理的任务。

风险：

- 简单任务可能过度思考。
- 输出更长，成本和延迟更高。
- 推理过程不一定完全忠实。
- 需要单独评估 answer correctness 和 reasoning trace。

## Chat Template 是什么

Chat template 是把多轮消息转换成模型实际看到的 token 序列的规则。

原始消息：

```json
[
  {"role": "system", "content": "你是一个助手"},
  {"role": "user", "content": "解释 KV Cache"},
  {"role": "assistant", "content": "KV Cache 是..."}
]
```

模型实际输入可能变成：

```text
<|system|>
你是一个助手
<|user|>
解释 KV Cache
<|assistant|>
KV Cache 是...
```

不同模型的 role token、BOS/EOS、assistant 起始标记、工具调用格式都可能不同。训练和推理必须一致。

## Chat Template 不一致会怎样

常见问题：

- 模型不按角色回答。
- assistant 输出里混入 user/system 标记。
- 多轮对话错乱。
- 工具调用 JSON 格式不稳定。
- EOS 位置错误，模型停不下来或过早停止。
- SFT loss mask 错，把 user prompt 也当成需要学习的答案。
- 迁移模型后 prompt 效果大幅变化。

面试表达：

> Chat template 是模型对话协议的一部分，不是无关的字符串。换模型时要同步检查 tokenizer、special tokens、BOS/EOS、role 格式、tool call schema 和 stop tokens。

## Tokenizer 选型看什么

Tokenizer 会影响：

- 中英文 token 数。
- 代码、数学符号、JSON 的切分。
- 长上下文 token budget。
- 特殊 token 和 chat template。
- 词表扩展和领域词表现。
- 多语言和 byte fallback。

常见追问：

### 中文词表重要吗？

重要，但不是唯一因素。中文切得更细会增加 token 数，影响成本和上下文；但模型能力还取决于预训练数据、指令数据和后训练。

### 代码 tokenizer 特殊在哪？

代码里有缩进、符号、长标识符、括号和重复模式。好的 tokenizer 能更高效表示代码，减少 token 数，并保留结构。

### 换 tokenizer 风险？

如果从头训练影响最大；微调时通常不能随便换 tokenizer。扩词表需要初始化新 token embedding，并处理旧数据兼容和训练稳定性。

Tokenizer 的 BPE / WordPiece / Unigram / SentencePiece、special tokens、token budget、扩词表和 TTFT 排查，集中看 [35_Tokenizer_BPE_SentencePiece与Token预算面试.md](35_Tokenizer_BPE_SentencePiece与Token预算面试.md)。

## 模型选型流程

### 1. 定义任务和约束

先问：

- 是聊天、RAG、Agent、代码、数学、OCR、embedding 还是 rerank？
- 输入和输出 token 分布是什么？
- 是否需要中文、多语言、代码、结构化 JSON、工具调用？
- P95/P99 延迟要求是多少？
- 单次成本上限是多少？
- 是否能走外部 API？
- 是否需要私有化部署？
- 是否有许可证和合规限制？
- 是否需要端侧离线？

### 2. 建候选池

候选池按能力和部署约束分层：

```text
强 API 模型
  -> 通用开放权重模型
  -> 专项模型：coder/math/VL/embedding/reranker
  -> 小模型/端侧模型
  -> safety/guard 模型
```

### 3. 公开 benchmark 初筛

公开 benchmark 可以看：

- 通用知识和推理。
- 数学。
- 代码。
- 多模态。
- 长上下文。
- Arena/人类偏好。

但它只能初筛，不能直接决定上线。

### 4. 私有 Eval

私有 eval 要按业务分桶：

- 常见问题。
- 困难问题。
- 长上下文。
- 多轮对话。
- 工具调用。
- 结构化输出。
- 拒答/安全。
- 低资源语言或领域术语。
- 线上 bad case 回放。

### 5. 成本和系统验证

同时看：

- TTFT / TPOT / P95 / P99。
- input/output tokens。
- GPU 显存和吞吐。
- batch 后延迟。
- prompt/cache 命中。
- 单次成本。
- 峰值流量下 fallback。

### 6. 灰度和回滚

上线要做：

- prompt/template/model 版本绑定。
- 少量流量灰度。
- 质量和安全采样。
- 成本告警。
- fallback 到旧模型。
- bad case 回流 eval。

## 为什么不能只看排行榜

公开榜单有价值，但有局限：

- 数据可能污染。
- 任务和业务不一致。
- 中文/领域/工具调用不一定覆盖。
- 长上下文位置偏置可能被平均分掩盖。
- 安全、拒答、格式遵循、低延迟不一定体现。
- 同一模型不同推理参数会影响结果。
- 榜单分数不等于部署成本。

面试表达：

> 我会用公开榜单做候选池初筛，但上线模型必须过私有 eval 和真实流量分桶。尤其是 RAG、Agent、代码和企业知识库，业务数据分布和公开 benchmark 差距很大。

## Dense 和 MoE 怎么影响选型

Dense model：

- 每个 token 激活全部参数。
- serving 更简单。
- 延迟更稳定。
- 小规模部署和端侧更直观。

MoE model：

- 总参数大，但每个 token 只激活部分 expert。
- 可能以较低计算成本获得更大容量。
- 训练和推理有路由、负载均衡、expert 通信问题。
- 小 batch 或部署不当时不一定省。

选型时不要只看“总参数”。要看：

- active parameters。
- tokens/s。
- 显存和 KV Cache。
- MoE 通信开销。
- 推理引擎支持。
- 质量和延迟尾部。

## Function Calling / Structured Output 怎么选模型

如果业务依赖工具调用，模型要评估：

- 是否支持官方 tool/function calling 格式。
- JSON/schema 遵循能力。
- 参数抽取准确率。
- 多工具选择准确率。
- 工具调用后是否能正确整合结果。
- 遇到高风险工具是否能拒绝或请求确认。
- 是否支持 constrained decoding / structured output。

不要只看普通问答能力。工具调用项目要单独建 eval：

```text
tool selection accuracy
argument exact match / semantic match
schema validity
multi-turn tool success
permission violation rate
fallback / clarification rate
```

## 长上下文模型怎么选

长上下文不是“标称 1M tokens 就好”。要看：

- 有效上下文长度。
- lost-in-the-middle。
- needle-in-a-haystack。
- 多文档引用。
- 长文档摘要。
- 长上下文下事实一致性。
- TTFT、prefill 成本和 KV Cache 显存。
- 是否真的比 RAG 更划算。

面试表达：

> 长上下文和 RAG 不是互斥。长上下文适合需要完整材料、跨段综合和减少检索漏召的场景；RAG 适合知识更新、权限过滤和成本控制。选模型时要同时评估质量和 serving 成本。

## Open Weight 和 API 模型怎么取舍

API 模型优势：

- 能力强，上手快。
- 不用维护 GPU 和推理引擎。
- 供应商持续升级。

API 模型风险：

- 成本和限流受供应商影响。
- 数据隐私和合规限制。
- 可控性、可复现性和回滚复杂。
- 模型版本变化可能影响输出。

开放权重优势：

- 私有化部署。
- 可微调和蒸馏。
- 成本可控。
- 可做深度优化和安全治理。

开放权重风险：

- 运维复杂。
- 需要 eval、serving、监控和安全团队能力。
- 可能落后于最强闭源模型。

## 模型迁移风险

从模型 A 切到模型 B，要检查：

- tokenizer 和 token 数变化。
- chat template。
- system prompt 效果。
- tool calling 格式。
- stop tokens。
- 采样参数。
- 安全拒答风格。
- RAG 引用格式。
- 长上下文位置偏置。
- 输出长度和成本。
- 许可证和合规。

迁移步骤：

```text
离线回放 eval
  -> prompt/template 适配
  -> shadow traffic
  -> 小流量灰度
  -> 指标对比
  -> 可回滚发布
```

## 项目里怎么讲模型选型

8 分钟讲稿：

1. 背景：业务任务、用户、风险和约束。
2. 候选池：API + 开放权重 + 专项模型。
3. Eval：公开 benchmark 初筛 + 私有 eval 分桶。
4. 指标：质量、格式遵循、工具调用、安全、延迟、成本。
5. 工程：chat template、prompt 版本、模型网关、路由、fallback。
6. 部署：vLLM/SGLang/TensorRT-LLM、量化、并发和监控。
7. 灰度：shadow、A/B、回滚和 bad case 回流。
8. 复盘：模型迁移风险和下一步优化。

项目表达示例：

> 我们没有直接按榜单选模型，而是先把业务问题分成知识问答、结构化抽取和工具调用三类。候选池包括强 API 模型、Qwen/Llama/DeepSeek 等开放权重模型和一个小模型。公开 benchmark 只做初筛，最终用私有 eval 分桶看准确率、引用正确率、JSON 有效率、工具参数准确率、拒答和 P95 延迟。上线时把模型、prompt、chat template、tool schema 和采样参数一起版本化，通过模型网关灰度和 fallback，线上 bad case 回流到 eval 集。

## 高频快答

### Base 和 Chat 模型区别？

Base 是预训练底座，擅长续写，不一定听指令；Chat/Instruct 做过指令微调和对齐，更适合对话、RAG、Agent 和生产应用。

### Chat template 为什么重要？

它定义 role、特殊 token、BOS/EOS、assistant 起始标记和工具调用格式。训练和推理不一致会导致格式错乱、停不下来、工具调用失败或 loss mask 错。

### 为什么不能只看榜单？

榜单不一定覆盖业务分布、中文、领域术语、工具调用、长上下文、安全、延迟和成本。只能初筛，不能替代私有 eval。

### 开放权重一定比 API 省钱吗？

不一定。要算 GPU、运维、人力、峰值容量、推理优化、质量损失和 fallback。流量足够大、隐私和可控性要求高时，开放权重更有吸引力。

### 换模型最容易踩什么坑？

tokenizer、chat template、tool schema、stop token、采样参数、安全策略、输出长度和许可证。

## 面试背诵版

模型选型要从任务和约束出发，而不是从榜单出发。先明确是 RAG、Agent、代码、数学、多模态、embedding 还是 rerank，再看中文/多语言、上下文长度、工具调用、结构化输出、安全、成本、延迟、部署和许可证。Base 模型适合继续训练，Instruct/Chat 适合对话应用，Reasoning/Coder/VL/Embedding/Reranker 是专项能力模型。Chat template 是模型对话协议，训练和推理必须一致，否则会导致角色错乱、EOS 错、工具调用失败和 loss mask 错。公开 benchmark 只能初筛，上线前必须做私有 eval、真实流量分桶、灰度和回滚。

## 延伸阅读

- Hugging Face Chat Templates 文档：[https://huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating)
- Hugging Face LLM Leaderboard：[https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- Qwen 官方文档：[https://qwen.readthedocs.io/](https://qwen.readthedocs.io/)
- Qwen GitHub：[https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- Meta Llama 官方页面：[https://www.llama.com/](https://www.llama.com/)
- DeepSeek GitHub：[https://github.com/deepseek-ai](https://github.com/deepseek-ai)
- Mistral AI 文档：[https://docs.mistral.ai/](https://docs.mistral.ai/)
- Google Gemma 文档：[https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- GLM / ChatGLM GitHub：[https://github.com/THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3)
- InternLM GitHub：[https://github.com/InternLM/InternLM](https://github.com/InternLM/InternLM)
