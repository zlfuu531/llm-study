# 答案版 21：开源模型生态、模型选型与 Chat Template

对应题号：441-460。建议先读 [26_开源模型生态_模型选型与ChatTemplate面试.md](../26_开源模型生态_模型选型与ChatTemplate面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 441. 主流开源/开放权重模型生态怎么分类？

30 秒版：

不要按品牌硬背，按能力分类：base、instruct/chat、reasoning、coder、math、VL、多模态、embedding、reranker、guard/safety。再按部署分 API、开放权重、端侧小模型和混合路由。

2 分钟版：

能力形态上，base model 是预训练底座，适合继续训练；instruct/chat 适合对话和生产应用；reasoning 强调复杂推理；coder/math 是专项模型；VL 处理图像/OCR/视频；embedding/reranker 服务 RAG；guard/safety 负责安全分类。

部署形态上，API 模型能力强、接入快但隐私和成本受限；开放权重模型可私有化、可微调但需要运维；端侧小模型适合低延迟、隐私和离线；混合路由把不同任务分给不同模型。

面试句：

> 我会先按任务和约束分类，再把 Qwen、Llama、DeepSeek、Mistral、Gemma、GLM 等放进候选池，而不是把模型名当答案。

## 442. Base、Instruct、Chat、Reasoning、Coder、VL、Embedding 模型有什么区别？

30 秒版：

Base 擅长续写，不一定听指令；Instruct/Chat 做过指令对齐，适合问答；Reasoning 强复杂推理；Coder 专注代码；VL 处理图像/视频；Embedding 生成向量，Reranker 做重排。

2 分钟版：

Base model 是预训练语言模型，适合继续预训练、领域适配和后训练起点。Instruct/Chat model 做过指令微调和偏好对齐，更适合 RAG、Agent、客服、助手。Reasoning model 适合数学、代码、规划，但可能输出长、成本高。Coder model 对代码数据、FIM、仓库上下文更友好。VL model 接收图像或视频 token。Embedding/reranker 不负责生成答案，而是负责召回和排序。

选型时不要拿不同类型直接比。比如 embedding 模型不能和 chat 模型比“谁更会回答”，coder 模型也不一定适合闲聊。

## 443. 模型选型流程怎么设计？

30 秒版：

先定义任务和约束，再建候选池，用公开 benchmark 初筛，用私有 eval 分桶验证，最后比较质量、成本、延迟、安全、许可证和部署，灰度上线并回流 bad case。

2 分钟版：

流程：

1. 定义任务：RAG、Agent、代码、数学、多模态、embedding、rerank。
2. 定义约束：中文、上下文、工具调用、延迟、成本、隐私、许可证。
3. 候选池：API 模型、开放权重模型、专项模型、小模型。
4. 公开 benchmark 初筛。
5. 私有 eval 分桶：常见、困难、长上下文、工具、安全、格式。
6. 系统验证：TTFT、TPOT、P95/P99、tokens/s、成本、显存。
7. 灰度上线：shadow traffic、A/B、fallback、回滚。

加分句：

> 模型选型不是离线一次性决策，而是 eval、灰度、监控和 bad case 回流的闭环。

## 444. 为什么不能只看排行榜选模型？

30 秒版：

榜单只能初筛。它可能有数据污染，任务和业务不一致，也不一定覆盖中文、领域、工具调用、长上下文、安全、格式遵循、延迟和成本。

2 分钟版：

公开 benchmark 的价值是快速缩小候选池，但上线任务通常有自己的数据分布。企业知识库问答要看引用正确率和权限过滤，Agent 要看工具选择和参数准确率，代码 Agent 要看测试通过率，长文档要看 lost-in-the-middle 和成本。

榜单还可能受提示词、采样参数、数据污染和评测方式影响。同一个模型在公开题上强，不代表在你的业务上强。

面试句：

> 我会用榜单做第一层过滤，但最终证据一定来自私有 eval、真实流量回放、P95 延迟、成本和安全指标。

## 445. Qwen、Llama、DeepSeek、Mistral、Gemma、GLM 这类模型怎么介绍？

30 秒版：

按生态定位说：Qwen 中文和专项生态强，Llama 社区生态广，DeepSeek 在代码/reasoning/MoE 热点高频，Mistral/Mixtral 常用于效率和 MoE 讨论，Gemma 是 Google 开放模型生态，GLM/ChatGLM 常见于中文和国产生态。

2 分钟版：

不要说“某模型绝对最好”。更稳的说法是：

- Qwen：中文、多语言、代码、多模态、embedding/reranker 覆盖较全。
- Llama：开放权重生态影响力大，社区工具和 serving 支持广。
- DeepSeek：代码、reasoning、MoE/MLA/GRPO 等追问高频。
- Mistral/Mixtral：小而强、MoE 和开放模型生态代表之一。
- Gemma：Google 开放模型生态，轻量和负责任 AI 文档常被提到。
- GLM/ChatGLM：中文企业应用和国产生态常见。

最后补一句：

> 具体选型必须回到任务、私有 eval、部署、许可证和成本，而不是模型名。

## 446. Dense 和 MoE 模型选型有什么区别？

30 秒版：

Dense 每个 token 激活全部参数，部署更简单、延迟更稳定；MoE 总参数大但每 token 只激活部分 expert，可能更省计算，但有路由、负载均衡和通信复杂度。

2 分钟版：

Dense model 的 serving 简单，性能更可预测。MoE model 用稀疏激活扩大容量，每 token 计算量可能低于同总参数 dense model，但实际部署要看 active parameters、expert parallel、all-to-all 通信、batch size 和引擎支持。

选型不要只看总参数。MoE 的总参数大不代表每 token 成本大，active params 才更接近计算成本。但 MoE 在小流量、小 batch 或通信差的环境下，可能不如 dense 稳。

面试句：

> MoE 是用系统复杂度换模型容量和计算效率，是否划算要看质量、tokens/s、P95、显存、通信和推理引擎支持。

## 447. 参数量、上下文长度、多模态和许可证怎么影响选型？

30 秒版：

参数量影响质量、显存和延迟；上下文长度影响能处理多少输入但也增加 prefill/KV 成本；多模态要看图像/OCR/视频能力；许可证决定能否商用、改造和分发。

2 分钟版：

参数量越大通常能力更强，但成本、延迟和部署难度更高。上下文长度不是越大越好，要看有效上下文、lost-in-the-middle 和 serving 成本。多模态模型要按 OCR、图表、定位、视频、grounding 分桶评估。许可证是生产上线必须检查的硬约束，包括商用、再分发、模型输出使用和合规要求。

选型表：

| 维度 | 影响 |
| --- | --- |
| 参数量 | 质量、显存、吞吐、延迟 |
| 上下文 | 长文档能力、KV Cache、TTFT |
| 多模态 | 输入形态和评估方式变化 |
| 许可证 | 是否能商用和私有化 |

## 448. Tokenizer 选型要看什么？

30 秒版：

看中英文 token 数、代码/JSON/数学符号切分、特殊 token、chat template、byte fallback、领域词和长上下文 token budget。Tokenizer 会影响成本和效果。

2 分钟版：

Tokenizer 决定文本如何变成 token。中文如果切得太碎，会增加 token 数、上下文占用和成本；代码如果切分差，会影响缩进、符号和标识符表示；JSON/工具调用也依赖稳定 tokenization。

还要检查 special tokens、BOS/EOS、role token、tool token、padding 和 stop tokens。微调时通常不能随便换 tokenizer，扩词表也要处理新 token embedding 初始化和训练稳定性。

面试句：

> Tokenizer 不是预处理小细节，它会影响上下文预算、成本、chat template、工具调用格式和模型迁移风险。

## 449. Chat template 是什么？

30 秒版：

Chat template 是把 system/user/assistant/tool 等消息转换成模型实际 token 序列的规则，包括 role token、BOS/EOS、assistant 起始标记和工具调用格式。

2 分钟版：

用户看到的是消息列表：

```json
{"role": "user", "content": "解释 KV Cache"}
```

模型看到的是经过模板拼接后的字符串和 token，比如 `<|user|>`、`<|assistant|>`、EOS 等。不同模型模板不同，训练和推理必须一致。

它影响：

- 多轮对话。
- system prompt。
- assistant 起始位置。
- 工具调用格式。
- stop token。
- SFT loss mask。

面试句：

> Chat template 是模型的对话协议，换模型时必须一起迁移，不能只换 model name。

## 450. Chat template 训练和推理不一致会有什么后果？

30 秒版：

会导致角色错乱、输出特殊 token、EOS 错、停不下来或过早停止、工具调用 JSON 不稳、SFT loss mask 错和线上效果大幅波动。

2 分钟版：

如果训练时用一种 role 格式，推理时用另一种，模型可能不知道哪里开始回答。比如 assistant 起始标记缺失，模型会继续模拟 user；EOS 错会导致停不下来；工具调用格式不一致会导致 schema invalid。

SFT 时如果 loss mask 没处理好，把 user prompt 也纳入 loss，模型可能学会复述问题或角色标记。迁移模型时，旧 prompt 在新 template 下 token 分布也会变。

回答模板：

> 我会把 chat template、tokenizer、special tokens、stop tokens、tool schema 和 prompt version 一起纳入版本管理，并在 eval 里检查格式遵循。

## 451. SFT 数据为什么要统一 chat template 和 loss mask？

30 秒版：

因为模型实际学的是模板化 token 序列。template 不统一会让角色格式混乱；loss mask 错会让模型学习 user 输入、system prompt 或 padding，而不是只学习 assistant 答案。

2 分钟版：

SFT 数据通常是多轮消息。训练前要统一成目标模型的 chat template，并且只对 assistant 需要学习的部分算 loss。System/user/tool 结果有时作为上下文输入，不应该全部作为预测目标。

常见错误：

- 丢 EOS。
- 多轮边界错。
- assistant 起始标记缺失。
- 把 prompt 也算 loss。
- padding 没 mask。
- tool call 和 tool result 格式不一致。

面试句：

> SFT 不是把 JSONL 拼起来就训，template 和 loss mask 直接决定模型学会怎样对话。

## 452. Thinking / non-thinking 模型怎么路由？

30 秒版：

简单问题、低成本场景走 non-thinking 或短思考；复杂推理、数学、代码、规划走 thinking/reasoning。路由要看任务难度、风险、成本、延迟和用户体验。

2 分钟版：

Reasoning 模型适合复杂问题，但可能输出长、慢、成本高。对 FAQ、简单抽取、固定格式任务，不一定需要长思考。可以设计路由：先用轻量分类器或小模型判断难度，简单任务走普通 chat，小部分复杂任务升级 reasoning。

指标：

- 正确率。
- token 成本。
- TTFT/TPOT。
- 用户等待。
- 过度推理比例。
- 安全风险。

面试句：

> reasoning 能力不是免费午餐，要把强推理用在真正需要的任务上。

## 453. Coder、Math、Reasoning 模型怎么选？

30 秒版：

代码看 HumanEval/MBPP/SWE-bench、仓库上下文和测试通过；数学看 GSM8K/MATH/GPQA 和私有题；reasoning 看复杂任务正确率、长思考成本和过程稳定性。

2 分钟版：

Coder 模型要看代码补全、生成、修复、FIM、仓库级上下文、patch 和测试闭环。Math 模型要看题型覆盖、符号推导、最终答案准确率和解题过程。Reasoning 模型要看复杂规划、工具使用和长链路任务。

不要只看一个 benchmark。代码模型在 HumanEval 好不代表 SWE-bench 好；数学模型强不代表工具调用强；reasoning 模型强不代表低延迟。

项目选型：

> 我会用公开 benchmark 初筛，再用公司真实 repo、真实数学题或真实任务轨迹做私有 eval。

## 454. 长上下文模型怎么评估？

30 秒版：

看有效上下文而不是标称长度。评估 needle、lost-in-the-middle、多文档引用、长文档摘要、事实一致性、TTFT、KV Cache 显存和是否比 RAG 更划算。

2 分钟版：

长上下文模型可能支持很长输入，但信息放在中间或多文档交叉时仍可能失败。要按位置、长度和任务类型分桶：开头/中间/末尾证据，多跳问题，跨文档综合，引用正确率。

系统层面还要看 prefill 时间和 KV Cache。长上下文请求 TTFT 高、显存大，不一定适合所有业务。很多企业知识库仍需要 RAG 做更新、权限过滤和成本控制。

面试句：

> 长上下文是能力，不是替代 RAG 的银弹。选型要同时看质量和 serving 成本。

## 455. Open weight 和闭源 API 怎么取舍？

30 秒版：

API 上手快、能力强、运维少，但隐私、成本、限流和可控性受限；开放权重可私有化、可微调、成本可控，但要承担部署、评测、安全和运维。

2 分钟版：

API 模型适合快速验证、强能力任务和团队 GPU 运维能力不足的阶段。开放权重适合数据敏感、流量大、需要定制、需要稳定版本和可控成本的场景。

真实成本要算：

- API 单价。
- GPU 机器。
- 峰值容量。
- 推理引擎运维。
- 监控和安全。
- 质量损失带来的人工成本。
- fallback 和灰度。

面试句：

> 不能简单说开源省钱。流量、隐私、团队能力和质量要求决定取舍。

## 456. 许可证和合规在模型选型里怎么看？

30 秒版：

上线前必须确认许可证是否允许商用、再分发、微调、模型输出使用、权重修改和私有化部署，还要看数据合规、隐私和地域要求。

2 分钟版：

很多开放权重模型不是无条件开源。选型时要检查 license、acceptable use policy、模型卡、数据声明和商用限制。企业场景还要看用户数据是否能出域，日志是否能保存，是否涉及个人信息、代码、财务、医疗等敏感数据。

工程动作：

- 记录模型版本和许可证。
- 法务/合规确认。
- 数据脱敏和访问控制。
- 日志隐私保护。
- 安全评测和红队。

面试句：

> 模型能力再强，如果许可证和数据合规不满足，也不能上线。

## 457. 开源模型做 Function Calling / Structured Output 要评估什么？

30 秒版：

评估 tool selection、参数准确率、schema validity、多轮工具成功率、拒绝高风险工具、clarification 能力和 constrained decoding 支持。

2 分钟版：

普通问答强不代表工具调用强。Function calling 要模型先选对工具，再抽对参数，还要输出合法 JSON/schema。工具返回后还要能整合结果，失败时要重试或澄清。

指标：

- tool selection accuracy。
- argument exact/semantic match。
- JSON/schema valid rate。
- task success rate。
- permission violation rate。
- unnecessary tool call rate。
- fallback/clarification rate。

面试句：

> 工具调用要单独评测轨迹，不能只看最终自然语言答案。

## 458. 中文/多语言能力怎么评估？

30 秒版：

看真实中文业务数据、领域术语、长文档、代码混合、检索问答、格式遵循和安全拒答，不只看通用中文 benchmark。

2 分钟版：

中文能力受 tokenizer、预训练数据、指令数据和对齐影响。评估要覆盖：日常问答、专业术语、表格/PDF、中文 RAG、口语表达、代码中英文混合、繁简转换、多语言切换。

还要看 token 成本。同一段中文在不同 tokenizer 下 token 数可能不同，影响上下文和成本。

面试句：

> 中文能力不是一句“支持中文”，要看业务语料、token 成本、领域词和私有 eval。

## 459. 从一个模型迁移到另一个模型有哪些风险？

30 秒版：

风险包括 tokenizer、chat template、stop token、tool schema、prompt 效果、安全拒答、输出长度、RAG 引用、长上下文位置偏置、成本、延迟和许可证变化。

2 分钟版：

迁移不是改 model name。不同模型的 tokenizer 会改变 token 数；template 会改变角色格式；stop token 会影响输出停止；tool schema 可能需要重写；同一 prompt 在新模型上风格和安全策略都可能不同。

迁移流程：

1. 离线 eval 回放。
2. prompt/template/tool schema 适配。
3. shadow traffic。
4. 小流量灰度。
5. 指标对比。
6. 保留 fallback 和回滚。

面试句：

> 模型迁移要把模型、tokenizer、template、prompt、tool schema 和采样参数作为一个整体迁移。

## 460. 模型选型项目怎么讲 8 分钟？

30 秒版：

按业务背景、候选池、eval 设计、指标、选型结论、工程接入、灰度上线和复盘讲。重点给出私有 eval、成本延迟、安全和 fallback 证据。

2 分钟版：

讲稿结构：

1. 背景：任务、用户、风险、成本和延迟约束。
2. 候选池：API、开放权重、专项模型、小模型。
3. Eval：公开榜单初筛，私有数据分桶。
4. 指标：准确率、引用、格式、工具、拒答、P95、成本。
5. 选型：为什么选这个模型，放弃哪些模型。
6. 工程：chat template、prompt 版本、模型网关、路由、fallback。
7. 灰度：shadow/A/B、监控和回滚。
8. 复盘：bad case 回流和下一步。

示例句：

> 我们把任务分成知识问答、结构化抽取和工具调用，候选模型包括强 API、Qwen/Llama/DeepSeek 等开放权重和小模型。公开 benchmark 只做初筛，最终用私有 eval 看引用正确率、JSON 有效率、工具参数准确率、安全拒答、P95 延迟和单次成本。上线时把模型、prompt、chat template、tool schema 和采样参数一起版本化，并通过模型网关灰度和 fallback。

## 本组题的复习顺序

1. 先背 441-445：生态分类、模型类型、选型流程和主流模型家族。
2. 再背 448-451：tokenizer、chat template、SFT loss mask。
3. 然后背 452-458：reasoning/coder/long-context/API/open-weight/license/tool/multilingual。
4. 最后背 459-460：模型迁移和项目表达。

## 延伸阅读

- Hugging Face Chat Templates：[https://huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating)
- Qwen 文档：[https://qwen.readthedocs.io/](https://qwen.readthedocs.io/)
- Qwen GitHub：[https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- Meta Llama：[https://www.llama.com/](https://www.llama.com/)
- DeepSeek GitHub：[https://github.com/deepseek-ai](https://github.com/deepseek-ai)
- Mistral AI 文档：[https://docs.mistral.ai/](https://docs.mistral.ai/)
- Google Gemma：[https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- ChatGLM3 GitHub：[https://github.com/THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3)
- InternLM GitHub：[https://github.com/InternLM/InternLM](https://github.com/InternLM/InternLM)
