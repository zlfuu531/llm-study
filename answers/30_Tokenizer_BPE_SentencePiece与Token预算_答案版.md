# 答案版 30：Tokenizer、BPE、SentencePiece 与 Token 预算

对应题目：`03_高频题单100题.md` 的 621-640。

用法：先按 30 秒版说清结论，再用 2 分钟版补算法、工程坑和项目排查。Tokenizer 题最怕答成“就是分词”，要主动讲 token 预算、special tokens、chat template、loss mask 和迁移风险。

## 621. Tokenizer 为什么是大模型输入协议？

30 秒版：

因为模型实际处理的是 token id，不是原始字符串。Tokenizer 决定文本怎么变成 id，special tokens、BOS/EOS/PAD、chat template、loss mask、生成停止和 token 成本都依赖它，所以它是模型输入协议的一部分。

2 分钟版：

同一段文本在不同 tokenizer 下会变成不同 token 序列。模型 embedding matrix 是按 token id 学的，id 和字符串绑定。如果换 tokenizer，模型看到的输入分布和 token id 含义都会变。微调、RAG、推理、工具调用都必须保证 tokenizer、chat template、special tokens 和模型权重匹配。

## 622. BPE 的训练和编码怎么讲？

30 秒版：

BPE 从字符或 byte 开始，反复合并语料里最高频的相邻 pair，直到达到目标词表大小。编码时按学到的 merge rank 合并输入，得到子词 token。

2 分钟版：

训练：

```text
初始 symbol -> 统计 pair -> 合并最高频 pair -> 更新语料 -> 重复
```

优点是词表可控、常见词可变成单 token、稀有词可拆子词。缺点是对训练语料、空格、Unicode 和大小写处理敏感。Byte-level BPE 用 byte 做基础符号，能覆盖任意字符串。

## 623. BPE、WordPiece、Unigram 有什么区别？

30 秒版：

BPE 按频率或 merge rank 合并 pair；WordPiece 也做子词合并，但更关注合并对语料似然的提升；Unigram 从大候选词表开始，逐步删掉贡献小的 token，并选择概率最高的切分。

2 分钟版：

BPE 常见于 GPT/Llama/Qwen 等 decoder LLM。WordPiece 常见于 BERT 系列，词内片段常带 `##`。Unigram 常和 SentencePiece 一起出现，一个词可以有多种切分，选概率最大的，也可以采样切分增强鲁棒性。

面试句：

> 三者都解决词表和 OOV 的折中，只是学习子词词表的准则不同。

## 624. SentencePiece 是什么？

30 秒版：

SentencePiece 是语言无关的 subword tokenizer 工具，常实现 BPE 或 Unigram。它直接处理原始文本，不强依赖空格分词，并用 `▁` 表示空格，适合中文、日文等无空格语言。

2 分钟版：

传统 BPE/WordPiece 常先按空格或规则预切词，但中文没有天然空格。SentencePiece 把文本看成原始字符流，空格也进入词表，例如 `"Hello world"` 可表示为 `["▁Hello", "▁world"]`。decode 时再把 `▁` 还原为空格。

## 625. Byte-level BPE / byte fallback 解决什么问题？

30 秒版：

它们解决未知字符覆盖问题。Byte-level BPE 用 256 个 byte 作为基础符号，byte fallback 在无法表示字符时退回 byte，因此任意 Unicode 字符串都能编码。

2 分钟版：

好处是减少 `<unk>`，对 emoji、罕见字、多语言、乱码更稳。代价是一些非英语文本、中文、特殊符号可能被拆得很碎，增加 token 数、成本和上下文占用。面试要说：覆盖性更强，但压缩率不一定更好。

## 626. BOS、EOS、PAD、UNK、SEP、MASK 分别是什么？

30 秒版：

BOS 是序列开始，EOS 是序列结束，PAD 是 padding，UNK 是未知 token，SEP 是分隔符，MASK 用于 MLM 掩码。Chat 模型还会有 role tokens 和 tool tokens。

2 分钟版：

常见坑：

- EOS 丢失导致模型停不下来或样本边界错。
- PAD 没 mask 污染 loss。
- tokenizer.pad_token_id 和 model.config.pad_token_id 不一致。
- chat template 自动加 BOS/EOS，手动重复加。
- stop token 和 EOS 不完全一样。

## 627. Token 预算怎么估算？

30 秒版：

把上下文拆成 system、history、user、RAG context、tool schema 和 reserved output。总和不能超过模型 context window，且越长 prefill、KV Cache、成本和 TTFT 越高。

2 分钟版：

公式：

```text
total = system + history + user + retrieved_context + tool_schema + output_reserve
```

如果模型 32k，上游 RAG 塞 24k，工具 schema 3k，历史 3k，就几乎没有输出空间。工程上要做 token budget 表、裁剪策略、RAG chunk 排序、固定前缀缓存和超长输入拒绝/降级。

## 628. 中文 tokenizer 要注意什么？

30 秒版：

看中文 token 数、专名、数字单位、中英混排、全角半角、繁简、标点和领域词。中文切得太碎会增加成本和上下文占用，但模型能力还取决于预训练和对齐数据。

2 分钟版：

不能只说“中文词表越大越好”。词表大可能减少 token 数，但 embedding 矩阵也更大，低频词学习不足。评估要用真实中文业务文本，统计 tokens/char、专有名词拆分、RAG chunk 长度、成本和下游效果。

## 629. 代码 tokenizer 为什么特殊？

30 秒版：

代码有缩进、换行、括号、路径、snake_case、camelCase、API 和稀有标识符。Tokenizer 会影响上下文长度、结构保真、补全边界和 FIM 特殊 token。

2 分钟版：

代码补全需要 prefix/suffix，Python 缩进有语义，路径和包名要稳定表示。切得太碎会浪费上下文，切得太粗又会让词表大和泛化差。代码模型还常需要 FIM 的 prefix/middle/suffix special tokens。

## 630. JSON、工具调用和数学 tokenizer 有什么坑？

30 秒版：

JSON 工具调用依赖 `{}`、引号、冒号、逗号和字段名稳定；数学有 LaTeX、数字、小数、单位；这些 tokenization 会影响结构化输出、约束解码、复制和计算。

2 分钟版：

工具 schema 常很长，占大量 token budget。字段名反复出现，可以压缩或缓存。数字可能被拆成多个 token，模型复制长数字和做精确计算会变难。Constrained decoding 也依赖当前前缀下哪些 token 合法。

## 631. 为什么 tokenizer 不能随便换？

30 秒版：

因为 embedding matrix 和 token id 绑定。换 tokenizer 会改变 token id 到字符串的映射、输入分布、special tokens、chat template 和旧 adapter 的语义，基本等于改模型接口。

2 分钟版：

微调时换 tokenizer 很危险：LoRA adapter 是在原 base/tokenizer 上训练的，旧 prompt、RAG chunk、缓存和日志也都按旧 tokenizer 统计。除非从头训练或明确做扩词表/迁移/继续训练，否则不建议换。

## 632. 扩词表怎么做？风险是什么？

30 秒版：

先 `tokenizer.add_special_tokens` 或 `add_tokens`，再 `model.resize_token_embeddings(len(tokenizer))`，然后继续训练新 token embedding。风险是新 token 表示随机、没训练好会很弱，保存时 tokenizer 和模型必须一起保存。

2 分钟版：

适合新增工具控制符、领域控制 token、特殊边界 token。不适合因为某个词被拆碎就盲目扩词表。扩词表后要检查 embedding resize、adapter 保存、推理 tokenizer、特殊 token 不被普通文本误触发，并用 eval 验证。

## 633. Tokenizer 和 chat template 是什么关系？

30 秒版：

Chat template 是把 messages 渲染成字符串或 token 序列的规则，tokenizer 决定这些 role/special tokens 怎么编码。两者共同定义 chat model 的对话协议。

2 分钟版：

换模型时不能只换 model name。要检查 role token、BOS/EOS、assistant generation prompt、tool call 格式、stop tokens 和 tokenizer special tokens。模板错会导致角色混乱、输出特殊 token、工具 JSON 不稳或 SFT loss mask 错。

## 634. Truncation、sliding window、packing 怎么处理长文本？

30 秒版：

Truncation 直接截断，简单但可能丢答案；sliding window 用重叠窗口保上下文；packing 把短样本拼成满长序列提高训练效率，但要处理 EOS 和 loss mask 边界。

2 分钟版：

训练时长样本要防止截断 assistant answer 或 EOS。RAG 文档可按 token 切 chunk，而不是按字符。Packing 适合短样本多的 SFT/CLM，但多轮对话和工具调用要小心样本边界泄漏。

## 635. Tokenization 延迟怎么优化？

30 秒版：

用 fast tokenizer、批量 tokenization、固定 system/tool schema 缓存、离线预 tokenize、prompt 压缩、RAG context 裁剪，并监控 tokenizer time 和 input tokens。

2 分钟版：

TTFT 高不一定是 GPU 慢。长 prompt、长 JSON schema、RAG 拼接和 CPU tokenizer 都可能拖慢。线上要拆 gateway、tokenizer、queue、prefill、first decode 和 network。静态前缀可以缓存 token ids 或 prefix cache。

## 636. tiktoken / API token counting 有什么用？

30 秒版：

它用于在调用模型前估算 token 数、成本和是否超上下文。不同模型 encoding 不同，不能用字符数粗估，要用对应 tokenizer 或官方计数方式。

2 分钟版：

应用里可以在请求前统计 system、messages、tools 和预留输出 token，超限就裁剪历史、压缩 RAG 或拒绝。注意 chat messages 和 tool schema 可能有额外包装 token，不同模型规则不同，所以要按对应模型计数。

## 637. Tokenizer 和 SFT labels mask 怎么联动？

30 秒版：

labels mask 必须按 token 边界做。要用 chat template 渲染后找到 assistant token 范围，把 prompt/padding 置 `-100`，只让 assistant answer 学习。

2 分钟版：

调试时要 decode input_ids 和 labels 中非 `-100` 的部分，确认模型学的是回答而不是 user prompt。常见坑是 assistant 起始边界错、EOS 被截断、padding side 不一致、packing 多样本边界错。

## 638. Tokenizer 质量怎么评估？

30 秒版：

看 tokens/char、tokens/word、中文/代码/JSON/数学/emoji 分桶、unknown 或 byte fallback 比例、专名拆分、decode 可逆性、tokenization 速度和下游效果。

2 分钟版：

评估不要只看 vocab size。真正要看真实业务文本的 token 成本、RAG chunk 长度、TTFT、格式遵循、工具调用成功率和微调效果。不同场景可以接受不同切分，不存在一个 tokenizer 对所有任务最优。

## 639. Tokenizer 相关项目 8 分钟怎么讲？

30 秒版：

按背景、分析、方案、评估、难点讲。背景可以是 token 成本高、上下文超限、模板迁移不稳、工具调用错误或中文/代码切分差。

2 分钟版：

模板：

```text
背景：模型迁移/RAG/微调中 token 成本高或格式错误。
分析：统计真实流量 tokens/char、input tokens、schema tokens、tokenizer time。
方案：统一 tokenizer/template，SFT mask 校验，RAG token budget，固定前缀缓存，必要时扩 special tokens。
评估：成本、TTFT、context overflow、格式遵循、工具调用成功率、bad case。
难点：不能随便换 tokenizer，扩词表要训练新 embedding，packing/truncation 容易切错边界。
```

## 640. Tokenizer 面试前最后怎么复习？

30 秒版：

最后背：Tokenizer 是输入协议；BPE/WordPiece/Unigram/SentencePiece 区别；special tokens；token budget；换 tokenizer 风险；扩词表流程；SFT labels mask；tokenization 延迟排查。

2 分钟版：

清单：

- BPE 合并高频 pair。
- WordPiece 关注似然提升。
- Unigram 从候选词表删 token。
- SentencePiece 直接处理原始文本，空格用 `▁`。
- Byte-level 覆盖任意字符串但可能更碎。
- BOS/EOS/PAD/UNK/role/tool tokens。
- Token budget 拆 system/history/RAG/tools/output。
- 换 tokenizer 会破坏 token id 和 embedding。
- 扩词表后 resize embedding 并继续训练。
- 线上看 tokenizer time 和 input token 分布。

背诵句：

> Tokenizer 不是分词小工具，而是模型输入协议、成本控制器和训练标签边界。
