# Tokenizer、BPE、SentencePiece 与 Token 预算面试

这一章面向所有大模型岗位，尤其是算法、微调、RAG、推理服务、代码模型和多语言应用。Tokenizer 看起来像预处理小细节，但它会影响上下文长度、显存、成本、中文/代码/JSON 表现、chat template、SFT loss mask、工具调用格式、模型迁移和线上延迟。

如果时间很紧，先背这句：

> Tokenizer 是文本和模型 token 空间之间的协议。它不只是把字切开，而是决定输入长度、词表、special tokens、BOS/EOS/PAD、chat template、loss mask、生成停止和模型迁移边界；换 tokenizer 基本等于改模型接口，不能随便换。

相关答案版：[answers/30_Tokenizer_BPE_SentencePiece与Token预算_答案版.md](answers/30_Tokenizer_BPE_SentencePiece与Token预算_答案版.md)

相邻章节：

- [26_开源模型生态_模型选型与ChatTemplate面试.md](26_开源模型生态_模型选型与ChatTemplate面试.md)：模型选型、chat template、迁移风险。
- [34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md](34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md)：AutoTokenizer、SFT labels mask、DataCollator。
- [24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md](24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md)：tokenizer 延迟、TTFT、prompt 长度。
- [23_代码大模型_CodeAgent与SWEbench面试.md](23_代码大模型_CodeAgent与SWEbench面试.md)：代码 tokenizer、FIM、仓库上下文。

## 1. Tokenizer 为什么重要

LLM 不是直接处理 Unicode 字符串，而是处理 token id 序列：

```text
text -> normalize/pre-tokenize -> subword tokenize -> token ids
token ids -> embedding lookup -> Transformer -> logits over vocab
```

Tokenizer 影响：

- token 数：同一段中文/代码/JSON，在不同 tokenizer 下 token 数可能差很多。
- 成本：API 和推理服务通常按 token 计费或按 token 消耗算力。
- 上下文：token 数越多，占用上下文越多，KV Cache 越大。
- 训练：SFT labels mask、padding、packing、truncation 都依赖 token 边界。
- 生成：EOS、stop token、工具调用格式依赖特殊 token。
- 迁移：换模型时 tokenizer 和 chat template 一起迁移。
- 性能：CPU tokenizer 可能成为 TTFT 瓶颈。

面试句：

> Tokenizer 决定“模型实际看到什么”，所以 prompt、数据、训练、推理和评估都要以 token 视角检查。

## 2. Tokenization pipeline

常见 tokenizer pipeline：

```text
normalizer
-> pre-tokenizer
-> model(BPE/WordPiece/Unigram)
-> post-processor
-> decoder
```

各层职责：

| 层 | 作用 | 例子 |
| --- | --- | --- |
| normalizer | 文本归一化 | lower-case、NFKC、去重音、空白处理 |
| pre-tokenizer | 初步切分 | 空格、标点、byte-level、正则 |
| model | 子词算法 | BPE、WordPiece、Unigram |
| post-processor | 加 special tokens | BOS/EOS、CLS/SEP、chat role tokens |
| decoder | token 还原文本 | byte decoder、SentencePiece `▁` 还原空格 |

工程里常见问题：

- normalization 改掉大小写、全角半角或空格。
- pre-tokenizer 对中文、代码、emoji、URL 表现差。
- post-processor 加错 special tokens。
- decoder 还原后空格错。

## 3. BPE 的训练和编码

BPE 的直觉：从字符或 byte 开始，反复合并最高频相邻 pair，直到达到目标词表大小。

训练过程简化：

```text
corpus -> 统计初始 symbol
-> 统计相邻 pair 频率
-> 合并最高频 pair
-> 更新语料表示
-> 重复直到 vocab size
```

例子：

```text
l o w
l o w e r
n e w e s t

如果 "l o" 高频，合并成 "lo"
如果 "lo w" 高频，合并成 "low"
```

编码时：

```text
把输入拆成初始 symbol
按学到的 merge rank 从高到低合并
得到 token 序列
```

BPE 优点：

- 词表可控。
- 常见词/片段可以成为单 token。
- 稀有词可以拆成子词。
- 适合 GPT 系列和很多 decoder-only LLM。

缺点：

- 对训练语料分布敏感。
- 对空格、大小写、Unicode 处理依赖实现。
- 没见过的字符如果 base vocab 不覆盖，可能 `<unk>`；byte-level BPE 可缓解。

## 4. Byte-level BPE 和 byte fallback

普通字符级 base vocab 很难覆盖全部 Unicode。Byte-level BPE 用 256 个 byte 作为基础符号：

```text
任何文本 -> UTF-8 bytes -> byte token / merge token
```

好处：

- 理论上任何字符串都能表示。
- 减少 `<unk>`。
- 对 emoji、罕见字符、混合语言更稳。

代价：

- 非英语、中文、emoji、特殊符号可能被拆得更碎。
- token 数增加，影响成本和上下文。
- 人眼看 token 更不直观。

byte fallback 类似思路：当字符或子词无法表示时，退回到 byte 表示，避免 unknown。

面试句：

> Byte-level 的目标不是让每个 token 都语义漂亮，而是保证任意输入都可编码。

## 5. WordPiece

WordPiece 常见于 BERT 系列。它也从小单元开始合并，但合并选择更偏最大化训练数据似然，而不是简单合并最高频 pair。

常见表示：

```text
unaffable -> un ##aff ##able
```

`##` 表示该 token 是词内部片段。

BPE vs WordPiece：

| 维度 | BPE | WordPiece |
--- | --- | --- |
| 合并依据 | 最高频 pair 或 merge rank | 提升似然/信息量的 pair |
| 常见模型 | GPT、Llama、Qwen 等很多 LLM | BERT、DistilBERT、Electra |
| OOV | 依赖 base vocab 或 byte-level | 可能 `[UNK]` |
| 表示 | merge rules | 子词词表 + continuation 标记 |

面试里不用深推 WordPiece 训练公式，重点说清：

```text
BPE 更像频率合并规则；
WordPiece 更关注合并后对语料似然的提升；
BERT 系列常用 WordPiece。
```

## 6. Unigram 和 SentencePiece

Unigram 从一个较大的候选子词集合开始，为每个 token 学一个概率，然后逐步删掉贡献小的 token，保留能高概率解释语料的词表。

直觉：

```text
一个词可以有多种切分
选择概率最大的切分
训练时可用 subword regularization 采样不同切分增强鲁棒性
```

SentencePiece 是一个 tokenizer 工具/框架，常用于 BPE 或 Unigram。它把输入当作原始字符流，不强依赖空格分词，对中文、日文这类没有空格的语言更自然。

SentencePiece 里的空格常被表示成：

```text
▁
```

比如：

```text
"Hello world" -> ["▁Hello", "▁world"]
```

面试句：

> SentencePiece 不是一种单独的子词算法本身，它常承载 BPE 或 Unigram，并把空格也作为可学习的一部分处理。

## 7. 特殊 token：BOS、EOS、PAD、UNK、SEP、MASK

常见 special tokens：

| token | 作用 | 常见风险 |
| --- | --- | --- |
| BOS | 序列开始 | 重复加 BOS 影响分布 |
| EOS | 序列结束 | 丢 EOS 会停不下来或训练边界错 |
| PAD | batch padding | 没 mask 会污染 loss |
| UNK | 未知 token | 过多说明词表覆盖差 |
| SEP | 分隔句子/段落 | BERT/检索任务常见 |
| MASK | MLM 掩码 | BERT 预训练常见 |
| role tokens | chat 角色 | 模板错会角色混乱 |
| tool tokens | 工具调用边界 | 工具 JSON 不稳 |

注意：

- `tokenizer.pad_token_id` 要和 `model.config.pad_token_id` 对齐。
- SFT labels 里 padding 一般设为 `-100`。
- 推理 stop token 和 EOS 不一定完全一样。
- chat template 可能自动添加 BOS/EOS，手动再加会重复。

## 8. Token 预算怎么估算

上下文预算不是字符预算。

```text
total_context =
system tokens
+ user prompt tokens
+ retrieved context tokens
+ tool schema tokens
+ conversation history tokens
+ reserved output tokens
```

示例：

```text
模型上下文 32k
系统 prompt 1k
工具 schema 3k
历史对话 4k
RAG chunks 18k
预留输出 4k
剩余用户输入约 2k
```

token 预算影响：

- prefill 时间。
- KV Cache 显存。
- API 成本。
- RAG context packing。
- 长上下文 lost-in-the-middle。
- 输出是否被截断。

面试答法：

> 我会把上下文预算显式拆表，而不是等模型报 context length exceeded。

## 9. 中文、多语言和领域词

中文 tokenizer 要看：

- 单字、词、子词切分比例。
- 同一段中文 token 数是否过高。
- 中英混排、数字、单位、公式。
- 专有名词、公司名、产品名。
- 繁简、全角半角、标点。
- 领域词是否被拆得太碎。

中文切得太碎的后果：

- 同样文本占更多上下文。
- RAG chunk token 数难控。
- 成本和 TTFT 上升。
- 长文本有效信息减少。

但不能只看“中文 token 少”：

- 预训练数据质量也重要。
- 指令对齐和领域数据也重要。
- token 粒度太粗会扩大词表和 embedding。

## 10. 代码、数学、JSON、工具调用 tokenizer

代码 tokenizer 特殊：

- 缩进和换行有语义。
- snake_case、camelCase、路径、包名多。
- 符号、括号、点号、冒号频繁。
- 长标识符和稀有 API 多。
- FIM 需要处理 prefix/middle/suffix 特殊 token。

JSON / tool calling：

- `{}`、`[]`、`"`、`:`、`,` 很关键。
- schema 长，token 预算高。
- 字段名反复出现，可缓存或压缩。
- constrained decoding 依赖 token 级合法性。

数学：

- LaTeX 符号、上下标、数字、小数、单位。
- 一个数字可能被拆成多个 token。
- 数字 tokenization 会影响精确复制和计算。

面试句：

> 代码和 JSON 的 tokenizer 不只是压缩率问题，它还影响结构保真、补全边界和格式遵循。

## 11. 换 tokenizer 为什么危险

微调或上线时通常不能随便换 tokenizer。

原因：

- embedding matrix 行数和 token id 对应词表绑定。
- 同一个 token id 在新 tokenizer 里可能对应不同字符串。
- chat template special tokens 可能不同。
- 数据 token 分布改变，训练分布漂移。
- LoRA adapter 绑定 base model 和 tokenizer。
- 旧 prompt、RAG chunk、缓存、日志不可直接复用。

什么时候可以换？

- 从头预训练。
- 明确做 tokenizer 迁移和 embedding 初始化。
- 扩词表后继续训练足够步数。
- 有完整 eval 验证新旧行为。

面试句：

> tokenizer 不是可插拔前处理，换它相当于改了模型输入协议和 embedding 查表规则。

## 12. 扩词表怎么做

有时领域词、特殊控制 token、工具 token 需要新增 token。

Hugging Face 常见流程：

```python
num_added = tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|tool_call|>", "<|tool_result|>"]
})
model.resize_token_embeddings(len(tokenizer))
```

注意：

- 新 token embedding 是新初始化的，需要训练。
- 如果只扩 tokenizer 不 resize model，会报错或越界。
- 如果只 resize 不训练，新 token 表示很弱。
- 保存时 tokenizer 和 model/adapter 要一起保存。
- special tokens 要避免被普通文本误触发。

扩词表适合：

- 明确控制符。
- 工具调用边界。
- 领域专用标记。

不适合：

- 只因为某些词被拆得碎就盲目扩。
- 没有继续训练预算。

## 13. Tokenizer 和 SFT loss mask

SFT 里常见结构：

```text
system + user + assistant
```

训练目标通常只让 assistant token 算 loss：

```text
labels:
system/user -> -100
assistant -> token id
padding -> -100
```

Tokenizer 相关坑：

- assistant 起始 token 边界找错。
- chat template 自动加 EOS，手动又加一次。
- padding side 不一致。
- left padding 对生成和 position ids 有影响。
- truncation 截断了 answer 或 EOS。
- packing 多样本时 EOS/边界错。

调试动作：

```text
打印原始 messages
打印 apply_chat_template 后文本
decode input_ids
decode labels 中非 -100 部分
检查 EOS/PAD/BOS
```

## 14. Tokenization 延迟和生产优化

线上 TTFT 里 tokenizer 可能是瓶颈，尤其是：

- prompt 很长。
- RAG context 很大。
- 每次都重新 tokenize system prompt/tool schema。
- Python tokenizer 慢。
- 并发高、CPU 打满。
- JSON/schema 很长。

优化：

- 使用 fast tokenizer。
- system prompt、tool schema、固定前缀做缓存。
- prompt 压缩和 context 裁剪。
- 离线预 tokenize 静态文档或训练数据。
- 批量 tokenization。
- 监控 tokenizer time、input tokens、queue time。

推理引擎里要分开看：

```text
gateway time
tokenizer time
queue time
prefill time
first decode time
network flush
```

## 15. Tokenizer 质量怎么评估

不能只看 vocab size。

评估维度：

- 平均 tokens/character 或 tokens/word。
- 中文、英文、代码、JSON、数学、emoji 分桶。
- `<unk>` 或 byte fallback 比例。
- 专有名词拆分情况。
- chat template 特殊 token 是否稳定。
- decode 是否可逆。
- tokenization 速度。
- 对下游任务效果和成本影响。

简单诊断表：

| 现象 | 可能原因 |
| --- | --- |
| 中文成本高 | 中文切得太碎 |
| 工具调用 JSON 不稳 | 特殊符号/模板/约束问题 |
| 代码补全缩进差 | 换行缩进 token 表示差 |
| 微调后复述 prompt | loss mask/token 边界错 |
| TTFT 高 | tokenizer 或 prompt 过长 |
| 换模型效果崩 | tokenizer/chat template 不一致 |

## 16. 项目 8 分钟讲法

```text
背景：
我们在 RAG/微调/模型迁移里发现 token 成本高、上下文不够、模板迁移不稳或工具调用格式错误。

分析：
先对真实流量分桶统计 token 数：中文、代码、JSON、长文档、工具 schema、历史对话。
再对比候选模型 tokenizer 的 tokens/char、特殊 token、chat template 和 decode 可逆性。

方案：
训练侧统一 tokenizer 和 chat template，SFT labels 只对 assistant token 算 loss；
推理侧做 token budget 表、RAG context 裁剪、固定前缀缓存、工具 schema 压缩；
如果确实要新增控制 token，扩词表后 resize embedding，并继续训练验证。

评估：
看 token 成本、TTFT、context overflow、格式遵循、引用正确率、工具调用成功率和 bad case。

难点：
不能随便换 tokenizer；扩词表要训练新 embedding；packing/truncation 容易切错 EOS 和 answer；tokenizer 快慢会影响线上延迟。
```

## 17. 面试前背诵版

Tokenizer 是模型输入协议。BPE 从字符或 byte 开始合并高频 pair，WordPiece 更关注合并后似然提升，Unigram 从候选词表里删掉贡献小的 token，SentencePiece 可直接处理原始文本并用 `▁` 表示空格，适合无空格语言。Byte-level BPE 用 256 byte 覆盖任意字符串，减少 unknown，但可能让中文和特殊符号 token 数变多。Special tokens 包括 BOS/EOS/PAD/UNK/SEP/MASK/role/tool tokens，训练和推理必须一致。Token 预算要拆 system、history、RAG、tool schema、user 和 output reserve。微调时 tokenizer 决定 labels mask 边界，换 tokenizer 会破坏 embedding id 映射和模型输入协议。生产中要监控 tokenizer time、input tokens、context overflow 和 token 成本。

## 本轮参考

- Hugging Face Tokenization algorithms：[https://huggingface.co/docs/transformers/tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- Hugging Face Tokenizers 文档：[https://huggingface.co/docs/tokenizers/index](https://huggingface.co/docs/tokenizers/index)
- OpenAI Cookbook: How to count tokens with tiktoken：[https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
- Neural Machine Translation of Rare Words with Subword Units：[https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
- SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing：[https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
