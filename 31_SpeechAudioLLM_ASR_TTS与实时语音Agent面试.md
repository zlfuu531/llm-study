# Speech / Audio LLM、ASR、TTS 与实时语音 Agent 面试

这一章补 LLM 学习里的语音与音频方向。它常出现在：

- 语音助手 / 实时对话 Agent。
- 多模态大模型。
- 智能客服 / 会议纪要 / 字幕翻译。
- ASR / TTS / 声纹 / 音频理解。
- 端侧 AI、车载、耳机、机器人和实时交互。

你需要能回答：

- 音频怎么变成模型能处理的特征？
- ASR 常见架构 CTC、RNN-T、encoder-decoder、Whisper 怎么区分？
- streaming ASR 和离线 ASR 有什么取舍？
- TTS 为什么从声学模型、vocoder 走向 neural codec language model？
- VALL-E / AudioLM 这类方法的直觉是什么？
- Speech-to-Speech 是级联系统还是端到端系统？
- 实时语音 Agent 为什么比文本 Agent 难？
- VAD、endpointing、barge-in、turn-taking、回声消除怎么讲？
- 语音系统怎么评估、怎么控延迟、怎么做安全隐私？

## 1. 一句话总览

语音大模型系统可以拆成三条链：

```text
ASR: speech -> text
TTS: text -> speech
Audio LLM: speech/audio <-> tokens <-> reasoning/generation
```

实时语音 Agent 常见级联架构：

```text
microphone
-> VAD / noise suppression / AEC
-> streaming ASR
-> LLM / agent / tool
-> streaming TTS
-> speaker
```

端到端 speech-to-speech 架构：

```text
speech/audio tokens
-> multimodal model
-> speech/audio tokens
-> codec decoder / vocoder
```

30 秒答案：

> 语音智能体不是简单“ASR 加 LLM 加 TTS”。实时交互里要处理音频特征、流式识别、端点检测、用户打断、噪声、回声、低延迟 TTS、情绪和安全隐私。架构上可以级联 ASR-LLM-TTS，也可以端到端 speech-to-speech。级联系统工程可控，端到端保留语音副语言信息更好，但训练、评估和安全更难。

## 2. 音频输入怎么表示

原始音频是 waveform：

```text
sample_rate: 16kHz / 24kHz / 48kHz
channels: mono / stereo
amplitude over time
```

常见预处理：

- resample。
- mono。
- normalize。
- voice activity detection。
- noise suppression。
- echo cancellation。

常见特征：

| 特征 | 含义 |
| --- | --- |
| waveform | 原始波形 |
| spectrogram | 时间-频率能量图 |
| mel spectrogram | 按人耳感知压缩频率 |
| log-mel | ASR 常用输入 |
| MFCC | 传统语音识别常见特征 |
| audio codec tokens | 离散音频 token，用于生成和 Audio LLM |

log-mel 直觉：

```text
waveform -> STFT -> power spectrum -> mel filter bank -> log
```

面试表达：

> 音频是一维时序信号，但语音模型通常不会直接吃裸波形，而是转成 log-mel spectrogram 或离散 codec tokens。log-mel 更适合 ASR 编码器，codec tokens 更适合把音频当成语言模型 token 来建模和生成。

## 3. ASR 任务和指标

ASR：Automatic Speech Recognition，把语音转文本。

输出可能是：

- 字符。
- subword。
- word。
- timestamps。
- speaker labels。
- punctuation。

核心指标：

```text
WER = (S + D + I) / N
```

其中：

- S：substitution。
- D：deletion。
- I：insertion。
- N：reference words。

中文常用 CER：

```text
CER = character-level edit distance / reference characters
```

其他指标：

- RTF，real-time factor。
- latency。
- endpoint delay。
- punctuation accuracy。
- speaker diarization error rate。
- domain-specific entity accuracy。

30 秒答案：

> ASR 最常用 WER/CER，看替换、删除、插入错误。工程上不能只看 WER，还要看实时率 RTF、首字延迟、端点延迟、标点、热词、人名地名和噪声场景。会议、客服、车载、字幕的指标侧重点不一样。

## 4. CTC、RNN-T、Encoder-Decoder ASR

### CTC

CTC 解决输入帧和输出字符不对齐的问题。

核心思想：

```text
audio frames: long sequence
text tokens: shorter sequence
CTC sums over all possible alignments
```

特点：

- 条件独立假设较强。
- 解码简单。
- 适合流式或低延迟。
- 常配合 external LM / prefix beam search。

### RNN-T

RNN-T 常用于 streaming ASR：

```text
encoder(audio frames)
prediction network(previous tokens)
joint network -> next token / blank
```

特点：

- 建模历史输出。
- 适合流式。
- 工程复杂度高于 CTC。

### Encoder-Decoder / Attention

典型 seq2seq：

```text
audio encoder -> text decoder with attention
```

特点：

- 表达能力强。
- 可以做多任务。
- 离线效果好。
- 流式要做 chunk / monotonic attention 等改造。

面试表达：

> CTC 简单高效，适合低延迟；RNN-T 兼顾流式和输出历史建模，是工业流式 ASR 常见方案；encoder-decoder 表达能力强，像 Whisper 这类模型更偏大规模弱监督、多任务、离线或准流式场景。

## 5. Whisper 怎么讲

Whisper 的核心亮点：

- 大规模弱监督语音数据。
- 多语言 ASR。
- speech translation。
- language ID。
- timestamp。
- robust to noise and accents。
- encoder-decoder Transformer。

典型流程：

```text
audio -> log-mel spectrogram -> encoder
decoder -> text tokens / timestamp tokens / language tokens
```

优点：

- 泛化强。
- 多语言能力好。
- 噪声和口音鲁棒。
- 工程易用。

局限：

- 不是天然低延迟 streaming。
- 可能 hallucinate。
- 专有领域热词、人名、术语仍要适配。
- 长音频要切片、对齐和去重。

面试表达：

> Whisper 可以看成大规模弱监督训练出来的 encoder-decoder ASR 模型。它不只做转写，还把语言识别、翻译、时间戳等任务统一成 token 生成。优势是鲁棒和多语言，缺点是流式低延迟和领域热词仍需要工程适配。

## 6. Streaming ASR 为什么难

离线 ASR 可以看到完整音频；流式 ASR 只能看到当前和少量未来上下文。

难点：

- 低延迟。
- endpointing。
- partial result 稳定性。
- punctuation 延迟。
- 上下文不足导致误识别。
- 噪声和打断。
- 热词和专名。

关键指标：

| 指标 | 含义 |
| --- | --- |
| first token latency | 首个识别结果时间 |
| partial latency | 中间结果延迟 |
| endpoint delay | 说完后多久判定结束 |
| RTF | 处理速度 / 音频时长 |
| stability | partial 结果是否频繁改 |

工程技巧：

- chunked encoder。
- limited right context。
- VAD + endpointing。
- hotword boosting。
- streaming punctuation。
- partial/final result 分离。

面试表达：

> Streaming ASR 的核心取舍是准确率和延迟。看更多未来上下文会更准，但延迟更高；更早出结果体验好，但 partial 可能频繁修正。工程上要把 partial result、final result、端点检测和热词处理分开设计。

## 7. TTS 基础：从文本到语音

TTS：Text-to-Speech，把文本转语音。

传统神经 TTS 常拆成：

```text
text -> phoneme / linguistic features
-> acoustic model -> mel spectrogram
-> vocoder -> waveform
```

常见组件：

- text normalization。
- G2P，grapheme-to-phoneme。
- duration / prosody prediction。
- acoustic model。
- vocoder。

评估：

- naturalness。
- intelligibility。
- speaker similarity。
- prosody。
- MOS。
- latency。

面试表达：

> TTS 不只是读字。它要做文本规范化、发音、韵律、时长、声学特征和波形生成。自然度、可懂度、音色相似度和延迟都很重要。

## 8. Vocoder 和 Neural Codec

Vocoder 把声学特征变成 waveform。

常见：

- WaveNet。
- WaveRNN。
- MelGAN。
- HiFi-GAN。
- DiffWave。

Neural codec 把音频压缩成离散 token：

```text
waveform -> codec encoder -> discrete codes
codes -> codec decoder -> waveform
```

常见思想：

- SoundStream。
- EnCodec。
- residual vector quantization。

为什么重要：

- 把连续音频变成 token。
- 方便用 language model 建模音频。
- 适合 TTS、voice conversion、AudioLM、VALL-E。

面试表达：

> 传统 TTS 常预测 mel 再用 vocoder 合成波形。新一代语音生成常把音频压成 codec tokens，再像语言模型一样预测离散音频 token，最后由 codec decoder 还原语音。这让音频生成更容易和 LLM 统一。

## 9. VALL-E / Neural Codec LM 怎么讲

VALL-E 类方法的直觉：

```text
text + speaker prompt audio
-> predict acoustic codec tokens
-> codec decoder -> speech
```

特点：

- zero-shot voice cloning。
- 保留说话人音色。
- 可以建模情绪和声学环境。
- 使用离散 codec tokens。

风险：

- 声音克隆滥用。
- 版权和身份冒充。
- 训练数据合规。
- watermark / speaker consent。

30 秒答案：

> VALL-E 的核心是把 TTS 变成 neural codec language modeling。它不直接预测 mel，而是预测离散音频 codec token；给一小段 speaker prompt，就能生成相似音色的语音。优势是音色迁移强，风险是声音克隆滥用，所以必须考虑授权、水印和检测。

## 10. AudioLM / AudioPaLM / Audio LLM

AudioLM 思路：

```text
audio -> semantic tokens + acoustic tokens
language modeling over audio tokens
```

AudioPaLM / speech-language model 思路：

```text
speech/text tokens
-> shared language model space
-> speech/text generation
```

Audio LLM 能做：

- speech recognition。
- speech translation。
- spoken QA。
- speech-to-speech translation。
- audio understanding。
- speech generation。

关键矛盾：

- 语义信息 vs 声学细节。
- 离散 token rate 太高导致上下文长。
- 音色、情绪、韵律和内容要分开建模。
- 实时交互要求低延迟。

面试表达：

> Audio LLM 的关键是把音频也 token 化或映射到 LLM 表示空间。ASR 只保留文本内容，Audio LLM 还可能保留语调、情绪、说话人和环境等副语言信息，但 token 速率、训练数据和实时延迟都是难点。

## 11. 级联语音 Agent vs 端到端 Speech-to-Speech

级联系统：

```text
ASR -> LLM -> TTS
```

优点：

- 模块可替换。
- 工程可控。
- 易调试。
- 文本 trace 方便审计。
- 可接现有 LLM / tools。

缺点：

- 丢失语气、情绪、停顿等音频信息。
- ASR 错误会传递。
- TTS 延迟叠加。
- 打断和 turn-taking 复杂。

端到端：

```text
speech -> speech
```

优点：

- 保留副语言信息。
- 延迟潜力更低。
- 更自然的交互。

缺点：

- 训练难。
- 评估难。
- 安全审计难。
- 工具调用和可控性更难。

面试表达：

> 级联 ASR-LLM-TTS 是当前工程上最稳的路线，方便调试和接工具；端到端 speech-to-speech 更自然，能保留情绪和语调，但训练、评估、安全和可解释性更难。实际产品可能混合使用。

## 12. 实时语音 Agent 的核心难点

实时语音 Agent 要处理：

- VAD。
- endpointing。
- barge-in。
- turn-taking。
- streaming ASR。
- streaming LLM。
- streaming TTS。
- echo cancellation。
- noise suppression。
- latency budget。
- tool call delay。

典型延迟预算：

```text
capture/VAD
+ ASR partial
+ LLM first token
+ TTS first audio
+ network
= perceived response latency
```

用户体验关键：

- 能不能快点开口。
- 能不能被打断。
- 能不能听懂半句话。
- 能不能处理背景噪声。
- 能不能自然接话。
- 工具调用等待时怎么填充。

面试表达：

> 实时语音 Agent 的难点不是把三个模型串起来，而是 turn-taking 和 latency。系统要在用户还没完全说完时判断是否该响应，同时允许用户打断模型输出，并在工具调用和 TTS 延迟下保持自然体验。

## 13. VAD、Endpointing、Barge-in

VAD：判断有没有人在说话。

Endpointing：判断这一轮话是否说完。

Barge-in：用户在系统说话时打断，系统要停止 TTS 并切回听用户。

常见错误：

- VAD 太敏感：噪声被当成说话。
- VAD 太迟钝：漏掉短句。
- endpoint 太早：用户停顿一下就被打断。
- endpoint 太晚：系统反应慢。
- barge-in 不灵：用户体验差。

策略：

- acoustic VAD + semantic endpointing。
- partial transcript 判断语义是否完整。
- TTS 播放可中断。
- echo cancellation 避免把系统声音识别成用户。

面试表达：

> VAD 看有没有语音，endpointing 看这一轮是否结束，barge-in 处理用户打断。实时 Agent 要把声学信号和语义信号结合起来，不能只靠静音时长。

## 14. 语音系统评估

ASR：

- WER / CER。
- entity accuracy。
- punctuation。
- timestamp accuracy。
- latency。
- RTF。

TTS：

- MOS。
- naturalness。
- intelligibility。
- speaker similarity。
- prosody。
- first audio latency。

Voice Agent：

- turn success。
- task success。
- interruption success。
- response latency。
- barge-in latency。
- tool success。
- user satisfaction。
- safety violation rate。

面试表达：

> 语音系统不能只看 ASR WER。实时 Agent 要看端到端任务成功率、响应延迟、打断成功率、TTS 首包延迟、工具调用成功率和用户满意度。

## 15. 安全、隐私和合规

语音系统风险：

- 录音包含 PII。
- 声纹可识别身份。
- 声音克隆冒充。
- 未授权录音。
- 儿童/医疗/金融等高敏场景。
- ASR 错误导致错误执行工具。
- TTS 生成冒充真人。

治理：

- 用户授权和录音提示。
- 数据最小化。
- 端侧 VAD / on-device preprocessing。
- PII 脱敏。
- 声纹和 voice cloning 权限。
- watermark。
- 高风险工具二次确认。
- 审计日志和保留期限。

面试表达：

> 语音比文本更敏感，因为它包含身份、情绪和环境信息。上线时要做授权、脱敏、访问控制、声纹/克隆风险治理和高风险操作确认。

## 16. 端侧语音和低延迟部署

端侧语音常见场景：

- 手机助手。
- 车载。
- 耳机。
- 会议字幕。
- 离线唤醒词。

约束：

- CPU/NPU 算力。
- 内存。
- 电量。
- 实时性。
- 网络不稳定。
- 隐私。

优化：

- 小模型 ASR。
- quantization。
- streaming encoder。
- keyword spotting。
- VAD on-device。
- local wake word。
- cloud fallback。
- cache prompt / TTS。

面试表达：

> 端侧语音通常采用端云协同：唤醒词、VAD、简单指令和隐私敏感处理尽量端侧做，复杂理解和生成上云。核心取舍是延迟、隐私、电量和效果。

## 17. 语音 Agent 项目讲法

8 分钟结构：

1. **场景**：客服、会议、车载、英语口语、AI 陪练、语音助手。
2. **输入链路**：麦克风、VAD、降噪、AEC、streaming ASR。
3. **理解链路**：LLM、上下文、工具、状态机、权限。
4. **输出链路**：streaming TTS、voice style、barge-in。
5. **延迟预算**：ASR 首字、LLM 首 token、TTS 首包、网络。
6. **评估**：WER/CER、任务成功率、打断成功率、用户满意度。
7. **安全**：录音授权、PII、声纹、工具确认、日志审计。
8. **优化**：热词、端点检测、缓存、端云协同、fallback。

收尾句：

> 这个项目的关键不是把 ASR、LLM、TTS 串起来，而是把实时交互、打断、延迟、安全和任务成功率做成闭环。

## 18. 高频追问清单

1. 音频 waveform、spectrogram、log-mel、codec token 有什么区别？
2. ASR 的 WER/CER 怎么算？
3. CTC、RNN-T、encoder-decoder ASR 怎么区分？
4. Whisper 的优势和局限是什么？
5. Streaming ASR 为什么比离线 ASR 难？
6. endpointing 和 VAD 有什么区别？
7. TTS 的 acoustic model 和 vocoder 分别做什么？
8. Neural codec 为什么重要？
9. VALL-E / codec language model 的直觉是什么？
10. AudioLM / Audio LLM 和普通 ASR 有什么区别？
11. ASR-LLM-TTS 级联和端到端 speech-to-speech 怎么选？
12. 实时语音 Agent 的延迟预算怎么拆？
13. Barge-in 怎么实现？
14. 语音 Agent 怎么评估？
15. 声音克隆有什么安全风险？
16. 端侧语音怎么做低延迟？
17. 热词和专有名词怎么处理？
18. ASR hallucination 怎么发现和治理？
19. 语音工具调用怎么防误触发？
20. 语音 Agent 项目怎么讲 8 分钟？

## 19. 推荐阅读

- Whisper：[https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- VALL-E：[https://arxiv.org/abs/2301.02111](https://arxiv.org/abs/2301.02111)
- AudioLM：[https://arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143)
- AudioPaLM：[https://arxiv.org/abs/2306.12925](https://arxiv.org/abs/2306.12925)
- SeamlessM4T：[https://arxiv.org/abs/2308.11596](https://arxiv.org/abs/2308.11596)
- GPT-4o：[https://openai.com/index/hello-gpt-4o/](https://openai.com/index/hello-gpt-4o/)
- OpenAI Realtime API：[https://platform.openai.com/docs/guides/realtime](https://platform.openai.com/docs/guides/realtime)

## 20. 本章复习顺序

第一遍：

1. 音频特征、ASR 指标。
2. CTC、RNN-T、Whisper、streaming ASR。
3. TTS、vocoder、neural codec、VALL-E。
4. Audio LLM、speech-to-speech、实时 Agent。
5. 评估、安全、端侧和项目讲法。

第二遍：

- 先背 541-546：音频特征和 ASR。
- 再背 547-551：TTS、codec、Audio LLM。
- 再背 552-557：实时 Agent、打断、评估、安全。
- 最后背 558-560：端侧、误触发和项目讲法。
