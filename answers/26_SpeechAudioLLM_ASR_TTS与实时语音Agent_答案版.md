# 答案版 26：Speech / Audio LLM、ASR、TTS 与实时语音 Agent

对应题号：541-560。建议先读 [31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md](../31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 541. 音频 waveform、spectrogram、log-mel、codec token 有什么区别？

30 秒版：

waveform 是原始波形；spectrogram 是时间-频率能量图；log-mel 是按人耳感知压缩后的频谱特征，ASR 常用；codec token 是神经音频 codec 把连续语音压成离散 token，适合 Audio LLM 和语音生成。

2 分钟版：

对比：

| 表示 | 含义 | 常见用途 |
| --- | --- | --- |
| waveform | 原始采样点 | 端到端音频模型、vocoder |
| spectrogram | 频谱随时间变化 | 音频分析 |
| log-mel | mel filter + log | ASR、Whisper 类模型 |
| codec token | 离散音频 token | VALL-E、AudioLM、speech generation |

流程：

```text
waveform -> STFT -> mel filter bank -> log-mel
waveform -> codec encoder -> discrete codes
```

面试表达：

> log-mel 更像给识别模型用的稳健声学特征，codec token 更像把音频转成语言模型能预测的离散符号。

## 542. ASR 的 WER/CER 怎么算？

30 秒版：

WER 是词错误率，公式是 `(S + D + I) / N`，S 是替换，D 是删除，I 是插入，N 是参考文本词数。中文常用 CER，按字符级编辑距离计算。工程上还要看实体准确率、延迟和实时率。

2 分钟版：

WER：

```text
WER = (Substitution + Deletion + Insertion) / ReferenceWords
```

CER：

```text
CER = character edit distance / reference characters
```

例子：

- 把“北京天气”识别成“背景天气”：替换错误。
- 漏掉一个词：删除错误。
- 多识别出一个词：插入错误。

不足：

- 不区分关键词重要性。
- 标点和大小写处理影响结果。
- 人名、地名、数字更重要。

面试表达：

> ASR 项目不能只报平均 WER，还要按噪声、口音、领域词、人名地名和实时场景分桶看。

## 543. CTC、RNN-T、encoder-decoder ASR 怎么区分？

30 秒版：

CTC 通过 blank 和对齐求和解决帧到文本不对齐，简单低延迟；RNN-T 有 encoder、prediction network 和 joint network，适合流式且能建模输出历史；encoder-decoder 表达能力强，像 Whisper，但低延迟流式要额外改造。

2 分钟版：

CTC：

```text
audio frames -> token/blank probabilities
sum over alignments
```

RNN-T：

```text
audio encoder + previous token predictor -> joint -> next token
```

Encoder-decoder：

```text
audio encoder -> text decoder with attention
```

取舍：

- CTC：快、简单，但独立性假设强。
- RNN-T：工业流式常见，但训练解码复杂。
- Encoder-decoder：效果强、任务统一，但流式延迟压力大。

## 544. Whisper 的优势和局限是什么？

30 秒版：

Whisper 是大规模弱监督训练的 encoder-decoder ASR 模型，优势是多语言、鲁棒、支持翻译和时间戳；局限是非天然 streaming，专有领域热词仍可能错，长音频需要切片和对齐，也可能 hallucinate。

2 分钟版：

优势：

- 多语言 ASR。
- speech translation。
- language ID。
- timestamp tokens。
- 噪声和口音鲁棒。
- 工程易用。

局限：

- 低延迟 streaming 不如专门流式模型。
- 长音频 chunk 拼接会有重复和漏字。
- 热词、人名、术语可能错。
- 静音或噪声下可能 hallucinate。

面试表达：

> Whisper 很适合做强 baseline，但如果是车载、同传、客服实时字幕，还要专门处理 streaming、endpoint、热词和延迟。

## 545. Streaming ASR 为什么比离线 ASR 难？

30 秒版：

离线 ASR 能看完整音频，流式只能看当前和少量未来上下文，所以准确率和延迟冲突。难点包括 partial 结果稳定性、endpointing、标点延迟、热词、噪声和首字延迟。

2 分钟版：

核心取舍：

```text
more future context -> more accurate -> higher latency
less future context -> lower latency -> less stable
```

难点：

- 首字延迟。
- endpoint delay。
- partial result 频繁改。
- 说话人停顿。
- 背景噪声。
- 领域词。

工程方案：

- chunked encoder。
- limited right context。
- VAD + endpointing。
- hotword boosting。
- partial/final 分离。

面试表达：

> 流式 ASR 不只是把离线模型切块跑，必须围绕延迟、稳定性和端点检测重新设计。

## 546. VAD、endpointing、barge-in 有什么区别？

30 秒版：

VAD 判断有没有人在说话；endpointing 判断这一轮话是否说完；barge-in 是用户在系统说话时打断，系统要停止 TTS 并重新听用户。实时语音 Agent 不能只靠静音时长，还要结合语义完整性和回声消除。

2 分钟版：

VAD：

```text
speech / non-speech
```

Endpointing：

```text
has user finished this turn?
```

Barge-in：

```text
user interrupts while assistant is speaking
stop playback -> listen -> update state
```

常见错误：

- 噪声误触发 VAD。
- 用户短暂停顿被当作说完。
- 系统自己的 TTS 被 ASR 识别成用户。

治理：

- acoustic VAD。
- semantic endpoint。
- echo cancellation。
- interruptible TTS。
- state machine。

## 547. TTS 的 acoustic model 和 vocoder 分别做什么？

30 秒版：

TTS 常拆成 acoustic model 和 vocoder。Acoustic model 把文本、音素和韵律信息预测成 mel 等声学特征；vocoder 把声学特征还原成 waveform。新一代方法也会用 codec token 直接做语音语言建模。

2 分钟版：

传统神经 TTS：

```text
text -> normalization/G2P -> acoustic model -> mel spectrogram -> vocoder -> waveform
```

Acoustic model 负责：

- 发音。
- 时长。
- 韵律。
- mel 特征。

Vocoder 负责：

- 把 mel 转波形。
- 决定音质和实时性。

评估：

- naturalness。
- intelligibility。
- speaker similarity。
- MOS。

## 548. Neural codec 为什么重要？

30 秒版：

Neural codec 把连续音频压缩成离散 codec tokens，再由 decoder 还原波形。它让语音生成可以像语言建模一样预测离散 token，是 VALL-E、AudioLM 等 audio language model 的关键基础。

2 分钟版：

流程：

```text
waveform -> codec encoder -> discrete codes
discrete codes -> codec decoder -> waveform
```

价值：

- 压缩音频。
- 离散化音频。
- 方便 LM 建模。
- 保留音色、韵律、环境等声学信息。

风险：

- token rate 高，上下文长。
- codec 质量限制生成音质。
- 多码本建模复杂。

面试表达：

> codec token 对音频的意义有点像 tokenizer 对文本的意义，但它还要保留声学细节。

## 549. VALL-E / codec language model 的直觉是什么？

30 秒版：

VALL-E 把 TTS 看成 neural codec language modeling：输入文本和一小段 speaker prompt audio，模型预测目标语音的 codec tokens，再由 codec decoder 合成语音。优势是 zero-shot voice cloning，风险是身份冒充和滥用。

2 分钟版：

流程：

```text
text + speaker prompt
-> acoustic codec tokens
-> codec decoder
-> speech
```

相比传统 TTS：

- 不只预测 mel。
- 用离散音频 token。
- 能从 prompt audio 学音色。
- 能保留一定声学环境。

风险：

- 声音克隆。
- 未授权模仿。
- deepfake。
- 水印和检测。

面试表达：

> 这种方法把语音生成变成了条件语言建模，所以和 LLM 技术栈更容易融合。

## 550. AudioLM / Audio LLM 和普通 ASR 有什么区别？

30 秒版：

ASR 主要把 speech 转 text，只保留文本内容；Audio LLM 会直接建模音频 token 或音频表示，可能保留语气、音色、情绪和环境信息，并支持 speech understanding、speech generation、speech-to-speech translation。

2 分钟版：

ASR：

```text
speech -> text
```

Audio LLM：

```text
speech/audio tokens <-> language model <-> text/audio tokens
```

能力：

- ASR。
- speech translation。
- spoken QA。
- TTS。
- speech-to-speech。
- audio event understanding。

难点：

- 音频 token rate 高。
- 多模态对齐。
- 情绪/音色/内容分离。
- 实时低延迟。

## 551. ASR-LLM-TTS 级联和端到端 speech-to-speech 怎么选？

30 秒版：

级联 ASR-LLM-TTS 工程可控、易调试、方便接工具和审计，但会丢失语气信息并有错误传递；端到端 speech-to-speech 更自然、能保留副语言信息，但训练、评估、安全和工具调用更难。实际产品常混合使用。

2 分钟版：

级联优点：

- 模块可替换。
- 文本 trace 可审计。
- 容易接现有 LLM 和工具。
- 成熟稳定。

级联缺点：

- ASR 错误传递。
- 情绪/语气丢失。
- 延迟叠加。

端到端优点：

- 更自然。
- 保留情绪和韵律。
- 延迟潜力低。

端到端难点：

- 数据和训练难。
- 安全审计难。
- 可控性弱。

## 552. 实时语音 Agent 的延迟预算怎么拆？

30 秒版：

端到端延迟包括采集/VAD、ASR partial、LLM 首 token、工具调用、TTS 首包、网络和播放缓冲。用户感知的是从说完到听到回应的时间，所以要优化 streaming ASR、streaming LLM、streaming TTS 和 endpointing。

2 分钟版：

预算：

```text
capture + VAD
+ ASR first partial / final
+ LLM first token
+ tool latency
+ TTS first audio
+ network jitter
+ playback buffer
```

优化：

- ASR partial 早出。
- LLM streaming。
- TTS streaming。
- endpoint 不要太晚。
- 工具调用异步或填充语。
- cache 常用回复。

面试表达：

> 实时语音体验主要看 first response latency 和 turn-taking，而不是只看模型平均推理时间。

## 553. Barge-in 怎么实现？

30 秒版：

Barge-in 是用户打断系统说话。实现上要让 TTS 播放可中断，同时用 VAD/ASR 监听用户新语音，并用 echo cancellation 避免把系统自己的声音识别成用户；状态机要取消或暂停当前生成。

2 分钟版：

流程：

```text
assistant speaking
-> detect user speech
-> stop TTS playback
-> cancel/interrupt LLM generation if needed
-> ASR user speech
-> update dialogue state
```

关键：

- full-duplex audio。
- acoustic echo cancellation。
- interruptible TTS。
- generation cancellation。
- state rollback。

常见坑：

- 系统声音回灌导致误触发。
- 用户短声被忽略。
- 停止 TTS 但后端还在生成。

## 554. 语音 Agent 怎么评估？

30 秒版：

除了 ASR WER 和 TTS MOS，还要看端到端任务成功率、响应延迟、打断成功率、turn-taking 自然度、工具调用正确率、安全违规率和用户满意度。语音 Agent 是交互系统，不是单模型评测。

2 分钟版：

指标：

- ASR：WER/CER、实体准确率、延迟。
- TTS：MOS、自然度、音色、首包延迟。
- Agent：task success、turn success、barge-in success、tool success。
- Safety：误触发、高风险操作、PII。
- Product：留存、用户评分、平均会话时长。

面试表达：

> 我会分模块评估，也会做端到端场景评估，因为 ASR 小错可能导致 Agent 大错。

## 555. 声音克隆有什么安全风险？

30 秒版：

声音克隆可能用于身份冒充、诈骗、伪造授权、deepfake 和名人滥用。治理要做用户授权、speaker consent、声纹保护、水印、克隆检测、敏感场景限制和审计。

2 分钟版：

风险：

- 冒充亲友或领导。
- 金融诈骗。
- 未授权复刻主播/名人。
- 绕过声纹认证。
- 合成虚假证据。

治理：

- 明确授权。
- speaker verification。
- watermark。
- provenance。
- sensitive speaker blocklist。
- human review。
- usage logging。

面试表达：

> TTS 能力越强，安全越不是附加项，而是产品能力的一部分。

## 556. 端侧语音怎么做低延迟？

30 秒版：

端侧常做唤醒词、VAD、简单 ASR、降噪和隐私敏感处理，复杂理解和生成上云。优化包括小模型、量化、streaming encoder、NPU、缓存和端云协同。核心取舍是效果、延迟、隐私、电量。

2 分钟版：

端侧适合：

- wake word。
- VAD。
- noise suppression。
- simple commands。
- offline ASR。

优化：

- int8 / int4 quantization。
- streaming model。
- small Conformer / RNN-T。
- on-device cache。
- cloud fallback。

取舍：

- 端侧隐私好、延迟低。
- 云端能力强、成本和网络依赖高。

## 557. 热词和专有名词怎么处理？

30 秒版：

热词问题常见在人名、地名、产品名和行业术语。处理方法包括 hotword boosting、custom vocabulary、contextual biasing、后处理纠错、领域微调和用户词典。要防止过度 boost 导致误插入。

2 分钟版：

方法：

- hotword list。
- contextual biasing。
- WFST / LM bias。
- pronunciation lexicon。
- domain fine-tuning。
- LLM post-correction。

注意：

- boost 太强会插入不存在的热词。
- 多音字和同音词难。
- 用户隐私词典要保护。
- 评估要看 entity recall/precision。

## 558. ASR hallucination 怎么发现和治理？

30 秒版：

ASR hallucination 指没有语音或不确定时生成不存在的文本。可以通过静音/噪声测试集、置信度、VAD、no-speech probability、重复片段检测和人工抽检发现；治理上要加强 VAD、阈值、分段、拒识和领域校准。

2 分钟版：

场景：

- 静音生成文字。
- 背景音乐被听成话。
- 长音频切片重复。
- 低置信内容强行输出。

治理：

- VAD gate。
- no-speech threshold。
- confidence threshold。
- segment dedup。
- noise augmentation。
- human review for low confidence。

面试表达：

> ASR 也会幻觉，尤其在静音、噪声和长音频分段场景，不能只看干净测试集 WER。

## 559. 语音工具调用怎么防误触发？

30 秒版：

语音输入容易误识别，所以高风险工具调用要二次确认。可以用 ASR confidence、semantic confirmation、tool risk level、用户身份校验、回读关键参数和撤销机制。不要让一次不确定识别直接执行支付、删除、发送等操作。

2 分钟版：

风险：

- ASR 听错金额。
- 背景人声触发命令。
- 用户没说完就执行。
- TTS 回声被识别成指令。

治理：

- high-risk tool confirmation。
- read back parameters。
- confidence threshold。
- speaker verification。
- permission and auth。
- cancel/undo。
- audit log。

面试表达：

> 语音 Agent 的工具调用比文本更需要确认机制，因为输入本身不稳定。

## 560. 语音 Agent 项目怎么讲 8 分钟？

30 秒版：

按场景、音频链路、ASR、LLM/工具、TTS、延迟预算、评估、安全和优化讲。重点证明你做的是实时交互闭环，不只是把 ASR、LLM、TTS 三个 API 串起来。

8 分钟结构：

1. 场景：

> 客服、会议、口语陪练、车载、语音助手。

2. 输入链路：

> 麦克风、VAD、降噪、AEC、streaming ASR、热词。

3. 理解链路：

> LLM、状态机、上下文、工具权限。

4. 输出链路：

> streaming TTS、音色、可打断播放。

5. 延迟：

> ASR partial、LLM first token、TTS first audio、endpoint delay。

6. 评估：

> WER/CER、任务成功率、打断成功率、用户满意度。

7. 安全：

> 录音授权、PII、声纹、工具二次确认、日志保留。

8. 优化：

> hotword、端侧 VAD、缓存、端云协同、fallback。

收尾：

> 语音 Agent 的核心是实时 turn-taking、低延迟、安全和任务成功率，而不是模型堆叠。

## 本组题的复习顺序

1. 先背 541-546：音频特征、ASR 指标、架构和流式难点。
2. 再背 547-551：TTS、vocoder、codec、Audio LLM 和架构选择。
3. 再背 552-559：实时延迟、barge-in、评估、安全、端侧和工具调用。
4. 最后背 560：项目 8 分钟讲法。

## 延伸阅读

- Whisper：[https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- VALL-E：[https://arxiv.org/abs/2301.02111](https://arxiv.org/abs/2301.02111)
- AudioLM：[https://arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143)
- AudioPaLM：[https://arxiv.org/abs/2306.12925](https://arxiv.org/abs/2306.12925)
- SeamlessM4T：[https://arxiv.org/abs/2308.11596](https://arxiv.org/abs/2308.11596)
- GPT-4o：[https://openai.com/index/hello-gpt-4o/](https://openai.com/index/hello-gpt-4o/)
- Realtime API：[https://platform.openai.com/docs/guides/realtime](https://platform.openai.com/docs/guides/realtime)
