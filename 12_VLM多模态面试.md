# VLM 多模态面试

这一章面向多模态算法、OCR、视觉问答、视频理解、多模态 Agent 和大模型应用岗位。你不需要把所有 VLM 论文都背下来，但要能讲清：

- 图像怎么变成 LLM 能读的 token。
- CLIP、BLIP-2、LLaVA 这条架构线怎么演进。
- projector、adapter、Q-Former、resampler 分别解决什么。
- OCR、grounding、视频理解、多图理解为什么难。
- 多模态幻觉怎么评估和治理。
- 多模态项目怎么做指标、bad case 和上线闭环。

如果岗位偏 AIGC、文生图、图像编辑、视频生成或可控生成，配合 [30_Diffusion_DiT_文生图视频生成与可控生成面试.md](30_Diffusion_DiT_文生图视频生成与可控生成面试.md) 看，那里集中讲 DDPM、CFG、Latent Diffusion、DiT、ControlNet、LoRA/DreamBooth、video diffusion、Rectified Flow 和生成模型评估安全。

如果岗位偏语音助手、实时对话、多模态语音、ASR/TTS 或 voice agent，配合 [31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md](31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md) 看，那里集中讲 log-mel、CTC/RNN-T、Whisper、streaming ASR、TTS/vocoder、neural codec、VALL-E、speech-to-speech、barge-in、延迟预算和语音安全。

如果面试官继续追高分辨率 OCR、AnyRes/tiling、dynamic resolution、bbox/IoU、视频采样、GUI Agent、OSWorld/ScreenSpot 或多模态项目排查，跳到进阶专题：[40_VLM进阶_高分辨率OCR_Grounding_VideoAgent面试.md](40_VLM进阶_高分辨率OCR_Grounding_VideoAgent面试.md)。

## 一句话总览

VLM，即 Vision-Language Model，多模态视觉语言模型。它把图像、视频、截图、文档等视觉信息编码成向量或 token，再和文本 token 一起交给语言模型理解和生成。

常见结构：

```text
image / video
  -> vision encoder
  -> projector / adapter / Q-Former / resampler
  -> visual tokens
  -> LLM
  -> answer / bbox / action / structured output
```

面试口语版：

> VLM 的核心是跨模态对齐。视觉 encoder 负责把图像变成 patch/token 表示，连接模块负责把视觉表示映射到语言模型能理解的 hidden space，LLM 负责结合文本指令做推理、问答、OCR、定位或工具调用。

## VLM 架构演进

### CLIP：图文对比学习

CLIP 的核心思想是：让匹配的图像和文本 embedding 更近，不匹配的更远。

```text
image_i -> image_encoder -> v_i
text_i  -> text_encoder  -> t_i
similarity(i, j) = cosine(v_i, t_j) / tau
```

常见损失可以理解成双向 InfoNCE：

```text
L_image_to_text = CE(similarity_matrix rows, labels)
L_text_to_image = CE(similarity_matrix cols, labels)
L = (L_image_to_text + L_text_to_image) / 2
```

直觉：

- 一批里第 `i` 张图应该和第 `i` 句文本最相似。
- 图搜文、文搜图、zero-shot 分类都能用这个对齐空间。

CLIP 的局限：

- 更像“对齐和检索模型”，不擅长长文本生成。
- 对细粒度 OCR、复杂推理、计数、空间关系不够稳。

### BLIP-2：Q-Former 连接视觉和语言

BLIP-2 的思路是冻结已有视觉模型和语言模型，中间加一个轻量 Q-Former，让少量 learnable query 从视觉特征里抽取和语言任务相关的信息。

```text
image -> frozen image encoder -> visual features
learnable queries + visual features -> Q-Former -> query tokens
query tokens -> projection -> frozen LLM
```

Q-Former 解决的问题：

- 视觉 patch 很多，直接塞给 LLM 成本高。
- 冻结大模型时，需要一个桥接模块适配视觉和语言空间。
- learnable query 可以压缩视觉信息，保留和语言任务更相关的部分。

### LLaVA：视觉指令微调

LLaVA 类模型常见做法是：

```text
CLIP/SigLIP/ViT vision encoder
  -> linear/MLP projector
  -> LLM embedding space
  -> multimodal instruction tuning
```

关键是两阶段：

1. 对齐阶段：训练 projector，让视觉特征能接到 LLM。
2. 指令微调阶段：用图文问答、多轮对话、推理数据训练模型遵循多模态指令。

面试口语版：

> LLaVA 的重点不是发明一个全新视觉模型，而是把强视觉 encoder 接到强 LLM 上，再用多模态指令数据让模型学会“看图按指令回答”。

### 2025-2026 VLM 新重点

近两年面试更爱追这些：

- 高分辨率和动态分辨率：小字、表格、票据、截图需要更细视觉 token。
- OCR 和文档理解：不只是看图描述，还要读文字、表格和版式。
- grounding：能输出框、点、区域，知道答案来自图中哪里。
- 视频理解：帧采样、时间顺序、长视频压缩。
- 多图理解：比较、排序、跨图引用。
- 多模态 Agent：看截图、点按钮、调用工具、读网页或文档。
- 多模态 RAG：图片/OCR/表格/文本混合检索。

## 图像怎么变成 token

以 ViT 为例，图像会先切成 patch。

```text
image shape: H x W x C
patch size: P x P
num_patches = (H / P) * (W / P)
```

例如 `448 x 448` 图像，patch size `14`：

```text
num_patches = 32 * 32 = 1024
```

每个 patch 经过线性投影变成一个视觉 token，再加位置编码，输入 vision transformer。

为什么这会影响成本？

- 图像分辨率越高，patch 数越多。
- 视觉 token 越多，LLM 上下文越长。
- 多图、视频会把 token 数继续放大。

所以 VLM 里经常需要：

- 动态分辨率。
- 图像裁剪或 tiling。
- visual token compression。
- resampler / Q-Former。
- 只保留任务相关区域。

## projector、adapter、Q-Former、resampler

它们都在解决同一个大问题：视觉表示和语言模型 hidden space 不一致，而且视觉 token 太多。

| 模块 | 做什么 | 优点 | 风险 |
| --- | --- | --- | --- |
| Linear projector | 线性映射视觉特征到 LLM hidden size | 简单便宜 | 表达能力有限 |
| MLP projector | 用小 MLP 做非线性映射 | 比线性强 | 参数和训练成本略高 |
| Adapter | 在模型层间插小模块适配 | 可控、参数少 | 设计复杂度更高 |
| Q-Former | learnable query 从视觉特征抽信息 | 能压缩视觉 token | query 数和训练目标要调 |
| Resampler | 把大量视觉 token 压成固定数量 token | 控制上下文成本 | 压缩过强会丢细节 |

面试答案：

> projector 负责空间对齐，Q-Former/resampler 还负责信息选择和 token 压缩。OCR 和定位任务更怕压缩过度，因为小字、坐标和局部细节容易丢。

## 训练流程

常见 VLM 训练可以拆成几层：

### 1. 图文对比预训练

目标：建立图像和文本的粗粒度语义对齐。

典型任务：

- image-text contrastive。
- image-text matching。
- captioning。

### 2. 视觉-语言连接对齐

目标：让视觉 token 能被 LLM 接收。

通常训练：

- projector。
- Q-Former。
- resampler。

很多时候会冻结 vision encoder 和 LLM，只训练连接层，降低成本。

### 3. 多模态指令微调

目标：让模型学会按用户指令回答。

数据包括：

- 图文问答。
- OCR 问答。
- 图表理解。
- grounding。
- 多图对比。
- 视频问答。
- 多轮对话。
- 多模态工具调用轨迹。

### 4. 任务强化和偏好优化

可选，用来提升：

- 拒答。
- 幻觉控制。
- 格式遵循。
- 安全。
- 工具调用。

## OCR 和文档理解为什么难

OCR/VLM 面试特别容易追，因为它能看出你是否真的做过多模态项目。

难点：

- 小字和低分辨率：patch 太粗会看不清。
- 中文、英文、数字、符号混排。
- 表格结构：行列关系比纯文本复杂。
- 版式信息：标题、页眉页脚、脚注、印章、手写体。
- 长文档：多页内容超出上下文。
- 文字相似：`0/O`、`1/l`、金额和日期容易错。
- OCR 结果和视觉证据不一致。

治理：

- 提高输入分辨率或动态切图。
- 对小字区域做局部 crop。
- 用专门 OCR 工具辅助，不全靠 VLM。
- 保留 bbox、页码、行列结构。
- 对金额、日期、编号做规则校验。
- 建 OCR bad case 集，按字段准确率评估。

面试口语版：

> OCR 不是简单问“图里有什么字”。真正难的是小字、版式、表格、字段结构和可验证性。生产里我会把 VLM、OCR 工具、规则校验和人工抽检结合起来。

## Grounding 和坐标

Grounding 指模型能把文本答案和图中区域对应起来，比如输出 bounding box、点坐标或区域。

常见坐标格式：

```text
[x1, y1, x2, y2]
```

为了适配不同图片大小，通常会归一化：

```text
x_norm = x / image_width
y_norm = y / image_height
```

评估：

- IoU：预测框和真实框重叠程度。
- point accuracy：点是否落在目标区域。
- phrase grounding accuracy：文本短语和区域是否对应。

难点：

- 输出坐标格式必须稳定。
- 多目标容易漏框或错框。
- UI 截图里按钮很小。
- 裁剪、缩放后坐标要映射回原图。

项目里要注意：

> 如果 VLM 用于点击 UI 或操作网页，坐标错误会变成真实动作风险，所以必须加校验、确认和回滚。

## 视频理解

视频 VLM 可以理解成多帧图像加时间维度。

核心问题：

- 帧采样：采少了漏事件，采多了 token 爆炸。
- 时间顺序：模型要理解前后因果。
- 长视频：需要分段摘要、关键帧检索、事件定位。
- 音频/字幕：视频理解常常不只靠画面。

常见流程：

```text
video -> shot/scene split -> keyframe sampling
-> frame encoder -> temporal aggregation
-> LLM reasoning -> answer / timestamp / event
```

面试答案：

> 视频理解比单图多了时间轴。不能把所有帧硬塞给模型，要做关键帧采样、分段摘要、时间位置编码和事件级评估。

## 多模态 RAG

多模态 RAG 不是把图片丢进向量库这么简单。

离线侧：

```text
image/pdf/video
  -> OCR / caption / layout parse / object detection
  -> text chunks + visual chunks + metadata
  -> text embedding + image embedding
  -> index
```

在线侧：

```text
query
  -> 判断是文本问题还是视觉问题
  -> text/image hybrid retrieval
  -> rerank
  -> context packaging
  -> VLM/LLM answer with citation
```

关键取舍：

- 文档类问题优先 OCR + layout。
- 图片相似检索可用 image embedding。
- 复杂图表需要保留原图区域，而不是只保留 caption。
- 多模态上下文很贵，要控制 image token 和文本 token。

## 多模态 Agent

多模态 Agent 常见任务：

- 看网页截图，决定点击哪里。
- 读票据/合同，调用系统录入字段。
- 看报表图，生成分析。
- 看视频片段，标注事件。

架构：

```text
screenshot / image / video
  -> VLM observation
  -> state
  -> tool selection
  -> coordinate/action validation
  -> execute tool
  -> new observation
```

风险：

- 坐标错导致误点击。
- OCR 错导致录错金额。
- 幻觉导致调用不存在的功能。
- 工具越权。
- 截图里含敏感信息。

工程措施：

- 高风险动作人工确认。
- 坐标映射和 UI 元素检测双重校验。
- 结构化字段用规则校验。
- 截图和工具调用脱敏审计。
- 每步记录 observation、action、坐标、工具返回。

## VLM 评估

不要只说“看回答对不对”。VLM 评估要按任务拆。

| 任务 | 指标 |
| --- | --- |
| 图文问答 | accuracy、人工评分、LLM-as-judge 辅助 |
| OCR | exact match、字段准确率、ANLS、编辑距离 |
| Grounding | IoU、point accuracy、bbox format error |
| 图表理解 | 数值误差、结论正确率、引用区域 |
| 多图对比 | pairwise accuracy、跨图引用正确率 |
| 视频理解 | temporal localization、事件召回、时间戳误差 |
| 多模态 Agent | task success、action accuracy、coordinate accuracy、安全违规 |
| 幻觉 | object hallucination rate、unsupported claim rate |

评估集要包含：

- 正常样本。
- 小字/低清。
- 表格和票据。
- 多图。
- 遮挡和噪声。
- 需要拒答的样本。
- prompt injection 图片或文档。
- 权限敏感样本。

## 多模态幻觉

常见幻觉：

- 图中没有的物体被说出来。
- 文字读错。
- 数量数错。
- 坐标定位错。
- 把常识当视觉证据。
- 图表数值编造。
- 视频事件顺序错。

治理：

- 要求答案引用视觉证据或 bbox。
- OCR/检测/规则工具交叉验证。
- 低置信度拒答或让用户提供更清晰图片。
- 训练和评估里加入 hard negative。
- 对金额、日期、数量做后验校验。
- 多轮追问时保留原图证据，不只保留摘要。

## 高频问答

### Q1：VLM 和普通 LLM 最大区别是什么？

VLM 多了视觉输入和跨模态对齐。普通 LLM 只处理文本 token，VLM 需要 vision encoder 把图像变成视觉 token，再用 projector/Q-Former/resampler 接到 LLM。

### Q2：为什么 VLM 做 OCR 容易错？

因为小字、低分辨率、复杂版式、表格、混排字符和视觉 token 压缩都会损失细节。生产里通常结合高分辨率输入、局部 crop、专门 OCR 工具、bbox 和规则校验。

### Q3：CLIP 为什么能 zero-shot 分类？

因为 CLIP 学到了图像和文本共享 embedding 空间。分类时可以把类别写成 prompt，比如 “a photo of a cat”，比较图像 embedding 和各类别文本 embedding 的相似度。

### Q4：projector 训练好了，为什么还要 instruction tuning？

projector 只解决视觉特征接入 LLM 的问题，不等于模型会按用户指令做多轮问答、OCR、推理和拒答。Instruction tuning 让模型学会在多模态输入下遵循指令。

### Q5：多模态项目怎么讲成工程项目？

按场景、数据、模型、评估、bad case、上线风险讲。比如 OCR 项目要说字段准确率、低清/小字 bad case、规则校验、人工抽检和回滚；多模态 Agent 要说坐标校验、权限、trace 和人工确认。

## 面试前背诵版

VLM 的核心是跨模态对齐：图像经过 vision encoder 变成视觉 token，再通过 projector、Q-Former 或 resampler 映射到 LLM hidden space，最后和文本指令一起输入语言模型。CLIP 用图文对比学习建立共享语义空间，BLIP-2 用 Q-Former 连接冻结视觉模型和 LLM，LLaVA 类模型用 projector 加多模态指令微调。VLM 高频追问集中在 OCR、grounding、视频、多图、多模态 RAG、Agent、幻觉和评估。项目表达要讲清 image token 成本、分辨率、视觉细节、字段/坐标准确率、bad case、权限和安全。

## 本轮参考来源

- CLIP 论文：https://arxiv.org/abs/2103.00020
- BLIP-2 论文：https://arxiv.org/abs/2301.12597
- LLaVA 论文：https://arxiv.org/abs/2304.08485
- LLaVA-OneVision 论文：https://arxiv.org/abs/2408.03326
- Qwen2.5-VL 官方博客：https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen2.5-VL GitHub：https://github.com/QwenLM/Qwen2.5-VL
- InternVL 官方仓库：https://github.com/OpenGVLab/InternVL
- 本地资料：`外部资料_GitHub/LLM-Agent-Interview-Guide/07-HotTopics/01-Hot-Topics-2025-2026.md`
