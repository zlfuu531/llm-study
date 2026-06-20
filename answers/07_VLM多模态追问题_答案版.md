# 答案版 07：VLM 多模态追问题

这一份对应 `03_高频题单100题.md` 的第 161-180 题。重点是把多模态/VLM 从“知道几个模型名”变成能解释结构、公式、指标和项目排查。

## 161. VLM 的基本架构是什么？

**面试答案：**  
典型 VLM 是 `image/video -> vision encoder -> projector/adapter/Q-Former/resampler -> visual tokens -> LLM -> answer`。vision encoder 把图像变成 patch/token 表示，连接模块把视觉表示映射到 LLM hidden space，LLM 再结合文本指令完成问答、OCR、定位、推理或工具调用。

## 162. CLIP 的对比学习目标怎么理解？

**面试答案：**  
CLIP 用图像 encoder 和文本 encoder 分别得到图像向量和文本向量，然后在一个 batch 内让匹配图文相似度最高，不匹配图文相似度更低。相似度常用 cosine similarity 除以温度系数，损失可以理解为 image-to-text 和 text-to-image 两个方向的交叉熵。

```text
similarity(i, j) = cosine(v_i, t_j) / tau
L = (CE(row) + CE(col)) / 2
```

## 163. ViT 怎么把图像变成 token？

**面试答案：**  
ViT 会把图像切成固定大小 patch，每个 patch 线性投影成一个 token，再加位置编码输入 Transformer。patch 数约等于 `(H/P) * (W/P)`。分辨率越高，视觉 token 越多，LLM 上下文和推理成本越高。

## 164. projector、Q-Former、resampler 有什么区别？

**面试答案：**  
projector 主要做空间映射，把视觉特征映射到 LLM hidden size；Q-Former 用 learnable query 从视觉特征中抽取和语言任务相关的信息；resampler 把大量视觉 token 压缩成固定数量 token。projector 简单便宜，Q-Former/resampler 更擅长信息选择和 token 压缩，但可能丢细节。

## 165. LLaVA 类模型通常怎么训练？

**面试答案：**  
通常分两步：先训练 projector，让视觉 encoder 的特征能接到 LLM；再做多模态 instruction tuning，让模型学会按用户指令做图文问答、多轮对话、OCR、推理和拒答。很多实现会冻结视觉 encoder 和 LLM，只训练连接层或少量参数。

## 166. 2025-2026 VLM 新模型更强调哪些能力？

**面试答案：**  
更强调高分辨率/动态分辨率、OCR 和文档理解、grounding、视频理解、多图理解、多模态 Agent 和更强评估。面试里不要只说图像描述，要能讲小字、表格、坐标、时间轴、视觉 token 成本和幻觉治理。

## 167. VLM 做 OCR 为什么难？

**面试答案：**  
难点包括小字、低分辨率、中文英文数字符号混排、表格结构、版式、手写体、长文档和相似字符。视觉 token 压缩过强也会丢小字细节。工程上常结合高分辨率输入、局部 crop、专门 OCR 工具、bbox、字段规则校验和人工抽检。

## 168. Grounding 是什么，怎么评估？

**面试答案：**  
Grounding 是把文本描述或答案和图中区域对应起来，常见输出是 bbox 或点坐标。评估可以看 IoU、point accuracy、bbox 格式错误率和 phrase grounding accuracy。UI Agent 场景下还要看坐标映射是否能正确点击目标。

## 169. 视频 VLM 和单图 VLM 有什么不同？

**面试答案：**  
视频多了时间维度。关键问题是帧采样、时间顺序、长视频压缩、事件定位和音频/字幕融合。不能把所有帧硬塞给模型，通常要做关键帧采样、分段摘要、时间位置编码和事件级评估。

## 170. VLM 幻觉有哪些类型？

**面试答案：**  
常见包括图中没有的物体被说出来、文字读错、数量数错、坐标定位错、图表数值编造、把常识当视觉证据、视频事件顺序错。治理可以用引用区域、bbox、OCR/检测工具交叉验证、低置信度拒答、规则校验和 hard negative 评估集。

## 171. 多模态 RAG 怎么设计？

**面试答案：**  
离线侧对图片、PDF、视频做 OCR、caption、layout parse、object detection，得到文本 chunk、视觉 chunk 和 metadata，再分别建 text/image embedding 索引。在线侧根据 query 判断文本检索还是视觉检索，做 hybrid retrieval、rerank、context packaging，最后用 VLM/LLM 生成答案和引用。

## 172. 多模态 Agent 看截图点击按钮时，风险在哪里？

**面试答案：**  
风险是坐标错、OCR 错、UI 元素识别错、工具越权、截图泄露敏感信息和误操作。工程上要做坐标映射校验、UI 元素检测、权限控制、高风险动作人工确认、脱敏审计和完整 trace。

## 173. 为什么 image token 会影响延迟和成本？

**面试答案：**  
图像被切成 patch 后会变成视觉 token，这些 token 会占用 LLM 上下文。分辨率越高、多图越多、视频帧越多，token 数越大，prefill 成本、显存和延迟都会上升。所以需要动态分辨率、裁剪、token compression 或只保留关键区域。

## 174. 高分辨率和动态分辨率解决什么问题？

**面试答案：**  
高分辨率帮助识别小字、表格、票据和 UI 细节；动态分辨率让模型根据图像实际尺寸和内容分配视觉 token，避免所有图片都缩放到固定尺寸导致细节丢失或成本浪费。风险是 token 增多后推理更慢、更贵。

## 175. 多图理解有什么难点？

**面试答案：**  
多图需要跨图比较、引用和状态管理。模型容易混淆图片顺序、把 A 图信息说成 B 图、遗漏细节。解决上要给图片编号，保留图级 metadata，必要时先做单图摘要再跨图推理，评估时看跨图引用正确率。

## 176. VLM 评估指标怎么选？

**面试答案：**  
按任务选。图文问答看 accuracy 和人工评分；OCR 看 exact match、字段准确率、编辑距离；grounding 看 IoU 和 point accuracy；视频看事件召回和时间戳误差；Agent 看 task success、action accuracy、coordinate accuracy 和安全违规率；幻觉看 unsupported claim rate。

## 177. 多模态数据清洗有哪些特殊问题？

**面试答案：**  
图文对不齐、caption 太泛、OCR 错、图片低清或带水印、隐私信息、重复图、标注框错位、多图顺序错、视频字幕不同步都会影响训练和评估。多模态数据清洗要同时检查图像质量、文本质量和图文对应关系。

## 178. VLM 微调时要注意什么？

**面试答案：**  
要注意分辨率、视觉 token 数、显存、是否冻结 vision encoder/LLM、projector 学习率、图文样本格式、loss mask、任务配比和过拟合。OCR/grounding 类任务要保留坐标、字段和结构化答案，不能只用普通 caption 数据。

## 179. VLM 在 OCR 上错了，怎么定位？

**面试答案：**  
先看输入图是否清晰、分辨率是否足够、目标区域是否被缩放或裁剪；再看 vision encoder/token compression 是否丢细节；然后看 prompt 是否明确要求逐字读取、输出格式是否稳定；最后对比专门 OCR 工具和人工标注，判断是视觉识别问题、语言生成问题还是后处理问题。

## 180. VLM 项目 8 分钟深挖版应该讲什么？

**面试答案：**  
讲场景和目标、数据来源、模型结构、视觉输入处理、训练/微调、评估指标、bad case、上线风险和复盘。比如 OCR 项目要说字段准确率、小字低清 bad case、规则校验、人工抽检；多模态 Agent 要说坐标校验、权限、trace、人工确认和回滚。
