# 答案版 35：VLM 进阶、高分辨率 OCR、Grounding、Video 与 GUI Agent

对应题单 721-740。建议先看 [40_VLM进阶_高分辨率OCR_Grounding_VideoAgent面试.md](../40_VLM进阶_高分辨率OCR_Grounding_VideoAgent面试.md)，再用本答案版做口头复述。

## 721. 高分辨率图片为什么会显著增加 VLM 成本？

图片通常会被切成 patch，再变成 visual tokens。粗略公式是：

```text
num_patches = ceil(H / P) * ceil(W / P)
```

如果分辨率从 `448 x 448` 提到 `896 x 896`，patch 数大约变成 4 倍。视觉 token 变多会增加 vision encoder 计算、LLM prefill 时间和 KV Cache 占用。

面试要补一句：高分辨率适合 OCR、截图、票据、小字和 GUI；普通图像描述不一定值得走最贵路径。

## 722. AnyRes / tiling 和直接 resize 有什么区别？

直接 resize 把所有图片缩到固定大小，工程简单、batch 稳定、成本可控，但容易丢小字、表格线、截图按钮和长图细节。

AnyRes / tiling 会保留长宽比例，把原图切成多个局部 tile，通常再加一张全局缩略图。全局图负责布局，局部 tile 负责细节。

风险是 token 数增加、tile 边界可能切断对象、跨 tile 对齐更难。生产里要设置 tile 上限、overlap、坐标映射和任务级 pixel budget。

## 723. Dynamic resolution / pixel budget 怎么讲？

动态分辨率的核心是按任务分配视觉 token，而不是所有图片都用同一个高分辨率。

```text
visual_budget = min(max_pixels, H * W)
num_visual_tokens ≈ visual_budget / patch_area
```

文字密集、票据、截图、图表给更高预算；普通物体识别和粗粒度 VQA 可以低分辨率。它解决的是“细节保留”和“成本延迟”之间的取舍。

项目里可以说：我会按业务类型、图片尺寸、OCR 字体大小、P95 延迟和失败率来调预算，而不是只看 benchmark 总分。

## 724. OCR 和文档理解为什么不能只看最终回答？

因为文档任务里“回答通顺”不代表字段正确。OCR 至少有三层：

- text recognition：读到了什么字。
- text localization：字在哪里。
- document reasoning：字段、表格、版式之间怎么推理。

评估要看字段准确率、exact match、表格结构、bbox、格式合法率、数值容差和人工复核率。金额、日期、编号、身份证这类字段还要规则校验。

一句话：文档 OCR 最怕“像真的但值错”，所以要字段级评估和证据校验。

## 725. VLM 做 OCR 为什么还常接传统 OCR 引擎？

VLM 擅长结合图像上下文推理和结构化解释，但传统 OCR 在批量文字定位、行级 bbox、置信度、速度和成本上仍然强。

生产常见组合是：

```text
OCR engine -> text/bbox/layout
VLM/LLM -> 字段理解、跨区域推理、纠错和结构化输出
规则/人工 -> 高风险字段复核
```

这样能降低 VLM 视觉 token 成本，也能让输出更可审计。

## 726. Grounding、bbox、point accuracy 和 IoU 怎么讲？

Grounding 是把文本答案和图中区域对齐。bbox 格式常见为：

```text
[x1, y1, x2, y2]
```

IoU 公式：

```text
IoU = area(pred ∩ gt) / area(pred ∪ gt)
```

目标检测和 bbox grounding 常用 IoU 判断是否定位正确。小按钮、小图标、GUI 点击任务里，point accuracy 或 click success 可能比 IoU 更贴近业务。

回答时要强调坐标系：resize、padding、crop、tile 都会改变坐标，必须能反变换回原图。

## 727. VLM 输出 JSON 为什么不等于可靠？

JSON 合法只说明语法能 parse，不说明语义正确。比如 bbox 越界、字段值错、金额单位错、坐标对应错对象，都可能是合法 JSON。

评估要拆开：

- parse success rate。
- schema valid rate。
- field exact match。
- numeric tolerance accuracy。
- bbox IoU / point accuracy。
- evidence consistency。

生产里要做 schema 校验、字段规则校验、坐标合法性检查、失败重试和高风险人工复核。

## 728. 视频理解为什么不是多图拼接？

视频除了空间信息，还有时间顺序和事件定位。难点包括帧数多、视觉 token 爆炸、短事件容易被采样漏掉、动作变化需要时序推理。

粗略预算：

```text
video_visual_tokens ≈ num_frames * tokens_per_frame
```

所以长视频通常不能全量塞进模型。更合理的是分层处理：先低成本采样或检索候选片段，再对候选片段高帧率/高分辨率细看，最后输出带时间戳的答案。

## 729. 视频帧采样策略怎么选？

常见策略：

- uniform sampling：适合全局概览，但可能漏短事件。
- scene / shot sampling：适合镜头切换明显的视频。
- query-aware sampling：根据问题找相关片段。
- hierarchical summarization：适合长视频，但摘要可能丢证据。
- event localization：适合需要时间戳的任务。

面试可答：先按业务问题决定是“全局理解”还是“找某个事件”。如果是长视频问答，我会先粗采样召回片段，再细看片段并输出时间戳。

## 730. 多图理解比单图难在哪里？

多图任务不只是把图片 concat。难点在于图与图之间的引用关系、顺序、每张图的角色、跨图对象对齐和 token 预算。

工程上要给每张图明确 ID、页码、顺序和元数据。文档多页要保留页码和版式；商品对比要先逐图抽结构化事实，再做差异比较；GUI 历史截图要保留状态变化。

一句话：多图理解要显式管理“哪张图提供了什么证据”。

## 731. GUI Agent 的基本链路是什么？

典型链路：

```text
screenshot / DOM / accessibility tree
  -> screen parsing / OCR / element detection
  -> task state
  -> action planning
  -> click / type / scroll / hotkey
  -> execute in sandbox
  -> observe next screenshot
```

和普通 Agent 相比，GUI Agent 的 action 会真实操作界面，所以 grounding、权限、沙箱、回滚和用户确认更重要。

## 732. GUI grounding 为什么难？

GUI 元素通常小、密、相似，截图里还不一定知道哪些区域可点击。模型如果直接输出坐标，resize/padding/crop 任何一步错都会导致点击偏移。

改进方式：

- OCR + icon detection + interactable region detection。
- DOM / accessibility tree / screenshot 混合。
- 给候选元素编号，让模型选 ID。
- 每步执行后 observe，再做错误恢复。
- 高风险动作前做 policy check 和用户确认。

评估不只看 bbox，还要看 step success 和 task success。

## 733. OSWorld、ScreenSpot、UI-TARS、OmniParser 这些怎么串起来讲？

可以按“任务、评测、模型、解析器”讲：

- OSWorld：真实桌面环境里的开放任务评测，关注最终任务成功。
- ScreenSpot / ScreenSpot-Pro：更偏 GUI grounding，考模型点得准不准。
- UI-TARS：面向 GUI 操作的原生视觉 Agent，学习截图到动作。
- OmniParser：把 UI 截图解析成结构化元素，帮助纯视觉 Agent 定位可交互区域。

面试重点不是背名字，而是说明 GUI Agent 需要感知、grounding、动作建模、执行反馈和安全控制。

## 734. 多模态 RAG 和普通文本 RAG 有什么区别？

多模态 RAG 要先把 PDF、图片、表格、截图、视频拆成可检索证据。

```text
PDF / image / video
  -> OCR / layout / caption / embedding / metadata
  -> retrieval / rerank
  -> evidence packing
  -> VLM or LLM answer
```

难点是检索粒度：太粗会噪声多、上下文贵；太细会丢跨区域关系。还要处理 OCR 错误、图片 embedding 不擅长数字、引用页码/bbox/时间戳等问题。

## 735. VLM 评测应该怎么分桶？

不要只看整体准确率。应该按任务和失败类型分桶：

- 图像类型：自然图、截图、票据、图表、文档。
- 文字：大小、语言、旋转、手写、低清。
- grounding：目标大小、遮挡、位置、相似干扰物。
- 视频：长度、事件持续时间、镜头切换。
- 输出：自由文本、JSON、bbox、表格。

指标按任务选：OCR 字段准确率、bbox IoU、point accuracy、视频时间戳误差、GUI task success、多模态 RAG citation accuracy。

## 736. VLM 幻觉有哪些类型？

常见类型：

- 图里没有的物体被说出来。
- OCR 数字、日期、单位读错。
- bbox 指错对象。
- 视频事件顺序说反。
- GUI Agent 点错按钮。
- 文档字段抽取错但格式正确。

治理方式是要求视觉证据：bbox、页码、时间戳、截图回显、字段校验和人工复核。高风险场景不能只相信模型自然语言回答。

## 737. 多模态线上延迟突然变高怎么排查？

先看输入分布：

- 图片数量是否增加。
- 分辨率是否变大。
- 视频帧数是否变多。
- OCR/检测/裁剪是否变慢。
- JSON retry 是否变多。

再看模型侧：

- visual tokens。
- prefill latency / TTFT。
- GPU 利用率和显存。
- batch 是否被大图拖慢。
- cache 是否命中。

优化可以从动态分辨率、局部 crop、离线 OCR、图片缓存、小模型前置过滤和失败重试策略入手。

## 738. 多模态项目 bad case 怎么讲？

按“现象 -> 分桶 -> 根因 -> 修复 -> 指标变化”讲。

例子：

```text
现象：发票金额字段经常错。
分桶：小字、反光、低清、表格边界。
根因：全图 resize 后金额区域像素太少，VLM 语言先验补全。
修复：金额区域 OCR crop + 规则校验 + 低置信人工复核。
指标：金额字段准确率提升，人工复核率下降。
```

这样比只说“换了更强模型”更像真实项目。

## 739. VLM 进阶面试项目 8 分钟怎么讲？

结构：

1. 业务问题：OCR、文档抽取、截图 Agent、视频检索或多模态 RAG。
2. 输入特点：分辨率、文字密度、多页、多图、视频长度。
3. 系统方案：OCR / detector / VLM / RAG / 规则 / 人工闭环。
4. 难点：小字、坐标、结构化 JSON、视频片段、GUI 小按钮。
5. 指标：字段准确率、IoU、task success、引用正确、P95 延迟。
6. 优化：dynamic resolution、crop、缓存、小模型前置。
7. bad case：讲 2-3 个具体失败和修复。

收束句：我关注的不只是模型能不能看懂，而是视觉证据、结构化校验、成本延迟和上线闭环。

## 740. VLM 进阶面试前最后怎么复习？

最后一天按四条线复习：

1. 视觉 token 成本：patch、resolution、tiling、dynamic resolution。
2. 证据定位：OCR、bbox、point、IoU、坐标系、JSON 校验。
3. 时序和操作：video sampling、GUI grounding、task success、安全。
4. 项目表达：输入特点、系统方案、评估指标、bad case、成本优化。

如果只剩 30 分钟，优先背这句：

> VLM 进阶不是“模型看图”，而是视觉 token 预算、证据定位、结构化校验、时序/动作闭环和线上成本安全的系统问题。
