# VLM 进阶：高分辨率 OCR、Grounding、Video 与 GUI Agent 面试

这一章是 [12_VLM多模态面试.md](12_VLM多模态面试.md) 的进阶补充，面向 2025-2026 LLM 学习和面试里更容易被追问的多模态细节：

- 高分辨率图片为什么贵，AnyRes / tiling / dynamic resolution 怎么讲。
- OCR、文档理解、表格抽取为什么不能只靠“看图问答”。
- grounding / bbox / point / segmentation 和结构化输出怎么评估。
- 视频理解为什么不是“多张图片拼起来”那么简单。
- GUI Agent / computer use 为什么对截图解析、坐标和动作空间要求更高。
- 多模态项目怎么做 eval、bad case、上线安全和成本控制。

推荐读法：

1. 先读 `12_VLM多模态面试.md`，确认 CLIP、BLIP-2、LLaVA、projector、Q-Former、OCR、grounding、视频理解这些基础词能说清。
2. 再读本章，把“图像 token 成本、动态分辨率、坐标评估、视频采样、GUI grounding、项目排查”练成能复述的系统答案。
3. 最后刷 [answers/35_VLM进阶_高分辨率OCR_Grounding_VideoAgent_答案版.md](answers/35_VLM进阶_高分辨率OCR_Grounding_VideoAgent_答案版.md) 的 721-740 题。

## 一句话总览

VLM 进阶面试的核心不是“模型会看图”，而是：

```text
视觉输入 -> 图像/视频 token 预算 -> 视觉细节保留 -> 坐标/文字/时间证据
       -> LLM 推理和结构化输出 -> 评估、成本、安全和上线闭环
```

你可以这样开场：

> 多模态系统的难点在于视觉信息既贵又容易丢细节。高分辨率、OCR、grounding、视频和 GUI Agent 都在追问同一件事：视觉 token 怎么生成、哪些细节被保留、答案能不能回到图中证据、线上如何评估和控成本。

## 1. 高分辨率图片为什么贵

把图片送进 VLM 之前，通常会被 vision encoder 切成 patch。

```text
num_patches = ceil(H / P) * ceil(W / P)
```

其中：

- `H, W` 是输入图像高和宽。
- `P` 是 patch size，例如 14 或 16。
- 每个 patch 会变成一个 visual token 或中间视觉特征。

例子：

```text
448 x 448, P = 14
num_patches = 32 * 32 = 1024

896 x 896, P = 14
num_patches = 64 * 64 = 4096
```

分辨率翻倍，token 数大约变成 4 倍。它会带来三类成本：

- vision encoder FLOPs 增加。
- 传给 LLM 的视觉 token 变多，上下文更长。
- prefill 更慢，KV Cache 更大，多图/视频会继续放大。

面试里要强调：高分辨率不是免费提高效果。OCR、截图、票据、小字和 GUI 需要高分辨率；普通物体识别、粗粒度 VQA 不一定需要。

## 2. AnyRes、Tiling、Dynamic Resolution

### 固定分辨率

最简单做法是把所有图片 resize 到固定大小，例如 `336 x 336` 或 `448 x 448`。

优点：

- batch 形状稳定。
- 工程简单。
- 成本可控。

缺点：

- 宽图、长图、截图、票据容易被压缩变形。
- 小字、表格线和局部细节会糊掉。

### AnyRes / Tiling

AnyRes 或 tiling 的思路是：保留原图长宽比例，把大图切成多个 tile，再加一个全局缩略图。

```text
original image
  -> global low-res view
  -> local tiles / crops
  -> visual tokens
  -> LLM
```

直觉：

- 全局图负责整体布局和上下文。
- 局部 tile 负责小字、图表、按钮、细节。
- 模型需要学会跨 tile 对齐同一个对象或文本。

典型追问：

1. tile 多了为什么成本爆炸？
2. tile 边界上的文字或对象被切断怎么办？
3. 模型怎么知道不同 tile 在原图里的位置？

回答：

- 通过 tile 数上限、overlap、坐标位置编码、全局图和局部图组合来控制。
- 对 OCR/GUI，可以先检测候选区域再局部 crop，而不是盲目全图高分辨率。

### Dynamic Resolution / Pixel Budget

动态分辨率会根据图片本身大小和任务难度决定输入 token 数。

```text
visual_budget = min(max_pixels, H * W)
num_visual_tokens ≈ visual_budget / patch_area
```

面试表达：

> 动态分辨率本质是在“视觉细节”和“token 成本”之间做预算。对文字密集、截图、票据和图表给更多视觉 token；对普通图片压低分辨率，避免所有请求都按最贵路径走。

## 3. OCR / 文档理解为什么是硬题

OCR 类多模态面试通常不是问“识别文字”这么简单，而是问：

- 文字能不能读对。
- bbox / 行列 / 表格结构能不能还原。
- 字段抽取能不能稳定输出 JSON。
- 金额、日期、编号、单位能不能校验。
- 多页文档、扫描件、手写体、印章、旋转文字能不能处理。

### OCR 三层能力

| 层次 | 问题 | 典型任务 |
| --- | --- | --- |
| Text recognition | 读到了什么字 | 票据文字、截图按钮、公式符号 |
| Text localization | 字在哪里 | bbox、line-level OCR、字段定位 |
| Document reasoning | 这些字说明什么 | 表格抽取、跨字段校验、问答推理 |

### 为什么 VLM 仍然会错

- patch 过粗，小字被平均掉。
- 图片压缩、模糊、倾斜、反光。
- 表格结构需要二维关系，不是线性文本。
- 视觉文字和语言先验冲突时，模型容易“补全”。
- 输出格式可能对但字段值错。

工程治理：

- 高分辨率或局部 crop。
- 专门 OCR engine + VLM reasoning 混合。
- 保留 bbox、页码、行列、置信度。
- 对金额、日期、身份证、手机号、发票号做规则校验。
- 建字段级 eval：字段准确率、完全匹配率、格式合法率、人工复核率。

面试口语版：

> 文档 OCR 最怕“看起来合理但值错”。我会把识别、定位、结构化抽取、规则校验和人工抽检分开评估，而不是只看模型最终回答是否通顺。

## 4. Grounding：答案要回到图中证据

Grounding 指模型不仅回答“是什么”，还要指出“在图哪里”。

常见输出：

```json
{"bbox_2d": [x1, y1, x2, y2], "label": "target"}
{"point_2d": [x, y], "label": "button"}
```

坐标可以是原图像素，也可以归一化：

```text
x_norm = x / image_width
y_norm = y / image_height
```

### IoU

目标检测和 bbox grounding 常用 IoU：

```text
IoU = area(pred ∩ gt) / area(pred ∪ gt)
```

常见阈值：

- `IoU >= 0.5`：粗定位正确。
- `IoU >= 0.75`：更严格。
- 小按钮、小文字、小图标任务里，point accuracy 有时比 IoU 更合适。

### Grounding 为什么难

- 坐标系容易错：resize、padding、crop、tile 坐标映射都可能出 bug。
- 语言短语和视觉对象一对多、多对一。
- 小目标和遮挡目标难。
- 模型输出 JSON 可能合法但坐标无效。
- GUI 场景里，一个像素点错一点就可能点错按钮。

工程排查顺序：

1. 确认输入图尺寸、resize、padding 和坐标反变换。
2. 检查 bbox 是否越界、x1/x2 是否反了。
3. 把预测框画回原图人工看。
4. 按目标大小、位置、类别、文字密度分桶。
5. 对 GUI/机器人动作，评估 click success，而不只评估 bbox。

## 5. 结构化输出：JSON 不等于可靠

VLM 常被要求输出：

- bbox JSON。
- 表格 JSON。
- 发票字段 JSON。
- UI action JSON。
- 图表数据 JSON。

但结构化输出有两层问题：

1. 语法合法：JSON 能 parse。
2. 语义正确：字段值、坐标、单位、证据都对。

所以 eval 要分开：

```text
parse_success_rate
schema_valid_rate
field_exact_match
numeric_tolerance_accuracy
bbox_iou / point_accuracy
evidence_consistency
```

面试表达：

> 结构化输出不是 prompt 写“请输出 JSON”就完了。生产里要做 schema 校验、字段级指标、数值容差、坐标合法性检查和失败重试；对高风险字段还要走规则或人工复核。

## 6. 视频理解为什么更难

视频可以看成一串帧，但不能只说“多图输入”。

难点：

- 帧数很多，视觉 token 爆炸。
- 事件发生在某个时间段，不一定在均匀采样帧里。
- 需要理解时间顺序、动作变化、因果关系。
- 音频、字幕、OCR、镜头切换会提供额外信息。
- 长视频问答需要定位相关片段，而不是全视频塞进上下文。

粗略 token 预算：

```text
video_visual_tokens ≈ num_frames * tokens_per_frame
tokens_per_frame ≈ tiles_per_frame * tokens_per_tile
```

比如：

```text
32 frames * 576 tokens/frame = 18,432 visual tokens
```

这还没算文本 prompt 和生成输出。

### 采样策略

| 策略 | 适合 | 风险 |
| --- | --- | --- |
| Uniform sampling | 全局概览 | 错过短事件 |
| Scene / shot sampling | 镜头切换明显的视频 | 依赖切分质量 |
| Query-aware sampling | 用户问题明确 | 需要先理解问题 |
| Hierarchical summarization | 长视频 | 摘要可能丢证据 |
| Event localization | 找时间段 | 标注和评估更难 |

面试口语版：

> 视频理解要先控制帧和 token 预算，再谈模型能力。长视频任务最好做分层：先粗采样找候选片段，再对片段高分辨率或高帧率细看，最后输出带时间戳的答案。

## 7. 多图理解

多图任务包括：

- 对比两张商品图差异。
- 多页文档问答。
- 多张病理/遥感/工业图联合判断。
- GUI Agent 的历史截图状态。
- 代码/设计图/表格之间交叉引用。

关键难点：

- 图与图之间的引用关系。
- 顺序是否重要。
- 每张图的 token 预算。
- 相似对象跨图对齐。
- 中间图容易被忽略。

回答模板：

> 多图不是简单 concat。我要先定义每张图的角色、顺序、ID 和预算；如果是文档，多页要保留页码和版式；如果是对比任务，要显式让模型逐图提取结构化事实，再做差异比较。

## 8. GUI Agent / Computer Use

GUI Agent 是 VLM 进阶高频，因为它把 OCR、grounding、tool calling、安全和多步规划绑在一起。

典型链路：

```text
screenshot
  -> screen parsing / OCR / element detection
  -> task state + instruction
  -> action planning
  -> action JSON: click / type / hotkey / scroll
  -> execute in sandbox
  -> observe new screenshot
  -> loop
```

动作空间：

```json
{"action": "click", "x": 512, "y": 384}
{"action": "type", "text": "query"}
{"action": "press", "key": "Enter"}
{"action": "scroll", "direction": "down"}
```

### GUI Agent 的核心问题

- 屏幕元素小、密、相似。
- 纯截图里没有 DOM，模型不知道哪些元素可点击。
- click 坐标必须准确。
- 多步任务会累积错误。
- 登录、支付、删除、外发消息等动作有安全风险。

常见工程增强：

- OCR + icon detection + interactable region detection。
- DOM / accessibility tree / screenshot 混合观察。
- 将候选元素编号，让模型选 ID 而不是直接输出坐标。
- 每步执行前做 policy check。
- 高风险动作要求用户确认。
- 用 OSWorld / AndroidWorld / WebArena 类 benchmark 或自建任务回放评估。

面试表达：

> GUI Agent 的难点不是让模型描述截图，而是把截图里的可交互元素稳定变成动作。真正上线要关注 grounding accuracy、task success、步数、错误恢复、权限和沙箱。

## 9. 多模态 RAG 和文档 RAG

多模态 RAG 不只是把图片丢给 VLM，它通常要把不同模态拆成可检索证据：

```text
PDF / 图片 / 表格 / 视频
  -> OCR / layout / caption / embedding / metadata
  -> hybrid retrieval
  -> rerank
  -> evidence packing
  -> VLM / LLM answer
```

索引对象可以是：

- 页面。
- 图片。
- 表格。
- 图表。
- OCR 行。
- bbox 区域。
- 视频片段。
- UI 截图状态。

关键问题：

- 检索粒度太粗：上下文贵，噪声多。
- 检索粒度太细：跨区域关系断掉。
- OCR 错误会污染索引。
- 图片 embedding 擅长语义，不一定擅长文字和数字。
- 生成答案必须引用页码、区域或时间戳。

项目讲法：

> 我会把多模态 RAG 拆成“模态解析、结构化索引、检索召回、证据组装、答案生成、引用校验”。评价不只看最终 QA，还看 OCR 字段、表格抽取、检索命中、引用正确和人工复核成本。

## 10. 多模态评测怎么设计

VLM eval 要按任务拆，不要只报一个总分。

| 任务 | 指标 |
| --- | --- |
| 普通 VQA | accuracy、LLM-as-judge、人工抽检 |
| OCR | 字符/字段准确率、exact match、格式合法率 |
| 文档抽取 | 字段 EM、表格结构 F1、数值容差 |
| Grounding | IoU、point accuracy、click success |
| 视频理解 | QA accuracy、temporal localization、timestamp error |
| GUI Agent | task success、step success、avg steps、unsafe action rate |
| 多模态 RAG | retrieval recall、evidence precision、citation accuracy |

分桶维度：

- 分辨率、文字大小、语言、旋转角度。
- 图像类型：截图、票据、图表、自然图、工业图。
- 目标大小、位置、遮挡。
- 视频长度、事件持续时间、镜头切换。
- 输出格式：自由文本、JSON、bbox、表格。

面试里可以说：

> 多模态评测一定要分桶。整体准确率可能掩盖“小字、表格、长视频、GUI 小按钮”这些真实业务最痛的失败点。

## 11. 线上成本和延迟

VLM 服务成本通常比纯文本高，因为 prefill 阶段要吃大量视觉 token。

排查链路：

1. 输入图片数量和分辨率是否变大。
2. 动态分辨率是否被误配置成全量高分辨率。
3. 多图/视频帧数是否过多。
4. OCR/检测/裁剪前处理是否串行阻塞。
5. VLM 输出 JSON 是否反复 retry。
6. 是否可以用专用 OCR / detector / 小模型前置过滤。

常见优化：

- 按任务选择分辨率。
- 先低分辨率判断，再局部高分辨率 crop。
- 对重复图片或文档页做缓存。
- OCR/embedding/thumbnail 离线预处理。
- 对视频做片段检索，不全量送 VLM。
- 对 GUI 用候选元素减少视觉搜索空间。

## 12. 多模态幻觉和安全

多模态幻觉常见形式：

- 图里没有的物体被说出来。
- OCR 数字读错但回答很自信。
- bbox 指错对象。
- 视频事件顺序说反。
- GUI Agent 点错按钮或执行危险操作。

治理：

- 要求引用视觉证据：页码、bbox、时间戳。
- 对关键字段做二次校验。
- 高风险动作 HITL。
- 对输出坐标画框回显。
- 对涉隐私图片做脱敏和权限控制。
- 建 bad case 回归集。

一句话：

> 多模态安全比文本更容易被忽视，因为错误常藏在像素、坐标和动作里。面试要把“看错、指错、点错、抽错字段”分开讲。

## 13. 高频追问快答

**Q：AnyRes 和直接 resize 有什么区别？**  
直接 resize 成本稳定但会丢细节；AnyRes/tiling 保留局部细节，但 token 更多、跨 tile 对齐更难。

**Q：为什么 VLM 做 OCR 还要接传统 OCR？**  
VLM 擅长结合上下文推理，但传统 OCR 在文字定位、批量抽取、置信度和成本上仍有优势。生产里常混合使用。

**Q：Grounding 输出 bbox 就可靠吗？**  
不一定。要检查坐标系、IoU、越界、是否对应正确对象，以及 bbox 画回原图是否能支持答案。

**Q：视频理解怎么控成本？**  
先做低成本采样或检索找候选片段，再对候选片段高帧率/高分辨率细看，输出时间戳证据。

**Q：GUI Agent 和普通 Agent 区别？**  
GUI Agent 的 observation 是截图/DOM/accessibility tree，action 是点击、输入、滚动等真实操作，所以 grounding、权限和执行安全更关键。

## 14. 项目 8 分钟讲法

如果你有 OCR、文档理解、截图 Agent 或多模态 RAG 项目，可以按这个结构讲：

```text
1. 业务问题：
   例如票据字段抽取、截图自动操作、视频事件检索、图文知识库问答。

2. 输入特点：
   图片分辨率、文字密度、是否多页、多图、视频长度、是否需要坐标或结构化输出。

3. 系统方案：
   OCR/视觉 encoder/VLM/检索/规则校验/人工复核怎么组合。

4. 关键难点：
   小字、表格、bbox、视频片段、GUI 小按钮、结构化 JSON、隐私安全。

5. 评估指标：
   字段准确率、IoU、point accuracy、task success、引用正确、P95 延迟、人工复核率。

6. 成本优化：
   动态分辨率、crop、缓存、离线解析、小模型前置、失败重试策略。

7. bad case：
   选 2-3 个真实失败样例，说清根因和修复。
```

面试收束句：

> 这个项目我不会只说“用了某个 VLM”，而是会讲清视觉输入怎么预算、证据怎么定位、结构化结果怎么校验，以及上线后怎么用 bad case 和指标闭环。

## 15. 参考资料

- Qwen2.5-VL Blog: [https://qwenlm.github.io/blog/qwen2.5-vl/](https://qwenlm.github.io/blog/qwen2.5-vl/)
- LLaVA-NeXT Blog: [https://llava-vl.github.io/blog/2024-01-30-llava-next/](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
- LLaVA-OneVision: [https://arxiv.org/abs/2408.03326](https://arxiv.org/abs/2408.03326)
- InternVL3: [https://arxiv.org/abs/2504.10479](https://arxiv.org/abs/2504.10479)
- MMMU: [https://arxiv.org/abs/2311.16502](https://arxiv.org/abs/2311.16502)
- MMBench: [https://arxiv.org/abs/2307.06281](https://arxiv.org/abs/2307.06281)
- OCRBench v2: [https://arxiv.org/abs/2501.00321](https://arxiv.org/abs/2501.00321)
- Video-MME: [https://arxiv.org/abs/2405.21075](https://arxiv.org/abs/2405.21075)
- OSWorld: [https://arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972)
- ScreenSpot-Pro: [https://arxiv.org/abs/2504.07981](https://arxiv.org/abs/2504.07981)
- UI-TARS: [https://arxiv.org/abs/2501.12326](https://arxiv.org/abs/2501.12326)
- OmniParser: [https://arxiv.org/abs/2408.00203](https://arxiv.org/abs/2408.00203)
