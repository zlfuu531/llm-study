# Diffusion、DiT、文生图/视频生成与可控生成面试

这一章补 LLM / AIGC 学习里的另一条生成主线：**扩散模型、DiT、文生图、视频生成和可控生成**。

如果你投的是大模型算法、多模态、AIGC、视觉生成、内容理解/生成、广告创意、设计工具、短视频生成、游戏/电商素材生成相关岗位，这章非常值得准备。面试官不一定要求你手推所有 SDE，但至少会希望你能讲清：

- Diffusion 和 GAN / 自回归生成有什么区别？
- DDPM 前向加噪和反向去噪公式是什么？
- 为什么训练时预测 noise？
- CFG 为什么能提升 prompt adherence？
- Latent Diffusion 为什么省算力？
- U-Net 和 DiT 有什么区别？
- ControlNet、LoRA、DreamBooth 分别解决什么？
- 图生图、局部重绘、视频生成怎么做？
- Rectified Flow / Flow Matching 为什么 2024-2026 常被提？
- 文生图/视频生成怎么评估、怎么控安全和成本？

## 1. 一句话框架

扩散模型可以用一句话理解：

```text
训练时：把真实图像逐步加噪，让模型学会从带噪图像里预测噪声/干净图像
生成时：从随机噪声出发，按很多步逐渐去噪，得到图像或视频
```

文生图系统通常是：

```text
prompt
-> text encoder
-> latent noise
-> denoising network(U-Net / DiT)
-> scheduler
-> VAE decoder
-> image
```

30 秒答案：

> Diffusion 是从噪声逐步去噪生成数据。训练时把真实图像加噪到不同时间步，让模型预测噪声或干净样本；推理时从高斯噪声开始反复去噪。现代文生图常在 latent 空间做扩散，用 text encoder 做条件，用 U-Net 或 DiT 做 denoiser，再用 VAE decoder 还原成图像。

## 2. Diffusion、GAN、自回归生成怎么区分

| 方法 | 生成方式 | 优点 | 缺点 |
| --- | --- | --- | --- |
| GAN | generator 一次生成，discriminator 对抗训练 | 采样快、图像锐利 | 训练不稳定、mode collapse |
| Autoregressive | 逐 token / patch 生成 | 概率建模清晰、适合离散序列 | 高分辨率图像生成慢 |
| Diffusion | 多步去噪生成 | 训练稳定、质量高、可控性强 | 推理步数多，成本较高 |
| Flow / Rectified Flow | 学习从噪声到数据的连续变换 | 路径更直、采样可更少步 | 训练和工程仍需调优 |

面试表达：

> GAN 是对抗学习，生成快但训练不稳定；自回归适合语言这类离散 token，但高分辨率图像逐 token 生成很慢；Diffusion 把生成拆成多步去噪，训练稳定、质量高、条件控制自然，但推理成本高。近年的 DiT、Rectified Flow、distillation 都在提升质量和速度。

## 3. DDPM 前向加噪公式

前向过程是固定的，不需要学习：

```text
q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I)
```

定义：

```text
alpha_t = 1 - beta_t
alpha_bar_t = product_{s=1}^t alpha_s
```

可以直接从 `x_0` 采样任意时间步：

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
epsilon ~ N(0, I)
```

直觉：

- `t` 越小，图像越清楚。
- `t` 越大，噪声越多。
- 到足够大的 `T`，`x_T` 接近标准高斯噪声。

面试口语：

> 前向过程就是逐步把真实图像加噪。关键公式是 `x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon`，它让我们训练时可以随机采一个时间步，不必真的一步步加噪到 t。

## 4. 反向去噪和训练目标

反向过程要学习：

```text
p_theta(x_{t-1} | x_t)
```

常见做法是让神经网络预测噪声：

```text
epsilon_theta(x_t, t, condition)
```

训练 loss：

```text
L = E_{x0, t, epsilon} || epsilon - epsilon_theta(x_t, t, c) ||^2
```

为什么预测 noise：

- 目标简单稳定。
- 每个时间步都有明确监督。
- 和 score matching 有联系。
- 实践效果好。

也可以预测：

| 预测目标 | 含义 |
| --- | --- |
| epsilon prediction | 预测加进去的噪声 |
| x0 prediction | 预测干净图像 |
| v prediction | 预测一种混合参数化，常用于更稳定训练 |
| score prediction | 预测数据分布 log density 的梯度 |

30 秒答案：

> DDPM 训练常让模型预测加噪时的 noise，loss 是预测噪声和真实噪声的 MSE。这样每个随机时间步都有监督信号，训练稳定。推理时用模型预测的噪声把 `x_t` 逐步还原到 `x_{t-1}`，最后得到图像。

## 5. Scheduler / Sampler：为什么推理要很多步

训练时通常有很多时间步；推理时从噪声一步步去噪。

常见采样器：

| Sampler | 直觉 |
| --- | --- |
| DDPM | 随机采样，接近原始反向扩散 |
| DDIM | 可确定性采样，步数可少一些 |
| Euler / Heun / DPM-Solver | 数值 ODE/SDE 求解视角，加速采样 |
| Distilled sampler | 用蒸馏减少步数 |

步数影响：

- 步数多：质量更稳，但慢。
- 步数少：快，但容易细节差、构图崩、prompt 不稳。

面试表达：

> Diffusion 推理慢是因为生成是迭代去噪。Sampler 本质是在近似反向生成路径，不同 scheduler 在质量、速度和随机性之间取舍。上线时要按场景调 steps、分辨率、CFG scale 和模型大小。

## 6. Classifier-Free Guidance

CFG 解决的是条件生成里“听不听 prompt”的问题。

训练时随机丢掉条件，让同一个模型同时学：

```text
conditional denoising: epsilon_theta(x_t, t, c)
unconditional denoising: epsilon_theta(x_t, t, empty)
```

推理时组合：

```text
epsilon_hat = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

其中 `s` 是 guidance scale。

直觉：

- `epsilon_cond - epsilon_uncond` 是条件方向。
- `s` 越大，越强迫模型贴近 prompt。
- `s` 太大可能过饱和、细节坏、图像不自然。

30 秒答案：

> CFG 不需要额外 classifier，而是在训练时让模型同时见到有条件和无条件输入。推理时把 conditional 和 unconditional 的预测差作为 prompt 方向放大。scale 越大 prompt adherence 越强，但过大可能牺牲多样性和画质。

## 7. Latent Diffusion：为什么在 latent 空间做

像素空间扩散很贵：

```text
512x512x3 pixel
```

Latent Diffusion 先用 VAE 把图像压缩：

```text
image -> VAE encoder -> latent
latent diffusion denoising
latent -> VAE decoder -> image
```

优点：

- latent 分辨率更小，训练和推理更省。
- 保留语义和主要视觉信息。
- 可以做高分辨率生成。
- 方便接 text condition、inpainting、super-resolution。

代价：

- VAE 会带来重建误差。
- 细节和文字可能受 decoder 影响。
- latent 空间压缩比影响质量和速度。

面试表达：

> Stable Diffusion 这类模型常用 Latent Diffusion，不直接在像素上去噪，而是在 VAE latent 里去噪。这样大幅降低空间维度和计算量，再用 VAE decoder 还原图像。代价是 VAE 质量会限制最终细节。

## 8. U-Net、DiT 和 MMDiT

早期扩散模型常用 U-Net：

- encoder-decoder。
- skip connection 保留空间细节。
- 适合图像多尺度结构。
- cross-attention 接文本条件。

DiT：Diffusion Transformer：

- 把 latent 切成 patch token。
- 用 Transformer 处理 token。
- 更符合 scaling law。
- 适合大规模训练和多模态 token 融合。

MMDiT / 多模态 DiT 思路：

- 文本 token 和图像 latent token 分别编码。
- 通过 attention 交互。
- 更强 text-image alignment。
- 更适合复杂文字、排版、长 prompt。

30 秒答案：

> U-Net 利用卷积和多尺度 skip connection，很适合图像局部结构；DiT 把 latent 当 patch token，用 Transformer 做 denoising，更容易随模型规模扩展。新一代文生图模型越来越多采用 Transformer/flow 结构，是为了提升 scaling、文本理解和多模态融合能力。

## 9. Text Conditioning 怎么接进扩散模型

文本条件通常来自 text encoder：

```text
prompt -> tokenizer -> text encoder(CLIP/T5/LLM encoder) -> text embeddings
```

注入方式：

| 方式 | 含义 |
| --- | --- |
| cross-attention | image latent query attends to text keys/values |
| AdaLN / modulation | 用文本条件调制归一化或层参数 |
| concat token | 文本 token 和图像 token 一起进 Transformer |
| pooled embedding | 全局语义条件 |

Prompt 相关问题：

- 文本 encoder 能不能理解复杂关系。
- 长 prompt 会不会被截断。
- token 权重和 negative prompt。
- 字体/文字生成依赖 text-image 对齐和训练数据。

面试表达：

> 文生图不是直接把中文 prompt 塞给 U-Net，而是先由 text encoder 变成条件向量，再通过 cross-attention 或 modulation 影响去噪过程。text encoder、数据标注质量和 attention 设计都会影响 prompt adherence。

## 10. ControlNet、T2I-Adapter 和可控生成

普通 prompt 控制不够精确。可控生成希望额外输入结构条件：

- edge。
- depth。
- pose。
- segmentation。
- scribble。
- reference image。
- layout / bounding boxes。

ControlNet：

```text
冻结原 diffusion model
复制一份可训练控制分支
把 control condition 注入各层
```

优点：

- 保留原模型能力。
- 训练相对稳定。
- 可以精确控制姿态、边缘、深度等。

T2I-Adapter：

- 更轻量的条件适配模块。
- 通过 adapter 注入控制信息。

面试表达：

> ControlNet 解决 prompt 控制不精确的问题。它把边缘、姿态、深度等结构条件编码后注入扩散模型，并常冻结原模型，只训练控制分支。这样既保留原模型生成能力，又能让输出更贴合结构约束。

## 11. LoRA、DreamBooth、Textual Inversion

这三类常被问“个性化生成怎么做”。

| 方法 | 训练什么 | 适合 |
| --- | --- | --- |
| Textual Inversion | 新 token embedding | 学一个概念，参数少 |
| DreamBooth | 微调整个模型或部分模块 | 学特定主体，效果强 |
| LoRA | 低秩增量权重 | 轻量风格/人物/产品适配 |

LoRA 在扩散里常用在：

- U-Net attention。
- text encoder。
- DiT attention / MLP。

风险：

- 过拟合主体。
- 风格污染。
- prompt 泛化差。
- 版权和肖像风险。
- 多 LoRA 叠加冲突。

面试表达：

> Textual Inversion 更像学一个新词向量，DreamBooth 更强但更容易过拟合，LoRA 是轻量低秩适配，适合风格、人物或商品定制。项目里要看样本数量、泛化、身份一致性、版权和推理成本。

## 12. 图生图、Inpainting、局部重绘

图生图：

```text
input image -> add noise to strength level -> denoise with prompt condition
```

`denoising strength`：

- 小：更保留原图。
- 大：更自由重绘。

Inpainting：

```text
image + mask + prompt
-> 只重绘 mask 区域
-> 非 mask 区域尽量保持
```

关键点：

- mask 边缘融合。
- 局部光照一致。
- prompt 是否只描述重绘区域。
- 保持人物/商品 identity。

面试表达：

> 图生图不是从纯噪声开始，而是把输入图加到某个噪声强度，再条件去噪。Inpainting 多了 mask，让模型只重绘局部。denoising strength 控制保留原图和创造性的平衡。

## 13. 视频生成为什么更难

视频比图像多一个时间维度：

```text
image: H x W
video: T x H x W
```

难点：

- 时间一致性。
- 物体身份保持。
- 运动合理性。
- 长视频记忆。
- 物理一致性。
- 计算和显存暴涨。
- 训练数据标注更难。

常见结构：

- 3D U-Net。
- temporal attention。
- space-time attention。
- latent video diffusion。
- DiT with spatiotemporal patches。
- image-to-video / text-to-video。

面试表达：

> 视频生成不是把每帧独立文生图拼起来。它必须建模时间一致性、运动和长程依赖，否则会闪烁、身份漂移和动作不连贯。工程上常在 latent 空间做时空去噪，用 temporal attention 或视频 DiT 处理时间维。

## 14. Sora / 视频生成模型怎么讲

面试里提 Sora，不要背营销描述，要抽象成通用能力：

- 文本到视频。
- 图像到视频。
- 视频延展。
- 长时间一致性。
- 对场景、物体、运动有一定建模。
- 仍可能出现物理、因果、空间关系错误。

可以这样说：

> Sora 类模型代表视频生成从短 clip 走向更长、更一致的世界模拟方向。它不只是逐帧生成，而是要在时空 latent 上建模对象、运动和场景变化。面试重点不是背某个产品，而是讲清视频生成的时间一致性、长程依赖、物理错误、评估和成本问题。

## 15. Rectified Flow / Flow Matching

近年文生图模型常提 flow / rectified flow。

Diffusion 的直觉：

```text
数据 -> 逐步加噪
噪声 -> 多步反向去噪
```

Rectified Flow 的直觉：

```text
学习从 noise 到 data 的连续速度场
希望路径更直、更容易采样
```

简化表达：

```text
x_t = (1 - t) x_0 + t x_1
learn v_theta(x_t, t) ~= x_1 - x_0
```

好处：

- 概念上路径更直接。
- 可以减少采样步数潜力。
- 适合 Transformer 扩展。

面试表达：

> Rectified Flow / Flow Matching 和传统 diffusion 都是在学习从噪声到数据的生成路径，但 flow 更强调学习连续速度场，让噪声到数据的路径更直，采样可能更高效。SD3/FLUX 这类模型会把 flow 和 Transformer 架构结合起来。

## 16. 文生图 / 视频生成怎么评估

自动指标：

| 指标 | 看什么 | 局限 |
| --- | --- | --- |
| FID | 生成分布和真实分布距离 | 不直接看 prompt 对齐 |
| CLIPScore | 图文相似度 | 会被 CLIP 偏差影响 |
| Aesthetic score | 美学偏好 | 主观偏差 |
| Human preference | 人类偏好 | 贵，协议要固定 |
| T2I-CompBench | 组合关系/属性对齐 | 覆盖有限 |
| VBench | 视频质量和时序能力 | 仍需结合业务 |

项目指标：

- prompt adherence。
- subject consistency。
- identity preservation。
- temporal consistency。
- edit fidelity。
- safety violation rate。
- generation latency。
- cost per image/video。
- user acceptance rate。

面试表达：

> 文生图不能只看 FID。真正上线要看 prompt 对齐、构图、主体一致性、审美、人类偏好、安全和成本。视频还要额外看时间一致性、闪烁、运动合理性和长程稳定性。

## 17. 安全、版权和合规

AIGC 生成模型风险：

- 生成违法/色情/暴力内容。
- 名人肖像和身份滥用。
- 版权风格争议。
- 商标和品牌误用。
- 深度伪造。
- 训练数据来源不清。
- 水印和溯源问题。

治理：

- prompt safety classifier。
- image/video safety classifier。
- negative prompt / safety guidance。
- 敏感主体黑名单。
- 人脸/商标/版权检测。
- watermark / provenance。
- 人工审核。
- 红队和日志审计。

面试表达：

> 生成模型上线必须做输入和输出双侧安全。输入侧拦截违规 prompt，输出侧做图像/视频安全检测、名人/商标/版权风险检查和水印溯源。高风险场景要加人工审核和灰度回滚。

## 18. 推理优化和部署

成本来自：

- denoising steps。
- 分辨率。
- batch size。
- text encoder。
- VAE encode/decode。
- U-Net / DiT 参数量。
- video frames。

优化手段：

- 减少 steps / 更好 scheduler。
- distillation / consistency model。
- latent 空间生成。
- mixed precision / quantization。
- attention 优化。
- VAE tiling。
- batch and queue。
- LoRA merge / adapter cache。
- resolution routing。
- safety model 异步化。

面试表达：

> 文生图推理优化先拆时间：text encoder、denoising、VAE decode、安全检测。denoising 通常最贵，可以通过减少 steps、蒸馏、更好 scheduler、低精度、attention 优化和分辨率路由来降成本。视频生成还要额外控制帧数和时序 attention 成本。

## 19. 项目讲法模板

如果你做过 AIGC / 文生图 / 图像编辑项目，可以按这个讲：

1. **场景**：广告图、商品图、头像、海报、视频素材、设计辅助。
2. **输入**：prompt、参考图、mask、pose/depth/edge、品牌规范。
3. **模型**：SD/LDM/DiT/ControlNet/LoRA/视频扩散。
4. **控制**：CFG、negative prompt、ControlNet、inpainting、identity reference。
5. **训练/微调**：LoRA/DreamBooth，数据清洗和标注。
6. **评估**：图文对齐、主体一致、审美、安全、人工偏好、成本。
7. **部署**：steps、分辨率、队列、GPU、缓存、安全审核。
8. **风险**：版权、肖像、违规生成、品牌一致性。

收尾句：

> 我没有只把模型接起来，而是把 prompt、结构控制、微调、评估、安全和推理成本做成闭环。

## 20. 高频追问清单

1. Diffusion 和 GAN / 自回归生成有什么区别？
2. DDPM 前向加噪公式是什么？
3. 为什么可以直接采样 `x_t`？
4. 为什么训练时常预测 noise？
5. epsilon prediction、x0 prediction、v prediction 有什么区别？
6. DDPM、DDIM、DPM-Solver 有什么区别？
7. CFG 公式和直觉是什么？
8. Latent Diffusion 为什么省算力？
9. U-Net 和 DiT 怎么区分？
10. 文本条件怎么注入扩散模型？
11. ControlNet 解决什么问题？
12. LoRA、DreamBooth、Textual Inversion 怎么选？
13. 图生图和 inpainting 怎么做？
14. 视频生成为什么比图像生成难？
15. Sora 类模型面试怎么讲？
16. Rectified Flow / Flow Matching 是什么？
17. 文生图怎么评估？
18. 生成模型安全怎么做？
19. 文生图推理慢怎么优化？
20. AIGC 项目怎么讲 8 分钟？

## 21. 推荐阅读

- DDPM：[https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Latent Diffusion Models：[https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- DiT：[https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- Classifier-Free Guidance：[https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
- ControlNet：[https://arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)
- Scaling Rectified Flow Transformers / Stable Diffusion 3：[https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
- Sora / Video generation models as world simulators：[https://openai.com/index/video-generation-models-as-world-simulators/](https://openai.com/index/video-generation-models-as-world-simulators/)

## 22. 本章复习顺序

第一遍：

1. DDPM 前向/反向和训练 loss。
2. CFG、scheduler、latent diffusion。
3. U-Net、DiT、text conditioning。
4. ControlNet、LoRA、DreamBooth、inpainting。
5. 视频生成、Rectified Flow、评估和安全。

第二遍：

- 先背 521-526：扩散基础公式。
- 再背 527-532：文生图架构和可控生成。
- 再背 533-536：视频生成、flow 和评估。
- 最后背 537-540：安全、部署和项目讲法。
