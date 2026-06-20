# 答案版 25：Diffusion、DiT、文生图/视频生成与可控生成

对应题号：521-540。建议先读 [30_Diffusion_DiT_文生图视频生成与可控生成面试.md](../30_Diffusion_DiT_文生图视频生成与可控生成面试.md)，再用本文件练 30 秒版和 2 分钟版。

## 521. Diffusion 和 GAN / 自回归生成有什么区别？

30 秒版：

GAN 是 generator 和 discriminator 对抗训练，采样快但训练不稳定；自回归生成逐 token 或 patch 生成，概率建模清晰但高分辨率图像慢；Diffusion 是从噪声多步去噪，训练稳定、质量高、可控性强，但推理步数多、成本高。

2 分钟版：

对比：

| 方法 | 生成方式 | 优点 | 问题 |
| --- | --- | --- | --- |
| GAN | 一次生成 + 对抗判别 | 快、锐利 | mode collapse、训练不稳 |
| Autoregressive | 逐 token 生成 | likelihood 清晰 | 图像高分辨率慢 |
| Diffusion | 多步去噪 | 稳定、高质量、可控 | 推理慢 |

Diffusion 的核心是：

```text
训练: image -> add noise -> predict noise
生成: noise -> denoise step by step -> image
```

面试表达：

> Diffusion 的优势是训练稳定和条件控制自然，所以文生图、图像编辑、视频生成里很常见。缺点是迭代去噪导致推理慢，所以近年很多工作在做 DiT、flow、scheduler 和 distillation 来加速。

## 522. DDPM 前向加噪公式是什么？

30 秒版：

前向过程是固定加噪：`q(x_t|x_{t-1}) = N(sqrt(1-beta_t)x_{t-1}, beta_t I)`。定义 `alpha_t=1-beta_t`、`alpha_bar_t=prod alpha_s` 后，可以直接写成 `x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon`。

2 分钟版：

一步加噪：

```text
q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I)
```

定义：

```text
alpha_t = 1 - beta_t
alpha_bar_t = product_{s=1}^t alpha_s
```

任意时间步采样：

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
epsilon ~ N(0, I)
```

直觉：

- `sqrt(alpha_bar_t)x_0` 是剩下的图像信号。
- `sqrt(1-alpha_bar_t)epsilon` 是加入的噪声。
- t 越大，图像越接近纯噪声。

面试加分：

> 这个闭式公式让训练时可以随机采时间步 t，而不需要真的从 1 加噪到 t。

## 523. 为什么训练时常预测 noise？epsilon、x0、v prediction 有什么区别？

30 秒版：

预测 noise 目标简单稳定，loss 常写成 `||epsilon - epsilon_theta(x_t,t,c)||^2`。x0 prediction 是直接预测干净图像；v prediction 是一种混合参数化，在一些噪声调度下更稳定。三者可以相互转换，但训练稳定性和采样表现不同。

2 分钟版：

epsilon prediction：

```text
model predicts added noise epsilon
L = ||epsilon - epsilon_theta(x_t,t,c)||^2
```

x0 prediction：

```text
model predicts clean sample x_0
```

v prediction：

```text
predicts a velocity-like combination of x0 and epsilon
```

为什么 epsilon 常见：

- 每个时间步监督明确。
- 和 denoising score matching 有联系。
- 实践稳定。

面试表达：

> 预测目标不是生成模型的本质差异，而是参数化选择。核心都是学会从带噪样本恢复数据分布。不同 prediction target 会影响训练稳定性、不同 timestep 的权重和采样效果。

## 524. Diffusion training loss 怎么写？为什么是 MSE？

30 秒版：

常见 DDPM 简化 loss 是随机采样图像、时间步和噪声，把图像加噪成 `x_t`，让模型预测噪声，最小化真实噪声和预测噪声的 MSE：`E||epsilon - epsilon_theta(x_t,t,c)||^2`。

2 分钟版：

训练流程：

```text
x0 from data
t ~ Uniform(1, T)
epsilon ~ N(0, I)
x_t = sqrt(alpha_bar_t)x0 + sqrt(1-alpha_bar_t)epsilon
epsilon_pred = model(x_t, t, condition)
loss = MSE(epsilon, epsilon_pred)
```

为什么 MSE：

- 高斯假设下自然。
- 预测连续噪声向量。
- 简化变分下界后得到可训练目标。
- 实现简单稳定。

项目坑：

- timestep 采样分布影响训练。
- loss 权重影响不同噪声级别。
- 条件 dropout 影响 CFG。
- VAE latent scaling 要一致。

## 525. DDPM、DDIM、DPM-Solver / scheduler 有什么区别？

30 秒版：

DDPM 更接近原始随机反向扩散；DDIM 可以做确定性采样，较少步数也能生成；DPM-Solver、Euler、Heun 等把采样看成数值求解，目标是更少步数保持质量。Scheduler 是质量、速度和随机性的取舍。

2 分钟版：

对比：

| 方法 | 特点 |
| --- | --- |
| DDPM | 随机、多步、原始扩散采样 |
| DDIM | 可确定性，步数可减少 |
| Euler/Heun | ODE/SDE 数值求解视角 |
| DPM-Solver | 更高阶求解，少步采样 |

上线影响：

- steps 多：质量更稳但慢。
- steps 少：快但细节和 prompt adherence 可能下降。
- scheduler 不同会影响风格、锐度和稳定性。

面试表达：

> Sampler 不是模型权重本身，而是推理时怎么走去噪路径。生产里会根据质量、延迟和成本选择 scheduler 和 steps。

## 526. Classifier-Free Guidance 的公式和直觉是什么？

30 秒版：

CFG 训练时随机丢条件，让模型同时学 conditional 和 unconditional denoising。推理时用 `eps_hat = eps_uncond + s * (eps_cond - eps_uncond)`，把条件方向放大。scale 越大越听 prompt，但太大会过饱和、降低多样性和画质。

2 分钟版：

训练：

```text
with condition: epsilon_theta(x_t,t,c)
without condition: epsilon_theta(x_t,t,empty)
```

推理：

```text
epsilon_hat = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```

直觉：

- `epsilon_cond - epsilon_uncond` 是 prompt 条件带来的方向。
- scale 放大这个方向。
- scale 太低：不听 prompt。
- scale 太高：画面僵硬、过饱和、质量差。

面试补一句：

> CFG 类似生成模型里的低温采样和 truncation，在 fidelity 和 diversity 间取舍。

## 527. Latent Diffusion 为什么省算力？

30 秒版：

Latent Diffusion 不在像素空间去噪，而是先用 VAE encoder 把图像压缩到 latent，在低维 latent 上扩散和去噪，最后用 VAE decoder 还原图像。latent 空间更小，所以训练和推理都省，但 VAE 会限制细节质量。

2 分钟版：

流程：

```text
image -> VAE encoder -> latent
latent denoising
latent -> VAE decoder -> image
```

为什么省：

- 空间分辨率更低。
- denoising network 计算量下降。
- 高分辨率生成可行。
- 条件控制仍然灵活。

代价：

- VAE reconstruction error。
- 细节和文字可能受影响。
- latent scaling、decoder 质量很重要。

面试表达：

> Stable Diffusion 的关键不是只靠 U-Net，而是把扩散搬到 latent 空间，在质量和成本之间做了很好的折中。

## 528. U-Net 和 DiT 有什么区别？

30 秒版：

U-Net 是卷积式 encoder-decoder，有多尺度和 skip connection，适合图像局部结构；DiT 把 latent 切成 patch token，用 Transformer 做 denoising，更适合大规模 scaling 和多模态 token 融合。新一代模型更多走 DiT/flow 方向。

2 分钟版：

U-Net：

- 卷积结构。
- 下采样/上采样。
- skip connection。
- 保留细节。
- cross-attention 接文本。

DiT：

- latent patch tokens。
- Transformer blocks。
- timestep / condition modulation。
- scaling law 更清晰。
- 文本和图像 token 交互更自然。

面试表达：

> U-Net 是图像生成时代的经典 denoiser，DiT 是把 diffusion 和 Transformer scaling 结合起来。它不一定小模型就更好，但大规模训练和复杂多模态条件下优势明显。

## 529. 文本条件怎么注入扩散模型？

30 秒版：

prompt 先经过 tokenizer 和 text encoder 得到文本 embedding，再通过 cross-attention、AdaLN/modulation、concat token 或 pooled embedding 注入 denoising network。text encoder 能力、prompt 截断和训练 caption 质量都会影响图文对齐。

2 分钟版：

流程：

```text
prompt -> tokenizer -> text encoder -> text embeddings
text embeddings -> denoiser conditioning
```

注入方式：

- cross-attention：image latent attends to text tokens。
- AdaLN/modulation：用文本调制层参数。
- concat token：文本和图像 token 一起进 Transformer。
- pooled embedding：全局语义。

风险：

- 长 prompt 被截断。
- 复杂关系理解差。
- 文字生成难。
- 中文能力受训练数据影响。

面试表达：

> 文本条件不是简单拼字符串，而是通过 text encoder 和 attention/modulation 影响每一步去噪。

## 530. ControlNet 解决什么问题？

30 秒版：

ControlNet 解决 prompt 控制不精确的问题。它把 edge、pose、depth、segmentation 等结构条件编码后注入扩散模型，常冻结原模型、训练控制分支，让输出既保留原模型能力，又符合姿态、边缘或布局约束。

2 分钟版：

普通 prompt 只能语义控制：

```text
"a person dancing"
```

但无法精确控制姿态。ControlNet 加结构条件：

```text
prompt + pose/depth/edge -> image
```

机制：

- 原模型保留。
- 新增 control branch。
- 条件特征注入 U-Net 多层。

适合：

- 姿态控制。
- 线稿上色。
- 深度控制。
- 室内设计。
- 商品图结构保持。

面试表达：

> ControlNet 的核心是结构控制，不是风格微调。它让生成结果遵守外部条件。

## 531. LoRA、DreamBooth、Textual Inversion 怎么选？

30 秒版：

Textual Inversion 学新 token embedding，参数少但能力有限；DreamBooth 微调模型学习特定主体，效果强但易过拟合；LoRA 学低秩增量，轻量、易分发，常用于风格、人物、商品定制。

2 分钟版：

对比：

| 方法 | 训练对象 | 优点 | 风险 |
| --- | --- | --- | --- |
| Textual Inversion | token embedding | 小、快 | 表达力有限 |
| DreamBooth | 模型部分/整体 | 主体一致强 | 过拟合、成本高 |
| LoRA | 低秩增量 | 轻量可组合 | 风格污染、冲突 |

怎么选：

- 少量概念：Textual Inversion。
- 强 identity：DreamBooth 或 LoRA。
- 轻量部署：LoRA。
- 多风格切换：LoRA 管理。

项目注意：

> 要看样本质量、版权授权、泛化 prompt、identity consistency 和安全风险。

## 532. 图生图和 inpainting 怎么做？

30 秒版：

图生图是把输入图像加噪到某个强度，再按 prompt 去噪；denoising strength 小更保留原图，大则更自由。Inpainting 额外输入 mask，只重绘 mask 区域，非 mask 区域尽量保持。

2 分钟版：

图生图：

```text
input image -> encode latent -> add noise by strength -> denoise with prompt
```

Inpainting：

```text
image + mask + prompt -> repaint masked region
```

关键参数：

- denoising strength。
- mask blur。
- CFG scale。
- steps。
- prompt 是否描述局部。

常见问题：

- mask 边缘不融合。
- 光照不一致。
- identity 漂移。
- 背景被污染。

面试表达：

> 图像编辑的关键是保留和重绘之间的平衡，不是每次从纯噪声生成。

## 533. 视频生成为什么比图像生成难？

30 秒版：

视频多了时间维度，需要保持帧间一致、身份一致、运动合理和长程依赖。不能逐帧独立文生图，否则会闪烁和身份漂移。视频模型通常用 latent video diffusion、temporal attention、3D U-Net 或视频 DiT。

2 分钟版：

图像：

```text
H x W
```

视频：

```text
T x H x W
```

难点：

- temporal consistency。
- object permanence。
- motion。
- physics。
- long context。
- compute/memory。

结构：

- 3D convolution。
- temporal attention。
- space-time attention。
- spatiotemporal latent tokens。
- text/image-to-video。

面试表达：

> 视频生成的核心不是单帧画质，而是时间一致性和运动建模。

## 534. Sora 类视频生成模型面试怎么讲？

30 秒版：

不要背产品参数，要抽象成长视频生成和世界模拟方向。它需要在时空 latent 上建模场景、对象、运动和长程一致性，能做文生视频、图生视频和视频延展，但仍可能有物理、因果和空间关系错误。

2 分钟版：

可讲能力：

- text-to-video。
- image-to-video。
- video extension。
- 长时间一致性。
- 镜头和运动建模。
- 多对象交互。

局限：

- 物理不稳定。
- 因果错误。
- 手部/文字/复杂交互错误。
- 长视频成本高。
- 安全和版权风险高。

面试表达：

> Sora 类模型说明视频生成正在从短 clip 走向更长的时空建模。面试重点是讲清 temporal consistency、world modeling、评估、安全和推理成本。

## 535. Rectified Flow / Flow Matching 是什么？

30 秒版：

Rectified Flow / Flow Matching 学习从噪声到数据的连续速度场，可以理解为让 noise 到 data 的路径更直接。相比传统 diffusion 多步反向去噪，flow 方法希望路径更直、采样更高效，常和 Transformer 架构结合用于新一代文生图。

2 分钟版：

简化：

```text
x_t = (1 - t)x_0 + t x_1
learn v_theta(x_t,t) ~= x_1 - x_0
```

直觉：

- diffusion 学去噪 score / reverse process。
- flow matching 学连续变换速度。
- rectified flow 希望路径更直。

优势：

- 采样步数有下降潜力。
- 概念简单。
- 和 DiT scaling 结合好。

注意：

> 不要把 flow 说成完全不同的魔法。它仍然是在学习从简单分布到数据分布的生成路径。

## 536. 文生图 / 视频生成怎么评估？

30 秒版：

不能只看 FID。文生图要看图文对齐、审美、主体一致性、文字/关系、人工偏好、安全和成本；视频还要看时间一致性、闪烁、运动合理性和长程稳定性。自动指标要结合人工评审和业务指标。

2 分钟版：

自动指标：

- FID。
- CLIPScore。
- aesthetic score。
- T2I-CompBench。
- VBench。

业务指标：

- prompt adherence。
- identity consistency。
- edit fidelity。
- temporal consistency。
- safety violation rate。
- user acceptance。
- generation latency。
- cost per asset。

局限：

- FID 不看 prompt。
- CLIPScore 有模型偏差。
- aesthetic 主观。
- 人工评审贵但必要。

面试表达：

> 生成模型评估必须是自动指标、人工偏好、安全和业务转化的组合。

## 537. 生成模型安全、版权和合规怎么做？

30 秒版：

做输入和输出双侧治理。输入侧拦截违规 prompt、名人肖像、版权和品牌风险；输出侧做图像/视频安全分类、深度伪造检测、水印溯源、人工审核和日志审计。高风险场景要灰度和回滚。

2 分钟版：

风险：

- 色情/暴力/违法。
- 名人肖像滥用。
- 版权风格争议。
- 商标误用。
- deepfake。
- 训练数据合规。

治理：

- prompt classifier。
- image/video safety classifier。
- face/logo/IP 检测。
- watermark/provenance。
- red team。
- human review。
- audit log。

面试表达：

> AIGC 安全不能只靠 negative prompt，要在输入、生成、输出、审核和追溯全链路做治理。

## 538. 文生图推理慢怎么优化？

30 秒版：

先拆耗时：text encoder、denoising、VAE decode、安全检测。denoising 通常最贵，可以减少 steps、换 scheduler、做蒸馏、低精度、attention 优化、batch/queue、分辨率路由。视频还要控制帧数和时序 attention。

2 分钟版：

成本来源：

- denoising steps。
- resolution。
- U-Net / DiT。
- VAE decode。
- text encoder。
- safety checker。
- video frames。

优化：

- fewer steps。
- DPM-Solver / better scheduler。
- model distillation。
- latent generation。
- mixed precision。
- xFormers/FlashAttention。
- quantization。
- VAE tiling。
- request batching。
- LoRA merge/cache。

面试表达：

> 不要只说换 GPU，要先 profile 哪个阶段慢，再分别优化 denoising、VAE、安全检测和队列调度。

## 539. AIGC / 文生图项目怎么讲 8 分钟？

30 秒版：

按场景、输入条件、模型架构、控制方式、微调数据、评估、安全、部署成本讲。重点证明你不只是调 API，而是做了 prompt、ControlNet/LoRA、评估、安全和推理成本闭环。

8 分钟结构：

1. 场景：

> 广告图、商品图、海报、头像、短视频素材。

2. 输入：

> prompt、参考图、mask、pose/depth/edge、品牌规范。

3. 模型：

> LDM/SD/DiT/ControlNet/LoRA/inpainting/video diffusion。

4. 控制：

> CFG、negative prompt、ControlNet、identity reference。

5. 数据：

> 商品图、风格图、清洗、去重、授权、caption。

6. 评估：

> prompt adherence、主体一致、审美、安全、人工偏好、成本。

7. 部署：

> steps、分辨率、GPU 队列、VAE、safety checker。

8. 风险：

> 版权、肖像、品牌一致性和违规内容。

收尾：

> 我把生成质量、可控性、安全和成本一起做闭环。

## 540. Diffusion 面试前最后怎么复习？

30 秒版：

先背 DDPM 前向公式、noise prediction loss、CFG 公式、Latent Diffusion 流程、U-Net vs DiT、ControlNet、LoRA/DreamBooth、视频生成难点、Rectified Flow、评估安全和推理优化。能把这些串成一个文生图系统就够稳。

2 分钟版：

最后清单：

- `x_t = sqrt(alpha_bar_t)x0 + sqrt(1-alpha_bar_t)epsilon`
- `L = ||epsilon - epsilon_theta||^2`
- CFG：`eps_u + s(eps_c - eps_u)`
- LDM：VAE latent 里去噪。
- U-Net：卷积多尺度。
- DiT：Transformer patch tokens。
- ControlNet：结构控制。
- LoRA/DreamBooth：个性化微调。
- Inpainting：mask 局部重绘。
- Video：时间一致性。
- Flow：学习 noise 到 data 的速度场。
- Eval：图文对齐 + 人类偏好 + 安全 + 成本。

面试口语：

> Diffusion 的主线是多步去噪，现代文生图的主线是 latent、条件控制和 Transformer scaling，工程主线是质量、速度、安全和成本平衡。

## 本组题的复习顺序

1. 先背 521-526：DDPM、loss、scheduler、CFG。
2. 再背 527-532：LDM、DiT、text conditioning、ControlNet、LoRA、inpainting。
3. 再背 533-536：视频生成、Sora、flow 和评估。
4. 最后背 537-540：安全、推理优化、项目讲法和收口清单。

## 延伸阅读

- DDPM：[https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Latent Diffusion Models：[https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- DiT：[https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- Classifier-Free Guidance：[https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
- ControlNet：[https://arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)
- Scaling Rectified Flow Transformers：[https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
- Sora / Video generation models as world simulators：[https://openai.com/index/video-generation-models-as-world-simulators/](https://openai.com/index/video-generation-models-as-world-simulators/)
