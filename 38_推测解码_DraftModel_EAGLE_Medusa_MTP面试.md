# 推测解码、Draft Model、EAGLE/Medusa/MTP 面试

这一章面向推理部署、AI Infra、模型服务平台、低延迟应用、代码模型和高阶系统设计岗位。Speculative Decoding 很容易被答成“小模型先生成，大模型验证”，但面试官继续追问时，真正要看的是：你是否理解为什么它可以保持目标模型分布、什么时候能加速、accept rate 怎么看、为什么高 QPS 反而收益不明显，以及 EAGLE、Medusa、MTP 和普通 draft model 到底差在哪里。

如果时间很紧，先背这句：

> 推测解码的核心是用更便宜的 proposer 先草拟多个 token，再让目标模型一次 forward 并行验证这些 token。它能加速的前提是 draft 足够便宜、目标模型验证多个 token 的代价接近验证一个 token、接受率足够高，并且端到端瓶颈主要在 decode 串行等待。经典 speculative sampling 可以通过接受/拒绝和修正采样保持目标模型分布不变；EAGLE、Medusa、MTP、n-gram/suffix 是不同的草拟 token 来源。

相关答案版：[answers/33_推测解码_DraftModel_EAGLE_Medusa_MTP_答案版.md](answers/33_推测解码_DraftModel_EAGLE_Medusa_MTP_答案版.md)

相邻章节：

- [24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md](24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md)：serving engine、prefill/decode、KV Cache、continuous batching。
- [36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md](36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md)：temperature、top-k/top-p、LogitsProcessor、EOS/stop 和 generation config。
- [18_DeepSeek_MoE_MLA与ReasoningModel面试.md](18_DeepSeek_MoE_MLA与ReasoningModel面试.md)：DeepSeek V3/R1、MTP、MLA、MoE 和推理效率。
- [37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md](37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md)：量化、FP8、KV Cache 量化和低比特部署。
- [25_GPU_CUDA_Triton与FlashAttention面试.md](25_GPU_CUDA_Triton与FlashAttention面试.md)：memory-bound、kernel、GPU 利用率和 profiler 排查。

## 1. 它解决什么瓶颈

自回归 decode 的基本形态是：

```text
for each step:
  target_model(prefix) -> next token
  append token
```

问题在于每个新 token 依赖上一个 token，所以 decode 是强串行的。即使每一步只处理一个 token，目标模型也要读取大量权重和 KV Cache。小 batch / 中低 QPS 场景下，GPU 可能没有被大矩阵乘充分填满，TPOT / ITL 会被逐 token 等待拖住。

Speculative Decoding 试图把：

```text
目标模型跑 gamma 次，每次前进 1 个 token
```

变成：

```text
便宜 proposer 先草拟 gamma 个 token
目标模型一次验证这 gamma 个 token
一次 forward 尽量前进多个 token
```

所以它优化的是 decode 阶段的每 token 延迟，不是 prefill。长 prompt 导致 TTFT 高时，speculative decoding 不会直接解决 prefill 过重的问题。

## 2. 基本流程

经典 draft-target speculative decoding：

```text
prefix x
-> draft model q 草拟 y_1...y_gamma
-> target model p 对 prefix + y_1...y_gamma 一次 forward
-> 从左到右验证 draft token
-> 接受连续前缀
-> 遇到拒绝则修正采样并停止本轮
-> 如果全部接受，目标模型还可以额外给出一个 token
```

伪代码直觉：

```python
while not stop:
    draft_tokens = draft.generate(prefix, gamma)
    target_logits = target.verify(prefix, draft_tokens)
    accepted = verify_left_to_right(draft_tokens, target_logits)
    prefix += accepted
```

关键点：

- draft 是 proposer，不是最终裁判。
- target 仍然决定哪些 token 被接受。
- target 对多个候选位置并行计算，所以可能一次推进多个 token。
- 如果 draft 质量差，很多 token 被拒绝，就回到接近普通 decode，甚至更慢。

## 3. 为什么可以保持目标模型分布

greedy 场景很直观：

```text
如果 draft token == target argmax token -> 接受
否则用 target argmax token -> 本轮停止
```

非 greedy sampling 场景更细。设：

- `p` 是目标模型对当前 token 的概率分布。
- `q` 是 draft/proposer 的概率分布。
- draft 抽到了 token `y`。

经典 speculative sampling 的接受概率可以写成：

```text
a(y) = min(1, p(y) / q(y))
```

如果接受，就输出 `y`。如果拒绝，则从修正后的剩余分布采样：

```text
r(x) ∝ max(0, p(x) - q(x))
```

这个接受/拒绝 + correction 的设计，使最终输出分布等价于直接从目标模型 `p` 采样。面试不用完整证明，但要能说清：

> 无损不是因为 draft 很准，而是因为 target 的验证和拒绝后的修正采样保证了目标分布。draft 越准，接受越多，加速越明显；draft 不准，仍能保持分布，但速度收益会掉。

注意：不同框架、不同方法在 greedy、sampling、typical acceptance、tree attention、guided decoding 中实现细节不完全一样。面试时把“理论保证”和“具体 runtime 支持”分开讲最稳。

## 4. 加速条件和粗略公式

设：

- `C_T`：普通目标模型生成 1 个 token 的成本。
- `C_D`：draft 生成 1 个 token 的成本。
- `gamma`：每轮草拟 token 数。
- `C_V`：target 一次并行验证 `gamma` 个 token 的成本。
- `A`：每轮平均接受 token 数。

普通 decode 生成约 `A + 1` 个 token 的成本：

```text
(A + 1) * C_T
```

推测解码一轮成本：

```text
gamma * C_D + C_V + overhead
```

粗略 speedup：

```text
speedup ≈ ((A + 1) * C_T) / (gamma * C_D + C_V + overhead)
```

能加速需要：

- `C_D` 足够小。
- `C_V` 接近一次 target forward，而不是 `gamma` 次 target forward。
- `A` 足够大，也就是 accept rate 高。
- overhead、KV Cache、调度、CUDA graph、通信、内存占用不能把收益吃掉。

TensorRT-LLM 文档也强调了两个直觉前提：多个 draft token 并行处理要接近单 token 成本；多个 draft token 要能在生成过程中被成功验证。面试就用这两个前提展开。

## 5. accept rate 怎么看

常见指标：

```text
token acceptance rate = accepted draft tokens / proposed draft tokens
mean accepted length = 每轮平均接受 token 数
tokens per target step = 每次 target 验证平均推进 token 数
```

accept rate 高说明 draft/proposer 和 target 对下一个 token 分布接近。影响因素：

- draft 模型能力和目标模型是否匹配。
- tokenizer / vocab 是否一致。
- temperature、top-p、top-k 等 sampling 参数。
- 任务类型：代码补全、模板文本、重复上下文通常更容易预测；开放写作、长摘要、复杂推理可能更难。
- 上下文分布：业务 prompt 和 draft 训练数据是否匹配。
- gamma 太大时，越往后的 token 越容易错，拒绝级联更明显。

低 accept rate 的典型现象：

- target 每轮只接受 0-1 个 token。
- draft 计算和显存开销变成负担。
- TPOT 没降，甚至上升。
- GPU 利用率和显存占用变差。

## 6. Draft Model 方法

经典方法使用两个模型：

```text
small draft model q
large target model p
```

要求：

- draft 比 target 便宜很多。
- 最好同 tokenizer / vocab，否则验证和 token 对齐很麻烦。
- draft 输出分布要和 target 足够接近。
- 服务系统要同时管理两套权重、KV Cache、batching 和调度。

优点：

- 概念清楚，论文和框架支持多。
- 目标模型不用改结构。
- 可以给不同 target 配不同 draft。

缺点：

- 需要维护 draft 模型。
- draft 太弱 accept rate 低。
- draft 太强成本高。
- 双模型调度和显存更复杂。
- 高 QPS 大 batch 时 target 已经饱和，收益可能下降。

面试句：

> draft model 的核心不是“小模型越小越好”，而是“足够便宜且足够像目标模型”。太小不准，太大不省。

## 7. N-gram / Suffix Decoding

有些场景不一定需要模型当 draft。n-gram / suffix 方法会从 prompt 或已生成文本里找重复片段，用已有 token 作为草稿，再让目标模型验证。

适合：

- 代码补全。
- 日志、表格、模板化文本。
- 文档里重复短语多的场景。
- 不想加载额外 draft 模型的服务。

优点：

- 不需要训练 draft。
- 开销很低。
- 容易作为保守优化。

局限：

- 候选来自已有上下文，不适合完全新内容。
- 对开放式创作和复杂推理帮助有限。
- 命中率依赖数据重复性。

一句话：

> n-gram speculative decoding 像是“从上下文里猜下一个重复片段”，便宜但适用面更窄。

## 8. Medusa

Medusa 的思路是：不给目标模型再配一个独立小模型，而是在 LLM 上加多个 decoding heads，让这些 heads 并行预测未来多个位置的 token。

直觉：

```text
原模型 head: 预测 t+1
Medusa heads: 预测 t+2, t+3, ... 的候选
tree attention: 构造多条候选路径
target backbone: 并行验证候选路径
```

优点：

- 不需要维护独立 draft model。
- 可以复用目标模型 backbone。
- tree-based attention 可以一次验证多条候选路径。

局限：

- 需要训练额外 heads，甚至联合训练。
- 工程实现比普通 generate 更复杂。
- tree 分支太多会增加计算和显存。
- 不同实现对 sampling 的 lossless 保证和质量取舍要具体看。

面试句：

> Medusa 把“草拟器”做成目标模型上的多头预测模块，用多个 decoding heads 和 tree attention 产生并验证候选，减少逐 token decode 步数。

## 9. EAGLE

EAGLE 的关键词是 feature-level speculation。它认为在 token 层直接预测未来 token 有不确定性，而在接近输出层的 feature 空间做预测可能更容易。

直觉：

```text
不是只猜未来 token
而是预测目标模型靠近输出层的 feature
再结合 shifted token 解决 feature uncertainty
最后用目标模型验证
```

优点：

- 常作为强通用 speculator。
- 可以在保持目标分布的前提下提高速度。
- 在 vLLM、SGLang、TensorRT-LLM 等生态里经常被提到。

局限：

- 通常需要 EAGLE draft 模型/权重。
- 参数如 draft depth、topk、num draft tokens 需要调。
- tree 变大时显存和计算会增加。
- 仍然要看任务、采样、硬件和 QPS。

一句话：

> EAGLE 把 speculative decoding 从 token 草拟推进到 feature 草拟，用 feature + shifted token 预测未来表示，再通过目标模型验证，目标是更高 accept rate 和更强加速。

## 10. MTP：Multi-Token Prediction

MTP 是 Multi-Token Prediction。它可以作为训练目标，也可以作为推理加速的草拟能力。

在 DeepSeek-V3/R1 语境里，MTP 常被问：

- 训练时让模型不只预测 next token，还预测后续多个 token。
- 帮助模型学习更远的未来 token 信号。
- 如果模型原生带 MTP 模块，推理时可用它作为 speculative decoding 的 proposer。
- 和独立 draft model 不同，MTP 能减少额外模型维护。

面试要分清：

```text
MTP as training objective:
  用多 token 预测辅助训练，可能提升训练和表示效果。

MTP as speculative proposer:
  用模型内置多 token 预测能力草拟未来 token，再让目标路径验证。
```

不要说“MTP 一定让模型一次生成多个 token”。更准确：

> MTP 提供多 token 预测信号或模块，能用于 speculative decoding，但最终是否加速仍取决于验证机制、接受率、runtime 支持和端到端瓶颈。

## 11. vLLM / SGLang / TensorRT-LLM 怎么讲

面试不要背命令，讲支持的类别和适用条件。

vLLM：

- speculative decoding 面向中低 QPS、memory-bound 场景降低 inter-token latency。
- 支持 EAGLE、MTP、draft model、MLP、n-gram、suffix 等多类方法。
- 真实收益依赖模型族、流量形态、硬件和采样设置。

SGLang：

- 强调 EAGLE-2/EAGLE-3、MTP、DFLASH、standalone draft model、NGRAM 等方法。
- 参数常围绕 `num_steps`、`topk`、`num_draft_tokens`、draft model path 和 OOM 调整。
- speculative tree 越大，潜在接受更多，但显存和计算压力更高。

TensorRT-LLM：

- 把 speculative sampling 解释为每次 forward 生成超过一个 token 的技术族。
- 支持 draft-target、Medusa、EAGLE、n-gram、lookahead 等思路。
- 强调小 batch GPU 利用不足时更可能降低平均 per-token latency。

项目表达：

> 我不会默认开 speculative decoding，而是先看 decode 是否 memory-bound、小 batch 是否 GPU 利用不足，再按任务测试 draft model/EAGLE/MTP/n-gram 的 accept rate、TPOT、吞吐、显存和质量。

## 12. 和解码参数 / 结构化输出的关系

Speculative decoding 不是独立于 generation config 的魔法。

会影响接受率的参数：

- temperature。
- top-p / top-k。
- repetition penalty。
- constrained decoding / guided decoding。
- stop tokens / max_new_tokens。
- tool calling / JSON schema。

低温 greedy：

- draft 更容易猜中。
- 验证简单。
- accept rate 通常更高。

高温 sampling：

- 分布更随机。
- draft 和 target 抽样差异更大。
- 需要严格的 speculative sampling 修正才能保持目标分布。

结构化输出：

- grammar / JSON schema 会限制合法 token。
- proposer 也要感知约束，否则草拟 token 容易被拒。
- CPU grammar 状态、GPU verification、CUDA graph 之间可能有额外协调开销。

## 13. 什么时候可能变慢

典型原因：

- accept rate 低，每轮只前进很少 token。
- draft model 太大，草拟成本高。
- gamma / tree 太大，验证和显存开销高。
- 高 QPS / 大 batch 下 target GPU 已经饱和，验证并行收益下降。
- 主要瓶颈是 prefill、网络、tokenizer、排队，而不是 decode。
- draft 和 target tokenizer / chat template / distribution 不匹配。
- speculative KV Cache、CUDA graph、batching 和调度引入额外 overhead。
- guided decoding / constrained decoding 让 proposer 经常生成非法 token。
- 多卡并行、PP/TP、MoE、LoRA、量化组合后兼容性和实现路径复杂。

面试句：

> Speculative decoding 优化的是 decode 串行等待。如果系统瓶颈不在这里，或者草拟成本和拒绝率太高，它就可能不加速甚至变慢。

## 14. 指标和评估

质量指标：

- 输出分布是否与 target baseline 一致。
- benchmark / 私有 eval 是否下降。
- 结构化输出有效率。
- 代码 pass@k / 单测通过率。
- RAG faithfulness / 引用正确率。

性能指标：

- TTFT：通常不直接改善长 prompt prefill。
- TPOT / ITL：重点指标。
- output tokens/s。
- request throughput / QPS。
- P50/P95/P99 延迟。
- GPU utilization、HBM bandwidth。
- draft latency、verification latency。
- accept rate、mean accepted length。
- 显存、KV Cache、OOM 率。

ablation：

```text
baseline normal decoding
-> draft model gamma=3/5/8
-> EAGLE / MTP / n-gram
-> 不同 temperature/top_p
-> 不同短输出/长输出/代码/RAG/JSON 桶
-> 不同 QPS / batch / concurrency
```

## 15. 调参和排查

调参顺序：

```text
1. 固定 target model、tokenizer、chat template、generation config
2. 跑 baseline：TTFT、TPOT、tokens/s、P95/P99、显存
3. 选择 proposer：draft / EAGLE / MTP / n-gram
4. 调 num_speculative_tokens / num_steps / topk / tree size
5. 看 accept rate 和 mean accepted length
6. 分桶评估质量和延迟
7. 决定是否按任务动态开启
```

低接受率排查：

- draft 模型是否太弱或训练域不匹配。
- draft 和 target tokenizer 是否一致。
- temperature/top-p 是否过高。
- gamma 是否太大。
- 是否结构化输出导致大量非法 token。
- 是否 prompt 类型本身难预测。

OOM 排查：

- 降低 draft tree size / num_draft_tokens。
- 降低 max running requests。
- 调整 static memory fraction。
- 检查 target + draft 双模型显存。
- 关闭不必要的高 topk 分支。

质量排查：

- 如果理论上 lossless，却质量变了，优先查实现、sampling 配置、随机种子、logits processor、stop 条件和框架兼容。
- 如果使用非严格 acceptance 或 typical acceptance，要明确这是质量/速度取舍。

## 16. 项目 8 分钟讲法

可以这样组织：

```text
背景：
  线上 decode TPOT 高，流式输出慢，GPU 在小 batch 下利用率不高。

baseline：
  普通 vLLM/SGLang/TensorRT-LLM decode，记录 TTFT、TPOT、tokens/s、P95、显存。

方案：
  比较 draft model、EAGLE/MTP、n-gram。
  对代码补全、RAG、JSON、开放问答分桶测试。

指标：
  accept rate、mean accepted length、TPOT、吞吐、质量分、格式有效率、OOM 率。

结果：
  某些任务开启 speculative decoding，某些高温开放生成或高 QPS 场景关闭。

上线：
  灰度、配置版本化、按模型/任务/负载动态路由、fallback 到普通 decode。

复盘：
  接受率比理论更关键；不是所有流量都适合一刀切。
```

## 17. 高频追问快答

**Q1：Speculative decoding 是不是无损？**  
经典 speculative sampling 可以保持目标模型分布，但前提是按正确接受/拒绝和修正采样实现。某些工程变体为了速度会做近似，要具体看实现。

**Q2：为什么代码补全常适合？**  
代码和模板文本局部可预测性强，重复结构多，draft 或 n-gram 更容易猜中，accept rate 更高。

**Q3：为什么高 QPS 收益可能小？**  
高 QPS 大 batch 时目标模型已经能充分利用 GPU，单次验证多个 token 的边际收益下降，draft 还会占显存和算力。

**Q4：EAGLE 和 Medusa 区别？**  
EAGLE 更偏 feature-level speculator；Medusa 给模型加多个 decoding heads 并用 tree attention 产生候选。二者都不是普通“小模型 draft”。

**Q5：MTP 和 speculative decoding 关系？**  
MTP 是多 token 预测目标/模块，可以给 speculative decoding 提供原生 proposer，但仍需要 target 验证。

**Q6：accept rate 低怎么办？**  
换更匹配 draft、降 temperature/top-p、减少 gamma/tree、按任务路由、使用 EAGLE/MTP 或 n-gram，必要时直接关闭。

## 18. 面试前背诵版

Speculative decoding 的目标是减少 decode 阶段逐 token 串行等待。它用 draft/proposer 先草拟多个 token，再让 target model 一次 forward 验证；经典 speculative sampling 用 `min(1, p/q)` 接受概率和拒绝后的修正分布保证最终分布等价于目标模型。它能加速的核心条件是 draft 便宜、target 并行验证多个 token 的成本接近单 token、accept rate 高、瓶颈确实在 decode。Draft model 方法需要维护小模型；n-gram/suffix 不要额外模型但适用面窄；Medusa 用多个 decoding heads 和 tree attention；EAGLE 做 feature-level speculation；MTP 是模型原生多 token 预测能力，可用于推测解码。线上评估要看 accept rate、mean accepted length、TTFT、TPOT、tokens/s、P95/P99、显存、OOM 和私有 eval。accept rate 低、draft 太重、高 QPS 大 batch、prefill 成为瓶颈或结构化约束冲突时，它可能不加速甚至变慢。

## 参考资料

- vLLM Speculative Decoding：[https://docs.vllm.ai/en/latest/features/speculative_decoding/](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- SGLang Speculative Decoding：[https://docs.sglang.ai/advanced_features/speculative_decoding.html](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
- TensorRT-LLM Speculative Sampling：[https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)
- Fast Inference from Transformers via Speculative Decoding：[https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
- Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads：[https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty：[https://arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)
- DeepSeek-V3 Technical Report：[https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- NVIDIA NeMo Megatron Bridge MTP：[https://docs.nvidia.com/nemo/megatron-bridge/latest/training/multi-token-prediction.html](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/multi-token-prediction.html)
