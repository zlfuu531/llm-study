# 答案版 29：Hugging Face 微调工程、Transformers、PEFT、TRL 与 Datasets

对应题目：`03_高频题单100题.md` 的 601-620。

用法：每题先说 30 秒版，再补工程链路。Hugging Face 面试题重点不是“会不会调用库”，而是你是否知道数据、模板、loss、adapter、checkpoint 和评估之间怎么互相影响。

## 601. Hugging Face 微调工程完整链路怎么讲？

30 秒版：

完整链路是任务定义、数据 schema、清洗去重脱敏、chat template、tokenize、labels mask、collator、base model/tokenizer、PEFT/量化配置、Trainer/SFTTrainer/DPOTrainer/ORPOTrainer、eval/checkpoint、adapter 保存合并和 bad case 回流。

2 分钟版：

我会按：

```text
数据 -> 模板 -> token -> loss -> 训练 -> 保存 -> 评估 -> 回流
```

来讲。先确定是 SFT、DPO、ORPO、分类还是继续预训练；再统一 messages/prompt/response 字段；然后用目标模型的 chat template 渲染文本；tokenize 后构造 labels mask；训练时选择 Trainer、SFTTrainer、DPOTrainer 或 ORPOTrainer；用 LoRA/QLoRA 控显存；最后保存 adapter/tokenizer/model card，并用 bad case 回流更新数据。

## 602. AutoTokenizer、AutoModel、AutoConfig 分别负责什么？

30 秒版：

AutoTokenizer 加载分词器和 special tokens，AutoModel/AutoModelForCausalLM 加载模型结构和权重，AutoConfig 加载模型配置。三者必须匹配，否则 token id、chat template、pad/eos、模型 head 都可能出问题。

2 分钟版：

Tokenizer 决定文本怎么变 token；model 决定网络结构和权重；config 记录 hidden size、layers、vocab、pad/eos、use_cache 等行为。微调时常见坑是 tokenizer 和 model 不是同一个 repo，pad token 缺失，chat template 没保存，base/instruct 模型混用。

面试句：

> Tokenizer 和 config 不是附属品，它们是模型接口的一部分。

## 603. Chat template 在 SFT 里为什么重要？

30 秒版：

Chat template 把 system/user/assistant/tool messages 渲染成模型训练时见过的格式。模板错会导致角色边界、assistant 起始、EOS、工具调用和 loss mask 都错。

2 分钟版：

训练数据可能是：

```json
[{"role":"user","content":"解释 KV Cache"},{"role":"assistant","content":"..."}]
```

模型实际看到的是 chat template 渲染后的 token 序列。SFT 时要包含 assistant 答案，推理时通常加 generation prompt。换模型时必须换模板，否则模型可能复述 user、输出特殊 token 或工具调用格式错误。

## 604. SFT 数据 labels mask 怎么做？

30 秒版：

SFT 通常只在 assistant 答案 token 上算 loss，system/user/padding/tool observation 位置设为 `-100`，让 CrossEntropyLoss 忽略这些位置。

2 分钟版：

基本规则：

```text
input_ids: 完整上下文
attention_mask: padding 为 0
labels: assistant token 保留 id，其余 -100
```

如果把 user prompt 也算 loss，模型可能学会复述问题；如果 padding 没 mask，loss 会被无效 token 污染；如果 assistant 边界错，模型格式会乱。面试加分是打印一条样本，检查渲染文本、tokens 和 labels 对齐。

## 605. Hugging Face Datasets 的 map/filter/shuffle/streaming 怎么用？

30 秒版：

`filter` 过滤低质样本，`map` 做清洗、模板渲染、tokenize、labels 构造，`shuffle` 打乱数据，streaming 适合超大数据流式读取但随机访问、shuffle 和 resume 更麻烦。

2 分钟版：

常见流程：

```python
ds = load_dataset("json", data_files=...)
ds = ds.filter(quality_fn)
ds = ds.map(tokenize_fn, batched=True, remove_columns=...)
ds = ds.shuffle(seed=42)
```

`batched=True` 通常更快。大规模预训练可以 streaming，但 SFT/DPO 如果需要精确分桶、去重、随机切分和多轮评估，普通 Dataset 更好操作。

## 606. DataCollator 在微调里解决什么？

30 秒版：

DataCollator 把样本列表拼成 batch，负责 padding、attention_mask、labels 对齐，有时还处理动态 padding、packing 或 seq2seq label padding。

2 分钟版：

Tokenizer 处理单条或批量文本，collator 处理训练时一个 batch 的张量形状。LLM SFT 里 collator 要确保 input_ids、attention_mask、labels 同长度，padding 的 labels 是 `-100`。如果 collator 错，训练可能不报错，但 loss 和模型行为会错。

## 607. Trainer 和手写 PyTorch loop 怎么取舍？

30 秒版：

Trainer 适合标准 Transformers 训练，能快速获得分布式、AMP、logging、checkpoint、eval；手写 loop 适合自定义 loss、复杂多模型交互、特殊数据生成或需要完全控制训练步骤。

2 分钟版：

Trainer 的优势是少写样板代码，TrainingArguments 控制 batch、accumulation、bf16、eval/save/logging 等。缺点是封装较深，出了问题要懂内部数据字段和 loss 计算。手写 loop 更透明，但要自己处理 AMP、DDP、checkpoint、日志和恢复。

好的回答：

> 项目初期用 Trainer 快速试验，复杂训练或排查时回到底层 PyTorch loop。

## 608. TrainingArguments 里哪些参数最常被问？

30 秒版：

常问 batch size、gradient_accumulation_steps、learning_rate、warmup、num_train_epochs/max_steps、bf16/fp16、logging/eval/save steps、load_best_model_at_end、gradient_checkpointing、report_to 和 output_dir。

2 分钟版：

这些参数对应训练关键维度：

- 显存：per_device_train_batch_size、gradient_accumulation_steps、max_length。
- 稳定性：learning_rate、warmup_ratio、weight_decay、max_grad_norm。
- 性能：bf16/fp16、gradient_checkpointing、dataloader_num_workers。
- 评估：eval_strategy、eval_steps、metric_for_best_model。
- 可恢复：save_steps、save_total_limit、resume checkpoint。

不要只背名字，要能说它影响显存、速度、稳定性还是评估。

## 609. SFTTrainer 和普通 Trainer 有什么区别？

30 秒版：

SFTTrainer 是 TRL 面向 LLM 指令微调的封装，更方便处理文本字段、chat 数据、packing、PEFT 配置和 SFTConfig；普通 Trainer 更通用，但很多 LLM SFT 细节要自己处理。

2 分钟版：

SFTTrainer 适合 prompt/response、messages、CausalLM SFT。它减少了你手写 dataset formatting、packing 和 PEFT 接入的工作。但它不是魔法，chat template、loss mask、max_length、EOS、packing 仍要检查。复杂任务仍可能需要自定义数据处理或 compute_loss。

## 610. Packing 是什么？有什么风险？

30 秒版：

Packing 把多个短样本拼进同一个 sequence，提高 token 利用率和训练吞吐。风险是样本边界、EOS、loss mask 和评估解释更复杂，不适合所有任务。

2 分钟版：

如果大量样本很短，不 packing 会让 padding 浪费很多 token。Packing 可以让一个 2048 长度序列装多个样本。但必须正确插入 EOS 或边界，避免模型把两个样本当成连续对话。多轮对话、工具调用或强格式任务要更谨慎。

## 611. PEFT / LoRA 在 Hugging Face 里怎么配置？

30 秒版：

用 `LoraConfig` 设置 `r`、`lora_alpha`、`target_modules`、`lora_dropout`、`bias`、`task_type`，再用 `get_peft_model` 包装 base model，并检查可训练参数。

2 分钟版：

示例：

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

面试重点是 target_modules。不同模型层名不同，没匹配上会导致几乎没有可训练参数。

## 612. LoRA 的 target_modules 怎么选？

30 秒版：

常见先选 attention 的 q/v 或 q/k/v/o；追求效果可加 MLP 的 gate/up/down；不确定结构时可用 all-linear，但显存和过拟合风险更高。最终靠验证集和 bad case 选。

2 分钟版：

选择策略：

- 小数据、低资源：q/v 或 q/k/v/o。
- 需要更强适配：attention + MLP。
- 领域差异大：考虑更高 rank 或更多模块。
- 分类头、扩词 embedding：用 modules_to_save。

加分点：先打印模型模块名，再确认 `print_trainable_parameters()`。

## 613. QLoRA 在 HF 里怎么落地？

30 秒版：

用 bitsandbytes 4-bit 加载 base model，常配 NF4、double quant、BF16 compute，再用 PEFT LoRA 训练 adapter。核心是 base 量化省显存，训练主要更新 LoRA。

2 分钟版：

典型配置：

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
```

注意量化 kernel、显存、速度、精度和保存/合并方式。QLoRA 不是全参微调，只更新 adapter。

## 614. Gradient checkpointing 和 `use_cache=False` 为什么常一起出现？

30 秒版：

Gradient checkpointing 训练时少存 activation、反向重算省显存；`use_cache=True` 是推理生成用 KV Cache，训练时会浪费显存并可能和 checkpointing 冲突，所以常设 `use_cache=False`。

2 分钟版：

训练需要反向传播，重点是 activation；推理生成需要 past key values，重点是 KV Cache。两者目标不同。SFT/LoRA 训练常：

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False
```

推理前再把 `use_cache=True`。

## 615. DPOTrainer / ORPOTrainer 的数据格式和核心逻辑是什么？

30 秒版：

DPOTrainer 通常需要 prompt、chosen、rejected。目标是让模型对 chosen 的相对 logprob 高于 rejected，同时通过 reference model 或隐式约束控制偏离程度。

ORPOTrainer 也通常需要 prompt、chosen、rejected，但它不需要 reference model。核心是把 SFT loss 和 odds-ratio preference loss 合在一起：chosen 用来做监督学习，chosen/rejected 用来做偏好拉开。

2 分钟版：

数据：

```json
{"prompt": "...", "chosen": "好回答", "rejected": "差回答"}
```

DPO 适合 SFT 后的偏好优化，比 PPO 简化。ORPO 适合想把 SFT 和偏好优化放在单阶段里、并减少 reference model 开销的场景。常见坑是 chosen/rejected 长度差异、偏好数据噪声、chat template 不一致、DPO 的 beta 或 ORPO 的 lambda 不合适，以及安全退化。评估不能只看偏好胜率，还要看任务正确率、安全和格式。

## 616. Adapter 保存、加载、合并怎么做？

30 秒版：

PEFT 通常保存 adapter 权重和配置；加载时先加载 base model，再加载 adapter；部署单一 adapter 时可 `merge_and_unload()` 合并成普通模型，但要注意许可证、量化和多 adapter 切换需求。

2 分钟版：

保存：

```python
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")
```

加载：

```python
base = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(base, "./adapter")
```

合并：

```python
model = model.merge_and_unload()
```

不合并适合多 adapter 动态切换；合并适合单模型部署简化。

## 617. `save_pretrained`、checkpoint、model card 分别是什么？

30 秒版：

`save_pretrained` 保存可复用模型/adapter/tokenizer；checkpoint 保存训练恢复状态；model card 记录模型来源、数据、用途、评估、限制和许可证。

2 分钟版：

checkpoint 要包含 optimizer、scheduler、scaler、step 等训练状态，用于 resume。`save_pretrained` 更偏发布和加载推理。model card 是交付文档，说明 base model、训练数据、指标、安全限制和使用边界。三者用途不同，不能混为一谈。

## 618. Trainer resume 失败或恢复后结果不一致怎么排查？

30 秒版：

查 checkpoint 是否完整、optimizer/scheduler/scaler 是否恢复、global step 是否对齐、数据顺序和 seed 是否变化、LoRA adapter 是否匹配 base model、TrainingArguments 是否改过。

2 分钟版：

排查：

```text
output_dir/checkpoint 是否存在
-> trainer state / optimizer / scheduler
-> adapter config 与 base model
-> tokenizer/chat template
-> max_steps/epochs 是否变化
-> dataset shuffle/seed
-> gradient accumulation step
```

恢复训练不是只加载权重。如果 scheduler 没恢复，学习率曲线会错；如果数据顺序变了，小差异会放大。

## 619. Hugging Face 微调常见 bug 怎么排查？

30 秒版：

先打印一条训练样本的原始 messages、渲染文本、input_ids、labels，再看 trainable parameters、单 batch overfit、loss/grad norm、eval bad case、checkpoint 和推理脚本是否一致。

2 分钟版：

典型链路：

```text
样本 schema
-> chat template
-> token truncation
-> labels mask
-> collator batch
-> trainable params
-> learning rate
-> eval generation
-> save/load
```

很多 bug 不会报错，只会训练出“看起来能说话但不听指令”的模型。最有效的动作是把一条样本完整 decode 出来，人眼检查。

## 620. HF 微调项目 8 分钟怎么讲？

30 秒版：

按背景、数据、训练、评估、难点、结果讲。一定要提 chat template、loss mask、PEFT/QLoRA、eval、adapter 保存和 bad case 回流。

2 分钟版：

模板：

```text
背景：通用模型在业务场景格式/术语/工具调用上不足。
数据：清洗 JSONL/messages，脱敏去重，统一 chat template，只训练 assistant token。
训练：用 SFTTrainer/Trainer + PEFT LoRA/QLoRA，设置 max_length、packing、bf16、checkpointing、eval/save steps。
评估：格式遵循、准确率、安全、人工偏好、bad case。
难点：模板错、mask 错、截断、target_modules、过拟合、DPO/ORPO 长度偏置、保存合并。
结果：指标提升、bad case 下降、adapter/model card 可复现交付。
```

背诵句：

> HF 微调项目的可信度来自可复现链路，而不是一句“我用 Trainer 训了 LoRA”。
