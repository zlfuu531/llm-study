# Hugging Face 微调工程：Transformers、PEFT、TRL 与 Datasets 面试

这一章面向大模型算法、微调、Post-training、应用算法和训练工程岗位。它补的是“真实微调脚本怎么搭”的工程主线：Tokenizer 和 chat template 怎么处理，Datasets 怎么清洗和 map，Trainer/SFTTrainer/DPOTrainer 什么时候用，PEFT/LoRA/QLoRA 怎么配，adapter 怎么保存和合并，以及线上评估和常见坑怎么排查。

如果时间很紧，先背这句：

> Hugging Face 微调不是 `trainer.train()` 一行代码，而是模型、tokenizer、chat template、数据字段、collator、loss mask、PEFT 配置、量化配置、TrainingArguments、评估、checkpoint 和 adapter 发布的一整条工程链路；面试要能说清每一层出了问题会怎么影响训练。

相关答案版：[answers/29_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets_答案版.md](answers/29_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets_答案版.md)

相邻章节：

- [26_开源模型生态_模型选型与ChatTemplate面试.md](26_开源模型生态_模型选型与ChatTemplate面试.md)：模型选型、tokenizer、chat template、迁移风险。
- [33_PyTorch训练工程_Autograd_DataLoader_AMP_DDP面试.md](33_PyTorch训练工程_Autograd_DataLoader_AMP_DDP面试.md)：训练循环、AMP、DDP、checkpoint 和训练排查。
- [35_Tokenizer_BPE_SentencePiece与Token预算面试.md](35_Tokenizer_BPE_SentencePiece与Token预算面试.md)：BPE/SentencePiece、special tokens、token budget、扩词表、SFT loss mask 和 tokenizer 排查。
- [37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md](37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md)：bitsandbytes、QLoRA、GPTQ/AWQ、FP8、量化加载和部署评估。
- [15_大模型数据工程与数据集构建面试.md](15_大模型数据工程与数据集构建面试.md)：SFT/偏好/RAG eval 数据构建。
- [modules/02_训练微调对齐_完整学习章.md](modules/02_训练微调对齐_完整学习章.md)：SFT、LoRA、QLoRA、DPO、GRPO 理论主线。

## 1. 微调工程总链路

真实项目不是从模型开始，而是从“任务和数据形态”开始：

```text
任务定义
-> 数据 schema
-> 清洗/去重/脱敏
-> chat template / prompt format
-> tokenize / truncation / packing
-> collator / labels mask
-> base model + tokenizer
-> PEFT / quantization / gradient checkpointing
-> Trainer / SFTTrainer / DPOTrainer
-> eval / logging / checkpoint
-> adapter save / merge / export
-> bad case 回流
```

面试时要先说清你做的是哪类任务：

| 任务 | 数据形态 | 常用工具 |
| --- | --- | --- |
| Causal LM SFT | prompt/response 或 messages | Transformers Trainer / TRL SFTTrainer |
| Chat SFT | system/user/assistant 多轮 | chat template + SFTTrainer |
| DPO/偏好优化 | prompt + chosen + rejected | TRL DPOTrainer |
| 分类/抽取 | text + label 或 instruction + label | Trainer / 自定义 loss |
| 领域继续预训练 | raw text | Causal LM + packing |
| PEFT/LoRA | 任意下游任务 | PEFT + Trainer/TRL |

一句话：

> 先把数据和目标定义清楚，再选 Trainer；不要用工具反推任务。

## 2. AutoTokenizer / AutoModel / AutoConfig

常见加载：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
```

常见坑：

- tokenizer 和 model 不匹配。
- pad token 缺失，batch padding 出问题。
- `model.config.pad_token_id` 没对齐 tokenizer。
- chat model 没用对应 chat template。
- base model 和 instruct model 混用。
- `trust_remote_code=True` 有安全风险，生产要审查来源。

pad token 处理：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

面试句：

> Tokenizer、chat template 和 special tokens 是模型接口的一部分，迁移模型时必须一起迁移。

## 3. Chat Template 和 SFT 数据格式

Chat 模型通常不是直接训练 `"用户: ... 助手: ..."`，而是把 messages 渲染成模型约定格式。

输入：

```python
messages = [
    {"role": "system", "content": "你是一个严谨助手。"},
    {"role": "user", "content": "解释 KV Cache"},
    {"role": "assistant", "content": "KV Cache 是..."},
]
```

渲染：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

训练时注意：

- SFT 要包含 assistant 答案。
- 推理时通常 `add_generation_prompt=True`，让模型知道开始回答。
- 多轮数据要保留 role 边界。
- tool call 数据要保留工具调用和工具返回的模板。
- 不同模型模板不同，不能直接复用同一 prompt。

如果 chat template 错：

```text
角色错乱
输出特殊 token
assistant 起始位置错
EOS 错
工具调用格式不稳
loss mask 错
```

## 4. Datasets：map、filter、shuffle、streaming

Hugging Face Datasets 常见流程：

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "train.jsonl", "valid": "valid.jsonl"})
ds = ds.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)
ds = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)
ds = ds.shuffle(seed=42)
```

`map` 常用于：

- 清洗字段。
- 应用 chat template。
- tokenize。
- 构造 labels。
- 截断/分桶/packing。

`batched=True` 通常更快：

```python
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=2048)
```

streaming 适合：

- 数据太大，不能完整下载或放入内存。
- 在线流式读取。
- 只做一遍或少量 epoch 的大规模预训练。

但 streaming 的 shuffle、random access、精确长度、resume 都更麻烦，面试要讲清取舍。

## 5. Data Collator 和 Labels Mask

Data collator 负责把样本列表拼成 batch。LLM SFT 最重要的是 labels mask。

常见规则：

```text
input_ids: 全部上下文 token
attention_mask: padding 位置为 0
labels: 只在需要学习的位置保留 token id，其余设为 -100
```

为什么设为 `-100`？

PyTorch CrossEntropyLoss 默认 `ignore_index=-100`，这些位置不算 loss。

指令微调常见 mask：

```text
system/user/tool observation: -100
assistant answer: token id
padding: -100
```

如果 mask 错：

- 模型学会复述 user prompt。
- 学 role token 或 padding。
- loss 很低但实际回答差。
- 多轮边界错，模型输出格式混乱。

面试句：

> SFT 的关键不是把文本拼起来，而是让模型只在应该学习的 assistant token 上算 loss。

## 6. Trainer 和 TrainingArguments

Transformers Trainer 提供 PyTorch 训练/评估循环，和 `TrainingArguments` 配合控制训练行为。

最小形态：

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./ckpt",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer,
)
trainer.train()
```

Trainer 适合：

- 标准 PyTorch/Transformers 模型。
- 常见监督训练。
- 需要快速接入分布式、AMP、logging、checkpoint。

Trainer 不适合无脑套：

- loss 很特殊。
- 多模型交互很复杂。
- 训练步骤需要强自定义。
- 数据动态生成或环境交互。

此时可以：

- 继承 Trainer，override `compute_loss`。
- 用 callback。
- 直接写 PyTorch loop。

## 7. SFTTrainer

TRL 的 SFTTrainer 更贴近 LLM 指令微调，能处理常见文本字段、chat 数据、packing、PEFT 配置等。

常见形态：

```python
from trl import SFTTrainer, SFTConfig

args = SFTConfig(
    output_dir="./sft",
    max_length=2048,
    packing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)
trainer.train()
```

packing 的取舍：

- 优点：把短样本拼进同一 sequence，提高 token 利用率。
- 风险：样本边界、EOS、loss mask、评估解释更复杂。
- 不适合所有任务，比如强依赖样本独立边界的任务要谨慎。

## 8. PEFT / LoRA 配置

LoRA 常用配置：

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

核心参数：

| 参数 | 含义 | 面试解释 |
| --- | --- | --- |
| `r` | LoRA rank | 可训练低秩子空间大小 |
| `lora_alpha` | 缩放 | 通常通过 `alpha/r` 或变体控制更新幅度 |
| `target_modules` | 注入层 | 常见 q/v/o/up/down/gate 等线性层 |
| `lora_dropout` | adapter dropout | 小数据防过拟合 |
| `bias` | 是否训练 bias | 通常 none，避免 base 行为变化 |
| `modules_to_save` | 额外保存模块 | 分类头、embedding 扩词等 |

target_modules 怎么选：

- Attention-only：省显存，常用 q/v 或 q/k/v/o。
- Attention + MLP：效果可能更好，训练参数更多。
- 全 linear：适合不确定结构时，但更吃显存。

## 9. QLoRA / bitsandbytes

QLoRA 常见目标：

```text
base model 4-bit 量化加载
LoRA adapter 保持可训练
反向只更新 adapter
```

典型配置：

```python
from transformers import BitsAndBytesConfig

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
```

常见配套：

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

面试要点：

- 4-bit base weights 省显存。
- LoRA adapter 是主要可训练部分。
- compute dtype 常用 BF16/FP16。
- double quantization 进一步省存储。
- 量化训练要注意精度、速度、kernel 和保存/合并方式。

## 10. Gradient Checkpointing 和 `use_cache=False`

训练 LLM 时常见：

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False
```

为什么？

- gradient checkpointing 通过 backward 重算减少 activation 显存。
- `use_cache=True` 主要服务生成推理，保存 past key values。
- 训练时开启 cache 可能和 checkpointing 冲突，也浪费显存。

推理时要恢复：

```python
model.config.use_cache = True
```

一句话：

> 训练看 activation 显存和反向传播，推理看 KV Cache 和生成速度；`use_cache` 是推理友好，不是训练必需。

## 11. DPOTrainer 和偏好数据

DPO 数据常见字段：

```json
{
  "prompt": "解释 KV Cache",
  "chosen": "更好的回答",
  "rejected": "较差的回答"
}
```

训练直觉：

```text
让模型对 chosen 的相对 logprob 高于 rejected
同时用 reference model 或隐式约束避免偏离 base 太远
```

TRL DPOTrainer 常用于：

- 已有 SFT 模型之后做偏好优化。
- 有人工/模型筛选的 chosen-rejected 数据。
- 希望避免 PPO 的复杂在线 RL 流程。

常见坑：

- chosen/rejected 长度差异导致长度偏置。
- 偏好数据噪声高。
- prompt 模板不一致。
- beta 太大或太小导致过强/过弱约束。
- 只看 win rate，不看安全和泛化退化。

## 12. 保存、加载、合并 Adapter

PEFT 常见保存：

```python
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")
```

加载 adapter：

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base, "./adapter")
```

合并 LoRA：

```python
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

什么时候合并？

- 推理部署希望减少 adapter 依赖和额外计算。
- 单一 adapter 长期使用。

什么时候不合并？

- 需要多 adapter 动态切换。
- 希望保留小体积 adapter。
- base 模型许可证/分发限制不允许直接发布合并权重。
- 量化模型合并可能受限制，要验证精度和加载方式。

## 13. Resume、Checkpoint 和 Hub 发布

Trainer 恢复：

```python
trainer.train(resume_from_checkpoint=True)
```

但要确认：

- output_dir 里 checkpoint 完整。
- optimizer/scheduler/scaler 状态存在。
- global step 对齐。
- 数据顺序和 seed 尽量一致。
- LoRA adapter checkpoint 和 base model 匹配。

发布时至少记录：

- base model。
- adapter 或 merged model。
- tokenizer/chat template。
- 训练数据来源和许可证。
- 超参。
- eval 结果。
- 安全限制。

Model Card 不是摆设，面试时可以说：

> 我会把训练数据、用途、限制、评估和许可证写进 model card，避免模型交付后没人知道它能不能用、该怎么用。

## 14. Trainer 常见坑

| 现象 | 可能原因 |
| --- | --- |
| loss 不降 | labels mask 错、学习率、数据格式、chat template 错 |
| 显存爆 | max_length 太大、packing 关、batch 大、没 checkpointing |
| eval 很慢 | 生成式评估太频繁、`predict_with_generate` 成本高 |
| 保存后不会聊天 | tokenizer/chat template 没保存或推理 prompt 不一致 |
| LoRA 没训练 | target_modules 没匹配、参数被冻结、`print_trainable_parameters` 没看 |
| resume 不对 | checkpoint 缺 optimizer/scheduler 或数据顺序变 |
| DPO 退化 | 偏好数据噪声、beta 不合适、长度偏置、安全退化 |

排查模板：

```text
数据样本打印
-> chat template 渲染后文本
-> token/labels 对齐
-> trainable params
-> 单 batch overfit
-> loss/grad norm
-> eval bad case
-> checkpoint 和推理脚本一致性
```

## 15. 项目 8 分钟讲法

```text
背景：
我们要把通用 instruct 模型适配到某业务场景，目标是格式遵循、领域术语、工具调用或客服回答质量提升。

数据：
先清洗 JSONL/messages，去重、脱敏、过滤低质样本，统一 chat template。SFT 只对 assistant 答案算 loss，padding 和 user/system token 设为 -100。

训练：
用 Transformers/TRL 组织训练，PEFT LoRA 或 QLoRA 降低显存。设置 max_length、packing、gradient accumulation、bf16、gradient checkpointing、eval/save/logging steps。

评估：
离线看格式遵循、准确率、拒答、安全、领域术语、人工偏好；线上灰度看解决率、转人工、投诉、延迟和成本。

难点：
chat template 不一致、loss mask 错、长样本截断、LoRA target_modules、过拟合、DPO 长度偏置、checkpoint 恢复和 adapter 合并。

结果：
用 bad case 回流迭代数据和超参，同时保留 base model、adapter、tokenizer、model card 和可复现实验记录。
```

## 16. 面试前背诵版

Hugging Face 微调工程要按“数据 -> 模板 -> token -> loss -> 训练 -> 保存 -> 评估”讲。Tokenizer 和 chat template 是模型接口的一部分，SFT 数据要渲染成目标模型格式，并用 labels mask 只训练 assistant token。Datasets 负责 map/filter/shuffle/streaming，DataCollator 负责 padding 和 batch。Trainer 适合标准训练，SFTTrainer 更贴近 LLM 指令微调，DPOTrainer 处理 chosen/rejected 偏好数据。PEFT/LoRA 通过 `LoraConfig` 选择 rank、alpha、target_modules 和 dropout，QLoRA 用 4-bit base model 加可训练 LoRA adapter 省显存。训练时常配 gradient checkpointing、`use_cache=False`、bf16、gradient accumulation。保存时要区分 adapter、merged model、tokenizer 和 model card。常见坑是 chat template 错、loss mask 错、target_modules 没匹配、packing/EOS 错、resume 不完整、DPO 数据噪声和推理脚本与训练不一致。

## 本轮参考

- Transformers Trainer 文档：[https://huggingface.co/docs/transformers/main_classes/trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- Transformers Chat Templates 文档：[https://huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating)
- Hugging Face Datasets process 文档：[https://huggingface.co/docs/datasets/process](https://huggingface.co/docs/datasets/process)
- PEFT LoRA 文档：[https://huggingface.co/docs/peft/package_reference/lora](https://huggingface.co/docs/peft/package_reference/lora)
- PEFT checkpoint format：[https://huggingface.co/docs/peft/developer_guides/checkpoint](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- TRL SFTTrainer 文档：[https://huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer)
- TRL DPOTrainer 文档：[https://huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer)
- Transformers bitsandbytes quantization 文档：[https://huggingface.co/docs/transformers/quantization/bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
