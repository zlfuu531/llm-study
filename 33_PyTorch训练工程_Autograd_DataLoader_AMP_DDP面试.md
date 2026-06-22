# PyTorch 训练工程、Autograd、DataLoader、AMP 与 DDP 面试

这一章面向大模型算法、训练工程、微调、Post-training、AI Infra 和需要手写训练循环的岗位。它补的是“能不能把模型真正训起来”的基本功：不是只会说 Transformer，也不是只会调用 `Trainer`，而是能解释 autograd、数据管线、混合精度、梯度累积、DDP、checkpoint、复现和训练故障排查。

如果时间很紧，先背这句：

> PyTorch 训练循环的主线是：数据进入 DataLoader，模型在 `train()` 模式下前向计算 loss，autograd 动态构图并在 `backward()` 时沿图求梯度，梯度累积到参数的 `.grad`，optimizer 根据 `.grad` 更新参数；生产训练还要处理 AMP、梯度裁剪、scheduler、checkpoint、DDP 同步、日志和异常恢复。

相关答案版：[answers/28_PyTorch训练工程_Autograd_DataLoader_AMP_DDP_答案版.md](answers/28_PyTorch训练工程_Autograd_DataLoader_AMP_DDP_答案版.md)

相邻章节：

- [17_大模型训练系统与分布式训练面试.md](17_大模型训练系统与分布式训练面试.md)：显存、DDP、ZeRO/FSDP、3D 并行、通信。
- [34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md](34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md)：Trainer、SFTTrainer、PEFT/LoRA/QLoRA、DPOTrainer/ORPOTrainer 和 adapter 保存合并。
- [25_GPU_CUDA_Triton与FlashAttention面试.md](25_GPU_CUDA_Triton与FlashAttention面试.md)：GPU profiler、kernel、NCCL、torch.compile。
- [28_ML_DL数学基础_损失函数优化器与指标面试.md](28_ML_DL数学基础_损失函数优化器与指标面试.md)：loss、optimizer、metric、校准和排查。
- [modules/05_手撕代码_完整学习章.md](modules/05_手撕代码_完整学习章.md)：LLM 相关手写题。

## 1. 一个合格训练循环长什么样

先看最小但不偷懒的版本：

```python
import torch
from torch.nn.utils import clip_grad_norm_

def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler=None,
                    grad_accum_steps=1, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=scaler is not None):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step + 1) % grad_accum_steps == 0
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(len(dataloader), 1)
```

面试官常追的点：

- `model.train()` 会影响 Dropout、BatchNorm 等层。
- `optimizer.zero_grad()` 要在下一轮 backward 前清梯度，因为梯度默认累积。
- 梯度累积时要把 loss 除以 `grad_accum_steps`，否则等效学习率变大。
- AMP 下 backward 前 loss 被 scale，梯度裁剪前要先 `unscale_`。
- `scheduler.step()` 放在每次 optimizer 更新后，而不是每个 micro-batch 后。
- `set_to_none=True` 通常更省内存，也能暴露未产生梯度的问题。

## 2. `train()`、`eval()`、`no_grad()`、`inference_mode()`

这四个概念容易混。

| API | 作用 | 是否影响 autograd | 典型场景 |
| --- | --- | --- | --- |
| `model.train()` | 切训练模式 | 不直接关闭梯度 | 训练 |
| `model.eval()` | 切评估模式 | 不直接关闭梯度 | 验证/推理 |
| `torch.no_grad()` | 不记录梯度图 | 关闭反向图构建 | 验证/推理 |
| `torch.inference_mode()` | 更强的推理模式 | 关闭梯度并减少额外开销 | 纯推理 |

常见误区：

```python
model.eval()
outputs = model(x)  # 仍然可能构建计算图
```

更稳的评估写法：

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        outputs = model(**batch)
```

如果确定不会把张量拿回训练图里，可以用：

```python
with torch.inference_mode():
    outputs = model(**batch)
```

面试句：

> `eval()` 管模块行为，`no_grad()` 管是否构图；两者解决的问题不同，验证时通常一起用。

## 3. Autograd 动态计算图

PyTorch autograd 是动态计算图。每次 forward 都按实际 Python 控制流构图，backward 时沿图反向传播。

关键概念：

- `requires_grad=True`：这个 tensor 需要梯度。
- `grad_fn`：记录这个 tensor 是由哪个操作产生的。
- leaf tensor：通常是用户创建的参数或输入；模型参数是 leaf。
- `.grad`：backward 后梯度累积的位置，主要出现在 leaf tensor 上。
- `detach()`：切断当前 tensor 和历史计算图的关系。
- `retain_graph=True`：保留图以便多次 backward，默认不需要。

简单例子：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * x + 3 * x
y.backward()
print(x.grad)  # 2*x + 3 = 7
```

为什么 `backward()` 后默认释放图？

```text
计算图里的中间激活会占显存。
大多数训练只需要一次 backward。
释放图可以及时回收内存。
```

什么时候需要 `retain_graph=True`？

- 同一张图上要反向多次。
- 多个 loss 共享中间图但没有先合并成一个 loss。

但面试里要主动说：

> `retain_graph=True` 经常是显存泄漏的信号。能把 loss 合并后一次 backward，就不要保留图。

## 4. 梯度为什么会累积

PyTorch 默认把新梯度加到参数已有的 `.grad` 上：

```text
param.grad = param.grad + new_grad
```

所以每轮更新后要清梯度：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

为什么默认累积是合理的？

- 支持多 loss 分开 backward。
- 支持 gradient accumulation。
- 支持手动控制复杂训练流程。

梯度累积的标准写法：

```python
loss = loss / grad_accum_steps
loss.backward()
if (step + 1) % grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

有效 batch size：

```text
effective_batch = micro_batch_per_gpu * grad_accum_steps * world_size
```

如果保持 optimizer step 次数不变，增大有效 batch 往往需要同步调整学习率、warmup 和训练步数。

## 5. DataLoader 和 `collate_fn`

训练慢不一定是 GPU 算得慢，也可能是数据喂不动。

PyTorch 数据管线常见组件：

```text
Dataset / IterableDataset
-> Sampler / DistributedSampler
-> DataLoader workers
-> collate_fn
-> host memory
-> device transfer
-> model forward
```

`Dataset` 适合随机访问：

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
```

`IterableDataset` 适合流式数据、超大文件、在线样本：

```python
class StreamDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for sample in stream():
            yield sample
```

`collate_fn` 负责把多个样本拼成 batch。NLP/LLM 常用它做 padding、attention mask、labels mask：

```python
def collate_fn(samples, pad_id=0):
    max_len = max(len(s["input_ids"]) for s in samples)
    input_ids, attention_mask = [], []
    for s in samples:
        ids = s["input_ids"]
        pad = [pad_id] * (max_len - len(ids))
        input_ids.append(ids + pad)
        attention_mask.append([1] * len(ids) + [0] * len(pad))
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }
```

常用参数：

- `num_workers`：多进程加载数据，过大可能抢 CPU/内存。
- `pin_memory=True`：页锁定内存，配合 `non_blocking=True` 加速 CPU 到 GPU 拷贝。
- `persistent_workers=True`：epoch 间保留 worker，减少重启开销。
- `prefetch_factor`：每个 worker 预取 batch 数。
- `drop_last=True`：让 batch shape 更稳定，DDP 中也常用。

## 6. AMP、FP16、BF16 和 GradScaler

混合精度的目标是省显存、提吞吐，同时尽量保持训练稳定。

常见 dtype：

| dtype | 优点 | 风险 |
| --- | --- | --- |
| FP32 | 稳定 | 慢、显存大 |
| FP16 | 省显存、Tensor Core 友好 | 动态范围小，容易 underflow/overflow |
| BF16 | 动态范围接近 FP32，训练更稳 | 部分硬件支持要求 |
| FP8 | 更省更快 | 训练和校准更复杂，硬件/kernel 依赖强 |

推荐写法使用 `torch.amp`：

```python
scaler = torch.amp.GradScaler("cuda")

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

如果用 BF16，很多场景不需要 GradScaler：

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss
loss.backward()
optimizer.step()
```

面试追问：

- autocast 通常包 forward 和 loss 计算，不包 optimizer step。
- FP16 需要 loss scaling，因为小梯度可能 underflow。
- `GradScaler` 会动态调整 scale，遇到 inf/nan 会跳过 step 并降低 scale。
- 梯度裁剪前要 `unscale_`，否则裁剪的是放大后的梯度。

## 7. 梯度裁剪、NaN 和训练稳定性

梯度裁剪常用于大模型训练和 RNN/Transformer 稳定训练：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

它解决的是“更新步子过大”的症状，不是万能药。NaN 常见原因：

- 学习率过大。
- FP16 overflow。
- loss mask 错，padding 或无效 token 参与 loss。
- logits 太大，手写 softmax 不稳定。
- 数据里有 NaN/Inf。
- 除零、log(0)、sqrt 负数。
- 梯度累积时忘了除以 accumulation steps。

排查顺序：

```text
输入数据是否 NaN/Inf
-> loss 是否第一步就异常
-> logits 范围是否爆炸
-> AMP scale 是否频繁下降
-> 学习率/warmup 是否过激
-> loss mask 是否正确
-> 梯度 norm 哪一层开始异常
```

## 8. Checkpoint 保存什么

只保存模型参数不够恢复训练。完整 checkpoint 通常包括：

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "scaler": scaler.state_dict() if scaler else None,
    "step": global_step,
    "epoch": epoch,
    "rng_state": torch.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state_all(),
    "config": config,
}
torch.save(ckpt, path)
```

恢复：

```python
ckpt = torch.load(path, map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
if scheduler and ckpt["scheduler"]:
    scheduler.load_state_dict(ckpt["scheduler"])
if scaler and ckpt["scaler"]:
    scaler.load_state_dict(ckpt["scaler"])
```

大模型训练还要考虑：

- rank0 保存，避免多进程同时写。
- 分片 checkpoint，避免单文件太大。
- 原子写入：先写临时文件，再 rename。
- 保留 latest 和 best。
- 记录 tokenizer、chat template、数据版本、代码 commit、超参。
- 恢复后 global step、scheduler、gradient accumulation 对齐。

## 9. Activation Checkpointing

Activation checkpointing 是用计算换显存：

```text
普通训练：forward 保存大量中间激活，backward 直接用。
checkpoint：forward 少保存，backward 时重算部分 forward。
```

适合：

- 长上下文。
- 大 batch。
- 深层 Transformer。
- activation 显存超过参数/优化器状态的场景。

代价：

- 训练更慢，因为 backward 里要重算 forward。
- 随机层和 dropout 要处理 RNG 状态。
- checkpoint 粒度影响收益和开销。

一句话：

> Checkpointing 不减少参数、梯度、optimizer state，它减少的是 forward 保存的 activation。

## 10. DDP 的最小正确姿势

DDP 是数据并行：每张卡一份完整模型，不同卡处理不同 mini-batch，backward 时同步梯度。

典型流程：

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)

device = setup()
model = MyModel().to(device)
model = DDP(model, device_ids=[device.index])

sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

DDP 面试必答：

- 每个 rank 有完整模型副本。
- 前向各算各的 batch。
- backward 时梯度通过 all-reduce 同步。
- optimizer 在每个 rank 上各自 step，但因为梯度同步，所以参数保持一致。
- `DistributedSampler.set_epoch(epoch)` 保证每个 epoch 的 shuffle 不一样。
- 日志和 checkpoint 通常只在 rank0 做。

DDP 不能解决：

- 单卡放不下模型参数的问题。
- 单层太大、attention activation 太大。

这些要用 ZeRO/FSDP、tensor parallel、pipeline parallel、activation checkpointing、LoRA/QLoRA 等。

## 11. DDP + 梯度累积

DDP 默认每次 backward 都同步梯度。梯度累积时，前几个 micro-step 不想同步，可以用 `no_sync()`：

```python
for micro_step, batch in enumerate(loader):
    sync = (micro_step + 1) % grad_accum_steps == 0
    ctx = model.no_sync() if not sync else contextlib.nullcontext()
    with ctx:
        loss = model(**batch).loss / grad_accum_steps
        loss.backward()

    if sync:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

为什么重要？

- 不用 `no_sync()` 时每个 micro-batch 都 all-reduce，通信开销大。
- 用 `no_sync()` 后只在真正 optimizer step 前同步一次。

注意：

- 最后不足 `grad_accum_steps` 的 batch 要处理。
- loss scaling、梯度裁剪、scheduler step 都要按真实 optimizer step 对齐。

## 12. 复现和随机性

复现训练结果常见动作：

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

还可以设置：

```python
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

但要诚实说明：

- 完全可复现会牺牲性能。
- 某些 GPU kernel 天然非确定。
- DDP 里数据顺序、worker seed、通信顺序都会影响细节。
- LLM 训练通常更关注可追踪和可回放，而不是 bitwise 完全一致。

项目里至少要记录：

- 代码版本。
- 数据版本。
- tokenizer/chat template。
- 模型初始化。
- 超参。
- seed。
- checkpoint。
- 环境和依赖版本。

## 13. 训练变慢怎么排查

训练慢要先定位瓶颈：

```text
step time = data time + H2D copy + forward + backward + optimizer + communication + logging/checkpoint
```

排查工具：

- 简单计时：记录 data_time、forward_time、backward_time、step_time。
- PyTorch Profiler：看 CPU/GPU op、CUDA kernel、memory、shape。
- Nsight Systems：看 CPU/GPU timeline、kernel launch、NCCL overlap。
- nvidia-smi：只能粗看 GPU utilization 和显存，不够定位根因。

常见现象：

| 现象 | 可能原因 |
| --- | --- |
| GPU 利用率低 | DataLoader 慢、CPU 预处理慢、H2D copy 阻塞 |
| step time 抖动大 | 数据样本长度差异、动态 padding、checkpoint、网络存储 |
| backward 很慢 | activation 大、checkpoint 重算、通信等待 |
| optimizer 慢 | 参数多、Adam 状态大、CPU offload |
| 多卡不加速 | batch 太小、通信占比高、DataLoader 每卡都慢 |

## 14. OOM 怎么排查

OOM 不要只说“减 batch size”。先判断是哪类显存：

```text
参数: params
梯度: grads
优化器状态: Adam m/v/master weights
激活: activations
临时 buffer / workspace
碎片
KV cache 或长序列缓存
```

常见处理：

- 减 micro-batch 或 sequence length。
- gradient accumulation 保持有效 batch。
- AMP / BF16。
- activation checkpointing。
- FlashAttention。
- LoRA/QLoRA。
- ZeRO/FSDP。
- 清理无用 tensor 引用，避免把 loss/list 保存整张图。
- `torch.cuda.empty_cache()` 不是根治，只是释放缓存给 allocator。

典型内存泄漏写法：

```python
losses.append(loss)        # 错：保存了带计算图的 tensor
losses.append(loss.item()) # 对：只保存数值
```

## 15. 手写训练循环面试模板

面试官让你“手写一个训练循环”，可以按这个顺序写：

```python
model.to(device)
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

for epoch in range(num_epochs):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader):
        batch = move_to_device(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            loss = model(**batch).loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        evaluate(model, val_loader)
```

然后主动补充：

- 如果用 BF16，可不用 GradScaler。
- 如果 DDP + gradient accumulation，非同步 micro-step 用 `no_sync()`。
- checkpoint 要保存 model/optimizer/scheduler/scaler/global_step/config。
- 验证时 `eval()` 和 `no_grad()` 都要用。
- loss mask、padding、DDP sampler、rank0 save/log 都是常见坑。

## 16. 高频追问答题骨架

### Q: 为什么验证集显存还很高？

可能只写了 `model.eval()`，但没有 `torch.no_grad()`，仍然在构建计算图。也可能把输出、loss 或 hidden states 放进列表保存，导致张量引用不释放。验证时应使用：

```python
model.eval()
with torch.no_grad():
    ...
```

### Q: 为什么 loss 没降？

按链路排查：

```text
数据和 label
-> loss mask
-> model.train/eval 模式
-> requires_grad / optimizer param groups
-> 学习率和 scheduler
-> 梯度是否为 0 或 NaN
-> AMP scale
-> batch/accumulation
-> 评估代码是否一致
```

### Q: 为什么多卡结果和单卡不同？

可能原因：

- 有效 batch size 变了。
- 学习率和 warmup 没按 step 对齐。
- DistributedSampler shuffle 不同。
- BatchNorm/Dropout/随机数。
- 梯度累积同步时机不同。
- 非确定 kernel。

面试回答要说：

> 我会先对齐有效 batch、optimizer step 数、数据顺序和 seed，再比较单步 loss/grad norm/参数更新。

## 17. 面试前背诵版

PyTorch 训练工程的核心是训练循环和 autograd。`model.train()` 控制模块行为，`model.eval()` 不等于关闭梯度，验证还要 `torch.no_grad()` 或 `inference_mode()`。Autograd 每次 forward 动态构图，`backward()` 后梯度累积到参数 `.grad`，所以要清梯度。梯度累积时 loss 要除以 accumulation steps，有效 batch 等于 micro-batch 乘以 accumulation 乘以 world size。DataLoader 里 Dataset、Sampler、collate_fn、num_workers、pin_memory 会影响吞吐。AMP 用 autocast 降精度计算，FP16 常配 GradScaler，梯度裁剪前要 unscale。Checkpoint 恢复训练要保存 model、optimizer、scheduler、scaler、step、rng 和 config。DDP 每卡完整模型，backward all-reduce 同步梯度，`DistributedSampler.set_epoch()`、rank0 保存日志、`no_sync()` 配合梯度累积都是常见考点。训练慢、OOM、NaN 要按数据、计算、通信、显存和数值稳定分层排查。

## 本轮参考

- PyTorch Autograd mechanics：[https://docs.pytorch.org/docs/stable/notes/autograd.html](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- PyTorch Automatic Mixed Precision：[https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html)
- PyTorch DataLoader：[https://docs.pytorch.org/docs/stable/data.html](https://docs.pytorch.org/docs/stable/data.html)
- PyTorch DistributedDataParallel：[https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- PyTorch DistributedSampler：[https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
- PyTorch Activation Checkpointing：[https://docs.pytorch.org/docs/stable/checkpoint.html](https://docs.pytorch.org/docs/stable/checkpoint.html)
- PyTorch Profiler：[https://docs.pytorch.org/docs/stable/profiler.html](https://docs.pytorch.org/docs/stable/profiler.html)
