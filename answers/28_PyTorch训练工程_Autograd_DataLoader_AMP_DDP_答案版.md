# 答案版 28：PyTorch 训练工程、Autograd、DataLoader、AMP 与 DDP

对应题目：`03_高频题单100题.md` 的 581-600。

用法：先自己手写训练循环，再用这里的 30 秒版和 2 分钟版补细节。PyTorch 题不要只背 API 名，要能说清每一步为什么放在那里。

## 581. PyTorch 标准训练循环怎么写？

30 秒版：

训练循环是 `model.train()`，取 batch 到 device，forward 算 loss，`zero_grad` 清旧梯度，`backward` 求新梯度，必要时裁剪，`optimizer.step()` 更新参数，`scheduler.step()` 更新学习率，周期性 eval 和 checkpoint。

2 分钟版：

基本顺序：

```python
model.train()
for batch in loader:
    batch = move_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

如果有 gradient accumulation，不是每个 batch 都 step；如果有 AMP，要用 autocast 和 GradScaler；如果有 DDP，要用 DistributedSampler、rank0 保存和必要时 `no_sync()`。

## 582. `model.train()`、`model.eval()`、`torch.no_grad()` 有什么区别？

30 秒版：

`train()` 和 `eval()` 控制模块行为，比如 Dropout、BatchNorm；`no_grad()` 控制是否记录计算图。`eval()` 不会自动关闭 autograd，验证时通常 `model.eval()` + `with torch.no_grad()`。

2 分钟版：

`model.train()` 开启训练模式，Dropout 会随机丢，BatchNorm 会更新统计量。`model.eval()` 切评估模式，Dropout 关闭，BatchNorm 使用已有统计量。但这两个 API 不控制梯度图。

所以验证推荐写：

```python
model.eval()
with torch.no_grad():
    outputs = model(**batch)
```

纯推理且不把结果接回训练图，可以用 `torch.inference_mode()`，开销更低。

## 583. Autograd 动态计算图怎么理解？

30 秒版：

PyTorch 每次 forward 根据实际执行的 tensor 操作动态构建计算图，带 `requires_grad=True` 的 tensor 会记录 `grad_fn`，`backward()` 沿图反向传播，把梯度累积到叶子参数的 `.grad`。

2 分钟版：

动态计算图意味着 Python 分支、循环、变长输入都能自然构图。模型参数通常是 leaf tensor，参与计算后产生的中间 tensor 有 `grad_fn`。反向传播时 autograd 从 loss 出发按链式法则计算梯度。

注意点：

- `.grad` 主要在 leaf tensor 上有。
- `detach()` 会切断历史图。
- 默认 backward 后释放图，省显存。
- 多次 backward 才需要 `retain_graph=True`，但它可能导致显存涨。

## 584. 为什么要 `optimizer.zero_grad()`？

30 秒版：

因为 PyTorch 的梯度默认累积，新 backward 的梯度会加到已有 `.grad` 上。如果每轮更新前不清梯度，就会把多轮 batch 的梯度混在一起。

2 分钟版：

默认行为是：

```text
param.grad = param.grad + new_grad
```

这支持多 loss backward 和梯度累积，但普通训练每次 optimizer step 后要清梯度。常见写法：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

`set_to_none=True` 通常更省内存，也能更容易发现某些参数没有收到梯度。

## 585. 梯度累积怎么写？loss 为什么要除以 accumulation steps？

30 秒版：

梯度累积把多个 micro-batch 的梯度加起来后再更新一次。为了让梯度尺度和大 batch 平均 loss 一致，每个 micro-batch 的 loss 要除以 `grad_accum_steps`。

2 分钟版：

写法：

```python
loss = loss / grad_accum_steps
loss.backward()
if (step + 1) % grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

有效 batch：

```text
effective_batch = micro_batch * grad_accum_steps * world_size
```

如果忘了除以 accumulation steps，梯度会变大，相当于学习率变大，可能不稳定。DDP 下还要配合 `no_sync()` 减少不必要通信。

## 586. DataLoader、Dataset、Sampler、collate_fn 分别做什么？

30 秒版：

Dataset 定义样本怎么取，Sampler 定义样本顺序，DataLoader 负责多进程加载和组 batch，collate_fn 把样本列表拼成模型需要的 batch，比如 padding、attention mask、label mask。

2 分钟版：

Map-style Dataset 实现 `__len__` 和 `__getitem__`，适合随机访问；IterableDataset 适合流式数据。Sampler 决定索引顺序，DDP 用 DistributedSampler 保证不同 rank 拿不同数据。collate_fn 处理变长样本。

LLM 训练里 collate_fn 很关键：

- 动态 padding。
- 构造 attention_mask。
- labels 里 padding 位置设为 `-100`。
- packing 或截断。

## 587. DataLoader 慢怎么排查？

30 秒版：

先记录 data time 和 step time。如果 GPU 等数据，就看 CPU 预处理、磁盘/网络 IO、num_workers、pin_memory、collate_fn、样本长度差异和 H2D copy。

2 分钟版：

排查路径：

```text
DataLoader 取 batch 时间
-> collate_fn 是否太慢
-> CPU tokenizer/图像解码是否在线做
-> num_workers 是否合适
-> pin_memory + non_blocking
-> 数据是否在网络盘
-> batch shape 是否抖动
```

常见优化：

- 提前 tokenize/cache。
- 增加 `num_workers`，但不要盲目过大。
- `pin_memory=True`。
- `persistent_workers=True`。
- 减少 Python 重逻辑。
- 按长度 bucket，减少 padding 浪费。

## 588. AMP / autocast / GradScaler 怎么用？

30 秒版：

AMP 用 autocast 在 forward 中自动选择低精度算子，省显存提吞吐。FP16 常用 GradScaler 做 loss scaling 防止梯度 underflow；梯度裁剪前要先 unscale。

2 分钟版：

典型写法：

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    loss = model(**batch).loss
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

BF16 动态范围更大，很多训练可不需要 scaler。autocast 通常包 forward 和 loss，不包 optimizer step。

## 589. FP16 和 BF16 有什么区别？

30 秒版：

FP16 精度位更多但指数范围小，容易溢出或下溢，所以常配 GradScaler；BF16 指数范围接近 FP32，训练更稳，但尾数精度少一些，依赖硬件支持。

2 分钟版：

FP16 更容易出现小梯度 underflow 或大值 overflow。GradScaler 通过放大 loss 来让梯度落在可表示范围。BF16 保留了更大的指数范围，大模型训练常更稳定，很多场景不需要 loss scaling。

选型：

- 新 GPU 支持 BF16 时，训练大模型常优先 BF16。
- 老硬件或特定 kernel 可能仍用 FP16。
- 不管哪种，都要看 loss、grad norm、吞吐和最终精度。

## 590. 梯度裁剪解决什么问题？

30 秒版：

梯度裁剪限制梯度 norm，防止某一步更新过大导致 loss 爆炸或 NaN。它是稳定训练的保险，不是根治手段，根因还要查学习率、数据、loss mask 和混合精度。

2 分钟版：

常用写法：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

AMP 下要先：

```python
scaler.unscale_(optimizer)
clip_grad_norm_(...)
```

如果裁剪频繁触发，说明训练本身可能不稳。继续排查 learning rate、warmup、异常样本、logits 范围、loss mask、梯度累积尺度。

## 591. Checkpoint 要保存哪些东西？

30 秒版：

恢复训练不只保存模型。要保存 model、optimizer、scheduler、scaler、global step、epoch、rng state、config、tokenizer/chat template 和数据版本。

2 分钟版：

基本结构：

```python
{
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "scheduler": scheduler.state_dict(),
  "scaler": scaler.state_dict(),
  "step": global_step,
  "rng_state": torch.get_rng_state(),
  "config": config,
}
```

多卡训练通常 rank0 保存，或者用分片 checkpoint。上线要记录代码版本、依赖版本、数据版本，否则 resume 后看似继续训练，实际实验不可复现。

## 592. `state_dict` 和直接保存整个 model 有什么区别？

30 秒版：

`state_dict` 只保存参数和 buffer，更稳定、可移植；直接保存整个 model 依赖 Python 类定义和代码路径，容易因为代码变化加载失败。生产里通常保存 `state_dict`。

2 分钟版：

推荐：

```python
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

完整训练恢复还要保存 optimizer/scheduler/scaler 等 state_dict。直接 `torch.save(model)` 会序列化对象，和代码结构强绑定，不适合长期维护和跨环境加载。

## 593. Activation checkpointing 为什么省显存？

30 秒版：

普通训练 forward 会保存很多中间激活供 backward 使用；activation checkpointing 少保存激活，在 backward 时重算部分 forward，用计算换显存。

2 分钟版：

它省的是 activation，不省参数、梯度和 optimizer state。适合长上下文、大 batch、深层 Transformer。代价是训练变慢，因为 backward 里要重算。checkpoint 粒度要权衡：粒度太小重算多，粒度太大省得少。

面试句：

> Checkpointing 是显存和计算的交换，不是免费午餐。

## 594. DDP 的核心流程是什么？

30 秒版：

DDP 每张卡一份完整模型，每个 rank 处理不同数据。forward 各算各的，backward 时梯度 all-reduce 同步，每个 rank 用同步后的梯度各自 optimizer step，所以参数保持一致。

2 分钟版：

流程：

```text
init_process_group
-> set local_rank device
-> model.to(device)
-> DDP(model)
-> DistributedSampler
-> forward/backward
-> gradient all-reduce
-> optimizer.step on each rank
```

注意：

- DDP 解决 batch 维并行，不解决单卡放不下模型。
- 日志和 checkpoint 通常 rank0 做。
- `sampler.set_epoch(epoch)` 保证每个 epoch shuffle 变化。

## 595. DistributedSampler 为什么要 `set_epoch()`？

30 秒版：

DistributedSampler 每个 epoch 要用不同随机种子打乱数据。调用 `set_epoch(epoch)` 能让各 rank 在当前 epoch 使用一致但变化的 shuffle，否则每个 epoch 可能顺序相同。

2 分钟版：

DDP 里不同 rank 要拿到不重叠的数据切片，同时整体 shuffle 要一致可控。`set_epoch()` 把 epoch 编进随机种子，让每个 epoch 的 shuffle 不同，但各 rank 仍能协同切分。

典型写法：

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

## 596. DDP + 梯度累积为什么要 `no_sync()`？

30 秒版：

DDP 默认每次 backward 都同步梯度。梯度累积时前几个 micro-batch 不需要同步，用 `no_sync()` 可以只在真正 optimizer step 前同步一次，减少通信开销。

2 分钟版：

写法：

```python
sync = (step + 1) % grad_accum_steps == 0
ctx = model.no_sync() if not sync else nullcontext()
with ctx:
    loss = loss / grad_accum_steps
    loss.backward()
if sync:
    optimizer.step()
```

注意 scheduler、clip、optimizer.step 都按真实更新步执行，而不是每个 micro-step。

## 597. 训练出现 NaN 怎么排查？

30 秒版：

先查输入和 label 是否有 NaN/Inf，再查 loss mask、学习率、AMP overflow、logits 是否爆炸、梯度 norm、异常 batch、手写 loss 数值稳定性。

2 分钟版：

排查顺序：

```text
数据 NaN/Inf
-> 第一轮 loss 是否正常
-> logits 范围
-> loss mask/padding
-> lr/warmup
-> AMP scaler 是否频繁降
-> grad norm 哪层异常
-> 最近改动的 kernel/loss
```

常用处理：降学习率、加 warmup、梯度裁剪、BF16 替代 FP16、检查 label、用稳定版 CE/BCEWithLogitsLoss。

## 598. 训练 OOM 怎么排查？

30 秒版：

先分清显存来自参数、梯度、optimizer state、activation、临时 buffer、碎片还是长序列缓存。处理不只是减 batch，还包括 AMP、gradient accumulation、checkpointing、FlashAttention、LoRA/QLoRA、ZeRO/FSDP。

2 分钟版：

排查：

```text
模型参数多吗
-> Adam 状态是否大
-> seq length/batch 导致 activation 大吗
-> 是否保存了带图 loss/output
-> 是否有临时大 tensor
-> 是否 DDP 每卡完整模型放不下
```

典型 bug：

```python
losses.append(loss)        # 保存计算图
losses.append(loss.item()) # 只保存数值
```

## 599. 多卡训练速度不线性提升怎么办？

30 秒版：

多卡不线性很正常。看 DataLoader、batch 是否太小、通信 all-reduce 占比、梯度 bucket、NCCL overlap、网络拓扑、负载不均、checkpoint/logging 和评估是否阻塞。

2 分钟版：

排查：

1. 先看单卡 step time 和多卡 step time 分解。
2. 看 data time 是否每卡都慢。
3. 看 NCCL/all-reduce 占比。
4. 看 batch 太小导致计算无法覆盖通信。
5. 看 gradient accumulation 和 `no_sync()`。
6. 看是否 rank0 logging/checkpoint 影响所有 rank。
7. 看样本长度导致某些 rank 慢。

面试句：

> 多卡慢要用 timeline 看通信、计算和数据加载，不要只看 GPU utilization。

## 600. PyTorch 训练工程面试前最后怎么复习？

30 秒版：

最后背一条训练循环、一条 autograd 主线、一个 AMP 模板、一个 DDP 模板、一个 checkpoint 清单，以及 NaN/OOM/慢训练三套排查链路。

2 分钟版：

清单：

- 手写 train/eval loop。
- 解释 `train/eval/no_grad/inference_mode`。
- 解释 autograd、leaf tensor、grad accumulation。
- 写 DataLoader/collate_fn 的职责。
- 写 AMP + GradScaler + clip 的顺序。
- 说明 FP16 vs BF16。
- 说明 checkpoint 保存内容。
- 说明 DDP all-reduce、DistributedSampler、set_epoch。
- 说明 DDP + accumulation 的 `no_sync()`。
- 准备 NaN、OOM、训练慢、多卡不加速的排查。

背诵句：

> PyTorch 工程题的重点不是 API 背诵，而是训练状态、梯度状态、数据状态、分布式状态能否对齐。
