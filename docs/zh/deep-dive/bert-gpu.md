# BERT GPU 保持 & 批量推理

GPU 路径中有两个主要优化点。

- **消除原版 SBV2 不必要的 CPU 转换**,将 BERT 输出保持为 GPU 张量
- **将多句打包为批次仅调用 1 次 BERT**,减少 kernel launch overhead(每次向 GPU 请求运算时产生的固定成本)

## 为什么是问题

原版 SBV2 基本上 **将文本整体一次性合成** (`line_split=False`)。

BERT 也只调用 1 次,不需要批量化。

HayaKoe 为了 prosody(韵律)稳定性引入了 **按标点句子分割**,由此产生了 BERT 按句子数重复调用的新问题。

### BERT 输出的不必要 CPU 转换

原版 SBV2 的 BERT feature 提取代码中有以下部分。

```python
# 原版 SBV2 (style_bert_vits2/nlp/japanese/bert_feature.py)
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```

BERT 在 GPU 上 forward 后,对输出张量调用 `.cpu()` **每次都转到 CPU**。

此输出之后传递给 Synthesizer,而 Synthesizer 在 GPU 上运行所以又要转回 GPU。

结果每个句子都产生 **GPU → CPU → GPU 往返**,这个不必要的往返本身就成为瓶颈。

### 逐句单独调用 BERT

多句分割后对每个句子单独调用 BERT 时,GPU kernel launch 按句子数重复。

kernel launch 是每次向 GPU 请求运算时产生的固定成本。

句子短时实际计算时间比 launch overhead 占比小,随句子数累积效率低下。

## 实现

### 移除 `.cpu()` — 保持 GPU 张量

移除原版的 `.cpu()` 调用,使 BERT 输出作为 GPU 张量直接传递给 Synthesizer。

```python
# 原版 SBV2
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()    # GPU → CPU

# HayaKoe
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].float()  # 保持 GPU
```

BERT 模型本身也在 `prepare()` 时加载到 GPU,直到推理结束保持不变。

BERT 模型作为 **全局单例** 管理,加载多个说话人时 BERT 也只加载一次,所有说话人共享。

### 多句 BERT 批量化

HayaKoe 使用的 BERT (DeBERTa) 是 HuggingFace Transformer 模型,天然支持 batch 输入。

利用此特性,多句合成时不逐句调用 BERT,而是 **将所有句子打包为一个批次 1 次处理**。

将多个句子一起送入 tokenizer 制作 padding 后的批次输入,BERT **仅 forward 1 次**。

ONNX 路径中也实现了相同的批量逻辑。

## 改善效果

### GPU 批量推理速度

同一硬件上 **顺序 (sequential) vs 批量 (batched)** 对比(5 次平均)。

| 句子数 | 顺序 | 批量 | 速度提升 |
|---|---|---|---|
| 2 | 0.447 s | 0.364 s | **1.23x** |
| 4 | 0.812 s | 0.566 s | **1.43x** |
| 8 | 1.598 s | 1.121 s | **1.43x** |
| 16 | 2.972 s | 2.264 s | **1.31x** |

kernel launch overhead 合并为 1 次的效果,带来 +23% ~ +43% 的速度提升。

### GPU 内存

确认批量化是否额外消耗内存。

| 句子数 | 顺序 peak | 批量 peak | 差异 |
|---|---|---|---|
| 2 | 1,662.2 MB | 1,661.9 MB | -0.3 MB |
| 4 | 1,661.8 MB | 1,662.2 MB | +0.4 MB |
| 8 | 1,697.7 MB | 1,699.0 MB | +1.3 MB |
| 16 | 1,934.3 MB | 1,934.3 MB | 0 MB |

顺序和批量间差异 **在 1.3 MB 以内**,实质上相同。

### CPU 上无效果

在 CPU (ONNX) 上重复同一实验,批量化效果几乎不出现。

| 句子数 | 顺序 | 批量 | 速度差异 |
|---|---|---|---|
| 2 | 2.566 s | 2.564 s | 1.00x |
| 4 | 5.464 s | 4.855 s | 1.13x |
| 8 | 10.647 s | 11.783 s | 0.90x |
| 16 | 24.559 s | 24.195 s | 1.01x |

ONNX Runtime 的图优化已经足够强,Python 层面的 dispatch overhead 不是瓶颈,批量时的 padding overhead 抵消了收益。

GPU 上保持批量化,CPU 上收益·损失都不大,为了后端间代码统一保持相同路径。
