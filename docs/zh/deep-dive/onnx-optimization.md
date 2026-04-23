# ONNX 优化 / 量化

原版 Style-Bert-VITS2 基于 PyTorch,仅用 CPU 难以实时推理。

HayaKoe 将 **BERT 导出为 Q8 量化 ONNX,Synthesizer 导出为 FP32 ONNX** 并在 ONNX Runtime 上运行,将 CPU 推理速度 **按文本长度提升了 1.5x ~ 3.3x**。

同时将 1 说话人加载时的 RAM 使用量从 **5,122 MB 降至 2,346 MB (-54%)**。

得益于同一路径,**不仅 x86_64,aarch64 (Raspberry Pi 等) 也能用相同代码运行**。

## 不足之处

原版 SBV2 (CPU, PyTorch FP32) 有两个不足。

- **速度** — 文本越长推理时间指数级增长。short (1.7 s 音频) 倍速 1.52x,而 xlong (38.5 s 音频) 推理 35.3 秒·倍速 1.09x,勉强跟上实时。
- **内存** — 单说话人推理 Peak 内存约 5 GB 以上,负担较重。

## 分析

先看模型参数分布,全部参数中约 **84%** 集中在 BERT ([DeBERTa-v2-Large-Japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm), 约 329 M),Synthesizer (VITS) 为 63 M 约 16%。

BERT 占模型大部分,因此量化 BERT 预计可以大幅减少内存。在 PyTorch 中仅对 BERT 应用 Q8 Dynamic Quantization (`torch.quantization.quantize_dynamic`) 进行验证。

| 配置 | 平均推理时间 | RAM |
|---|---|---|
| PyTorch BERT FP32 | 4.796 s | +1,698 MB |
| PyTorch BERT Q8 | 4.536 s | **+368 MB** (-78%) |

确认 BERT 量化虽不改善速度,但能大幅减少内存使用。

在此基础上通过 **转到 ONNX Runtime** 进一步确保速度改善。

ONNX Runtime 在加载模型时自动应用 [图级优化](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)。

- **Kernel fusion** — 将连续多个运算合并为一个。例如将 Conv → BatchNorm → Activation 三步合为一步,消除中间结果的内存写入和再读取,减少内存访问从而加速。
- **Constant folding** — 将不论输入如何始终产生相同值的运算在加载时预先计算好,推理时使用预计算值加速。
- **删除不必要节点** — 找出未使用、重复或执行无意义运算的节点并删除。

总之,重构为数学上等价但推理更优化的运算。

Synthesizer 参数量仅 63 M,量化的内存收益有限,且 Flow 层(`rational_quadratic_spline`)在 FP16 以下数值不稳定,因此排除在量化对象之外。仅导出为 ONNX 以获取图优化收益。

### BERT 优化

为确认量化是否影响音质,在同一文本·同一说话人下比较了 FP32 · Q8 · Q4 三种 BERT 配置的输出(Synthesizer 均为 FP32 固定)。

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。
>
> (旅途中来到了一个神奇的小镇。稍微绕道走走吧。)

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="BERT dtype"
  :defaultIndex="0"
  :samples='[
    { value: "FP32", caption: "原版 baseline", src: "/hayakoe/deep-dive/quantization/fp32_med_ja.wav" },
    { value: "Q8",   caption: "INT8 dynamic quantization", src: "/hayakoe/deep-dive/quantization/q8_med_ja.wav" },
    { value: "Q4",   caption: "INT4 weight-only (MatMulNBits)", src: "/hayakoe/deep-dive/quantization/q4_med_ja.wav" }
  ]'
/>

FP32 和 Q8 在直接听辨时难以一致地区分。

Q4 在大部分区间与 FP32 · Q8 相似,但在尾部可以听到微小差异。

| 配置 | BERT 大小 | RAM (1 说话人) |
|---|---|---|
| FP32 | 1,157 MB | 1,599 MB |
| Q8 | 497 MB | 1,079 MB (-33%) |
| Q4 | 394 MB | 958 MB (-40%) |

判断进一步 Q4 量化获得的内存收益不足以弥补音质差异,决定 **默认使用 Q8**。

### Synthesizer 优化

BERT 占参数的 96%,按理快速化 BERT 就能加速整体。

但实际分别测量 BERT 和 Synthesizer 的推理时间后发现,**CPU 时间的大部分消耗在 Synthesizer 侧**。

PyTorch FP32 CPU 下的实测结果(5 次平均)。

| 文本 | BERT | Synthesizer | BERT 占比 | Synth 占比 |
|---|---|---|---|---|
| short (1.7 s) | 0.489 s | 0.885 s | 36% | **64%** |
| medium (5.3 s) | 0.602 s | 2.504 s | 19% | **81%** |
| long (7.8 s) | 0.690 s | 3.714 s | 16% | **84%** |
| xlong (30 s) | 1.074 s | 11.410 s | 9% | **91%** |

文本越长 Synthesizer 占比越大,因为 BERT 对文本长度相对不敏感,而 Synthesizer 时间与要生成的音频长度成正比增长。

实际仅量化 BERT Q8 时整体推理时间仅减少约 5%。

也就是说,**要提速必须优化 Synthesizer 部分**。

Synthesizer 采用 **仅 ONNX 转换** 而非量化。

- VITS 的 Flow 层(`rational_quadratic_spline`)在 FP16 以下因浮点误差导致 assertion error,无法量化。
- 参数量仅 63 M,量化的内存收益也有限。

通过转换到 ONNX Runtime,将前述图级优化(kernel fusion · constant folding · 删除不必要节点)同样应用到 Synthesizer。

### ONNX Runtime + `CPUExecutionProvider`

BERT 量化和 Synthesizer 图优化都在 ONNX Runtime 上运行。

此外通过 [intra-op parallelism](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) 将单个运算分散到多个 CPU 核心,即使只有一个请求也能利用全部 CPU。

## 改善效果

### CPU 性能对比(倍速,同一硬件)

倍速 = 音频时长 / 推理时间(值越大越快)。

| 配置 | short (1.7 s) | medium (7.6 s) | long (10.7 s) | xlong (38.5 s) |
|---|---|---|---|---|
| SBV2 PyTorch FP32 | 1.52x | 2.27x | 2.16x | 1.09x |
| SBV2 ONNX FP32 | 1.76x | 3.09x | 3.26x | 2.75x |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2.50x** | **3.35x** | **3.33x** | **3.60x** |

相比 PyTorch FP32 的速度提升为 **按文本长度 1.5x ~ 3.3x**。

### 内存(1 说话人加载基准)

| 配置 | RAM |
|---|---|
| SBV2 PyTorch FP32 | 5,122 MB |
| SBV2 ONNX FP32 | 2,967 MB |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2,346 MB** (-54%) |
