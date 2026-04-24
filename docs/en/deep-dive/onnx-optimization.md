# ONNX Optimization / Quantization

The original Style-Bert-VITS2 was PyTorch-based, making real-time CPU-only inference difficult.

HayaKoe **exports BERT as Q8 quantized ONNX and the Synthesizer as FP32 ONNX**, running them on ONNX Runtime to achieve **1.5x to 3.3x CPU inference speedup depending on text length**.

Simultaneously, RAM usage for a single speaker was reduced from **5,122 MB to 2,346 MB (−54%)**.

Thanks to the same runtime path, **the same code works on aarch64 (Raspberry Pi, etc.) as well as x86_64**.

## The Problem

The original SBV2 (CPU, PyTorch FP32) had two shortcomings.

- **Speed** — Inference time grows dramatically with text length. For short text (1.7s audio), it runs at 1.52x speed, but for extra-long text (38.5s audio), it drops to 35.3 seconds inference at 1.09x — barely keeping up with real time.
- **Memory** — Peak memory above 5 GB for single-speaker inference is a substantial burden.

## Analysis

Looking at the model parameter distribution, approximately **84%** is concentrated in BERT ([DeBERTa-v2-Large-Japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm), ~329M), while the Synthesizer (VITS) is 63M, about 16%.

Since BERT dominates the model, we expected quantizing BERT would significantly reduce memory. We verified this by applying Q8 Dynamic Quantization (`torch.quantization.quantize_dynamic`) to BERT alone in PyTorch.

| Configuration | Avg Inference Time | RAM |
|---|---|---|
| PyTorch BERT FP32 | 4.796 s | +1,698 MB |
| PyTorch BERT Q8 | 4.536 s | **+368 MB** (−78%) |

BERT quantization did not improve speed but confirmed it can significantly reduce memory usage.

We then proceeded to additionally **switch to ONNX Runtime** to also capture speed improvements.

ONNX Runtime automatically applies [graph-level optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) when loading a model.

- **Kernel fusion** — Merges consecutive operations into one. For example, fusing Conv -> BatchNorm -> Activation into a single operation eliminates intermediate memory writes and reads, reducing memory access for faster execution.
- **Constant folding** — Pre-computes operations that always produce the same value regardless of input at load time, using the pre-computed values during inference for speed.
- **Dead node elimination** — Finds and removes unused, redundant, or no-op nodes.

In summary, it reconstructs mathematically equivalent operations optimized for inference, enabling faster execution.

The Synthesizer was excluded from quantization because its parameter count of 63M is small enough that quantization offers limited memory benefit, and the Flow layer (`rational_quadratic_spline`) is numerically unstable at FP16 or below. Instead, it was exported to ONNX only to capture graph optimization benefits.

### BERT Optimization

To check whether quantization affects audio quality, we compared outputs from FP32, Q8, and Q4 BERT configurations using the same text and speaker (Synthesizer fixed at FP32 for all).

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。
>
> (I arrived at a mysterious town during my journey. Let us take a short detour.)

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="BERT dtype"
  :defaultIndex="0"
  :samples='[
    { value: "FP32", caption: "Original baseline", src: "/hayakoe/deep-dive/quantization/fp32_med_ja.wav" },
    { value: "Q8",   caption: "INT8 dynamic quantization", src: "/hayakoe/deep-dive/quantization/q8_med_ja.wav" },
    { value: "Q4",   caption: "INT4 weight-only (MatMulNBits)", src: "/hayakoe/deep-dive/quantization/q4_med_ja.wav" }
  ]'
/>

FP32 and Q8 were consistently indistinguishable by ear.

Q4 sounds similar to FP32 and Q8 throughout most of the audio, but a subtle difference is audible near the end.

| Configuration | BERT Size | RAM (1 speaker) |
|---|---|---|
| FP32 | 1,157 MB | 1,599 MB |
| Q8 | 497 MB | 1,079 MB (−33%) |
| Q4 | 394 MB | 958 MB (−40%) |

We determined that the additional memory savings from Q4 did not outweigh the quality trade-off, so we chose **Q8 as the default**.

### Synthesizer Optimization

Since BERT accounts for 84% of parameters, it seems like making BERT faster would speed up everything.

However, when measuring BERT and Synthesizer inference times separately, **the CPU time is mostly consumed by the Synthesizer**.

Actual measurements on PyTorch FP32 CPU (5-run average).

| Text | BERT | Synthesizer | BERT Share | Synth Share |
|---|---|---|---|---|
| short (1.7 s) | 0.489 s | 0.885 s | 36% | **64%** |
| medium (5.3 s) | 0.602 s | 2.504 s | 19% | **81%** |
| long (7.8 s) | 0.690 s | 3.714 s | 16% | **84%** |
| xlong (30 s) | 1.074 s | 11.410 s | 9% | **91%** |

The Synthesizer's share grows with longer text because BERT is relatively insensitive to text length while the Synthesizer's time grows proportionally with the audio length to generate.

In practice, quantizing only BERT reduced total inference time by only about 5%.

In other words, **improving speed requires optimizing the Synthesizer portion**.

The Synthesizer used **ONNX conversion only** instead of quantization.

- VITS's Flow layer (`rational_quadratic_spline`) causes floating-point assertion errors at FP16 or below, making quantization impossible.
- The parameter count of 63M is small enough that quantization offers limited memory benefit.

Instead, converting to ONNX Runtime applies the same graph-level optimizations (kernel fusion, constant folding, dead node elimination) to the Synthesizer as well.

### ONNX Runtime + `CPUExecutionProvider`

Both BERT quantization and Synthesizer graph optimization run on ONNX Runtime.

Additionally, [intra-op parallelism](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) distributes a single operation across multiple CPU cores, utilizing the full CPU even with just one request.

## Improvement Results

### CPU Performance Comparison (Speed Factor, Same Hardware)

Speed factor = audio length / inference time (higher is faster).

| Configuration | short (1.7 s) | medium (7.6 s) | long (10.7 s) | xlong (38.5 s) |
|---|---|---|---|---|
| SBV2 PyTorch FP32 | 1.52x | 2.27x | 2.16x | 1.09x |
| SBV2 ONNX FP32 | 1.76x | 3.09x | 3.26x | 2.75x |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2.50x** | **3.35x** | **3.33x** | **3.60x** |

Speed improvement over PyTorch FP32 is **1.5x to 3.3x depending on text length**.

### Memory (Single Speaker Load)

| Configuration | RAM |
|---|---|
| SBV2 PyTorch FP32 | 5,122 MB |
| SBV2 ONNX FP32 | 2,967 MB |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2,346 MB** (−54%) |
