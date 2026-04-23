# Deep Dive

HayaKoe is a TTS engine built by trimming Style-Bert-VITS2 down to Japanese-only and restructuring it into a practical form for CPU inference and server operations.

This section documents **what was changed, how it was done, and how much the results improved**, with actual measurements.

You can start from any topic that interests you.

## Summary at a Glance

Here are the measured improvements HayaKoe achieved over the original SBV2 (see individual pages for details).

| Category | Original SBV2 | HayaKoe | Difference |
|---|---|---|---|
| CPU speed (short text, ~2s) | 1.13 s | 0.68 s | **1.67x faster** |
| CPU speed (medium text, ~8s) | 3.35 s | 2.44 s | **1.37x faster** |
| CPU speed (long text, ~38s) | 35.33 s | 10.43 s | **3.39x faster** |
| CPU memory | 5,122 MB | 2,346 MB | **54% reduction** |
| GPU VRAM | 3,712 MB | 1,661 MB | **55% reduction** |
| Supported architecture | x86_64 | x86_64 · aarch64 Linux | **ARM board support** |

## Page Structure

Each page follows a **why it matters -> implementation -> improvement results** flow as a baseline, with flexible structure depending on the topic.

## Table of Contents

### Big Picture
- [Architecture Overview](./architecture) — Full structure of the TTS engine

### Making CPU Inference Real-time
- [ONNX Optimization / Quantization](./onnx-optimization) — Q8 BERT + FP32 Synthesizer, arm64 support
- [Sentence Boundary Pause — Duration Predictor](./duration-predictor) — Restoring natural pauses in multi-sentence synthesis

### Further GPU Inference Optimization
- [BERT GPU Retention & Batch Inference](./bert-gpu) — Eliminating PCIe round-trips and multi-sentence batching

### Operational Convenience
- [Source Abstraction (HF / S3 / Local)](./source-abstraction) — Unifying speaker sources under URIs
- [OpenJTalk Dictionary Bundling](./openjtalk-dict) — Eliminating first-import delay and network dependency
- [arm64 Support](./arm64) — Raspberry Pi 4B measurements

### Other
- [Issue Reporting & License](./contributing)

::: info Recommended reading order
If this is your first time, we recommend skimming [Architecture Overview](./architecture) first, then selectively reading topics that interest you.
:::
