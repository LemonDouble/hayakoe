# Backend Selection (CPU vs GPU)

HayaKoe supports two backends: CPU (ONNX Runtime) and GPU (PyTorch). At the code level, it is a single `device` parameter difference.

```python
tts_cpu = TTS(device="cpu").load("tsukuyomi").prepare()
tts_gpu = TTS(device="cuda").load("tsukuyomi").prepare()
```

However, **installation profiles differ from the start** — CPU works with just `pip install hayakoe`, while GPU additionally requires `hayakoe[gpu]` + a PyTorch CUDA build. Installing both on the same environment and running them together is possible, but in actual deployments you typically install **only one** matching the target environment (see [Installation — CPU vs GPU](/en/quickstart/install) for details).

The underlying architecture is also entirely different. Here is a summary of criteria for deciding which fits your deployment environment.

## When CPU (ONNX) Is the Right Choice

- **Server environments without a GPU** — Works immediately on general web hosting, VPS, and managed container platforms without CUDA support.
- **When image size must be minimized** — The PyTorch + CUDA stack runs several GB, while an ONNX Runtime-only image shrinks to hundreds of MB.
- **Low-concurrency workloads** — For personal projects or internal tools with modest concurrent load, CPU alone provides sufficient throughput.
- **When cold start must be short** — On the ONNX path, `prepare()` finishes as soon as the process starts and synthesis is available immediately. The GPU path is also fast by default, but with `compile=True` enabled the first `prepare()` must absorb about 80 seconds of BERT compilation, which makes a noticeable difference in autoscale or serverless environments.

::: details CPU path composition
- **BERT** — `bert_q8.onnx` (Q8 quantized DeBERTa), ONNX Runtime `CPUExecutionProvider`
- **Synthesizer** — `synthesizer.onnx` (ONNX-exported VITS decoder)
- **Duration Predictor** — `duration_predictor.onnx`
:::

## When GPU (PyTorch) Is the Right Choice

- **Real-time services requiring low latency** — User-facing responses, conversational UIs, and scenarios where single-request response time directly impacts perceived quality.
- **Environments needing high concurrent throughput** — Multiple speakers can be synthesized in parallel on a single GPU, providing much greater concurrent request capacity than CPU.
- **Environments with existing GPU infrastructure** — Leverage existing resources without additional investment for better latency and throughput at the same cost.
- **Long-running server workloads** — Applying `torch.compile` to BERT via `prepare(compile=True)` trades about 80 seconds of warmup for roughly 20% faster steady-state synthesis per sentence.

::: details GPU path composition
- **BERT** — FP32 DeBERTa loaded in GPU VRAM for embedding computation. Slightly higher precision than the CPU ONNX path due to no quantization. The only component `torch.compile` is applied to when `prepare(compile=True)` is passed.
- **Synthesizer** — PyTorch VITS decoder (eager execution).
- **Duration Predictor** — Same PyTorch path as the Synthesizer (eager execution).
:::

::: tip Reducing GPU backend cold start
The first `prepare()` on the GPU backend can take a long time when model downloads overlap, and with `compile=True` enabled it adds about 80 seconds of BERT compilation. For production services, the following two practices are recommended to pay this cost upfront.

- **`pre_download()` at Docker build time** — Baking weights into the image at build time means runtime `prepare()` loads from cache with no HF/S3 access. Initialization proceeds with no network latency as soon as the image starts. (-> [Docker Image](/en/deploy/docker))
- **`prepare(warmup=True, compile=True)`** — Running a dummy inference at prepare time pulls BERT compilation and cuDNN initialization forward into prepare. Prepare itself takes a bit longer, but **the first real request does not absorb the warmup cost**. (-> [FastAPI Integration](/en/deploy/fastapi))
:::

## Side-by-side Comparison

| Item | CPU (ONNX) | GPU (PyTorch) |
|---|---|---|
| Installation | `pip install hayakoe` | `pip install hayakoe[gpu]` |
| Image size | Hundreds of MB | Several GB |
| Cold start | Fast (seconds) | Fast (seconds) — ~80s with `compile=True` |
| Single request latency | Moderate | Lowest |
| Concurrent throughput | Limited by core count | Parallel on 1 GPU |
| Memory (1 speaker loaded) | ~1.7 GB RAM | ~1.3 GB RAM + 1.8 GB VRAM |
| Memory (per additional speaker) | +300-400 MB RAM | +250-300 MB VRAM |
| Required hardware | Any CPU | NVIDIA GPU + CUDA |

::: info Specific numbers are in the benchmarks
Speed factor, memory, and latency figures are heavily hardware-dependent.

- Speed factor measurement — [Benchmark on Your Machine](/en/quickstart/benchmark)
- Memory measurement (actual tables and reproduction scripts) — [FAQ — How much more memory does loading multiple speakers use](/en/faq/#how-much-more-memory-does-loading-multiple-speakers-use)
:::
