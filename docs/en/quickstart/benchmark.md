# Benchmark on Your Machine

This page lets you **measure how fast HayaKoe runs on your hardware**.

If you have the repository cloned, it takes just one command.

## Why Measure Yourself?

TTS inference speed heavily depends on CPU generation, memory bandwidth, ONNX Runtime version, and background load.

Documentation numbers are for reference only — running the benchmark on your own PC is far more accurate.

Having a measurement on hand also helps greatly when deciding where and how to use it.

## Terminology — Speed Factor

HayaKoe uses **speed factor** as its performance metric.

```text
speed factor = generated audio length (seconds) / inference time (seconds)
```

A speed factor of `1.0x` or higher is where it starts to feel near-real-time.

Of course, faster is always better.

For example, imagine generating 10 seconds of audio.

- `1.0x` — You must wait exactly 10 seconds to get all the audio
- `5.0x` — 2 seconds is enough
- `10.0x` — 1 second is enough

## Running the Benchmark

Use the benchmark CLI included in `dev-tools` in the repository.

```bash
uv run poe cli benchmark
```

An interactive menu lets you choose between **CPU (ONNX)**, **GPU (torch.compile)**, or **CPU + GPU**.

It runs 2 warm-up rounds + 5 measurement rounds across short, medium, and long text lengths, then presents results in a table.

After measurement, the same content is also saved as an HTML report under `benchmarks/`, which opens directly in your browser.

::: tip Requires a repo clone
`dev-tools` is not included in the pip package.

Clone from the [project GitHub](https://github.com/LemonDouble/hayakoe) and install dev dependencies first.
:::

## Reading the Results

When running both CPU + GPU, the terminal output looks roughly like this.

Numbers vary by machine (below is from a Ryzen 3950X / RTX 3080 setup, measured casually while other tasks were running).

```text
  speed factor = audio length / inference time (higher is faster)
  e.g.: 10.0x -> generates 1 second of audio in 0.1 seconds

                           Benchmark Results
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┓
┃ Backend              ┃ Text   ┃ Inference ┃ Audio Len   ┃ Speed ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━┩
│ ONNX (CPU)           │ Short  │    0.393s │        0.8s │  2.1x │
│ ONNX (CPU)           │ Medium │    1.297s │        4.1s │  3.2x │
│ ONNX (CPU)           │ Long   │    3.318s │       11.0s │  3.3x │
│ torch.compile (CUDA) │ Short  │    0.182s │        0.9s │  4.7x │
│ torch.compile (CUDA) │ Medium │    0.547s │        4.9s │  8.9x │
│ torch.compile (CUDA) │ Long   │    0.747s │       12.4s │ 16.6x │
└──────────────────────┴────────┴───────────┴─────────────┴───────┘

Benchmark complete!
  benchmarks/benchmark_20260414_211131.html

? Open in browser? Yes
```

### Longer Sentences Yield Better Speed Factors

Each model call has a fixed overhead, which gets amortized better with longer sentences.

However, since HayaKoe internally splits long texts into individual sentences, having more sentences does not automatically make things faster.

### Warm-up Must Be Excluded

The first call includes ONNX session initialization or (on GPU) `torch.compile` graph compilation time.

This is why the CLI discards the first 2 runs with `WARMUP = 2`.

### Measure with Minimal Background Activity

Close browsers, builds, and background downloads before measuring for stable results.

::: tip Run multiple times to check variance
Run the same benchmark 2-3 times.

If the speed factor reproduces within a few percent, you can trust that number more.

If variance is high, another program may be competing for resources.
:::

## Raspberry Pi 4B

HayaKoe works as-is on arm64 environments. For reference, here are the benchmark results from a Raspberry Pi 4B.

- Linux 6.8 · aarch64 · Python 3.10 · ONNX Runtime 1.23.2

| Text   | Inference | Audio Length | Speed Factor |
|:--|--:|--:|--:|
| Short  | 3.169s | 0.8s | 0.3x |
| Medium | 13.042s | 4.1s | 0.3x |
| Long   | 35.119s | 10.8s | 0.3x |

At 0.3x speed (about 3 times slower than real-time), it is tight for interactive real-time UI use, but perfectly usable for offline batch synthesis or personal projects.

::: warning First-run model download may take a while (~10 minutes)
On environments with slow SD card I/O and network like the Raspberry Pi, downloading BERT, Synthesizer, and style vector weights from HuggingFace alone can take **close to 10 minutes**.

Once downloaded, they are cached in `hayakoe_cache/` and subsequent runs start immediately.
:::

::: info ORT / HF warnings in the logs can be safely ignored
You may see the following two warnings during execution. Neither affects CPU inference behavior or results, so just proceed.

```text
Warning: You are sending unauthenticated requests to the HF Hub. ...
[W:onnxruntime:Default, device_discovery.cc:...] GPU device discovery failed: ...
```

- **HF anonymous request warning** — A notification that you are making tokenless requests to HuggingFace. Only affects download speed and rate limits. Setting the `HF_TOKEN` environment variable will suppress it.
- **ORT GPU discovery failure warning** — ONNX Runtime scanned for a GPU in the system and failed. This is normal behavior on GPU-less environments like Raspberry Pi and has no effect on CPU inference.
:::

## If the Speed Factor Is Near `1.0x`

If you measured on CPU and the speed factor is close to `1.0x`, it will be tight for real-time production use.

There are several options.

- **Switch to GPU** — The most reliable solution.
- **Sentence-level streaming** — For conversational UIs, you can stream the first sentence without waiting for the full synthesis ([Sentence-level Streaming](./streaming)).
