# Installation — CPU vs GPU

HayaKoe supports two installation profiles: **CPU only** and **GPU (CUDA)**.

Just pick the one that matches your environment.

## Which Should I Choose?

- **CPU** — When you don't have a GPU, or just want to try it out first
- **GPU** — When you need batch processing or real-time performance matters

::: tip Default when in doubt
If you are unsure, start with **CPU**.

You can always add the GPU extras later.
:::

## CPU Installation (Default)

No PyTorch required, so installation is quick and image size stays small.

::: code-group
```bash [pip]
pip install hayakoe
```
```bash [uv]
uv add hayakoe
```
```bash [poetry]
poetry add hayakoe
```
:::

::: tip arm64 works out of the box
On aarch64 Linux environments like Raspberry Pi (4B or later), the same single command installs and CPU inference runs without issues.

See the [Raspberry Pi 4B Benchmark](./benchmark#raspberry-pi-4b) for real-world numbers.
:::

### Verification

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
audio = tts.speakers["jvnv-F1-jp"].generate("テスト、テスト。")
audio.save("test.wav")
print("OK")
```

On first run, the BERT, Synthesizer, and style vectors are automatically downloaded from the [official HuggingFace repo](https://huggingface.co/lemondouble/hayakoe) into the cache folder.

The default cache path is `hayakoe_cache/` in the current directory.

## GPU Installation (CUDA)

### Prerequisites

GPU mode uses the PyTorch CUDA build.

All you need is an **NVIDIA driver**.

- You do not need to install the CUDA Toolkit separately — the PyTorch wheel bundles the required CUDA runtime.
- However, your driver must support the CUDA version you are installing.

Check if a driver is installed:

```bash
nvidia-smi
```

If properly installed, you should see output like this:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:06:00.0 Off |                  N/A |
| 53%   33C    P8             38W /  390W |    1468MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

The `CUDA Version: 13.0` on the top right is the **maximum CUDA version** your driver supports (13.0 in the example above).

::: tip Choosing a CUDA version
Pick a PyTorch CUDA build that is at or below the version shown by `nvidia-smi`.

Replace the `cu126` in the install command below with the version that matches your setup (e.g., `cu118`, `cu121`, `cu124`, `cu128`).

You can find the supported combinations on the [official PyTorch installation page](https://pytorch.org/get-started/locally/).
:::

### Installation

The `hayakoe[gpu]` extras only add `safetensors` and do not pull in `torch`.

Install them in two lines — the order does not matter.

::: code-group
```bash [pip]
pip install hayakoe[gpu]
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
```bash [uv]
uv add hayakoe --extra gpu
uv add torch --index https://download.pytorch.org/whl/cu126
```
```bash [poetry]
poetry add hayakoe -E gpu
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
:::

### Verification

```python
from hayakoe import TTS

tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ完了。").save("gpu_test.wav")
```

::: warning The first request may be slow
In GPU mode, the first `generate()` call may take a few extra seconds compared to normal.

From the second call onward, performance returns to normal speed.

If you are running this as a server, it is recommended to fire a dummy `generate()` right after startup to "warm up".
:::

::: details Why is the first call slow? (torch.compile background)
HayaKoe automatically applies PyTorch's `torch.compile` during `prepare()` when in GPU mode.

`torch.compile` is a JIT compiler introduced in PyTorch 2.0 that traces the model execution graph, compiles it once, and reuses the result for subsequent calls.

This improves inference speed, but at the cost of **extra time spent tracing and compiling the graph on the first call**.

Once compiled, the graph is cached for the lifetime of the process, so the second call onward runs without that overhead. In production, it is common practice to run a short dummy call right after the container or process starts to finish the warm-up.

```python
# In FastAPI lifespan, Celery worker init, etc.
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ")  # result can be discarded
```

In CPU (ONNX) mode, `torch.compile` is not used, so this warm-up step is not needed.
:::

Once you are done here, proceed to the next step: [First Voice -->](./first-voice)
