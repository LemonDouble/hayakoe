# 安装 — CPU vs GPU

HayaKoe 支持 **CPU 专用** 和 **GPU (CUDA)** 两种安装配置。

根据自己的环境选择其中一种即可。

## 应该选择哪种?

- **CPU** — 没有 GPU,或者有 GPU 但想先试试看的时候
- **GPU** — 需要批量处理,或者对实时性要求较高的时候

::: tip 犹豫时的默认选择
纠结的话就从 **CPU** 开始。

之后只需额外安装 GPU extras 即可。
:::

## CPU 安装(默认)

不需要 PyTorch,安装更快,镜像也更轻量。

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

::: tip arm64 同样可以运行
在 Raspberry Pi (4B 及以上) 等 aarch64 Linux 环境下,同样可以用一条命令安装并进行 CPU 推理。

实测数据请参考 [树莓派 4B 基准测试](./benchmark#拉-raspberry-pi-4b-实测)。
:::

### 验证

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
audio = tts.speakers["jvnv-F1-jp"].generate("テスト、テスト。")
audio.save("test.wav")
print("OK")
```

首次运行时,BERT · Synthesizer · 风格向量会从 [HuggingFace 官方仓库](https://huggingface.co/lemondouble/hayakoe) 自动下载到缓存目录。

默认缓存路径为当前目录下的 `hayakoe_cache/`。

## GPU 安装 (CUDA)

### 前置准备

GPU 模式使用 PyTorch CUDA 构建版本。

只需要 **NVIDIA 驱动** 即可。

- 不需要单独安装 CUDA Toolkit — PyTorch wheel 中已包含所需的 CUDA 运行时。
- 但需确保您的驱动支持要安装的 CUDA 版本。

检查驱动是否已安装:

```bash
nvidia-smi
```

如果正常安装,您将看到如下输出。

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

第一行右侧的 `CUDA Version: 13.0` 是您的驱动支持的 **最大 CUDA 版本**(上例中为 13.0)。

::: tip 选择 CUDA 版本
选择不超过 `nvidia-smi` 显示版本的 PyTorch CUDA 构建即可。

请在下方安装示例的 `cu126` 位置填入适合您的版本(例如:`cu118`、`cu121`、`cu124`、`cu128`)。

支持的组合可在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择。
:::

### 安装

`hayakoe[gpu]` extras 仅添加 `safetensors`,不会引入 `torch`。

分两行安装即可,顺序不限。

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

### 验证

```python
from hayakoe import TTS

tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ完了。").save("gpu_test.wav")
```

::: warning 首次请求可能较慢
在 GPU 模式下,第一次 `generate()` 调用可能会比平时多花几秒。

从第二次调用开始就会恢复正常速度。

如果要作为服务器运行,建议在启动后立即调用一次虚拟 `generate()` 进行"预热"。
:::

::: details 为什么首次调用会慢?(torch.compile 背景)
HayaKoe 在 GPU 模式下会在 `prepare()` 时自动应用 PyTorch 的 `torch.compile`。

`torch.compile` 是 PyTorch 2.0 中新增的 JIT 编译器,它追踪模型执行图,编译一次后复用其结果。

因此推理速度会提升,但代价是 **首次调用时需要额外的图追踪和编译时间**。

一旦编译完成的图在进程存活期间会被缓存,因此从第二次调用开始没有此开销,可以直接执行。所以在实际服务中,通常在容器/进程启动后用短文本进行一次虚拟调用来完成预热。

```python
# 在 FastAPI lifespan, Celery worker 初始化等场景
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ")  # 结果可以丢弃
```

CPU (ONNX) 模式不使用 `torch.compile`,因此不需要此预热步骤。
:::

到这里就完成了,接下来:[生成第一条语音 →](./first-voice)
