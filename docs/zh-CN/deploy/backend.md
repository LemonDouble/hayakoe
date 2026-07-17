# 后端选择 (CPU vs GPU)

HayaKoe 支持 CPU (ONNX Runtime) 和 GPU (PyTorch) 两种后端。在代码层面只是 `device` 参数的区别。

```python
tts_cpu = TTS(device="cpu").load("tsukuyomi").prepare()
tts_gpu = TTS(device="cuda").load("tsukuyomi").prepare()
```

但 **从安装配置就开始不同** — CPU 只需 `pip install hayakoe`,GPU 还需要 `hayakoe[gpu]` + PyTorch CUDA 构建。在同一环境中同时安装两者运行也是可能的,但实际部署中通常根据目标环境 **只安装一个**(详见 [安装 — CPU vs GPU](/zh-CN/quickstart/install))。

底层结构也完全不同。以下整理帮助您判断哪种适合自己的部署环境。

## CPU (ONNX) 适用的场景

- **没有 GPU 的服务器环境** — 在一般 Web 托管、VPS、托管容器平台等没有 CUDA 支持的环境中直接运行。
- **需要最小化镜像大小的场景** — PyTorch + CUDA 栈在数 GB 级别,而仅包含 ONNX Runtime 的镜像可缩减至数百 MB。
- **低并发请求的工作负载** — 个人项目或内部工具等并发负载不大的情况下,仅 CPU 也能确保足够的处理量。
- **需要短冷启动时间时** — ONNX 路径在进程启动后 `prepare()` 立即完成,可以马上开始合成。GPU 路径的默认配置同样很快,但如果开启了 `compile=True`,首次 `prepare()` 需要承受约 80 秒的 BERT 编译时间,在自动扩缩·无服务器环境中体感差异很大。

::: details CPU 路径构成
- **BERT** — `bert_q8.onnx` (Q8 量化 DeBERTa), ONNX Runtime `CPUExecutionProvider`
- **Synthesizer** — `synthesizer.onnx` (导出为 ONNX 的 VITS 解码器)
- **Duration Predictor** — `duration_predictor.onnx`
:::

## GPU (PyTorch) 适用的场景

- **要求低延迟的实时服务** — 面向用户的响应、对话式 UI 等单请求响应时间直接影响体验质量的场景。
- **需要高并发处理量的环境** — 一块 GPU 上可以并行合成多个说话人,比 CPU 的并发请求容纳能力大得多。
- **已有 GPU 基础设施的环境** — 无需额外投资即可利用现有资源,以相同成本获得更好的延迟和处理量。
- **长期运行的服务器工作负载** — 通过 `prepare(compile=True)` 对 BERT 应用 `torch.compile`,以约 80 秒的预热为代价,steady-state 合成每句提速约 20%。

::: details GPU 路径构成
- **BERT** — FP32 DeBERTa 加载到 GPU VRAM 中计算嵌入。因未量化,精度比 CPU ONNX 路径略高。是 `prepare(compile=True)` 时唯一应用 `torch.compile` 的组件。
- **Synthesizer** — PyTorch VITS 解码器(eager 执行)。
- **Duration Predictor** — 与 Synthesizer 相同的 PyTorch 路径(eager 执行)。
:::

::: tip 缩短 GPU 后端冷启动
GPU 后端的首次 `prepare()` 可能因夹带模型下载而耗时较久,开启 `compile=True` 时还会再加上约 80 秒的 BERT 编译。在实际服务中建议通过以下两种方式提前支付此成本。

- **Docker 构建时 `pre_download()`** — 在构建阶段将权重烘焙到镜像中,运行时 `prepare()` 无需 HF · S3 访问直接从缓存加载。镜像启动后立即无网络延迟地进行初始化。(→ [Docker 镜像](/zh-CN/deploy/docker))
- **`prepare(warmup=True, compile=True)`** — 在 prepare 时预先执行虚拟推理,将 BERT 编译和 cuDNN 初始化提前到 prepare 阶段。prepare 本身更久但 **第一个实际请求不承担 warmup 成本**。(→ [FastAPI 集成](/zh-CN/deploy/fastapi))
:::

## 并排对比

| 项目 | CPU (ONNX) | GPU (PyTorch) |
|---|---|---|
| 安装 | `pip install hayakoe` | `pip install hayakoe[gpu]` |
| 镜像大小 | 数百 MB | 数 GB |
| 冷启动 | 快 (秒级) | 快 (秒级) — `compile=True` 时 ~80 秒 |
| 单请求延迟 | 一般 | 最低 |
| 并发处理量 | 受核心数限制 | 1 块 GPU 上并行 |
| 内存 (加载 1 个说话人) | ≈ 1.7 GB RAM | ≈ 1.3 GB RAM + 1.8 GB VRAM |
| 内存 (每说话人增加) | +300~400 MB RAM | +250~300 MB VRAM |
| 所需硬件 | 任何 CPU | NVIDIA GPU + CUDA |

::: info 具体数值请看基准测试
倍速·内存·延迟数值高度依赖硬件。

- 倍速测量 — [在我的机器上做基准测试](/zh-CN/quickstart/benchmark)
- 内存测量(实测表和复现脚本) — [FAQ — 加载多个说话人时内存增加多少](/zh-CN/faq/#加载多个说话人时内存增加多少)
:::

