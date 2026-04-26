# 服务器部署

HayaKoe 采用 **单例 TTS 实例 + 构建时内嵌权重** 模式,为服务器环境量身设计。可以用 FastAPI · Docker 组合搭建简洁的 API 服务器。

## 设计要点

### 1. 模型只加载一次(单例)

TTS 模型加载一次需要相当长的时间。在 GPU 环境下包括编译阶段可能需要数十秒。如果每个请求都创建新实例,实际服务就不可行了,因此需要在 **进程生命周期内只维护一个**,让所有请求共享。

实际代码在 FastAPI 的 lifespan 钩子中用 `TTS(...).load(...).prepare(warmup=True)` 构建单例并保存到 `app.state.tts`,处理器复用这一个实例。

并发无需担心。`Speaker` 内部持有 `threading.Lock`,同一说话人的并发请求会自动串行化,不同说话人之间则并行运行 — 无需额外的池·队列实现。

::: details GPU 后端会同时准备 torch.compile
`TTS.prepare()` 在 CUDA 后端不仅加载模型,还会对所有说话人和 BERT 统一应用 `torch.compile`。

`warmup=True` 时会预先执行 1 次虚拟推理,将编译成本提前到 prepare 阶段。此过程本身可能需要数十秒,因此必须在应用启动时只做一次。**每个请求都新建 TTS 会导致每次重新编译**,服务实际上会瘫痪。

CPU 后端使用 ONNX Runtime 因此没有单独的编译步骤,prepare 快得多。
:::

→ 实现见 [FastAPI 集成](/zh-CN/deploy/fastapi)

### 2. 权重在构建时烘焙到镜像中

HayaKoe 推荐的运维模式是 **将模型权重全部打包到 Docker 镜像中,运行时无需外部网络即可启动**。

为此提供了 `TTS.pre_download(device=...)` — "不初始化,只填充缓存"的方法。在 Docker 构建阶段调用将所需的说话人文件全部烘焙到镜像中,运行时容器就不需要访问 HF · S3。

在离线环境、防火墙内部、不想在运行时容器中暴露 HF·S3 凭证的情况下特别整洁。

→ 实现见 [Docker 镜像](/zh-CN/deploy/docker)

## 板块构成

| 页面 | 内容 |
|---|---|
| [FastAPI 集成](/zh-CN/deploy/fastapi) | lifespan 中加载单例,`agenerate` / `astream`,并发 |
| [Docker 镜像](/zh-CN/deploy/docker) | 构建时 `pre_download`,BuildKit secret,多阶段构建 |
| [后端选择](/zh-CN/deploy/backend) | CPU(ONNX) vs GPU(PyTorch) 权衡 |

