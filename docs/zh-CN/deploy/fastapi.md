# FastAPI 集成

在 lifespan 中构建 `TTS` 单例,处理器通过 `speaker.agenerate()` 委托合成。两个文件即可启动服务器。

## 完整示例

**`app/tts.py`** — TTS 引擎构建 + 路由

```python
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from hayakoe import TTS

HAYAKOE_VOICES_SOURCE = "hf://me/my-voices"
SPEAKERS = ("tsukuyomi",)

router = APIRouter(prefix="/api/tts")


def build_tts_engine() -> TTS:
    tts = TTS(device="cuda")
    for name in SPEAKERS:
        tts.load(name, source=HAYAKOE_VOICES_SOURCE)
    return tts


@router.get("")
async def synthesize(
    request: Request,
    text: str,
    speaker_name: str = "tsukuyomi",
    speed: float = 1.0,
):
    tts: TTS = request.app.state.tts
    if speaker_name not in tts.speakers:
        raise HTTPException(status_code=404, detail=f"Unknown speaker: {speaker_name}")
    audio = await tts.speakers[speaker_name].agenerate(text, speed=speed)
    return Response(audio.to_bytes(), media_type="audio/wav")
```

**`app/main.py`** — lifespan 中加载单例

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.tts import build_tts_engine, router as tts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    tts = build_tts_engine()
    tts.prepare(warmup=True)
    app.state.tts = tts
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(tts_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

运行:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 关键要点

### 1. 在 lifespan 中创建单例

`build_tts_engine()` 是将说话人注册顺序集中在一处的工厂方法。在 lifespan 中调用后,跑完 `prepare(warmup=True)` 并挂到 `app.state.tts`。

`warmup=True` 在 CUDA 后端会预先执行 1 次虚拟推理,将 `torch.compile` · Triton JIT · CUDA graph 捕获成本提前到 prepare 阶段 — **第一个实际请求的延迟会显著降低**。

每个请求都新建 `TTS()` 会导致每次编译,实际服务不可行。请务必 **在应用生命周期内只维护一个**。

### 2. 异步处理器使用 `agenerate` / `astream`

在 FastAPI async 处理器中直接调用同步 `generate()` 会导致事件循环在合成期间停顿。HayaKoe 提供了 async 包装器。

| 同步 | 异步 |
|---|---|
| `speaker.generate(text)` | `await speaker.agenerate(text)` |
| `speaker.stream(text)` | `async for chunk in speaker.astream(text)` |

在异步处理器中请务必使用 async 包装器。

::: details 内部机制 — async 包装器做了什么
`agenerate` / `astream` 内部通过 `asyncio.to_thread` 将同步函数卸载到工作线程。流式传输按句子 yield,中间中间让出事件循环,使其他请求不会等待。
:::

### 3. 并发自动安全

`Speaker` 内部持有 `threading.Lock`,同一说话人的并发合成请求会自动串行化。

- **同一说话人** → 串行(共享同一 GPU/CPU 资源的必然结果)
- **不同说话人** → 并行(各说话人持有各自的锁)

想并行运行多个说话人的话,在 `build_tts_engine()` 中预先全部 `load` 即可。无需额外的池·队列实现。

::: warning 流式生成器请消耗到底
`astream` 在生成器存活期间持有 per-speaker lock。

如果中途退出 `async for`,其他请求可能会永远等待。请用 `try / finally` 包裹或遍历完毕。

FastAPI 的 `StreamingResponse` 在客户端连接断开时会自动关闭生成器,因此大多数情况下没有问题。
:::

## 流式响应

想按句子立即发送给客户端的话,组合使用 `astream` 和 `StreamingResponse`。

```python
from fastapi.responses import StreamingResponse


@router.get("/stream")
async def synthesize_stream(
    request: Request,
    text: str,
    speaker_name: str = "tsukuyomi",
):
    tts: TTS = request.app.state.tts
    speaker = tts.speakers[speaker_name]

    async def body():
        async for chunk in speaker.astream(text):
            yield chunk.to_bytes()

    return StreamingResponse(body(), media_type="audio/wav")
```

## 测试

```bash
# 一次性接收 WAV
curl "http://localhost:8000/api/tts?text=こんにちは&speaker_name=tsukuyomi" \
  --output hello.wav

# 流式接收
curl "http://localhost:8000/api/tts/stream?text=こんにちは、はじめまして。&speaker_name=tsukuyomi" \
  --output hello_stream.wav

# 健康检查
curl http://localhost:8000/health
```

## 常见问题

::: details 同时服务多个说话人内存开销大吗?
BERT 由所有说话人共享,每个说话人增加的仅是 synthesizer 部分。比想象的要轻量得多。

大致增长趋势(参考指标,因硬件·torch 版本而异):

- **CPU RAM** — 每说话人 +300~400 MB
- **GPU VRAM** — 每说话人 +250~300 MB

实测表和复现脚本在 [FAQ — 加载多个说话人时内存增加多少](/zh/faq/#加载多个说话人时内存增加多少) 中整理。
:::

::: details `prepare()` 太慢了
原因分两路。

**1. 如果正在下载模型文件**

缓存中没有文件时 `prepare()` 会从 HF · S3 重新下载权重。由于是数 GB 级别,在网络慢的环境中可能需要数分钟。

→ **解决方法**: 在 Docker 镜像构建时用 `pre_download()` 将权重烘焙到镜像中([Docker 镜像](/zh/deploy/docker) 参考)。运行时的 `prepare()` 完全不走网络。

**2. 模型已有但 prepare 本身很慢**

CUDA 后端在 `prepare()` 中运行 `torch.compile`。首次编译耗时数十秒属于正常,此后请求使用编译好的图会很快。

有两个选择。

- **`prepare(warmup=True)`** — 将虚拟推理 1 次也包含在 prepare 中。prepare 更久但 **第一个实际请求很快**。服务时推荐。
- **`prepare(warmup=False)`** (默认) — prepare 快速完成。但 **第一个实际请求承担 warmup 成本**。
:::

::: details 增加 worker 数能提升性能吗?
用 `uvicorn --workers N` 增加 worker 时,每个 worker 会 **单独加载模型**。内存变为 N 倍,如果只有 1 块 GPU 则 worker 之间会争抢 CUDA 上下文。

**CPU / GPU 都建议默认 `--workers 1`。**

- **GPU** — per-speaker lock 已经处理了并发,1 块 GPU 资源本来就是共享的,增加 worker 无益。
- **CPU** — ONNX Runtime 仅用 1 个请求就已通过 intra-op parallelism 利用全部核心。增加 worker 总处理量几乎不变,只增加内存。

没有特殊原因就用 1 个 worker。
:::

## 下一步

- 烘焙到镜像:[Docker 镜像](/zh/deploy/docker)
- CPU / GPU 选哪个:[后端选择](/zh/deploy/backend)
