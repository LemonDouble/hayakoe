# FastAPI Integration

The structure builds a `TTS` singleton in the lifespan and delegates synthesis to `speaker.agenerate()` in the handlers. Two files and the server is up.

## Complete Example

**`app/tts.py`** — TTS engine build + router

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

**`app/main.py`** — Singleton load in lifespan

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

Running:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Key Points

### 1. Build the Singleton in Lifespan

`build_tts_engine()` is a factory that centralizes speaker registration order. Call it inside the lifespan, run `prepare(warmup=True)`, and attach to `app.state.tts`.

`warmup=True` runs a dummy inference pass in the CUDA backend to shift `torch.compile`, Triton JIT, and CUDA graph capture costs into the prepare phase — **noticeably reducing latency on the first real request**.

Creating a new `TTS()` per request triggers recompilation every time, making production service virtually impossible. Always maintain **exactly one instance for the app lifetime**.

### 2. Async Handlers Use `agenerate` / `astream`

Calling synchronous `generate()` directly in a FastAPI async handler blocks the event loop for the entire synthesis duration. HayaKoe provides async wrappers.

| Sync | Async |
|---|---|
| `speaker.generate(text)` | `await speaker.agenerate(text)` |
| `speaker.stream(text)` | `async for chunk in speaker.astream(text)` |

Always use the async wrappers in async handlers.

::: details Internal details — What async wrappers do
`agenerate` / `astream` internally offload the sync function to a worker thread via `asyncio.to_thread`. Streaming yields sentence by sentence, yielding control back to the event loop in between so other requests are not blocked.
:::

### 3. Concurrency Is Automatically Safe

`Speaker` has an internal `threading.Lock` that automatically serializes concurrent synthesis requests for the same speaker.

- **Same speaker** -> Serial (inevitable since one GPU/CPU resource is shared)
- **Different speakers** -> Parallel (each speaker has its own lock)

To run multiple speakers in parallel, just `load` them all in `build_tts_engine()` upfront. No additional pool or queue implementation is needed.

::: warning Fully consume streaming generators
`astream` holds a per-speaker lock for the generator's lifetime.

Breaking out of `async for` midway can leave other requests waiting indefinitely. Wrap with `try / finally` or iterate to completion.

FastAPI's `StreamingResponse` automatically closes the generator when the client disconnects, so this is rarely an issue in practice.
:::

## Streaming Response

To deliver audio to the client sentence by sentence in real time, combine `astream` with `StreamingResponse`.

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

## Testing

```bash
# Receive WAV at once
curl "http://localhost:8000/api/tts?text=こんにちは&speaker_name=tsukuyomi" \
  --output hello.wav

# Receive streaming
curl "http://localhost:8000/api/tts/stream?text=こんにちは、はじめまして。&speaker_name=tsukuyomi" \
  --output hello_stream.wav

# Health check
curl http://localhost:8000/health
```

## Frequently Asked Questions

::: details Does serving multiple speakers at once use a lot of memory?
Since BERT is shared across all speakers, the per-speaker increase is only the synthesizer portion. It is much lighter than you might expect.

Approximate increase trends (reference only, varies by hardware and torch version):

- **CPU RAM** — +300-400 MB per speaker
- **GPU VRAM** — +250-300 MB per speaker

Actual measurements and reproduction scripts are documented in [FAQ — How much more memory does loading multiple speakers use](/en/faq/#how-much-more-memory-does-loading-multiple-speakers-use).
:::

::: details `prepare()` is too slow
There are two possible causes.

**1. If it is downloading model files**

If files are not in cache, `prepare()` downloads weights from HF or S3. At several GB, this can take minutes on slow networks.

-> **Fix**: Bake weights into the image at Docker build time with `pre_download()` ([Docker Image](/en/deploy/docker)). Runtime `prepare()` will not touch the network at all.

**2. If models are already cached but prepare itself is slow**

The CUDA backend runs `torch.compile` inside `prepare()`. The first compilation taking tens of seconds is normal behavior, and subsequent requests use the compiled graph for fast execution.

Two options:

- **`prepare(warmup=True)`** — Includes a dummy inference pass in prepare. Prepare takes longer but **the first real request is fast**. Recommended for serving.
- **`prepare(warmup=False)`** (default) — Finishes prepare quickly. But **the first real request absorbs the warmup cost**.
:::

::: details Does increasing the number of workers improve performance?
Increasing workers with `uvicorn --workers N` causes each worker to **load the model separately**. Memory multiplies by N, and with a single GPU, workers compete for the CUDA context.

**For both CPU and GPU, the default recommendation is `--workers 1`.**

- **GPU** — Per-speaker locks already handle concurrency, and a single GPU's resources are shared regardless, so there is no benefit from more workers.
- **CPU** — ONNX Runtime already uses all cores via intra-op parallelism even for a single request. Adding workers is unlikely to increase total throughput and will mostly just increase memory.

Unless you have a specific reason, keep it at 1 worker.
:::

## Next Steps

- Bake into an image: [Docker Image](/en/deploy/docker)
- CPU or GPU: [Backend Selection](/en/deploy/backend)
