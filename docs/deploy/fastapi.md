# FastAPI 통합

lifespan 에서 `TTS` 싱글톤을 빌드하고, 핸들러는 `speaker.agenerate()` 로 합성을 위임하는 구조입니다. 파일 두 개로 서버가 돕니다.

## 완성된 예제

**`app/tts.py`** — TTS 엔진 빌드 + 라우터

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

**`app/main.py`** — lifespan 에서 싱글톤 로드

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

실행:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 핵심 포인트

### 1. lifespan 에서 싱글톤 만들기

`build_tts_engine()` 은 화자 등록 순서를 한곳에 모아둔 팩토리입니다. lifespan 안에서 호출한 뒤 `prepare(warmup=True)` 까지 돌려 `app.state.tts` 에 붙입니다.

`warmup=True` 는 CUDA 백엔드에서 더미 추론을 1회 선행해 `torch.compile` · Triton JIT · CUDA graph 캡처 비용을 prepare 단계로 앞당깁니다 — **첫 실제 요청의 지연이 눈에 띄게 줄어듭니다**.

요청마다 `TTS()` 를 새로 만들면 매번 컴파일이 돌아 실서비스가 사실상 불가능합니다. 반드시 **앱 수명 동안 하나만** 유지하세요.

### 2. 비동기 핸들러는 `agenerate` / `astream`

FastAPI async 핸들러에서 동기 `generate()` 를 그대로 부르면 이벤트 루프가 합성 시간 내내 멈춥니다. HayaKoe 는 async 래퍼를 제공합니다.

| 동기 | 비동기 |
|---|---|
| `speaker.generate(text)` | `await speaker.agenerate(text)` |
| `speaker.stream(text)` | `async for chunk in speaker.astream(text)` |

비동기 핸들러에서는 반드시 async 래퍼를 쓰세요.

::: details 내부 동작 — async 래퍼가 하는 일
`agenerate` / `astream` 은 내부적으로 `asyncio.to_thread` 로 동기 함수를 워커 스레드에 오프로드합니다. 스트리밍은 문장 단위로 yield 하며 중간중간 이벤트 루프를 양보해, 다른 요청이 대기하지 않습니다.
:::

### 3. 동시성은 자동으로 안전

`Speaker` 는 내부에 `threading.Lock` 을 가지고 있어, 같은 화자에 대한 동시 합성 요청을 자동 직렬화합니다.

- **같은 화자** → 직렬 (하나의 GPU/CPU 리소스를 공유하므로 필연적)
- **다른 화자** → 병렬 (각 화자가 각자의 락을 가짐)

여러 화자를 병렬로 돌리고 싶다면 `build_tts_engine()` 에서 미리 전부 `load` 해두면 됩니다. 추가 풀·큐 구현은 필요 없습니다.

::: warning 스트리밍 제너레이터는 끝까지 소진하세요
`astream` 은 제너레이터가 살아있는 동안 per-speaker lock 을 잡고 있습니다.

`async for` 를 중간에 끊어 빠져나가면 다른 요청이 영원히 대기할 수 있습니다. `try / finally` 로 감싸거나 끝까지 돌리세요.

FastAPI 의 `StreamingResponse` 는 클라이언트 연결이 끊기면 제너레이터를 자동으로 닫아주므로 대부분의 경우 문제가 없습니다.
:::

## 스트리밍 응답

문장 단위로 바로바로 클라이언트에 내려주려면 `astream` 과 `StreamingResponse` 를 조합합니다.

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

## 테스트

```bash
# WAV 한 번에 받기
curl "http://localhost:8000/api/tts?text=こんにちは&speaker_name=tsukuyomi" \
  --output hello.wav

# 스트리밍 받기
curl "http://localhost:8000/api/tts/stream?text=こんにちは、はじめまして。&speaker_name=tsukuyomi" \
  --output hello_stream.wav

# 헬스체크
curl http://localhost:8000/health
```

## 자주 묻는 질문

::: details 여러 화자를 동시에 서빙하면 메모리가 많이 드나요?
BERT 는 모든 화자가 공유하므로, 화자당 늘어나는 건 synthesizer 분량뿐입니다. 체감보다 훨씬 가볍습니다.

대략적인 증가 경향 (참고 지표, 하드웨어·torch 버전에 따라 달라집니다):

- **CPU RAM** — 화자당 +300~400 MB
- **GPU VRAM** — 화자당 +250~300 MB

실측 표와 재현 스크립트는 [FAQ — 화자를 여러 명 올리면 메모리가 얼마나 더 드나요](/faq/#화자를-여러-명-올리면-메모리가-얼마나-더-드나요) 에 정리되어 있습니다.
:::

::: details `prepare()` 가 너무 느립니다
원인이 두 갈래입니다.

**1. 모델 파일을 그때 내려받고 있다면**

캐시에 파일이 없으면 `prepare()` 가 HF · S3 에서 가중치를 새로 내려받습니다. 수 GB 단위이므로 네트워크가 느린 환경에서는 수 분까지 걸릴 수 있습니다.

→ **해결**: Docker 이미지 빌드 시점에 `pre_download()` 로 가중치를 이미지 안에 박아 두세요 ([Docker 이미지](/deploy/docker) 참고). 런타임의 `prepare()` 는 네트워크를 전혀 타지 않게 됩니다.

**2. 모델은 이미 있는데 prepare 자체가 느리다면**

CUDA 백엔드는 `prepare()` 안에서 `torch.compile` 을 돌립니다. 첫 컴파일이 수십 초 걸리는 건 정상이고, 이후 요청부터는 컴파일된 그래프로 빠르게 돕니다.

선택지는 두 가지입니다.

- **`prepare(warmup=True)`** — 더미 추론 1회까지 prepare 에 포함. prepare 는 더 걸리지만 **첫 실제 요청이 빠릅니다**. 서빙 시 권장.
- **`prepare(warmup=False)`** (기본값) — prepare 를 빠르게 끝냄. 대신 **첫 실제 요청이 warmup 비용을 떠안습니다**.
:::

::: details 워커 수를 늘리면 성능이 올라가나요?
`uvicorn --workers N` 으로 워커를 늘리면 각 워커가 **모델을 따로 로드** 합니다. 메모리는 N 배로 늘고, GPU 1장이면 워커끼리 CUDA 컨텍스트를 다투게 됩니다.

**CPU / GPU 어느 쪽이든 기본은 `--workers 1` 을 권장합니다.**

- **GPU** — per-speaker lock 이 이미 동시성을 처리하고, 1장의 GPU 자원은 어차피 공유되므로 워커 증설 이득이 없습니다.
- **CPU** — ONNX Runtime 이 요청 1건으로도 intra-op parallelism 으로 코어 전체를 활용합니다. 워커를 늘려도 총 처리량은 거의 그대로이고 메모리만 늘 가능성이 높습니다.

특별한 이유가 없으면 1 워커로 두세요.
:::

## 다음 단계

- 이미지로 굽기: [Docker 이미지](/deploy/docker)
- CPU / GPU 중 어느 쪽: [백엔드 선택](/deploy/backend)
