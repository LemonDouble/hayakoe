# FastAPI 整合

在 lifespan 中建構 `TTS` 單例,處理器透過 `speaker.agenerate()` 委派合成。

兩個檔案即可啟動伺服器。

## 完整範例

**`app/tts.py`** — TTS 引擎建構 + 路由

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

**`app/main.py`** — lifespan 中載入單例

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

執行:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 關鍵要點

### 1. 在 lifespan 中建立單例

`build_tts_engine()` 是將說話人註冊順序集中在一處的工廠方法。

在 lifespan 中呼叫後,跑完 `prepare(warmup=True)` 並掛到 `app.state.tts`。

`warmup=True` 在 CUDA 後端會預先執行 1 次虛擬推論,將 `torch.compile` · Triton JIT · CUDA graph 捕獲成本提前到 prepare 階段 — **第一個實際請求的延遲會顯著降低**。

每個請求都新建 `TTS()` 會導致每次編譯,實際服務不可行。

請務必 **在應用生命週期內只維護一個**。

### 2. 非同步處理器使用 `agenerate` / `astream`

在 FastAPI async 處理器中直接呼叫同步 `generate()` 會導致事件循環在合成期間停頓。

HayaKoe 提供了 async 包裝器。

| 同步 | 非同步 |
|---|---|
| `speaker.generate(text)` | `await speaker.agenerate(text)` |
| `speaker.stream(text)` | `async for chunk in speaker.astream(text)` |

在非同步處理器中請務必使用 async 包裝器。

::: details 內部機制 — async 包裝器做了什麼
`agenerate` / `astream` 內部透過 `asyncio.to_thread` 將同步函式卸載到工作執行緒。

串流傳輸按句子 yield,中間中間讓出事件循環,使其他請求不會等待。
:::

### 3. 並發自動安全

`Speaker` 內部持有 `threading.Lock`,同一說話人的並發合成請求會自動串列化。

- **同一說話人** → 串列(共享同一 GPU/CPU 資源的必然結果)
- **不同說話人** → 並行(各說話人持有各自的鎖)

想並行執行多個說話人的話,在 `build_tts_engine()` 中預先全部 `load` 即可。

無需額外的池·佇列實作。

::: warning 串流產生器請消耗到底
`astream` 在產生器存活期間持有 per-speaker lock。

如果中途退出 `async for`,其他請求可能會永遠等待。

請用 `try / finally` 包裹或遍歷完畢。

FastAPI 的 `StreamingResponse` 在客戶端連線斷開時會自動關閉產生器,因此大多數情況下沒有問題。
:::

## 串流回應

想按句子立即發送給客戶端的話,組合使用 `astream` 和 `StreamingResponse`。

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

## 測試

```bash
# 一次性接收 WAV
curl "http://localhost:8000/api/tts?text=こんにちは&speaker_name=tsukuyomi" \
  --output hello.wav

# 串流接收
curl "http://localhost:8000/api/tts/stream?text=こんにちは、はじめまして。&speaker_name=tsukuyomi" \
  --output hello_stream.wav

# 健康檢查
curl http://localhost:8000/health
```

## 常見問題

::: details 同時服務多個說話人記憶體開銷大嗎?
BERT 由所有說話人共享,每個說話人增加的僅是 synthesizer 部分。

比想像的要輕量得多。

大致增長趨勢(參考指標,因硬體·torch 版本而異):

- **CPU RAM** — 每說話人 +300~400 MB
- **GPU VRAM** — 每說話人 +250~300 MB

實測表和重現腳本在 [FAQ — 載入多個說話人時記憶體增加多少](/zh-TW/faq/#載入多個說話人時記憶體增加多少) 中整理。
:::

::: details `prepare()` 太慢了
原因分兩路。

**1. 如果正在下載模型檔案**

快取中沒有檔案時 `prepare()` 會從 HF · S3 重新下載權重。

由於是數 GB 級別,在網路慢的環境中可能需要數分鐘。

→ **解決方法**: 在 Docker 映像檔建置時用 `pre_download()` 將權重烘焙到映像檔中([Docker 映像檔](/zh-TW/deploy/docker) 參考)。執行時的 `prepare()` 完全不走網路。

**2. 模型已有但 prepare 本身很慢**

CUDA 後端在 `prepare()` 中執行 `torch.compile`。

首次編譯耗時數十秒屬於正常,此後請求使用編譯好的圖會很快。

有兩個選項。

- **`prepare(warmup=True)`** — 將虛擬推論 1 次也包含在 prepare 中。prepare 更久但 **第一個實際請求很快**。服務時推薦。
- **`prepare(warmup=False)`** (預設) — prepare 快速完成。但 **第一個實際請求承擔 warmup 成本**。
:::

::: details 增加 worker 數能提升效能嗎?
用 `uvicorn --workers N` 增加 worker 時,每個 worker 會 **單獨載入模型**。

記憶體變為 N 倍,如果只有 1 塊 GPU 則 worker 之間會爭搶 CUDA 上下文。

**CPU / GPU 都建議預設 `--workers 1`。**

- **GPU** — per-speaker lock 已經處理了並發,1 塊 GPU 資源本來就是共享的,增加 worker 無益。
- **CPU** — ONNX Runtime 僅用 1 個請求就已透過 intra-op parallelism 利用全部核心。增加 worker 總處理量幾乎不變,只增加記憶體。

沒有特殊原因就用 1 個 worker。
:::

## 下一步

- 烘焙到映像檔:[Docker 映像檔](/zh-TW/deploy/docker)
- CPU / GPU 選哪個:[後端選擇](/zh-TW/deploy/backend)
