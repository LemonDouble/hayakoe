# FastAPI 統合

lifespan で `TTS` シングルトンをビルドし、ハンドラーは `speaker.agenerate()` で合成を委譲する構造です。ファイル2つでサーバーが動きます。

## 完成例

**`app/tts.py`** — TTS エンジンビルド + ルーター

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

**`app/main.py`** — lifespan でシングルトンロード

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

実行：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 重要ポイント

### 1. lifespan でシングルトンを作る

`build_tts_engine()` は話者登録順を一箇所にまとめたファクトリです。lifespan 内で呼び出した後 `prepare(warmup=True)` まで実行して `app.state.tts` に付けます。

`warmup=True` は CUDA バックエンドでダミー推論を1回先行して `torch.compile` ・ Triton JIT ・ CUDA graph キャプチャのコストを prepare 段階に前倒しします — **最初の実リクエストのレイテンシが目に見えて減ります**。

リクエストごとに `TTS()` を新規作成すると毎回コンパイルが走り本番サービスが事実上不可能になります。必ず **アプリの存続期間中ひとつだけ** 維持してください。

### 2. 非同期ハンドラーは `agenerate` / `astream`

FastAPI async ハンドラーで同期の `generate()` をそのまま呼ぶとイベントループが合成時間中ずっと停止します。HayaKoe は async ラッパーを提供します。

| 同期 | 非同期 |
|---|---|
| `speaker.generate(text)` | `await speaker.agenerate(text)` |
| `speaker.stream(text)` | `async for chunk in speaker.astream(text)` |

非同期ハンドラーでは必ず async ラッパーを使ってください。

::: details 内部動作 — async ラッパーがやること
`agenerate` / `astream` は内部的に `asyncio.to_thread` で同期関数をワーカースレッドにオフロードします。ストリーミングは文単位で yield しつつ途中でイベントループに制御を戻すため、他のリクエストが待たされません。
:::

### 3. 並行性は自動的に安全

`Speaker` は内部に `threading.Lock` を持っており、同じ話者への同時合成リクエストを自動シリアライズします。

- **同じ話者** → シリアル（ひとつの GPU/CPU リソースを共有するため必然）
- **異なる話者** → パラレル（各話者がそれぞれのロックを持つ）

複数話者を並列で動かしたければ `build_tts_engine()` で事前にすべて `load` しておけば良いです。追加のプール・キュー実装は不要です。

::: warning ストリーミングジェネレータは最後まで消費してください
`astream` はジェネレータが生きている間 per-speaker lock を保持しています。

`async for` を途中で抜けると他のリクエストが永遠に待機する可能性があります。`try / finally` で囲むか最後まで回してください。

FastAPI の `StreamingResponse` はクライアント接続が切れるとジェネレータを自動的に閉じてくれるため、ほとんどの場合問題ありません。
:::

## ストリーミングレスポンス

文単位ですぐにクライアントに送りたい場合は `astream` と `StreamingResponse` を組み合わせます。

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

## テスト

```bash
# WAV を一度に受け取る
curl "http://localhost:8000/api/tts?text=こんにちは&speaker_name=tsukuyomi" \
  --output hello.wav

# ストリーミングで受け取る
curl "http://localhost:8000/api/tts/stream?text=こんにちは、はじめまして。&speaker_name=tsukuyomi" \
  --output hello_stream.wav

# ヘルスチェック
curl http://localhost:8000/health
```

## よくある質問

::: details 複数話者を同時にサービングするとメモリが多くかかりますか？
BERT はすべての話者が共有するため、話者あたり増えるのは synthesizer 分だけです。体感よりはるかに軽量です。

おおよその増加傾向（参考指標、ハードウェア・torch バージョンによって異なります）：

- **CPU RAM** — 話者あたり +300~400 MB
- **GPU VRAM** — 話者あたり +250~300 MB

実測表と再現スクリプトは [FAQ — 話者を複数ロードするとメモリはどれくらい増えますか](/ja/faq/#話者を複数ロードするとメモリはどれくらい増えますか) にまとめてあります。
:::

::: details `prepare()` が遅すぎます
原因は2つに分かれます。

**1. モデルファイルをその場でダウンロードしている場合**

キャッシュにファイルがなければ `prepare()` が HF・S3 から重みを新たにダウンロードします。数 GB 単位なのでネットワークが遅い環境では数分かかる場合があります。

→ **解決策**：Docker イメージビルド時に `pre_download()` で重みをイメージ内に焼き込んでおいてください（[Docker イメージ](/ja/deploy/docker) 参照）。ランタイムの `prepare()` はネットワークをまったく経由しなくなります。

**2. モデルは既にあるのに prepare 自体が遅い場合**

CUDA バックエンドは `prepare()` 内で `torch.compile` を実行します。初回コンパイルが数十秒かかるのは正常で、以降のリクエストからはコンパイル済みグラフで高速に動作します。

選択肢は2つです。

- **`prepare(warmup=True)`** — ダミー推論1回まで prepare に含む。prepare はより長くかかりますが **最初の実リクエストが速くなります**。サービング時に推奨。
- **`prepare(warmup=False)`**（デフォルト） — prepare を早く終わらせる。代わりに **最初の実リクエストが warmup コストを負担します**。
:::

::: details ワーカー数を増やすとパフォーマンスが上がりますか？
`uvicorn --workers N` でワーカーを増やすと各ワーカーが **モデルを別々にロード** します。メモリは N 倍に増え、GPU 1枚なら CUDA コンテキストの競合が発生します。

**CPU / GPU どちらでもデフォルトは `--workers 1` を推奨します。**

- **GPU** — per-speaker lock が既に並行性を処理し、1枚の GPU リソースはどのみち共有されるのでワーカー増設の利点がありません。
- **CPU** — ONNX Runtime がリクエスト1件でも intra-op parallelism ですべてのコアを活用します。ワーカーを増やしても総スループットはほぼ同じでメモリだけ増える可能性が高いです。

特別な理由がなければ1ワーカーにしてください。
:::

## 次のステップ

- イメージに焼き込み：[Docker イメージ](/ja/deploy/docker)
- CPU / GPU のどちら：[バックエンド選択](/ja/deploy/backend)
