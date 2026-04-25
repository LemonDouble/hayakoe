**[한국어](./README.md)** | **日本語** | **[中文](./README.zh.md)** | **[English](./README.en.md)**

# HayaKoe

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)をベースにした高速日本語TTSライブラリ。

**[ドキュメントサイト](https://lemondouble.github.io/hayakoe/ja/)** · **[音声サンプルを聴く](https://lemondouble.github.io/hayakoe/ja/quickstart/#このような音声を自由に作れます)** · **[ディープダイブ](https://lemondouble.github.io/hayakoe/ja/deep-dive/)**

> **📖 ドキュメントサイトをぜひご覧ください！** インストールからパラメータ調整、話者学習、サーバーデプロイ、アーキテクチャ詳細まですべてまとめています。
>
> [한국어](https://lemondouble.github.io/hayakoe/) · [日本語](https://lemondouble.github.io/hayakoe/ja/) · [中文](https://lemondouble.github.io/hayakoe/zh/) · [English](https://lemondouble.github.io/hayakoe/en/)

> Based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02 (AGPL-3.0)

## 特徴

- **ONNX最適化** — CPUリアルタイム推論（PyTorch比1.6倍高速化、47% RAM削減）
- **torch不要** — CPU推論時はPyTorchなしで動作（軽量インストール）
- **簡潔なAPI** — チェイニング1行 `TTS().load(...).prepare()`
- **ソースプラグ型** — HuggingFace / S3 / ローカルパスを混在利用可能
- **Thread-safe** — シングルトンサービング（FastAPIなど）で同期/非同期の両方に対応
- **JP-Extraモデル** — Style-Bert-VITS2 JP-Extra (v2.7.0), DeBERTa JP
- **英語→カタカナ自動変換** — 22万エントリの外来語辞書ルックアップ（依存なし）

## インストール

### CPU（デフォルト、torch不要）

<details open>
<summary>pip</summary>

```bash
pip install hayakoe
```
</details>

<details>
<summary>uv</summary>

```bash
uv add hayakoe
```
</details>

<details>
<summary>Poetry</summary>

```bash
poetry add hayakoe
```
</details>

### GPU（PyTorch CUDAの別途インストールが必要）

<details open>
<summary>pip</summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
</details>

<details>
<summary>uv</summary>

```bash
uv add torch --index https://download.pytorch.org/whl/cu126
uv add hayakoe --extra gpu
```
</details>

<details>
<summary>Poetry</summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
poetry add hayakoe -E gpu
```
</details>

デフォルトモデルは[HuggingFace](https://huggingface.co/lemondouble/hayakoe)から自動ダウンロードされます。
独自に学習した話者はprivate HF repo / S3 / ローカルパスのどこにでも配置できます。

## 使い方

### 基本

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

GPU推論（CUDAでは `prepare()` が自動的に `torch.compile` を適用）：

```python
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

複数話者 + 独自ソース混在：

```python
tts = (
    TTS(device="cuda")
    .load("jvnv-F1-jp")                                 # 公式repo
    .load("my-voice", source="hf://me/private-voices")  # private HF
    .load("client-a", source="s3://tts-prod/voices")    # S3
    .load("dev-voice", source="file:///mnt/experiments") # ローカル
    .prepare()
)
```

パラメータ調整：

```python
speaker = tts.speakers["jvnv-F1-jp"]
audio = speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。楽しみですね。",
    style="Happy",
    speed=0.9,
    sdp_ratio=0.2,
    noise=0.6,
    noise_w=0.8,
    pitch_scale=1.0,
    intonation_scale=1.0,
    style_weight=1.0,
)
```

### 利用可能な公式話者

| 名前 | 説明 | スタイル |
|------|------|----------|
| `jvnv-F1-jp` | 女性話者 1 | Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust |
| `jvnv-F2-jp` | 女性話者 2 | 〃 |
| `jvnv-M1-jp` | 男性話者 1 | 〃 |
| `jvnv-M2-jp` | 男性話者 2 | 〃 |
| `tsukuyomi_chan` | つくよみちゃん — アニメ風 | Neutral |
| `amitaro_normal` | あみたろ — ノーマル | Neutral |
| `amitaro_runrun` | あみたろ — ワクワク | Neutral |
| `amitaro_yofukashi` | あみたろ — 落ち着き | Neutral |
| `amitaro_punsuka` | あみたろ — 怒り | Neutral |
| `amitaro_sasayaki_a` | あみたろ — ささやきA | Neutral |
| `amitaro_sasayaki_b` | あみたろ — ささやきB | Neutral |

各話者の音声サンプルは**[ドキュメントサイトで直接聴くことができます](https://lemondouble.github.io/hayakoe/ja/quickstart/#このような音声を自由に作れます)**。

### FastAPIシングルトンサービング

`Speaker` は内部の `threading.Lock` で同時呼び出しをシリアライズするため、1つの
`TTS` インスタンスを `app.state` に載せてすべてのリクエストで共有しても安全です。
同期ハンドラでは `generate()` / `stream()` を、asyncハンドラでは `agenerate()` /
`astream()` を呼び出してください（非同期バージョンは自動的に別スレッドにオフロード）。

```python
from enum import Enum
from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from hayakoe import TTS, Speaker

SPEAKERS = ["jvnv-F1-jp", "jvnv-M1-jp"]

class SpeakerName(str, Enum):
    F1 = "jvnv-F1-jp"
    M1 = "jvnv-M1-jp"

app = FastAPI()

@app.on_event("startup")
def _load_tts() -> None:
    tts = TTS(device="cuda")
    for name in SPEAKERS:
        tts.load(name)
    tts.prepare(warmup=True)  # 話者materialize + torch.compile + Tritonウォームアップ
    app.state.tts = tts

def get_speaker(name: SpeakerName, request: Request) -> Speaker:
    return request.app.state.tts.speakers[name.value]

@app.post("/tts/{name}")
async def tts_async(text: str, speaker: Speaker = Depends(get_speaker)):
    audio = await speaker.agenerate(text)
    return Response(audio.to_bytes(), media_type="audio/wav")

@app.post("/tts/{name}/stream")
async def tts_stream(text: str, speaker: Speaker = Depends(get_speaker)):
    async def body():
        async for chunk in speaker.astream(text):
            yield chunk.to_bytes()
    return StreamingResponse(body(), media_type="audio/wav")
```

### Docker / サーバー環境

ビルド時にはGPUなしでモデルをキャッシュにダウンロードするだけにし、ランタイムイメージで
同じ `cache_dir` から即座にロードします：

```dockerfile
# ビルド時 — イメージにモデルを含める（GPU不要）
RUN python -c "\
from hayakoe import TTS; \
TTS().load('jvnv-F1-jp').pre_download(device='cuda')"

# 実行時 — キャッシュから即座にロード
CMD ["python", "server.py"]
```

キャッシュルートはデフォルトで `$CWD/hayakoe_cache` であり、`HAYAKOE_CACHE` 環境変数または
`TTS(cache_dir=...)` で上書きできます。HuggingFace / S3 / ローカルソースは
すべて同じルート配下に保存されます。

| メソッド | 役割 | GPU必要 | 用途 |
|----------|------|---------|------|
| `TTS(device=...).load(...)` | 話者スペック登録（ダウンロードなし） | X | 宣言 |
| `tts.pre_download(device=...)` | キャッシュにダウンロードのみ | X | Dockerビルド、CI |
| `tts.prepare()` | モデルロード +（CUDAなら）torch.compile | 選択 | ランタイム初期化 |

### Private / 社内ソース

`hayakoe[s3]` extraをインストールすると `s3://` スキームが使用できます。
S3互換エンドポイント（MinIO、R2など）は `AWS_ENDPOINT_URL_S3` 環境変数で指定します。

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",          # BERTも社内ミラーから
        hf_token="hf_...",                        # private HF用
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

## ユーザー辞書

pyopenjtalkが認識しない固有名詞の発音を登録できます。

```python
tts = TTS().load("jvnv-F1-jp").prepare()

# 読みのみ登録（アクセントは平板）
tts.add_word(surface="担々麺", reading="タンタンメン")

# アクセント位置も指定（3番目のモーラでピッチ下降）
tts.add_word(surface="担々麺", reading="タンタンメン", accent=3)
```

## アーキテクチャ

```
TTS (エンジン)
├── BERT DeBERTa Q8 (ONNX)  ← 自動ダウンロード
│
├── speakers["jvnv-F1-jp"]  → Synthesizer ONNX + style vectors
├── speakers["jvnv-F2-jp"]  → ...
└── ...
```

- **CPU**: ONNX Runtime (BERT Q8 + Synthesizer FP32)
- **GPU**: PyTorch FP32 + `torch.compile` — `prepare()` が自動適用

## 開発ツール (Dev Tools)

モデル学習からデプロイ準備までをサポートするインタラクティブCLIです。

```bash
uv run poe cli
```

| ステップ | 機能 | 説明 |
|----------|------|------|
| ① 学習 | データ前処理 + モデル学習 | 音声データでTTSモデルを学習します |
| ② 品質レポート | チェックポイントごとの音声比較 | 学習したチェックポイントの音声を比較試聴します（HTML） |
| ③ ONNXエクスポート | CPU推論用モデル変換 | GPUのない環境で推論するには必要です。GPUのみで推論するならスキップ可 |
| ④ ベンチマーク | CPU/GPU推論速度測定 | リアルタイム比の倍速を測定します（HTMLレポート） |
| ⑤ デプロイ (Publish) | HF / S3 / ローカルへモデルアップロード | 学習した話者をprivate repoやバケットにアップロードし、`TTS(...).load(source=...)` で取得できるようにします |

## ライセンス

- コード: AGPL-3.0（オリジナルのStyle-Bert-VITS2）
- JVNV音声モデル: CC BY-SA 4.0 ([JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus))
- 事前学習モデル (DeBERTa): MIT
- 英語→カタカナ辞書データ: GPL-3.0 ([loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo))

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
- [JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) — 日本語感情音声コーパス
- [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) by James O'Hortle — 英語→カタカナ辞書データ
