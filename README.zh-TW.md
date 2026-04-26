**[한국어](./README.md)** | **[日本語](./README.ja.md)** | **[简体中文](./README.zh-CN.md)** | **繁體中文** | **[English](./README.en.md)**

# HayaKoe

基於 [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 的高速日語 TTS 函式庫。

**[文件網站](https://lemondouble.github.io/hayakoe/zh-TW/)** · **[試聽語音範例](https://lemondouble.github.io/hayakoe/zh-TW/quickstart/#可以隨意製作這樣的語音)** · **[深入閱讀](https://lemondouble.github.io/hayakoe/zh-TW/deep-dive/)**

> **📖 請先閱讀文件網站！** 從安裝到參數調整、話者訓練、伺服器部署、架構詳解,所有內容都已整理完畢。
>
> [한국어](https://lemondouble.github.io/hayakoe/) · [日本語](https://lemondouble.github.io/hayakoe/ja/) · [简体中文](https://lemondouble.github.io/hayakoe/zh-CN/) · [繁體中文](https://lemondouble.github.io/hayakoe/zh-TW/) · [English](https://lemondouble.github.io/hayakoe/en/)

> Based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02 (AGPL-3.0)

## 特色

- **ONNX 最佳化** — CPU 即時推論(相較於 PyTorch 速度提升 1.6 倍,RAM 節省 47%)
- **無需 torch** — CPU 推論時無需 PyTorch 即可運行(輕量安裝)
- **簡潔 API** — 鏈式呼叫一行搞定 `TTS().load(...).prepare()`
- **資料來源可插拔** — HuggingFace / S3 / 本機路徑混合使用
- **Thread-safe** — 單例部署(FastAPI 等)支援同步/非同步兩種方式
- **JP-Extra 模型** — Style-Bert-VITS2 JP-Extra (v2.7.0), DeBERTa JP
- **英語→片假名自動轉換** — 22 萬條外來語字典查詢(無額外相依)

## 安裝

### CPU(預設,無需 torch)

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

### GPU(需另行安裝 PyTorch CUDA)

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

預設模型會從 [HuggingFace](https://huggingface.co/lemondouble/hayakoe) 自動下載。
自行訓練的話者可以放在 private HF repo / S3 / 本機路徑的任意位置。

## 使用方式

### 基本用法

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

GPU 推論(CUDA 下 `prepare()` 會自動套用 `torch.compile`):

```python
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

多話者 + 自訂資料來源混合:

```python
tts = (
    TTS(device="cuda")
    .load("jvnv-F1-jp")                                 # 官方 repo
    .load("my-voice", source="hf://me/private-voices")  # private HF
    .load("client-a", source="s3://tts-prod/voices")    # S3
    .load("dev-voice", source="file:///mnt/experiments") # 本機
    .prepare()
)
```

參數調整:

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

### 可用的官方話者

| 名稱 | 說明 | 風格 |
|------|------|------|
| `jvnv-F1-jp` | 女性話者 1 | Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust |
| `jvnv-F2-jp` | 女性話者 2 | 〃 |
| `jvnv-M1-jp` | 男性話者 1 | 〃 |
| `jvnv-M2-jp` | 男性話者 2 | 〃 |
| `tsukuyomi_chan` | つくよみちゃん — 動漫風 | Neutral |
| `amitaro_normal` | あみたろ — 普通 | Neutral |
| `amitaro_runrun` | あみたろ — 興奮 | Neutral |
| `amitaro_yofukashi` | あみたろ — 沉穩 | Neutral |
| `amitaro_punsuka` | あみたろ — 生氣 | Neutral |
| `amitaro_sasayaki_a` | あみたろ — 耳語A | Neutral |
| `amitaro_sasayaki_b` | あみたろ — 耳語B | Neutral |

各話者的語音範例可以在 **[文件網站直接試聽](https://lemondouble.github.io/hayakoe/zh-TW/quickstart/#可以隨意製作這樣的語音)**。

### FastAPI 單例部署

`Speaker` 內部使用 `threading.Lock` 對並行呼叫進行序列化,因此可以將一個
`TTS` 實例放在 `app.state` 中讓所有請求共享,完全安全。
同步處理器使用 `generate()` / `stream()`,非同步處理器使用 `agenerate()` /
`astream()`(非同步版本會自動在獨立執行緒中執行)。

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
    tts.prepare(warmup=True)  # 話者 materialize + torch.compile + Triton 預熱
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

### Docker / 伺服器環境

建置階段無需 GPU,僅將模型下載到快取中;執行時映像檔使用相同的
`cache_dir` 即可立即載入:

```dockerfile
# 建置階段 — 將模型包含在映像檔中(無需 GPU)
RUN python -c "\
from hayakoe import TTS; \
TTS().load('jvnv-F1-jp').pre_download(device='cuda')"

# 執行階段 — 從快取立即載入
CMD ["python", "server.py"]
```

快取根目錄預設為 `$CWD/hayakoe_cache`,可透過 `HAYAKOE_CACHE` 環境變數或
`TTS(cache_dir=...)` 覆寫。HuggingFace / S3 / 本機資料來源全部儲存在同一根目錄下。

| 方法 | 作用 | 需要 GPU | 用途 |
|------|------|----------|------|
| `TTS(device=...).load(...)` | 註冊話者規格(不下載) | 否 | 宣告 |
| `tts.pre_download(device=...)` | 僅下載到快取 | 否 | Docker 建置, CI |
| `tts.prepare()` | 載入模型 +(CUDA 時)torch.compile | 可選 | 執行時初始化 |

### Private / 內部資料來源

安裝 `hayakoe[s3]` extra 後即可使用 `s3://` 協定。
S3 相容端點(MinIO, R2 等)透過 `AWS_ENDPOINT_URL_S3` 環境變數指定。

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",          # BERT 也從內部映像取得
        hf_token="hf_...",                        # 用於 private HF
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

## 使用者字典

可以為 pyopenjtalk 不認識的專有名詞註冊讀音。

```python
tts = TTS().load("jvnv-F1-jp").prepare()

# 僅註冊讀音(重音為平板型)
tts.add_word(surface="担々麺", reading="タンタンメン")

# 同時指定重音位置(在第 3 個音拍處下降)
tts.add_word(surface="担々麺", reading="タンタンメン", accent=3)
```

## 架構

```
TTS (引擎)
├── BERT DeBERTa Q8 (ONNX)  ← 自動下載
│
├── speakers["jvnv-F1-jp"]  → Synthesizer ONNX + style vectors
├── speakers["jvnv-F2-jp"]  → ...
└── ...
```

- **CPU**: ONNX Runtime (BERT Q8 + Synthesizer FP32)
- **GPU**: PyTorch FP32 + `torch.compile` — `prepare()` 自動套用

## 開發工具 (Dev Tools)

從模型訓練到部署準備的互動式 CLI。

```bash
uv run poe cli
```

| 階段 | 功能 | 說明 |
|------|------|------|
| ① 訓練 | 資料前處理 + 模型訓練 | 使用語音資料訓練 TTS 模型 |
| ② 品質報告 | 依檢查點比較語音 | 比較試聽已訓練檢查點的語音(HTML) |
| ③ ONNX 匯出 | 轉換為 CPU 推論模型 | 在無 GPU 環境下推論時需要此步驟。若僅使用 GPU 推論則可跳過 |
| ④ 基準測試 | 測量 CPU/GPU 推論速度 | 測量相對於即時的倍速(HTML 報告) |
| ⑤ 發佈 (Publish) | 上傳模型至 HF / S3 / 本機 | 將訓練好的話者上傳到 private repo 或儲存桶,以便透過 `TTS(...).load(source=...)` 取得 |

## 授權

- 程式碼: AGPL-3.0(原版 Style-Bert-VITS2)
- JVNV 語音模型: CC BY-SA 4.0 ([JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus))
- 預訓練模型 (DeBERTa): MIT
- 英語→片假名字典資料: GPL-3.0 ([loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo))

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
- [JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) — 日語情感語音語料庫
- [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) by James O'Hortle — 英語→片假名字典資料
