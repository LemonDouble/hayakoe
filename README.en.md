**[한국어](./README.md)** | **[日本語](./README.ja.md)** | **[简体中文](./README.zh-CN.md)** | **[繁體中文](./README.zh-TW.md)** | **English**

# HayaKoe

A high-speed Japanese TTS library based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2).

**[Documentation](https://lemondouble.github.io/hayakoe/en/)** · **[Listen to Voice Samples](https://lemondouble.github.io/hayakoe/en/quickstart/#voices-you-can-create)** · **[Deep Dive](https://lemondouble.github.io/hayakoe/en/deep-dive/)**

> **📖 Check out the docs site first!** Everything from installation to parameter tuning, speaker training, server deployment, and architecture deep dives — all in one place.
>
> [한국어](https://lemondouble.github.io/hayakoe/) · [日本語](https://lemondouble.github.io/hayakoe/ja/) · [简体中文](https://lemondouble.github.io/hayakoe/zh-CN/) · [繁體中文](https://lemondouble.github.io/hayakoe/zh-TW/) · [English](https://lemondouble.github.io/hayakoe/en/)

> Based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02 (AGPL-3.0)

## Features

- **ONNX Optimized** — Real-time CPU inference (1.6x faster than PyTorch, 47% RAM savings)
- **No torch required** — Runs without PyTorch for CPU inference (lightweight installation)
- **Simple API** — One-line chaining `TTS().load(...).prepare()`
- **Pluggable Sources** — Mix and match HuggingFace / S3 / local paths
- **Thread-safe** — Supports both sync and async in singleton serving (FastAPI, etc.)
- **JP-Extra Model** — Style-Bert-VITS2 JP-Extra (v2.7.0), DeBERTa JP
- **Auto English-to-Katakana Conversion** — 220K-entry loanword dictionary lookup (no dependencies)

## Installation

### CPU (default, no torch required)

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

### GPU (requires separate PyTorch CUDA installation)

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

The default model is automatically downloaded from [HuggingFace](https://huggingface.co/lemondouble/hayakoe).
Custom-trained speakers can be stored anywhere — private HF repo / S3 / local path.

## Usage

### Basic

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

GPU inference (on CUDA, `prepare()` automatically applies `torch.compile`):

```python
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

Multiple speakers + mixed sources:

```python
tts = (
    TTS(device="cuda")
    .load("jvnv-F1-jp")                                 # Official repo
    .load("my-voice", source="hf://me/private-voices")  # Private HF
    .load("client-a", source="s3://tts-prod/voices")    # S3
    .load("dev-voice", source="file:///mnt/experiments") # Local
    .prepare()
)
```

Parameter tuning:

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

### Available Official Speakers

| Name | Description | Styles |
|------|-------------|--------|
| `jvnv-F1-jp` | Female Speaker 1 | Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust |
| `jvnv-F2-jp` | Female Speaker 2 | 〃 |
| `jvnv-M1-jp` | Male Speaker 1 | 〃 |
| `jvnv-M2-jp` | Male Speaker 2 | 〃 |
| `tsukuyomi_chan` | Tsukuyomi-chan — Anime-style | Neutral |
| `amitaro_normal` | Amitaro — Normal | Neutral |
| `amitaro_runrun` | Amitaro — Excited | Neutral |
| `amitaro_yofukashi` | Amitaro — Calm | Neutral |
| `amitaro_punsuka` | Amitaro — Angry | Neutral |
| `amitaro_sasayaki_a` | Amitaro — Whisper A | Neutral |
| `amitaro_sasayaki_b` | Amitaro — Whisper B | Neutral |

You can **[listen to voice samples for each speaker on the documentation site](https://lemondouble.github.io/hayakoe/en/quickstart/#voices-you-can-create)**.

### FastAPI Singleton Serving

`Speaker` serializes concurrent calls with an internal `threading.Lock`, so it is
safe to place a single `TTS` instance on `app.state` and share it across all requests.
Use `generate()` / `stream()` for sync handlers and `agenerate()` / `astream()` for
async handlers (async versions automatically offload to a separate thread).

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
    tts.prepare(warmup=True)  # Materialize speakers + torch.compile + Triton warmup
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

### Docker / Server Environments

At build time, download models to cache without a GPU. At runtime, load instantly
from the same `cache_dir`:

```dockerfile
# Build time — embed models in image (no GPU required)
RUN python -c "\
from hayakoe import TTS; \
TTS().load('jvnv-F1-jp').pre_download(device='cuda')"

# Runtime — load instantly from cache
CMD ["python", "server.py"]
```

The cache root defaults to `$CWD/hayakoe_cache` and can be overridden with the
`HAYAKOE_CACHE` env variable or `TTS(cache_dir=...)`. All sources (HuggingFace /
S3 / local) are stored under the same root.

| Method | Role | GPU Required | Use Case |
|--------|------|--------------|----------|
| `TTS(device=...).load(...)` | Register speaker spec (no download) | No | Declaration |
| `tts.pre_download(device=...)` | Download to cache only | No | Docker build, CI |
| `tts.prepare()` | Load models + (CUDA) torch.compile | Optional | Runtime init |

### Private / Internal Sources

Install `hayakoe[s3]` extra to use the `s3://` scheme.
For S3-compatible endpoints (MinIO, R2, etc.), set `AWS_ENDPOINT_URL_S3` env variable.

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",          # BERT from internal mirror
        hf_token="hf_...",                        # For private HF repos
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

## User Dictionary

You can register pronunciations for proper nouns that pyopenjtalk does not recognize.

```python
tts = TTS().load("jvnv-F1-jp").prepare()

# Register reading only (flat accent)
tts.add_word(surface="担々麺", reading="タンタンメン")

# Also specify accent position (pitch drops at the 3rd mora)
tts.add_word(surface="担々麺", reading="タンタンメン", accent=3)
```

## Architecture

```
TTS (Engine)
├── BERT DeBERTa Q8 (ONNX)  ← Auto-downloaded
│
├── speakers["jvnv-F1-jp"]  → Synthesizer ONNX + style vectors
├── speakers["jvnv-F2-jp"]  → ...
└── ...
```

- **CPU**: ONNX Runtime (BERT Q8 + Synthesizer FP32)
- **GPU**: PyTorch FP32 + `torch.compile` — automatically applied by `prepare()`

## Dev Tools

An interactive CLI that supports the full workflow from model training to deployment.

```bash
uv run poe cli
```

| Stage | Feature | Description |
|-------|---------|-------------|
| 1. Training | Data preprocessing + model training | Train a TTS model from voice data |
| 2. Quality Report | Compare audio across checkpoints | Listen and compare generated audio from trained checkpoints (HTML) |
| 3. ONNX Export | Convert model for CPU inference | Required for inference without a GPU. Can be skipped if using GPU only |
| 4. Benchmark | Measure CPU/GPU inference speed | Measures real-time speed ratio (HTML report) |
| 5. Publish | Upload model to HF / S3 / local | Upload trained speakers to a private repo or bucket for use with `TTS(...).load(source=...)` |

## License

- Code: AGPL-3.0 (original Style-Bert-VITS2)
- JVNV Voice Models: CC BY-SA 4.0 ([JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus))
- Pretrained Model (DeBERTa): MIT
- English-to-Katakana Dictionary Data: GPL-3.0 ([loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo))

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
- [JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) — Japanese emotional speech corpus
- [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) by James O'Hortle — English-to-Katakana dictionary data
