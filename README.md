**한국어** | **[日本語](./README.ja.md)** | **[中文](./README.zh.md)** | **[English](./README.en.md)**

# HayaKoe

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 기반으로 한 고속 일본어 TTS 라이브러리.

**[문서 사이트](https://lemondouble.github.io/hayakoe/)** · **[음성 샘플 듣기](https://lemondouble.github.io/hayakoe/quickstart/#공식-화자-음성-샘플)** · **[깊이 읽기](https://lemondouble.github.io/hayakoe/deep-dive/)**

> **📖 문서 사이트를 먼저 읽어보세요!** 설치부터 파라미터 조절, 자체 화자 학습, 서버 배포, 아키텍처 상세까지 모든 내용을 정리해 두었습니다.
>
> **📖 ドキュメントサイトをぜひご覧ください！** インストールからパラメータ調整、話者学習、サーバーデプロイ、アーキテクチャ詳細まですべてまとめています。
>
> **📖 请先阅读文档站点！** 从安装到参数调节、话者训练、服务器部署、架构详解，所有内容都已整理完毕。
>
> **📖 Check out the docs site first!** Everything from installation to parameter tuning, speaker training, server deployment, and architecture deep dives — all in one place.
>
> [한국어](https://lemondouble.github.io/hayakoe/) · [日本語](https://lemondouble.github.io/hayakoe/ja/) · [中文](https://lemondouble.github.io/hayakoe/zh/) · [English](https://lemondouble.github.io/hayakoe/en/)

> Based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02 (AGPL-3.0)

## 특징

- **ONNX 최적화** — CPU 실시간 추론 (PyTorch 대비 1.6x 속도 향상, 47% RAM 절감)
- **torch 불필요** — CPU 추론 시 PyTorch 없이 동작 (경량 설치)
- **간결한 API** — 체이닝 한 줄 `TTS().load(...).prepare()`
- **소스 플러그형** — HuggingFace / S3 / 로컬 경로를 섞어서 사용 가능
- **Thread-safe** — 싱글톤 서빙 (FastAPI 등) 에서 동기/비동기 양쪽 지원
- **JP-Extra 모델** — Style-Bert-VITS2 JP-Extra (v2.7.0), DeBERTa JP
- **영어→카타카나 자동 변환** — 22만 엔트리 외래어 사전 룩업 (의존성 없음)

## 설치

### CPU (기본, torch 불필요)

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

### GPU (PyTorch CUDA 별도 설치 필요)

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

기본 모델은 [HuggingFace](https://huggingface.co/lemondouble/hayakoe)에서 자동 다운로드됩니다.
자체 학습한 화자는 private HF repo / S3 / 로컬 경로 어디든 둘 수 있습니다.

## 사용법

### 기본

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

GPU 추론 (CUDA 에서는 `prepare()` 가 자동으로 `torch.compile` 적용):

```python
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

여러 화자 + 자체 소스 혼합:

```python
tts = (
    TTS(device="cuda")
    .load("jvnv-F1-jp")                                 # 공식 repo
    .load("my-voice", source="hf://me/private-voices")  # private HF
    .load("client-a", source="s3://tts-prod/voices")    # S3
    .load("dev-voice", source="file:///mnt/experiments") # 로컬
    .prepare()
)
```

파라미터 조절:

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

### 사용 가능한 공식 화자

| 이름 | 설명 | 스타일 |
|------|------|--------|
| `jvnv-F1-jp` | 여성 화자 1 | Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust |
| `jvnv-F2-jp` | 여성 화자 2 | 〃 |
| `jvnv-M1-jp` | 남성 화자 1 | 〃 |
| `jvnv-M2-jp` | 남성 화자 2 | 〃 |
| `tsukuyomi_chan` | つくよみちゃん — 애니메이션풍 | Neutral |
| `amitaro_normal` | あみたろ — 노멀 | Neutral |
| `amitaro_runrun` | あみたろ — 설렘 | Neutral |
| `amitaro_yofukashi` | あみたろ — 차분 | Neutral |
| `amitaro_punsuka` | あみたろ — 화남 | Neutral |
| `amitaro_sasayaki_a` | あみたろ — 속삭임A | Neutral |
| `amitaro_sasayaki_b` | あみたろ — 속삭임B | Neutral |

각 화자의 음성 샘플은 **[문서 사이트에서 직접 들어볼 수 있습니다](https://lemondouble.github.io/hayakoe/quickstart/#공식-화자-음성-샘플)**.

### FastAPI 싱글톤 서빙

`Speaker` 는 내부 `threading.Lock` 으로 동시 호출을 직렬화하므로, 하나의
`TTS` 인스턴스를 `app.state` 에 올려 모든 요청이 공유해도 안전합니다.
동기 핸들러는 `generate()` / `stream()`, async 핸들러는 `agenerate()` /
`astream()` 을 호출하세요 (비동기 버전은 자동으로 별도 스레드로 오프로딩).

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
    tts.prepare(warmup=True)  # 화자 materialize + torch.compile + Triton 워밍업
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

### Docker / 서버 환경

빌드 시점에는 GPU 없이 모델을 캐시로 내려받기만 하고, 런타임 이미지에서
같은 `cache_dir` 로 즉시 로드합니다:

```dockerfile
# 빌드 시점 — 이미지에 모델 포함 (GPU 불필요)
RUN python -c "\
from hayakoe import TTS; \
TTS().load('jvnv-F1-jp').pre_download(device='cuda')"

# 실행 시점 — 캐시에서 즉시 로드
CMD ["python", "server.py"]
```

캐시 루트는 기본 `$CWD/hayakoe_cache` 이며 `HAYAKOE_CACHE` env 또는
`TTS(cache_dir=...)` 로 덮어쓸 수 있습니다. HuggingFace / S3 / 로컬 소스
모두 같은 루트 아래에 저장됩니다.

| 메서드 | 역할 | GPU 필요 | 용도 |
|--------|------|----------|------|
| `TTS(device=...).load(...)` | 화자 스펙 등록 (다운로드 X) | X | 선언 |
| `tts.pre_download(device=...)` | 캐시에만 다운로드 | X | Docker 빌드, CI |
| `tts.prepare()` | 모델 로드 + (CUDA 면) torch.compile | 선택 | 런타임 초기화 |

### Private / 사내 소스

`hayakoe[s3]` extra 를 설치하면 `s3://` 스킴을 사용할 수 있습니다.
S3-호환 엔드포인트 (MinIO, R2 등) 는 `AWS_ENDPOINT_URL_S3` env 로 지정합니다.

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",          # BERT 도 사내 미러에서
        hf_token="hf_...",                        # private HF 용
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

## 유저 사전

pyopenjtalk가 모르는 고유명사의 발음을 등록할 수 있습니다.

```python
tts = TTS().load("jvnv-F1-jp").prepare()

# 읽기만 등록 (악센트는 평판)
tts.add_word(surface="担々麺", reading="タンタンメン")

# 악센트 위치도 지정 (3번째 모라에서 피치 하강)
tts.add_word(surface="担々麺", reading="タンタンメン", accent=3)
```

## 아키텍처

```
TTS (엔진)
├── BERT DeBERTa Q8 (ONNX)  ← 자동 다운로드
│
├── speakers["jvnv-F1-jp"]  → Synthesizer ONNX + style vectors
├── speakers["jvnv-F2-jp"]  → ...
└── ...
```

- **CPU**: ONNX Runtime (BERT Q8 + Synthesizer FP32)
- **GPU**: PyTorch FP32 + `torch.compile` — `prepare()` 가 자동 적용

## 개발 도구 (Dev Tools)

모델 학습부터 배포 준비까지를 지원하는 인터랙티브 CLI입니다.

```bash
uv run poe cli
```

| 단계 | 기능 | 설명 |
|------|------|------|
| ① 학습 | 데이터 전처리 + 모델 학습 | 음성 데이터로 TTS 모델을 학습합니다 |
| ② 품질 리포트 | 체크포인트별 음성 비교 | 학습된 체크포인트의 음성을 비교 시청합니다 (HTML) |
| ③ ONNX 내보내기 | CPU 추론용 모델 변환 | GPU 없는 환경에서 추론하려면 필요합니다. GPU로만 추론한다면 건너뛰어도 됩니다 |
| ④ 벤치마크 | CPU/GPU 추론 속도 측정 | 실시간 대비 배속을 측정합니다 (HTML 리포트) |
| ⑤ 배포 (Publish) | HF / S3 / 로컬로 모델 업로드 | 학습한 화자를 private repo 나 버킷에 올려 `TTS(...).load(source=...)` 로 받을 수 있게 만듭니다 |

## 라이선스

- 코드: AGPL-3.0 (원본 Style-Bert-VITS2)
- JVNV 음성 모델: CC BY-SA 4.0 ([JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus))
- 사전학습 모델 (DeBERTa): MIT
- 영어→카타카나 사전 데이터: GPL-3.0 ([loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo))

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
- [JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) — 일본어 감정 음성 코퍼스
- [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) by James O'Hortle — 영어→카타카나 사전 데이터
