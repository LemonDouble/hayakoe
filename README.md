# HayaKoe

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)를 기반으로 한 고속 일본어 TTS 라이브러리.

> Based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02 (AGPL-3.0)

## 특징

- **ONNX 최적화** — CPU 실시간 추론 (PyTorch 대비 1.6x 속도 향상, 47% RAM 절감)
- **torch 불필요** — CPU 추론 시 PyTorch 없이 동작 (경량 설치)
- **3줄 추론** — 모델 자동 다운로드, 설정 불필요
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

모델은 [HuggingFace](https://huggingface.co/lemondouble/hayakoe)에서 자동 다운로드됩니다.

## 사용법

```python
from hayakoe import TTS

speaker = TTS().load("jvnv-F1-jp")
speaker.generate("こんにちは").save("output.wav")
```

GPU 추론:

```python
speaker = TTS(device="cuda").load("jvnv-F1-jp")
speaker.generate("こんにちは").save("output.wav")
```

GPU 추론 + torch.compile 최적화 (10-25% 향상):

```python
tts = TTS(device="cuda")
speaker = tts.load("jvnv-F1-jp")
tts.optimize()  # torch.compile 적용 (초회 워밍업 발생)
speaker.generate("こんにちは").save("output.wav")
```

파라미터 조절:

```python
audio = speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。楽しみですね。",
    style="Neutral",
    speed=0.9,
    sdp_ratio=0.2,
    noise=0.6,
    noise_w=0.8,
    pitch_scale=1.0,
    intonation_scale=1.0,
    style_weight=1.0,
)
```

### 사용 가능한 화자

| 이름 | 설명 |
|------|------|
| `jvnv-F1-jp` | 여성 화자 1 |
| `jvnv-F2-jp` | 여성 화자 2 |
| `jvnv-M1-jp` | 남성 화자 1 |
| `jvnv-M2-jp` | 남성 화자 2 |

각 화자는 7개 스타일을 지원합니다: `Neutral`, `Happy`, `Sad`, `Angry`, `Fear`, `Surprise`, `Disgust`

### Docker / 서버 환경

```dockerfile
# 빌드 시점 — 이미지에 모델 포함 (GPU 불필요)
RUN python -c "from hayakoe import TTS; TTS.prepare()"

# 실행 시점 — 다운로드 없이 바로 서빙
CMD ["python", "server.py"]
```

| 메서드 | 역할 | GPU 필요 | 용도 |
|--------|------|----------|------|
| `TTS.prepare()` | 모델 사전 다운로드 | X | Docker 빌드, CI |
| `TTS(device=...)` | 엔진 초기화 + 모델 로드 | 선택 | 추론 |
| `tts.optimize()` | torch.compile 적용 (10-25% 향상) | O (CUDA) | 서버 반복 추론 |

## 유저 사전

pyopenjtalk가 모르는 고유명사의 발음을 등록할 수 있습니다.

```python
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
- **GPU**: PyTorch FP32 (BERT + Synthesizer)
- **GPU + torch.compile**: CUDA Graphs + Triton 최적화 (`tts.optimize()`)

## 개발 도구 (Dev Tools)

모델 학습부터 배포 준비까지를 지원하는 인터랙티브 CLI입니다.

```bash
cd dev-tools
python -m cli
```

| 단계 | 기능 | 설명 |
|------|------|------|
| ① 학습 | 데이터 전처리 + 모델 학습 | 음성 데이터로 TTS 모델을 학습합니다 |
| ② 품질 리포트 | 체크포인트별 음성 비교 | 학습된 체크포인트의 음성을 비교 시청합니다 (HTML) |
| ③ ONNX 내보내기 | CPU 추론용 모델 변환 | GPU 없는 환경에서 추론하려면 필요합니다. GPU로만 추론한다면 건너뛰어도 됩니다 |
| ④ 벤치마크 | CPU/GPU 추론 속도 측정 | 실시간 대비 배속을 측정합니다 (HTML 리포트) |

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
