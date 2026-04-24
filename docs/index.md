---
layout: home

hero:
  name: HayaKoe
  text: "내가 좋아하는 목소리로 만든 TTS를,<br>CPU만 가지고도 준실시간으로."
  tagline: 동영상이나 녹음본만 있으면, 데이터 준비·학습·벤치마크·배포까지 저희가 다 해드릴게요.
  actions:
    - theme: brand
      text: 10분 만에 써보기
      link: /quickstart/
    - theme: alt
      text: 학습
      link: /training/
    - theme: alt
      text: 배포
      link: /deploy/
    - theme: alt
      text: 깊이 읽기
      link: /deep-dive/

features:
  - title: CPU 실시간 추론
    details: "ONNX 최적화로 Style-Bert-VITS2 대비 짧은 텍스트는 1.5배, 긴 텍스트는 3.3배 빠른 CPU 단독 추론.<br>GPU에서는 torch.compile로 한층 더 빨라집니다."
    link: /deep-dive/onnx-optimization
    linkText: 어떻게 했나
  - title: AMD64 · ARM64 어디서나
    details: "x86_64 · aarch64 Linux 모두 같은 명령 하나로 설치.<br>Raspberry Pi 같은 ARM 보드에서도 CPU 추론이 그대로 돕니다."
    link: /quickstart/benchmark#라즈베리파이-4b-에서는-어떨까
    linkText: 라즈베리파이 벤치마크
  - title: 메모리 47% 절감
    details: "BERT Q8 양자화로 PyTorch 대비 RAM 47% 절감.<br>CPU 모드 약 2.0 GB RAM, GPU 모드 약 1.7 GB VRAM."
    link: /deep-dive/onnx-optimization
    linkText: 어떻게 했나
  - title: 다화자도 가볍게
    details: "BERT를 모든 화자가 공유하는 구조.<br>화자 하나를 추가해도 RAM은 ~300 MB만 더 듭니다."
    link: /deploy/fastapi
    linkText: 다화자 서빙
  - title: 문장 단위 스트리밍
    details: "astream()으로 문장이 합성되는 대로 흘려보냅니다.<br>전체 합성을 기다리는 것보다 첫 음성을 더 빨리 받을 수 있습니다."
    link: /deploy/fastapi
    linkText: 스트리밍 예제
  - title: 내가 원하는 목소리로
    details: "좋아하는 목소리가 담긴 영상만 준비하세요.<br>전처리·학습·품질 비교·최적화·배포까지, 저희가 다 해드릴게요."
    link: /training/
    linkText: 학습 가이드
  - title: HF · S3 호환 · 로컬 플러그형
    details: "CLI 배포는 HuggingFace · S3 호환 · 로컬 중 어디로든.<br>런타임 로드도 같은 세 경로를 동일하게 지원합니다."
    link: /deep-dive/source-abstraction
    linkText: Source 추상화
---

## 이런 목소리를 만들 수 있어요

기본 제공 화자들이 같은 문장 ("こんにちは、はじめまして。") 을 말하는 샘플입니다.

<SpeakerSample badge="JVNV" name="jvnv-F1-jp  —  여성 화자 1" src="/hayakoe/samples/hello_jvnv-F1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-F2-jp  —  여성 화자 2" src="/hayakoe/samples/hello_jvnv-F2-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M1-jp  —  남성 화자 1" src="/hayakoe/samples/hello_jvnv-M1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M2-jp  —  남성 화자 2" src="/hayakoe/samples/hello_jvnv-M2-jp.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="tsukuyomi_chan  —  애니메이션풍" src="/hayakoe/samples/hello_tsukuyomi_chan.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_normal  —  노멀" src="/hayakoe/samples/hello_amitaro_normal.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_runrun  —  설렘" src="/hayakoe/samples/hello_amitaro_runrun.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_yofukashi  —  차분" src="/hayakoe/samples/hello_amitaro_yofukashi.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_punsuka  —  화남" src="/hayakoe/samples/hello_amitaro_punsuka.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_a  —  속삭임A" src="/hayakoe/samples/hello_amitaro_sasayaki_a.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_b  —  속삭임B" src="/hayakoe/samples/hello_amitaro_sasayaki_b.wav" />

내 노트북에서, CPU 만으로, 위 샘플을 직접 만들어 보고 싶다면 [10분 만에 써보기](/quickstart/) 로.

## 짧게 써보면

### 설치

::: code-group
```bash [CPU]
pip install hayakoe
```
```bash [GPU (CUDA)]
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
:::

CPU 프로파일은 PyTorch가 필요 없어 설치가 짧고 이미지도 가벼워집니다.

GPU 프로파일은 추가 의존성을 설치하는 대신, 더 빠르게 추론합니다.

### 추론

```python
from hayakoe import TTS

text = "こんにちは、はじめまして。"

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate(text).save("hello.wav")
```

바로 `hello.wav` 를 들어보세요!

기본으로 제공되는 화자는 11명입니다.

- `jvnv-F1-jp` / `jvnv-F2-jp` / `jvnv-M1-jp` / `jvnv-M2-jp` — JVNV 코퍼스 기반
- `tsukuyomi_chan` — つくよみちゃんコーパス 기반
- `amitaro_normal` / `amitaro_runrun` / `amitaro_yofukashi` / `amitaro_punsuka` / `amitaro_sasayaki_a` / `amitaro_sasayaki_b` — あみたろ ITAコーパス 기반

위 코드의 `"jvnv-F1-jp"` 자리만 바꾸면 다른 목소리도 바로 들어볼 수 있습니다.

GPU 프로파일로 설치했다면 `TTS(device="cuda")` 파라미터만 넣으면 GPU로 추론할 수 있습니다.

## 어떤 문서를 읽어야 하나요?

1. **먼저 [퀵스타트](/quickstart/)** 를 따라가 보세요. 설치부터 첫 합성, 벤치마크까지 이 TTS가 얼마나 빠른지·음질은 어떤지 직접 확인할 수 있습니다.
2. **욕심이 생겼다면 [자체 화자 학습](/training/)** 으로. 좋아하는 목소리가 담긴 영상 하나로 데이터 준비부터 배포까지 전 과정을 안내합니다.
3. **혼자 쓰기엔 아깝다면 [서버로 배포](/deploy/)** 로. FastAPI·Docker 위에 API로 공개하는 방법을 정리했습니다.
4. **기술적으로 깊이 들어가 보고 싶다면 [깊이 읽기](/deep-dive/)** 로. 어디를 어떻게 건드려서 이만큼의 속도·메모리 개선을 얻었는지 개선 포인트를 하나씩 해설합니다.
5. **막히는 부분이 있다면 [FAQ](/faq/)** 로. 캐시 경로·Private HF·S3·다화자 메모리 같은 고급 설정을 모아뒀습니다.

## 음성 데이터 크레딧

본 프로젝트의 음성합성에는 아래 음성 데이터를 사용하고 있습니다.

- **つくよみちゃんコーパス** (CV.夢前黎, © Rei Yumesaki) — https://tyc.rei-yumesaki.net/material/corpus/
- **あみたろの声素材工房** ITAコーパス読み上げ音声 — https://amitaro.net/
