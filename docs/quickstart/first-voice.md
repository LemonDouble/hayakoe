# 첫 음성 만들기

설치가 끝났다면, 이제 실제로 wav 파일 하나를 만들어볼 차례입니다.

이 페이지는 가장 기본이 되는 "텍스트 한 줄 → wav 한 개" 흐름과, 거기서 자연스럽게 이어지는 몇 가지 변형을 다룹니다.

## 기본 — 한 문장을 wav 로 저장하기

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()

speaker = tts.speakers["jvnv-F1-jp"]
speaker.generate("こんにちは、はじめまして。").save("hello.wav")
```

`TTS()` 는 엔진 스펙만 등록합니다.

실제로 모델이 디스크에 내려받아지고 메모리에 올라오는 건 `prepare()` 가 호출되는 시점입니다.

`speakers` 는 준비가 끝난 화자들을 이름으로 꺼내쓸 수 있는 dict 입니다.

::: tip GPU 로 돌리고 싶다면
먼저 GPU extras 가 설치되어 있어야 합니다 ([설치 — GPU 설치 (CUDA)](./install#gpu-설치-cuda)).

그 다음에는 `TTS(device="cuda")` 로 한 글자만 바꾸면 됩니다. 나머지 코드는 동일합니다.

첫 호출이 조금 느릴 수 있다는 점만 기억하세요.
:::

## 여러 문장은 그대로 넣으면 됩니다

문장 경계(`。`, `！`, `？`, `!`, `?`, 개행) 에서 자동으로 분할되고, 문장 사이에는 자연스러운 길이의 무음이 들어갑니다.

```python
text = """
こんにちは。今日はいい天気ですね。
散歩でもしましょうか？
"""

speaker.generate(text).save("long.wav")
```

문장 사이 pause 의 길이는 Duration Predictor 가 예측해 붙여주는데, 이 예측기는 **화자마다 따로** 학습되어 있습니다.

즉, 어떤 화자는 문장 사이를 길게 쉬고 어떤 화자는 짧게 쉬는, 그 화자 고유의 호흡이 그대로 반영됩니다.

## wav 파일 말고 bytes 로 받기

FastAPI 같은 웹 서버에서 바로 응답으로 돌려주려면 `to_bytes()` 를 씁니다.

반환값은 WAV 포맷 바이트 스트림 입니다.

```python
audio = speaker.generate("テストです。")
payload: bytes = audio.to_bytes()
```

## 여러 화자를 한 번에 올리기

`load()` 는 체이닝이 되기 때문에 여러 화자를 한 번에 등록할 수 있습니다.

```python
tts = (
    TTS()
    .load("jvnv-F1-jp")
    .load("jvnv-F2-jp")
    .load("jvnv-M1-jp")
    .load("jvnv-M2-jp")
    .prepare()
)

for name, speaker in tts.speakers.items():
    speaker.generate("おはようございます。").save(f"{name}.wav")
```

BERT 는 모든 화자가 공유하기 때문에, 화자를 하나 더 얹어도 메모리가 두 배·세 배로 뛰지는 않습니다.

화자당 늘어나는 양은 BERT 가 아니라 훨씬 작은 synthesizer 만큼이라, 4 명을 올려도 RAM 이 4 배가 되지 않습니다.

실제 측정치와 재현 방법은 [FAQ — 화자를 여러 명 올리면 메모리가 얼마나 더 드나요?](/faq/#화자를-여러-명-올리면-메모리가-얼마나-더-드나요) 에 따로 정리해 두었습니다.

## 속도나 피치만 살짝 바꾸고 싶을 때

`generate()` 는 말속도·피치·억양을 직접 조절할 수 있는 파라미터를 받습니다.

```python
speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。",
    speed=0.95,        # 살짝 느리게
    pitch_scale=1.05,  # 살짝 높게
).save("tuned.wav")
```

사용 가능한 파라미터와 권장 범위는 [속도·운율 조절](./parameters) 에서 정리합니다.
