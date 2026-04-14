# 문장 단위 스트리밍

HayaKoe 는 긴 텍스트를 **한꺼번에 다 합성하지 않고, 문장이 나오는 대로 하나씩 흘려보내는** 스트리밍 모드를 지원합니다.

대화형 UI 나 실시간 응답이 필요한 상황에서 첫 음성을 빠르게 내보낼 수 있습니다.

## 언제 쓰면 좋나요?

여러 문장짜리 텍스트를 합성한다고 생각해 보세요.

`generate()` 는 모든 문장이 다 합성될 때까지 기다려야 wav 를 돌려줍니다.

15초짜리 오디오라면 사용자는 그 합성이 끝날 때까지 아무 소리도 듣지 못합니다.

반면 `stream()` 은 첫 문장이 완성되는 순간 그 chunk 를 바로 넘겨줍니다.

첫 문장을 재생하는 동안 뒤 문장이 이어서 합성되기 때문에, 체감 지연이 크게 줄어듭니다.

## 기본 — 문장별로 chunk 받기

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

text = "こんにちは。私はイレイナ。旅の魔女です。"

for i, chunk in enumerate(speaker.stream(text)):
    chunk.save(f"chunk_{i:02d}.wav")
```

`speaker.stream()` 은 각 문장을 `AudioResult` 하나로 yield 하는 파이썬 제너레이터입니다.

`generate()` 가 돌려주는 것과 같은 타입이라, `.save()` · `.to_bytes()` · `.data` 전부 똑같이 쓸 수 있습니다.

예시 텍스트는 `。` 에서 3 문장으로 쪼개져 3 개의 chunk 가 순서대로 나옵니다.

## 파라미터는 `generate()` 와 동일

`speed`, `pitch_scale`, `style`, `intonation_scale` 같은 [속도·운율 조절](./parameters) 파라미터를 그대로 사용할 수 있습니다.

```python
for chunk in speaker.stream(text, speed=0.95, style="Happy"):
    chunk.save(...)  # 또는 바로 오디오 디바이스에 write
```

## 문장 사이 무음은 자동으로 들어갑니다

두 번째 chunk 부터는 앞쪽에 **앞 문장과의 사이를 채울 무음** 이 자동으로 포함되어 나옵니다.

덕분에 chunk 를 순서대로 이어 붙이거나 재생하기만 해도, `generate()` 로 한 번에 받은 결과와 동일한 호흡이 나옵니다.

CPU (ONNX) 와 GPU (PyTorch) 둘 다 Duration Predictor 로 문장 경계 pause 를 미리 예측해서, 각 chunk 사이 gap 에 반영합니다.

ONNX 백엔드는 학습 시 같이 export 해 둔 별도의 `duration_predictor.onnx` 모델을 CPU 세션으로 돌립니다.

두 경로 모두 최소 80ms 의 floor 가 적용돼서, 예측값이 너무 짧게 나와도 부자연스럽게 붙지는 않습니다.

::: info Duration Predictor 가 뭐냐면
화자마다 문장 사이를 얼마나 쉬는지는 고유의 습관이 있습니다.

HayaKoe 는 이 pause 길이를 화자별로 학습된 작은 모델로 예측합니다.
:::

## 비동기 버전 — 웹 서버용

FastAPI 처럼 async 런타임에서 쓰려면 `astream()` 을 쓰세요.

```python
async for chunk in speaker.astream(text):
    await send(chunk.to_bytes())
```

내부적으로 별도 스레드에서 `stream()` 의 각 chunk 를 꺼내와 yield 하므로, 이벤트 루프를 블로킹하지 않습니다.

::: warning 제너레이터는 끝까지 소진해 주세요
`stream()` / `astream()` 은 내부적으로 per-speaker lock 을 잡습니다.

같은 화자에 다른 요청이 몰릴 수 있는 환경이라면, 제너레이터를 중간에 버리지 말고 `for` 문으로 다 돌리거나 `try/finally` 로 close 를 보장해 주세요.

소진되거나 close 되는 시점에 lock 이 풀립니다.
:::
