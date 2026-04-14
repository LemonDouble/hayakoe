# 속도·운율 조절

::: tip 퀵스타트 권장: 일단 `speed` 만
대부분의 경우 **`speed` 외의 파라미터는 그대로 두는 쪽이 가장 자연스럽게 들립니다**.

`pitch_scale` 이나 `intonation_scale` 은 1.0 에서 벗어나면 약간의 음질 저하까지 따라옵니다.

"좀 빠르게/느리게만 하고 싶다" 면 `speed` 하나만 만져 보고, 나머지는 필요해졌을 때 [고급 파라미터](#고급-파라미터) 섹션으로 내려가시면 됩니다.
:::

`generate()` 는 말속도·피치·억양·변동성을 조절할 수 있는 키워드 인자 여섯 개를 받습니다.

모두 생략 가능하고, 아무것도 넘기지 않으면 학습된 기본값으로 합성됩니다.

아래 샘플은 모두 같은 문장 ("今日はどんな国に辿り着くのでしょうか。新しい出会いが楽しみです。") 을 `jvnv-F2-jp` 화자로, 해당 파라미터만 바꿔서 뽑은 결과입니다.

파라미터 동작은 HayaKoe 가 포크한 [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 의 `infer()` 와 동일합니다.

## `speed` — 말속도

기본값 `1.0` 을 기준으로, **작을수록 빠르고 클수록 느립니다**.

내부적으로 Duration Predictor 가 예측한 음소 길이에 `speed` 를 그대로 곱하는 방식이라, 발음 자체는 잘 유지됩니다.

<SpeakerSampleGroup
  label="speed"
  :defaultIndex="2"
  :samples='[
    { value: "0.5", caption: "2배 빠르게 — 극단", src: "/hayakoe/samples/params/speed_0.5.wav" },
    { value: "0.75", caption: "빠르게", src: "/hayakoe/samples/params/speed_0.75.wav" },
    { value: "1.0", caption: "기본", src: "/hayakoe/samples/params/speed_1.0.wav" },
    { value: "1.25", caption: "느리게", src: "/hayakoe/samples/params/speed_1.25.wav" },
    { value: "1.5", caption: "아주 느리게 — 극단", src: "/hayakoe/samples/params/speed_1.5.wav" }
  ]'
/>

0.8 보다 아래로 내려가면 발음이 뭉개지고, 1.3 을 넘기면 "느리다" 보다 "늘어진다" 쪽에 가까워집니다.

실전에서는 0.9 ~ 1.1 정도가 가장 자연스럽습니다.

```python
speaker.generate(text, speed=0.9)   # 약간 빠르게
speaker.generate(text, speed=1.1)   # 약간 느리게
```

## 고급 파라미터

여기서부터는 기본값으로도 충분히 자연스럽지만, 세밀한 튜닝이 필요할 때 건드리는 설정입니다.

### 전체 요약

| 파라미터           | 기본값 | 권장 범위 | 효과                                                |
| ------------------ | -----: | --------- | --------------------------------------------------- |
| `pitch_scale`      |  `1.0` | 0.95 ~ 1.05 | 음높이 배율. 1.0 에서 벗어나면 약간 음질 저하       |
| `intonation_scale` |  `1.0` | 0.8 ~ 1.3   | 억양의 기복. 1.0 에서 벗어나면 약간 음질 저하       |
| `sdp_ratio`        |  `0.2` | 0.0 ~ 0.5   | 결정적 DP 와 확률적 SDP 의 혼합 비율                |
| `noise`            |  `0.6` | 0.3 ~ 0.9   | DP 쪽에 들어가는 노이즈                             |
| `noise_w`          |  `0.8` | 0.5 ~ 1.2   | SDP 쪽에 들어가는 노이즈                            |

한 번에 하나씩만 움직여 보는 걸 권합니다.

아래 샘플에서는 차이를 귀로 확인할 수 있도록 일부러 권장 범위 바깥까지 밀어봤습니다.

### `pitch_scale` — 음높이

음높이를 전반적으로 올리거나 내리는 단순한 배율입니다.

`1.0` 에서 벗어나면 약간의 음질 저하가 따라와서, **다른 파라미터보다 좁게 움직이는 걸 권합니다**.

<SpeakerSampleGroup
  label="pitch_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.8", caption: "많이 낮게 — 극단", src: "/hayakoe/samples/params/pitch_0.8.wav" },
    { value: "0.9", caption: "낮게", src: "/hayakoe/samples/params/pitch_0.9.wav" },
    { value: "1.0", caption: "기본", src: "/hayakoe/samples/params/pitch_1.0.wav" },
    { value: "1.1", caption: "높게", src: "/hayakoe/samples/params/pitch_1.1.wav" },
    { value: "1.2", caption: "많이 높게 — 극단", src: "/hayakoe/samples/params/pitch_1.2.wav" }
  ]'
/>

0.95 ~ 1.05 범위는 화자 정체성이 거의 유지되지만, 극단 값에서는 "다른 사람" 이 되거나 음질이 눈에 띄게 떨어집니다.

```python
speaker.generate(text, pitch_scale=1.05)
```

### `intonation_scale` — 억양 기복

억양 변화의 "폭" 을 조절합니다.

`0.0` 은 거의 완전히 단조로운 로봇 톤, `2.0` 은 과장된 낭독 톤입니다.

`pitch_scale` 과 마찬가지로 1.0 에서 벗어나면 약간의 음질 저하가 있습니다.

<SpeakerSampleGroup
  label="intonation_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.0", caption: "완전 단조 — 극단", src: "/hayakoe/samples/params/intonation_0.0.wav" },
    { value: "0.5", caption: "평탄", src: "/hayakoe/samples/params/intonation_0.5.wav" },
    { value: "1.0", caption: "기본", src: "/hayakoe/samples/params/intonation_1.0.wav" },
    { value: "1.5", caption: "기복 크게", src: "/hayakoe/samples/params/intonation_1.5.wav" },
    { value: "2.0", caption: "과장 — 극단", src: "/hayakoe/samples/params/intonation_2.0.wav" }
  ]'
/>

실전에서는 0.85 ~ 1.3 정도가 자연스럽습니다.

```python
speaker.generate(text, intonation_scale=1.2)
```

### `sdp_ratio` — 결정적/확률적 길이 예측 혼합

HayaKoe (와 Style-Bert-VITS2) 는 두 종류의 길이 예측기를 함께 씁니다.

- **DP (Deterministic Duration Predictor)** — 같은 텍스트에 대해 항상 같은 길이를 내놓습니다
- **SDP (Stochastic Duration Predictor)** — 호출할 때마다 조금씩 다른 길이를 내놓습니다

`sdp_ratio` 는 두 예측기의 혼합 비율이고, **`0.0` 은 DP 만 쓰고, `1.0` 은 SDP 만 쓴다** 는 뜻입니다.

높을수록 문장 내 리듬 기복이 커지고, 같은 텍스트를 여러 번 돌렸을 때 결과가 매번 달라집니다.

<SpeakerSampleGroup
  label="sdp_ratio"
  :defaultIndex="1"
  :samples='[
    { value: "0.0", caption: "DP 만 — 항상 같은 길이", src: "/hayakoe/samples/params/sdp_0.0.wav" },
    { value: "0.25", caption: "DP 주도", src: "/hayakoe/samples/params/sdp_0.25.wav" },
    { value: "0.5", caption: "절반", src: "/hayakoe/samples/params/sdp_0.5.wav" },
    { value: "0.75", caption: "SDP 주도", src: "/hayakoe/samples/params/sdp_0.75.wav" },
    { value: "1.0", caption: "SDP 만 — 매번 달라짐", src: "/hayakoe/samples/params/sdp_1.0.wav" }
  ]'
/>

재현성이 중요한 서비스 (예: 자막 타이밍 고정) 에서는 `0.0` 으로 두고, 일회성 생성이라면 `0.2 ~ 0.4` 가 자연스럽습니다.

```python
speaker.generate(text, sdp_ratio=0.0)   # 항상 똑같이
```

### `noise` / `noise_w` — 두 예측기의 노이즈

둘 다 **길이 예측기** 쪽에 들어가는 노이즈의 크기입니다 (음소 자체의 노이즈가 아닙니다).

- `noise` — DP (결정적 예측기) 에 들어가는 노이즈
- `noise_w` — SDP (확률적 예측기) 에 들어가는 노이즈

즉 `sdp_ratio` 를 0 으로 해두면 `noise_w` 는 영향이 거의 없고, 반대로 1 로 해두면 `noise` 가 거의 영향이 없는 관계입니다.

아래 샘플은 나머지 파라미터를 전부 기본값으로 두고, 해당 노이즈 하나만 바꿔서 뽑은 결과입니다.

<SpeakerSampleGroup
  label="noise"
  :defaultIndex="1"
  :samples='[
    { value: "0.3", caption: "작게", src: "/hayakoe/samples/params/noise_0.3.wav" },
    { value: "0.6", caption: "기본", src: "/hayakoe/samples/params/noise_0.6.wav" },
    { value: "0.9", caption: "크게", src: "/hayakoe/samples/params/noise_0.9.wav" }
  ]'
/>

<SpeakerSampleGroup
  label="noise_w"
  :defaultIndex="1"
  :samples='[
    { value: "0.5", caption: "작게", src: "/hayakoe/samples/params/noise_w_0.5.wav" },
    { value: "0.8", caption: "기본", src: "/hayakoe/samples/params/noise_w_0.8.wav" },
    { value: "1.2", caption: "크게", src: "/hayakoe/samples/params/noise_w_1.2.wav" }
  ]'
/>

대부분은 기본값 (`0.6`, `0.8`) 을 그대로 두면 가장 자연스럽습니다.

"자꾸 흔들리는 것 같다" 싶으면 해당 쪽 노이즈를 살짝 내려 보고, "너무 기계적이다" 싶으면 조금 올려 보는 식으로 쓰면 됩니다.
