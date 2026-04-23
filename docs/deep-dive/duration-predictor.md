# 문장 경계 pause — Duration Predictor

다문장을 분할 합성하면 **문장 사이의 쉼 (pause) 이 소실되는** 부작용이 발생합니다.

HayaKoe 는 Duration Predictor 를 재활용하여 각 문장 경계의 자연스러운 pause 시간을 직접 예측합니다.

Flow · Decoder 를 건너뛰고 **TextEncoder + Duration Predictor 만 실행** 하므로, 추가 비용이 낮습니다.

## 왜 문제인가

### 문장 분할의 이점

[아키텍처 한눈에](./architecture#_1-%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC-%E1%84%87%E1%85%AE%E1%86%AB%E1%84%92%E1%85%A1%E1%86%AF) 에서 설명한 것처럼, HayaKoe 는 다문장 입력을 문장 단위로 쪼개 개별 합성합니다.

긴 텍스트를 통째로 넣으면 억양이 뭉개지거나 불안정해지는 경향이 있습니다.

문장 단위로 끊으면 각 문장마다 안정적인 prosody (운율) 가 보장됩니다.

### 분할의 부작용 — pause 소실

하지만 분할에는 부작용이 있습니다.

원본 SBV2 는 통문 합성에서 `.`, `!`, `?` 같은 구두점 뒤에 자연스러운 쉼을 넣어줍니다.

문장 단위 분할을 하면 각 문장이 구두점에서 끝나고 다음 문장이 처음부터 시작되므로, **구두점 뒤의 쉼이 함께 사라집니다.**

초기 구현에서는 문장 사이에 고정 80 ms 무음을 삽입했습니다.

실제로 Duration Predictor 가 예측하는 문장 경계 pause 는 0.3 ~ 0.6 초 수준이므로, 80 ms 는 이에 비해 매우 짧습니다.

결과적으로 "숨 돌릴 틈이 없는" 부자연스러운 발화가 만들어졌습니다.

## 원리 분석

이 절을 읽기 전에 Synthesizer 내부 흐름을 먼저 짚고 가겠습니다 (상세는 [아키텍처 한눈에 — Synthesizer](./architecture#_4-synthesizer-—-음소-bert-→-파형) 참고).

<PipelineFlow
  :steps="[
    {
      num: '1',
      title: 'Text Encoder',
      content: 'Transformer 인코더가 음소 시퀀스를 192차원 벡터로 임베딩합니다. BERT 특징이 여기서 음소 레벨 임베딩과 결합되어, 문장 맥락이 처음으로 음소에 주입됩니다.'
    },
    {
      num: '2',
      title: 'Duration Predictor',
      content: '각 음소를 몇 프레임 동안 발음할지 예측합니다. 안정적이지만 단조로운 DDP (결정적) 와, 자연스럽지만 불안정한 SDP (확률적) 두 predictor 의 출력을 sdp_ratio 로 블렌딩하여 안정성과 자연스러움의 균형을 잡습니다. 이 단계에서 음소 시퀀스가 시간 축으로 늘어납니다.'
    },
    {
      num: '3',
      title: 'Flow',
      content: 'Normalizing Flow (가역 신경망) 의 역변환을 거쳐, Text Encoder 가 만든 가우시안 분포 (평균·분산) 를 실제 음성의 복잡한 분포로 변형하여 latent z 벡터를 생성합니다. 학습 때는 정방향 (음성 → 텍스트 공간), 추론 때는 역방향 (텍스트 → 음성 공간) 으로 동작합니다.'
    },
    {
      num: '4',
      title: 'Decoder',
      content: 'HiFi-GAN 계열의 보코더로, latent z 를 ConvTranspose 업샘플링과 잔차 블록 (ResBlock) 을 거쳐 시간 도메인의 실제 파형 (44.1 kHz) 으로 생성합니다. Synthesizer 서브 모듈 중 연산량이 가장 크며, CPU 추론 시간의 대부분이 여기서 소비됩니다.'
    }
  ]"
/>

이 문서에서 다루는 핵심은 **1 · 2 단계 (Text Encoder + Duration Predictor) 까지만 따로 실행** 하는 것입니다.

3 · 4 단계 (Flow + Decoder) 를 건너뛰므로 비용이 매우 낮습니다.

### 원본 모델은 pause 를 어떻게 만들었나

원본 SBV2 가 통문 합성에서 자연스러운 pause 를 생성하는 원리를 추적한 결과, **Duration Predictor 가 구두점 음소의 frame 수를 예측하는 부수 효과** 였습니다.

Duration Predictor 는 원래 "각 음소를 몇 프레임 동안 발음할지" 를 예측하는 모듈입니다.

"안" 은 5 프레임, "녕" 은 4 프레임 같은 식입니다.

그런데 `.`, `!`, `?` 같은 구두점도 음소 시퀀스에 포함됩니다.

Duration Predictor 가 구두점에 대해 예측한 frame 수가 곧 **해당 구두점 위치에서의 쉼 길이** 가 됩니다.

예를 들어 `.` 에 20 프레임이 예측되면 Synthesizer 는 그 구간 동안 무음 또는 묵음에 가까운 파형을 생성합니다.

분할 합성에서는 구두점 위치에서 합성이 끊기므로 이 정보가 그대로 폐기되고 있었습니다.

### Duration Predictor 의 내부 동작

Duration Predictor 의 예측 흐름을 더 자세히 보면, 두 개의 서브 모듈이 병렬로 동작합니다.

**DDP (Deterministic Duration Predictor)** 는 같은 입력에 대해 항상 같은 duration 을 출력합니다.

안정적이지만 발화가 기계적으로 단조롭게 들릴 수 있습니다.

**SDP (Stochastic Duration Predictor)** 는 같은 입력에 대해 매번 약간 다른 duration 을 출력합니다.

확률 샘플링 기반이라 자연스러운 변동이 생기지만, 그만큼 결과가 덜 안정적입니다.

두 predictor 의 출력은 `sdp_ratio` 파라미터로 블렌딩됩니다.

`sdp_ratio=0.0` 이면 DDP 만, `1.0` 이면 SDP 만, `0.5` 면 반반 섞은 결과를 사용합니다.

`length_scale` (= speed 파라미터) 은 예측된 duration 전체에 곱해져 말속도를 조절합니다.

최종적으로 `ceil()` 로 올림하면 각 음소의 **정수 프레임 수** 가 결정됩니다.

### blank token 과 구두점

pause 계산 시 한 가지 주의할 점이 있습니다.

원본 SBV2 는 음소 시퀀스의 모든 음소 사이에 **blank token (공백 토큰, ID = 0)** 을 삽입하는 구조입니다. HayaKoe 도 이 동작을 그대로 따릅니다.

```
원본:  [は, い, .]
삽입 후: [0, は, 0, い, 0, ., 0]
```

blank token 에도 duration 이 예측되므로, 구두점 `.` 의 pause 를 구할 때는 **구두점 자체 + 앞뒤 blank 의 duration 을 합산** 해야 합니다.

예: `.` = 20 프레임, 앞 blank = 3 프레임, 뒤 blank = 5 프레임 → 총 28 프레임

## 구현

### 핵심 아이디어

핵심은 간단합니다.

**전체 원문 텍스트를 TextEncoder + Duration Predictor 까지만 통과시켜, 구두점 위치의 frame 수를 얻는 것** 입니다.

Flow 와 Decoder 는 건너뜁니다.

Synthesizer 전체 pass 에서 비용의 대부분은 Flow 와 Decoder 에서 발생하므로 ([ONNX 최적화](./onnx-optimization#synthesizer-최적화) 참고), Duration Predictor 까지만 실행하는 비용은 상대적으로 낮습니다.

```
전체 텍스트 (분할 전 원문)
  │
  ├─ TextEncoder (G2P → 음소열 → 임베딩)
  │
  ├─ Duration Predictor (음소별 frame 수 예측)
  │     └─ 구두점 위치의 frame 수만 추출
  │
  └─ pause 시간 계산
        frames × hop_length / sample_rate = 초
```

전체 합성에서는 이미 분할된 개별 문장을 각각 TextEncoder → Duration Predictor → Flow → Decoder 로 통과시킵니다.

pause 예측에서는 **분할 전 원문을 통째로** TextEncoder → Duration Predictor 만 통과시킵니다.

분할 전 원문을 사용하는 이유는, 문장 경계의 구두점이 원문에서만 온전히 존재하기 때문입니다.

개별 문장으로 쪼갠 뒤에는 마지막 문장의 구두점 외에는 경계 구두점이 사라지거나 위치가 달라집니다.

### pause 시간 계산

구두점 위치의 frame 수를 얻었으면 초 단위로 변환합니다.

```
pause (초) = frames × hop_length / sample_rate
```

HayaKoe 의 기본 설정에서 `hop_length = 512`, `sample_rate = 44100` 이므로, 1 프레임은 약 11.6 ms 에 해당합니다.

예를 들어 구두점 + 인접 blank 의 합산 frame 수가 35 라면:

```
35 × 512 / 44100 ≈ 0.41 초
```

실제 구현 (`durations_to_boundary_pauses()`) 에서는 다음 과정을 거칩니다.

1. 전체 음소 시퀀스에서 **문장 경계 구두점의 위치** 를 찾습니다 (`.`, `!`, `?` 에 해당하는 음소 ID).
2. 각 구두점 위치에서 해당 음소의 duration 을 가져옵니다.
3. 앞쪽 인접 토큰이 blank (ID = 0) 이면 그 duration 도 더합니다.
4. 뒤쪽 인접 토큰이 blank (ID = 0) 이면 그 duration 도 더합니다.
5. 합산된 frame 수를 `frames × hop_length / sample_rate` 로 변환합니다.

문장이 N 개면 경계는 N − 1 개이므로, 결과는 N − 1 개의 pause 시간 리스트입니다.

### trailing silence (꼬리 무음) 보상

한 가지 더 고려해야 할 점이 있습니다.

Synthesizer 가 각 문장을 합성할 때, 문장 끝부분에 이미 **짧은 무음이 포함** 될 수 있습니다.

이 trailing silence 를 무시하고 예측된 pause 를 그대로 삽입하면 실제 쉼이 과도하게 길어집니다.

HayaKoe 는 합성된 오디오의 뒤쪽에서 무음 구간을 직접 측정합니다.

측정 방식은 오디오 끝에서부터 10 ms 윈도우를 한 칸씩 앞으로 옮기며, **피크 진폭의 2 % 이하인 구간** 을 무음으로 판정합니다.

이후 pause 삽입 시에는 예측된 목표 pause 시간에서 trailing silence 를 빼서, **부족분만 무음 샘플로 보충** 합니다.

```
추가 무음 = max(0, 예측 pause − trailing silence)
```

모델이 이미 충분한 무음을 생성했다면 추가 삽입은 0 이 됩니다.

목표 pause 시간 자체를 최소 80 ms 로 하한을 두기 때문에, 예측값이 아무리 짧아도 문장 사이 총 무음은 항상 80 ms 이상이 됩니다.

### ONNX 지원

PyTorch 경로에서는 모델 내부 모듈을 개별 호출할 수 있어 Duration Predictor 만 따로 실행하면 됩니다.

반면 `synthesizer.onnx` 는 Synthesizer 전체를 하나의 종단 간 그래프로 내보낸 형태라 중간 출력을 꺼낼 수 없습니다.

이를 해결하기 위해 **TextEncoder + Duration Predictor 만 포함하는 별도 ONNX 모델** (`duration_predictor.onnx`, ~30 MB, FP32) 을 추가로 export 했습니다.

## 개선 효과

### pause 시간 분포

동일 텍스트에 대해 자동 예측된 문장 경계 pause 입니다.

| 백엔드 | pause 범위 |
|---|---|
| GPU (PyTorch) | 0.41 s ~ 0.55 s |
| CPU (ONNX) | 0.38 s ~ 0.57 s |

두 백엔드의 차이는 SDP 의 stochastic sampling (확률적 샘플링) 특성상 발생하는 편차 수준입니다.

SDP 는 확률 샘플링 기반이라 같은 입력이라도 호출마다 결과가 약간 달라집니다.

GPU 와 CPU 의 차이가 이 자연 변동폭 안에 들어오므로, ONNX 변환으로 인한 품질 손실은 무시할 수 있습니다.

### Before / After

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。きっと楽しい発見がありますよ。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pause 방식"
  :defaultIndex="1"
  :samples='[
    { "value": "Before (80 ms 고정)", "caption": "모든 문장 경계가 동일한 짧은 쉼", "src": "/hayakoe/samples/duration-predictor/before.wav" },
    { "value": "After (DP 예측)", "caption": "Duration Predictor 가 문장 경계 pause 를 자동 예측", "src": "/hayakoe/samples/duration-predictor/after.wav" }
  ]'
/>

### 비용

추가 비용은 TextEncoder + Duration Predictor 1 회 실행입니다.

[ONNX 최적화 — Synthesizer 비중](./onnx-optimization#synthesizer-최적화) 에서 확인할 수 있듯, Synthesizer 가 전체 CPU 추론 시간의 64 ~ 91 % 를 차지하며, 그 중 대부분은 Flow + Decoder 에서 소비됩니다.

Duration Predictor 까지만 실행하는 비용은 이에 비해 낮으므로, pause 예측으로 인한 체감 지연은 거의 없습니다.

## 관련 커밋

- `c57e0ad` — Duration Predictor 기반 pause 예측으로 다문장 합성 자연스럽게 개선
- `5522db1` — ONNX `duration_predictor` 추가로 CPU 백엔드도 자연스러운 문장 경계 무음 지원

## 향후 과제

- **감정별 pause 길이 분화** — 기쁨은 짧게, 슬픔은 길게 등 감정 스타일에 따라 pause 분포를 다르게 적용
- **쉼표 · 콜론 등 세분화** — 현재는 문장 종결 구두점 (`. ! ?`) 만 대상이지만, 쉼표 (`,`, `、`) 나 콜론 등 긴 호흡이 필요한 위치에 대한 추가 세분화
- **pause 직접 제어 API** — 사용자가 특정 문장 경계의 pause 길이를 명시적으로 지정할 수 있는 인터페이스
