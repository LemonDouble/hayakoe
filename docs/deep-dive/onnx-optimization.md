# ONNX 최적화 / 양자화

원본 Style-Bert-VITS2 는 PyTorch 기반이라 CPU 단독으로 실시간 추론이 어려웠습니다.

HayaKoe 는 **BERT 를 Q8 양자화된 ONNX 로, Synthesizer 를 FP32 ONNX 로** 내보내고 ONNX Runtime 위에서 실행하여, CPU 추론 속도를 **텍스트 길이에 따라 1.5× ~ 3.3× 까지 향상** 시켰습니다.

동시에 1 화자 로드 시 RAM 사용량을 **5,122 MB → 2,346 MB (−54 %)** 로 줄였습니다.

동일한 경로 덕분에 **x86_64 뿐 아니라 aarch64 (Raspberry Pi 등) 에서도 같은 코드로 동작** 합니다.

## 아쉬운 점

원본 SBV2 (CPU, PyTorch FP32) 에는 두 가지 아쉬운 점이 있었습니다.

- **속도** — 텍스트가 길어질수록 추론 시간이 기하급수적으로 증가합니다. short (1.7 s 분량) 에서는 배속 1.52× 수준이지만, xlong (38.5 s 분량) 에서는 추론 35.3 초 · 배속 1.09× 로 실시간을 간신히 따라가는 수준까지 떨어집니다.
- **메모리** — 1 화자 추론 시 Peak 메모리 약 5 GB 이상은 부담스러운 규모입니다.

## 분석

모델 파라미터 분포를 먼저 보면, 전체 중 약 **84 %** 가 BERT ([DeBERTa-v2-Large-Japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm), 약 329 M) 에 집중되어 있고, Synthesizer (VITS) 는 63 M 으로 약 16 % 수준입니다.

BERT 가 모델의 대부분을 차지하므로, BERT 를 양자화하면 메모리를 크게 줄일 수 있을 것으로 예상했습니다. PyTorch 에서 BERT 만 Q8 Dynamic Quantization (`torch.quantization.quantize_dynamic`) 을 적용해 검증했습니다.

| 구성 | 평균 추론 시간 | RAM |
|---|---|---|
| PyTorch BERT FP32 | 4.796 s | +1,698 MB |
| PyTorch BERT Q8 | 4.536 s | **+368 MB** (−78 %) |

BERT 양자화가 속도는 개선시켜주진 않지만, 메모리 사용량을 크게 줄일 수 있음을 확인했습니다.

여기에 추가로 **ONNX Runtime 전환**을 통해 속도 개선까지 확보하는 방향으로 진행했습니다.

ONNX Runtime 은 모델을 로드할 때 [그래프 수준 최적화](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) 를 자동으로 적용합니다.

- **Kernel fusion** — 연속된 여러 연산을 하나의 연산으로 합칩니다. 예를 들어 Conv → BatchNorm → Activation 세 단계를 하나로 묶으면, 중간 결과를 메모리에 쓰고 다시 읽는 과정이 사라져 메모리 접근이 줄어들어 빨라집니다.
- **Constant folding** — 입력에 관계없이 항상 같은 값을 내는 연산을 로드 시점에 미리 계산해 두고, 추론 시에는 미리 계산한 값을 사용해 속도를 높입니다.
- **불필요한 노드 제거** — 사용하지 않거나, 중복되거나, 무의미한 연산을 하는 노드를 찾아 제거합니다.

결론적으로, 추론에 최적화된 수학적으로 동일한 연산을 재구성해 더 빠르게 추론할 수 있게 만들어줍니다.

Synthesizer 는 파라미터가 63 M 으로 작아 양자화의 메모리 이득이 제한적이고, Flow 레이어 (`rational_quadratic_spline`) 가 FP16 이하에서 수치적으로 불안정하여 양자화 대상에서 제외했습니다. 대신 ONNX 로만 export 하여 그래프 최적화 이득을 확보했습니다.

### BERT 최적화

양자화가 음질에 영향을 주는지 확인하기 위해, 동일 텍스트·동일 화자로 FP32 · Q8 · Q4 BERT 세 구성의 출력을 비교했습니다 (Synthesizer 는 모두 FP32 고정).

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。
>
> (여행 도중 신비로운 마을에 도착했습니다. 잠시 들러볼까요.)

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="BERT dtype"
  :defaultIndex="0"
  :samples='[
    { value: "FP32", caption: "원본 baseline", src: "/hayakoe/deep-dive/quantization/fp32_med_ja.wav" },
    { value: "Q8",   caption: "INT8 dynamic quantization", src: "/hayakoe/deep-dive/quantization/q8_med_ja.wav" },
    { value: "Q4",   caption: "INT4 weight-only (MatMulNBits)", src: "/hayakoe/deep-dive/quantization/q4_med_ja.wav" }
  ]'
/>

FP32 와 Q8 은 직접 청취 시 일관되게 구분하기 어려운 수준이었습니다.

Q4 는 대부분의 구간에서 FP32 · Q8 과 유사하지만, 끝부분에서 미세한 차이가 들립니다.

| 구성 | BERT 크기 | RAM (1 화자) |
|---|---|---|
| FP32 | 1,157 MB | 1,599 MB |
| Q8 | 497 MB | 1,079 MB (−33 %) |
| Q4 | 394 MB | 958 MB (−40 %) |

Q4 로 추가 양자화해도 얻을 수 있는 메모리 이득이 음질보다 크지 않다고 판단해, **기본값으로 Q8 을 사용**하기로 결정했습니다.

### Synthesizer 최적화

BERT 가 파라미터의 96 % 를 차지하니, BERT 를 빠르게 만들면 전체가 빨라질 것 같습니다.

하지만 실제로 BERT 와 Synthesizer 의 추론 시간을 따로 측정해 보면, **CPU 시간의 대부분은 Synthesizer 쪽에서 소비**됩니다.

PyTorch FP32 CPU 에서의 실측 결과입니다 (5 회 평균).

| 텍스트 | BERT | Synthesizer | BERT 비중 | Synth 비중 |
|---|---|---|---|---|
| short (1.7 s) | 0.489 s | 0.885 s | 36 % | **64 %** |
| medium (5.3 s) | 0.602 s | 2.504 s | 19 % | **81 %** |
| long (7.8 s) | 0.690 s | 3.714 s | 16 % | **84 %** |
| xlong (30 s) | 1.074 s | 11.410 s | 9 % | **91 %** |

텍스트가 길어질수록 Synthesizer 의 비중이 커지는데, BERT 는 텍스트 길이에 비교적 둔감한 반면 Synthesizer 는 생성할 오디오 길이에 비례하여 시간이 늘어나기 때문입니다.

실제로 BERT 만 Q8 양자화했을 때 전체 추론 시간은 약 5 % 밖에 줄지 않았습니다.

즉, **속도를 높이려면 Synthesizer 구간을 최적화해야 합니다**.

Synthesizer 는 양자화 대신 **ONNX 변환만** 적용했습니다.

- VITS 의 Flow 레이어 (`rational_quadratic_spline`) 가 FP16 이하에서 부동소수점 오차로 인한 assertion error 를 일으켜 양자화가 불가능합니다.
- 파라미터 수가 63 M 으로 작아 양자화의 메모리 이득도 제한적입니다.

대신 ONNX Runtime 으로 변환하여 앞서 설명한 그래프 수준 최적화 (kernel fusion · constant folding · 불필요 노드 제거) 를 Synthesizer 에도 동일하게 적용했습니다.

### ONNX Runtime + `CPUExecutionProvider`

BERT 양자화와 Synthesizer 그래프 최적화는 모두 ONNX Runtime 위에서 동작합니다.

또한 [intra-op parallelism](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) 으로 단일 연산을 여러 CPU 코어에 분산하여, 요청이 하나뿐이어도 CPU 전체를 활용할 수 있습니다.

## 개선 효과

### CPU 성능 비교 (배속, 동일 하드웨어)

배속 은 오디오 길이 / 추론 시간 (값이 클수록 빠름).

| 구성 | short (1.7 s) | medium (7.6 s) | long (10.7 s) | xlong (38.5 s) |
|---|---|---|---|---|
| SBV2 PyTorch FP32 | 1.52× | 2.27× | 2.16× | 1.09× |
| SBV2 ONNX FP32 | 1.76× | 3.09× | 3.26× | 2.75× |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2.50×** | **3.35×** | **3.33×** | **3.60×** |

PyTorch FP32 대비 속도 향상은 **텍스트 길이에 따라 1.5× ~ 3.3×** 입니다.

### 메모리 (1 화자 로드 기준)

| 구성 | RAM |
|---|---|
| SBV2 PyTorch FP32 | 5,122 MB |
| SBV2 ONNX FP32 | 2,967 MB |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2,346 MB** (−54 %) |
