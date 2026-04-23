# 용어 정리

이 페이지는 deep-dive 전반에 자주 나오는 TTS · 추론 용어를 정리합니다.

TTS 가 처음이신 분이 **architecture 이후 페이지를 읽을 때 막히지 않도록** 구성했습니다.

순서대로 훑어도 되고, 필요한 용어만 검색해 찾아 읽어도 괜찮습니다.

## 파이프라인의 큰 그림

### TTS (Text-to-Speech)

글(텍스트)을 사람 목소리 오디오로 변환하는 기술의 총칭입니다.

입력은 보통 문장, 출력은 WAV · MP3 같은 음성 파일입니다.

TTS 시스템은 내부적으로 "글자를 발음으로 해석" → "그 발음을 파형으로 합성" 하는 여러 단계를 거칩니다.

HayaKoe 도 이 범주에 속하며, 일본어 입력을 받아 WAV 파형을 생성합니다.

### 음소 (Phoneme)

말의 뜻을 구별하는 가장 작은 소리 단위입니다.

"발" 과 "달" 은 첫 소리 (ㅂ vs ㄷ) 만 달라도 완전히 다른 뜻이 됩니다. 이렇게 **의미를 바꾸는 소리 단위** 가 음소입니다.

한국어는 글자와 실제 발음이 다른 경우가 많습니다. 예를 들어 "같이" 는 두 글자지만 발음은 "가치" 이므로, 글자 수와 음소 수가 꼭 같지는 않습니다.

TTS 모델은 글자가 아니라 음소를 입력으로 받습니다. 글자를 그대로 받으면 "어떤 조건에서 어떻게 읽히는가" 하는 발음 규칙까지 모델이 전부 학습해야 하기 때문입니다. 음소로 미리 변환해 넣으면 모델은 "이 소리를 어떻게 들리게 만들지" 에만 집중할 수 있습니다.

이 변환을 담당하는 모듈이 바로 **G2P** 입니다.

### G2P (Grapheme-to-Phoneme)

글자 (Grapheme) 를 음소 (Phoneme) 로 변환하는 과정, 또는 그 모듈입니다.

한국어의 "같이 → 가치" (구개음화), "독립 → 동닙" (비음화) 같은 규칙, 일본어의 한자 독법 · 연음 규칙 등 언어별 발음 규칙을 전부 여기서 처리합니다.

TTS 파이프라인에서 모델에 입력을 넣기 직전 단계에 해당합니다.

HayaKoe 는 일본어 전용이라 일본어 G2P 를 [pyopenjtalk](./openjtalk-dict) 에 위임합니다.

### 파형 (Waveform)

공기의 압력 변화를 시간 축에 따라 기록한 숫자열입니다. 스피커가 재생할 수 있는 "실제 소리" 그 자체를 의미합니다.

각 숫자는 **특정 순간의 공기 압력 (진폭, amplitude)** 을 나타냅니다. 값이 양수면 기준보다 공기가 압축된 상태, 음수면 팽창된 상태를 의미하고, 절댓값이 클수록 소리가 크게 들립니다. 0 은 무음 (기준 압력) 에 해당합니다.

샘플링 레이트 (sample rate) 가 22,050 Hz 면 1 초 = 22,050 개의 이런 숫자로 표현됩니다. HayaKoe 의 출력은 44,100 Hz 이므로 1 초당 44,100 개입니다.

TTS 의 최종 출력물이 바로 이 파형입니다.

## 모델 구성 요소

### VITS

2021 년 발표된 음성 합성 모델 구조입니다.

이전까지 두 단계 (Acoustic Model + Vocoder) 로 나뉘어 있던 TTS 파이프라인을 **하나의 End-to-End 모델** 로 통합한 것이 핵심 기여입니다.

텍스트 → 파형 변환을 단일 모델이 직접 수행하며, 내부적으로 Text Encoder · Duration Predictor · Flow · Decoder 로 구성됩니다.

HayaKoe 는 VITS 계보의 연장선에 있는 모델입니다.

- **VITS (2021)** — End-to-End TTS 의 출발점.
- **[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)** — Fish Audio 팀이 VITS 에 BERT 를 붙여 **문맥 기반 prosody** 를 보강한 오픈소스 프로젝트.
- **[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)** — litagin02 가 Bert-VITS2 를 포크해 **Style Vector** 를 추가, 같은 화자의 다양한 톤 · 감정을 표현 가능하게 확장. 일본어 특화 변형인 **JP-Extra** 가 품질 우위를 보임.
- **HayaKoe** — Style-Bert-VITS2 JP-Extra 를 **일본어 전용으로 축소** 하고, CPU · 서버 운영에 실용적인 형태로 재구성.

Synthesizer 자체의 모델 구조는 Style-Bert-VITS2 를 그대로 사용하며, HayaKoe 가 추가한 변경은 주로 그 바깥 (ONNX 경로, 추론 파이프라인, 배포 · 소스 단순화) 에 집중되어 있습니다.

### Synthesizer

HayaKoe 에서 **VITS 본체 (Text Encoder + Duration Predictor + Flow + Decoder)** 를 통칭하는 이름입니다.

음소 시퀀스와 BERT 특징을 입력으로 받아, 최종 파형을 만들어내는 부분입니다.

BERT 는 Synthesizer **밖** 에 별도로 존재하고 모든 화자가 공유합니다. 화자마다 달라지는 건 Synthesizer 의 가중치 (약 300 MB) 입니다.

### BERT

Google 이 2018 년 발표한 Transformer 기반 사전학습 언어모델입니다. 문장을 읽고 각 토큰의 문맥 임베딩을 만들어줍니다.

TTS 에서는 **문장의 의미 · 문맥 정보를 합성에 반영** 하기 위해 사용됩니다. 같은 음소열이라도 BERT 덕분에 더 자연스러운 억양 · 강세가 생성됩니다.

HayaKoe 는 일본어 전용 DeBERTa v2 모델 (`ku-nlp/deberta-v2-large-japanese-char-wwm`) 을 사용합니다.

CPU 경로에서는 이 BERT 를 INT8 로 양자화해 ONNX 로 실행합니다.

### Text Encoder

Synthesizer 내부 모듈입니다. 음소 시퀀스를 입력받아 각 음소에 해당하는 **192 차원 은닉 벡터** 를 출력합니다.

Transformer encoder 구조이며, self-attention 으로 음소가 앞뒤 문맥을 참조하면서 합성에 필요한 임베딩을 만듭니다.

개념적으로 BERT 의 축소판이라고 볼 수 있습니다. BERT 는 단어 · 문장 수준, Text Encoder 는 음소 수준이라는 차이가 있습니다.

### Duration Predictor (SDP)

각 음소를 **몇 프레임 동안** 발음할지 예측하는 모듈입니다. "안" 은 5 프레임, "녕" 은 4 프레임 같은 식입니다.

"SDP" 는 **Stochastic Duration Predictor** 의 약자입니다. 결정적 (deterministic) 이 아니라 확률 분포에서 샘플링하기 때문에 같은 문장이라도 매 호출마다 억양 · 속도가 조금씩 달라집니다.

HayaKoe 는 이 모듈을 본래 용도 외에 **문장 경계 pause 예측** 에도 재활용합니다. 자세한 내용은 [문장 경계 pause — Duration Predictor](./duration-predictor) 에서 다룹니다.

### Flow

Synthesizer 내부 모듈입니다. **가역 (invertible) 변환** 이라 정방향 · 역방향 모두 계산 가능한 신경망입니다.

학습 시에는 "정답 음성의 latent → 텍스트 임베딩 공간" 으로 맞춰지고, 추론 시에는 그 역방향을 타서 텍스트 임베딩으로부터 음성 latent 를 생성합니다.

정식 명칭은 **Normalizing Flow** 입니다.

::: warning Flow 와 양자화
HayaKoe 가 Synthesizer 를 FP16 으로 낮추지 않는 주된 이유가 Flow 에 있습니다. Flow 의 `rational_quadratic_spline` 연산이 FP16 에서 부동소수점 오차로 인한 assertion error 를 일으킵니다.

Synthesizer INT8 는 별도의 이유로 제외됐습니다 — Conv1d 중심 구조라 PyTorch dynamic quantization 이 자동 적용되지 않고, static quantization 은 구현 복잡도가 높습니다.
:::

### Decoder (HiFi-GAN)

Synthesizer 의 마지막 모듈입니다. Flow 가 만든 latent 벡터를 입력받아 **실제 파형 (waveform)** 을 생성합니다.

과거에 독립된 Vocoder 로 쓰이던 HiFi-GAN 구조를 VITS 가 모델 안에 통합한 것입니다.

**VITS 가 End-to-End 로 작동할 수 있는 핵심 모듈** 이며, 동시에 TTS 추론 시간의 상당 비중을 차지하는 부분이기도 합니다.

### Style Vector

화자의 "톤 · 말투" 같은 스타일 정보를 하나의 벡터로 압축한 것입니다.

같은 화자라도 "평온", "기쁨", "화남" 등 스타일을 바꿔가며 합성할 수 있습니다.

Style-Bert-VITS2 계열 특유의 구성 요소로, 화자별 safetensors 와 함께 `style_vectors.npy` 로 제공됩니다.

HayaKoe 는 현재 단순화를 위해 **Normal 스타일만** 사용합니다. 다양한 스타일 선택 지원은 추후 개선 예정입니다.

### Prosody (운율)

말의 **억양 · 리듬 · 강세 · 쉼** 을 통틀어 부르는 말입니다.

음소가 "무엇을 발음하는가" 라면, prosody 는 "어떻게 발음하는가" 에 해당합니다.

"진짜?" (올라가는 억양 — 의문) 과 "진짜." (내려가는 억양 — 확언) 은 음소는 같지만 prosody 가 다릅니다.

TTS 가 "로봇 같이" 들리는 가장 흔한 원인이 바로 prosody 가 자연스럽지 못할 때입니다.

Bert-VITS2 계열이 BERT 를 사용하는 주된 이유 중 하나가, 문장 맥락으로부터 prosody 힌트를 얻기 위함입니다.

## 성능 · 실행 용어

### ONNX · ONNX Runtime

**ONNX (Open Neural Network Exchange)** 는 신경망 모델을 **프레임워크 독립적으로 저장** 할 수 있는 표준 포맷입니다.

PyTorch · TensorFlow 등 어디서 학습했든 ONNX 로 export 하면 동일한 그래프로 취급됩니다.

**ONNX Runtime** 은 ONNX 모델을 실제로 실행하는 추론 엔진입니다. C++ 로 작성되어 있어 Python 오버헤드가 적고, 모델 그래프를 분석해 다양한 최적화를 미리 수행합니다.

CPU · CUDA · ARM (aarch64) 등 다양한 실행 장치를 지원합니다.

HayaKoe 의 CPU 경로는 전체가 ONNX Runtime 위에서 동작합니다. 동일한 코드가 x86_64 와 aarch64 에서 공통으로 작동하는 것도 이 덕분입니다.

### 양자화 (Quantization)

모델 가중치의 숫자 표현 정밀도를 낮춰서, 메모리와 연산을 아끼는 기법입니다.

딥러닝 모델 가중치는 보통 다음 정밀도 중 하나로 저장됩니다.

- **FP32** — 32 비트 부동소수. 기본값. 가장 정확하지만 크기가 크다.
- **FP16** — 16 비트 부동소수. FP32 대비 절반 크기.
- **INT8** — 8 비트 정수. FP32 의 약 1/4 크기. 흔히 "Q8" 이라고도 부른다.
- **INT4** — 4 비트 정수. FP32 의 약 1/8 크기. LLM 분야에서 최근 활발히 사용.

비트 수가 줄면 모델 파일 크기와 RAM 사용량도 거의 비례해 줄고, 특정 하드웨어에서는 연산도 빨라집니다.

대신 **정밀도가 떨어지므로 출력 품질이 나빠질 수 있습니다.** 어디까지 양자화해도 품질이 괜찮은지는 모델마다, 또 연산 종류마다 다릅니다.

HayaKoe 는 **BERT 의 MatMul 만 INT8 로 동적 양자화 (Q8 Dynamic Quantization)**, Synthesizer 는 FP32 를 유지하는 선택을 했습니다. 자세한 이유와 실측 효과는 [ONNX 최적화](./onnx-optimization) 에서 다룹니다.

### Kernel Launch Overhead

CPU 에서 GPU 에게 "이 커널을 실행하라" 고 요청하는 데 드는 고정 비용입니다. 실제 계산 시간과는 별개로, 커널 호출 한 건당 수 μs ~ 수십 μs 정도 발생합니다.

커널 하나가 무겁게 계산하는 워크로드에서는 이 비용이 묻힙니다. 하지만 **TTS 처럼 작은 Conv1d 연산이 수백 회 반복되는 경우**, kernel launch overhead 가 전체 시간의 상당 비중을 차지할 수 있습니다.

CUDA Graph · kernel fusion · torch.compile 등이 이 비용을 줄이기 위한 기법들입니다.

### Eager Mode

PyTorch 의 기본 실행 방식입니다. 파이썬 코드가 한 줄씩 실행되면서 그때그때 GPU 커널을 개별 호출합니다.

디버깅이 쉽다는 장점이 있지만, 매 커널마다 Python 디스패치 오버헤드와 kernel launch overhead 가 누적됩니다.

`torch.compile` 은 이 오버헤드를 그래프 수준 최적화로 제거하기 위한 대안입니다.

### torch.compile

PyTorch 2.0 부터 제공되는 **JIT 컴파일러** 입니다.

모델을 첫 호출 시 그래프로 추적하고, 커널을 융합 · 재컴파일해서 이후 호출부터 더 빠르게 실행합니다.

HayaKoe 는 GPU 경로에서 `torch.compile` 을 사용합니다.

첫 호출에는 컴파일 시간이 소요되므로, `prepare(warmup=True)` 로 이 비용을 서빙 시작 단계로 옮길 수 있습니다.

## 기타

### OpenJTalk

나고야공업대학에서 개발한 오픈소스 일본어 TTS 프론트엔드입니다.

일본어 텍스트를 받아 **음소열 · 억양 정보** 를 생성해줍니다. 한자 독법 · 연음 등 일본어 특유의 규칙이 여기에 포함됩니다.

HayaKoe 는 Python 바인딩인 [pyopenjtalk](./openjtalk-dict) 를 통해 이 기능을 사용합니다.
