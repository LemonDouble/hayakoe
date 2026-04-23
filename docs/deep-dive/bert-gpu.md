# BERT GPU 유지 & 배치 추론

GPU 경로에서는 두 가지가 주요 최적화 포인트입니다.

- **원본 SBV2 의 불필요한 CPU 전환을 제거** 하여 BERT 출력을 GPU 텐서로 유지하는 것
- **다문장을 배치로 묶어 BERT 를 1 회만 호출** 하여 kernel launch overhead (GPU 에 연산을 요청할 때마다 발생하는 고정 비용) 를 줄이는 것

## 왜 문제인가

원본 SBV2 는 기본적으로 **텍스트를 통째로 한 번에 합성** 합니다 (`line_split=False`).

BERT 도 1 회만 호출되므로 배치화가 필요 없는 구조였습니다.

HayaKoe 는 prosody (운율) 안정성을 위해 **구두점 기준 문장 분할** 을 도입했고, 이에 따라 BERT 가 문장 수만큼 반복 호출되는 문제가 새로 생겼습니다.

### BERT 출력의 불필요한 CPU 전환

원본 SBV2 의 BERT feature 추출 코드에는 다음과 같은 부분이 있습니다.

```python
# 원본 SBV2 (style_bert_vits2/nlp/japanese/bert_feature.py)
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```

BERT 를 GPU 에서 forward 한 뒤, 출력 텐서에 `.cpu()` 를 호출하여 **매번 CPU 로 내리고 있었습니다.**

이 출력은 이후 Synthesizer 에 전달되는데, Synthesizer 는 GPU 에서 동작하므로 다시 GPU 로 올려야 합니다.

결과적으로 문장마다 **GPU → CPU → GPU 왕복** 이 발생하며, 이 불필요한 왕복 자체가 병목이 됩니다.

### 문장별 개별 BERT 호출

다문장을 분할한 뒤 각 문장에 대해 BERT 를 따로 호출하면, GPU kernel launch 가 문장 수만큼 반복됩니다.

kernel launch 는 GPU 에 연산을 요청할 때마다 발생하는 고정 비용입니다.

문장이 짧으면 실제 연산 시간보다 launch overhead 의 비중이 커져, 문장 수에 비례해 비효율이 누적됩니다.

## 구현

### `.cpu()` 제거 — GPU 텐서 유지

원본의 `.cpu()` 호출을 제거하여 BERT 출력이 GPU 텐서 그대로 Synthesizer 에 전달되도록 수정했습니다.

```python
# 원본 SBV2
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()    # GPU → CPU

# HayaKoe
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].float()  # GPU 유지
```

BERT 모델 자체도 `prepare()` 시점에 GPU 로 로드되어 추론이 끝날 때까지 유지됩니다.

BERT 모델은 **글로벌 싱글턴** 으로 관리되므로, 화자를 여러 명 로드해도 BERT 는 한 번만 올라가고 모든 화자가 공유합니다.

### 다문장 BERT 배치화

HayaKoe 가 사용하는 BERT (DeBERTa) 는 HuggingFace Transformer 모델이라 기본적으로 batch 입력을 지원합니다.

이를 활용하여, 다문장 합성 시 각 문장의 BERT 를 개별 호출하는 대신 **모든 문장을 하나의 배치로 묶어 1 회에 처리** 합니다.

여러 문장을 tokenizer 에 한꺼번에 넣어 padding 된 배치 입력을 만들고, BERT 를 **1 회만** forward 합니다.

ONNX 경로에서도 동일한 배치 로직이 구현되어 있습니다.

## 개선 효과

### GPU 배치 추론 속도

동일 하드웨어에서 **순차 (sequential) vs 배치 (batched)** 비교입니다 (5 회 평균).

| 문장 수 | 순차 | 배치 | 속도 향상 |
|---|---|---|---|
| 2 | 0.447 s | 0.364 s | **1.23×** |
| 4 | 0.812 s | 0.566 s | **1.43×** |
| 8 | 1.598 s | 1.121 s | **1.43×** |
| 16 | 2.972 s | 2.264 s | **1.31×** |

kernel launch 오버헤드가 1 회로 통합된 효과로, +23 % ~ +43 % 의 속도 향상이 나타납니다.

### GPU 메모리

배치화가 메모리를 추가로 소비하지 않는지 확인했습니다.

| 문장 수 | 순차 peak | 배치 peak | 차이 |
|---|---|---|---|
| 2 | 1,662.2 MB | 1,661.9 MB | −0.3 MB |
| 4 | 1,661.8 MB | 1,662.2 MB | +0.4 MB |
| 8 | 1,697.7 MB | 1,699.0 MB | +1.3 MB |
| 16 | 1,934.3 MB | 1,934.3 MB | 0 MB |

순차와 배치 간 차이가 **1.3 MB 이내** 로, 사실상 동일합니다.

### CPU 에서는 효과 없음

동일 실험을 CPU (ONNX) 에서 반복하면 배치화 효과가 거의 나타나지 않습니다.

| 문장 수 | 순차 | 배치 | 속도 차이 |
|---|---|---|---|
| 2 | 2.566 s | 2.564 s | 1.00× |
| 4 | 5.464 s | 4.855 s | 1.13× |
| 8 | 10.647 s | 11.783 s | 0.90× |
| 16 | 24.559 s | 24.195 s | 1.01× |

ONNX Runtime 의 그래프 최적화가 이미 충분히 강해 Python 레벨 dispatch overhead 가 병목이 아니며, 배치 시 padding 오버헤드가 이득을 상쇄합니다.

GPU 에서는 배치화를 유지하고, CPU 에서는 이득 · 손해 모두 크지 않아 백엔드 간 코드 단일성을 위해 동일 경로를 유지합니다.
