# 백엔드 선택 (CPU vs GPU)

HayaKoe 는 CPU (ONNX Runtime) 와 GPU (PyTorch + `torch.compile`) 두 백엔드를 지원합니다. 코드 레벨에서는 `device` 파라미터 하나 차이입니다.

```python
tts_cpu = TTS(device="cpu").load("tsukuyomi").prepare()
tts_gpu = TTS(device="cuda").load("tsukuyomi").prepare()
```

다만 **설치 프로파일부터 다릅니다** — CPU 는 `pip install hayakoe` 만으로 되지만, GPU 는 `hayakoe[gpu]` + PyTorch CUDA 빌드가 추가로 필요합니다. 한 환경에 양쪽을 같이 깔아 돌리는 것도 가능하지만, 실제 배포에서는 대개 타겟 환경에 맞춰 **한쪽만** 설치합니다 (자세한 건 [설치 — CPU vs GPU](/quickstart/install)).

그 아래 구조도 완전히 다릅니다. 어느 쪽이 내 배포 환경에 맞는지 결정하는 기준을 정리합니다.

## CPU (ONNX) 가 맞는 경우

- **GPU 가 없는 서버 환경** — 일반적인 웹 호스팅, VPS, 관리형 컨테이너 플랫폼처럼 CUDA 지원이 없는 환경에서 바로 동작합니다.
- **이미지 크기를 최소화해야 하는 경우** — PyTorch + CUDA 스택은 수 GB 수준이지만, ONNX Runtime 만 포함한 이미지는 수백 MB 대로 줄어듭니다.
- **낮은 동시 요청을 처리하는 워크로드** — 개인 프로젝트나 사내 툴처럼 동시 부하가 크지 않은 경우, CPU 만으로도 충분한 처리량을 확보할 수 있습니다.
- **콜드 스타트가 짧아야 할 때** — ONNX 경로는 `torch.compile` 컴파일 단계가 없어, 프로세스가 뜨는 즉시 `prepare()` 가 끝나고 바로 합성을 받을 수 있습니다. GPU 경로는 첫 `prepare()` 에서 수십 초의 그래프 컴파일 시간을 감수해야 하므로, 오토스케일·서버리스 환경에서 체감 차이가 큽니다.

::: details CPU 경로 구성
- **BERT** — `bert_q8.onnx` (Q8 양자화 DeBERTa), ONNX Runtime `CPUExecutionProvider`
- **Synthesizer** — `synthesizer.onnx` (ONNX 로 export 된 VITS 디코더)
- **Duration Predictor** — `duration_predictor.onnx`
:::

## GPU (PyTorch) 가 맞는 경우

- **낮은 지연이 요구되는 실시간 서비스** — 사용자 대면 응답, 대화형 UI 등 단일 요청의 응답 시간이 체감 품질에 직결되는 경우.
- **높은 동시 처리량이 필요한 환경** — 하나의 GPU 에서 여러 화자를 병렬로 합성할 수 있어, CPU 대비 동시 요청 수용 폭이 큽니다.
- **이미 GPU 인프라가 구축된 환경** — 별도 투자 없이 기존 자원을 활용할 수 있으므로, 동일 비용으로 더 나은 지연·처리량을 얻을 수 있습니다.
- **긴 문장을 반복 합성하는 워크로드** — `torch.compile` 의 그래프 최적화 이득이 합성 길이에 비례해 커집니다.

::: details GPU 경로 구성
- **BERT** — FP32 DeBERTa 가 GPU VRAM 에 올라가 임베딩을 계산합니다. 양자화하지 않아 CPU ONNX 경로보다 정밀도가 약간 높습니다.
- **Synthesizer** — PyTorch VITS 디코더. `torch.compile` 이 적용됩니다.
- **Duration Predictor** — Synthesizer 와 같은 PyTorch 경로이며, `torch.compile` 대상에 함께 포함됩니다.
:::

::: tip GPU 백엔드의 콜드 스타트 줄이기
GPU 백엔드의 첫 `prepare()` 는 모델 다운로드 + `torch.compile` 초기화가 맞물려 수십 초가 걸릴 수 있습니다. 실제 서비스에서는 아래 두 가지로 이 비용을 미리 치러두는 것이 권장됩니다.

- **Docker 빌드 시 `pre_download()`** — 빌드 단계에서 가중치를 이미지 안에 박아 두면, 런타임 `prepare()` 는 HF · S3 접근 없이 캐시에서 바로 로드합니다. 이미지가 뜨자마자 네트워크 지연 없이 초기화가 진행됩니다. (→ [Docker 이미지](/deploy/docker))
- **`prepare(warmup=True)`** — prepare 시점에 더미 추론을 선행해 `torch.compile` 컴파일과 CUDA graph 캡처까지 prepare 로 앞당깁니다. prepare 자체는 조금 더 걸리지만 **첫 실제 요청이 warmup 비용을 떠안지 않게** 됩니다. (→ [FastAPI 통합](/deploy/fastapi))
:::

## 나란히 비교

| 항목 | CPU (ONNX) | GPU (PyTorch + compile) |
|---|---|---|
| 설치 | `pip install hayakoe` | `pip install hayakoe[gpu]` |
| 이미지 크기 | 수백 MB | 수 GB |
| 콜드 스타트 | 빠름 (초) | 느림 (수십 초, 첫 compile) |
| 단일 요청 지연 | 보통 | 최저 |
| 동시 처리량 | 코어 수 제한 | GPU 1대에서 병렬 |
| 메모리 (화자 1개 로드) | ≈ 1.7 GB RAM | ≈ 1.3 GB RAM + 1.8 GB VRAM |
| 메모리 (화자당 증가) | +300~400 MB RAM | +250~300 MB VRAM |
| 필요 하드웨어 | 어떤 CPU 든 | NVIDIA GPU + CUDA |

::: info 구체 수치는 벤치마크에서
배속·메모리·지연 수치는 하드웨어에 강하게 의존합니다.

- 배속 측정 — [내 머신에서 벤치마크](/quickstart/benchmark)
- 메모리 측정 (실측 표와 재현 스크립트) — [FAQ — 화자를 여러 명 올리면 메모리가 얼마나 더 드나요](/faq/#화자를-여러-명-올리면-메모리가-얼마나-더-드나요)
:::

