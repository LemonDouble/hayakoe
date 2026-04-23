# 서버로 배포

HayaKoe 는 **싱글톤 TTS 인스턴스 + 빌드 타임 가중치 내장** 패턴으로 서버 환경에 맞게 설계되어 있습니다. FastAPI · Docker 조합으로 간결한 API 서버를 띄울 수 있습니다.

## 설계 요점

### 1. 모델은 한 번만 로드 (싱글톤)

TTS 모델은 한 번 로드하는 데 시간이 꽤 걸립니다. GPU 환경에서는 컴파일 단계까지 포함해 수십 초가 걸릴 수도 있습니다. 요청마다 새 인스턴스를 만들면 실서비스가 사실상 불가능해지므로, **프로세스 수명 동안 하나만** 유지해서 모든 요청이 공유해야 합니다.

실제 코드는 FastAPI 의 lifespan 훅에서 `TTS(...).load(...).prepare(warmup=True)` 로 싱글톤을 빌드해 `app.state.tts` 에 저장해 두고, 핸들러들은 이 하나의 인스턴스를 재사용하는 구조입니다.

동시성은 걱정하지 않아도 됩니다. `Speaker` 는 내부에 `threading.Lock` 을 가지고 있어, 같은 화자에 대한 동시 요청은 자동으로 직렬화되고 다른 화자끼리는 병렬로 돌아갑니다 — 별도의 풀·큐 구현이 필요 없습니다.

::: details GPU 백엔드는 torch.compile 까지 함께 준비합니다
`TTS.prepare()` 는 CUDA 백엔드에서 모델 로드 뿐 아니라 `torch.compile` 을 모든 화자와 BERT 에 일괄 적용합니다.

`warmup=True` 를 주면 더미 추론을 1회 선행해 컴파일 비용을 prepare 단계로 앞당깁니다. 이 과정 자체가 수십 초 걸릴 수 있어서 반드시 앱 부팅 시점에 한 번만 해야 합니다. **요청마다 TTS 를 새로 만들면 매번 재컴파일이 일어나** 서버가 사실상 마비됩니다.

CPU 백엔드는 ONNX Runtime 을 쓰므로 별도 컴파일 단계가 없고, prepare 가 훨씬 빠릅니다.
:::

→ 구현은 [FastAPI 통합](/deploy/fastapi)

### 2. 가중치는 빌드 타임에 이미지로 굽기

Docker 이미지 하나에 **모델 가중치까지 전부 담아 두고**, 런타임에는 외부 네트워크 없이 바로 띄울 수 있게 하는 것이 HayaKoe 의 권장 운영 방식입니다.

이를 위해 `TTS.pre_download(device=...)` — "초기화는 하지 않고 캐시만 채우는" 메서드를 제공합니다. Docker 빌드 단계에서 호출해 필요한 화자 파일 전부를 이미지 안에 박아두면, 런타임 컨테이너는 HF · S3 에 접근할 필요가 없습니다.

오프라인 환경, 방화벽 안쪽, HF·S3 자격 증명을 런타임 컨테이너에 노출하고 싶지 않은 경우에 특히 깔끔한 패턴입니다.

→ 구현은 [Docker 이미지](/deploy/docker)

## 섹션 구성

| 페이지 | 내용 |
|---|---|
| [FastAPI 통합](/deploy/fastapi) | lifespan 에서 싱글톤 로드, `agenerate` / `astream`, 동시성 |
| [Docker 이미지](/deploy/docker) | 빌드 타임 `pre_download`, BuildKit secret, 멀티 스테이지 |
| [백엔드 선택](/deploy/backend) | CPU(ONNX) vs GPU(PyTorch) 트레이드오프 |

