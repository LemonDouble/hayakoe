# 설치 — CPU vs GPU

HayaKoe 는 **CPU 전용**과 **GPU(CUDA)** 두 가지 설치 프로파일을 지원합니다.

본인 환경에 맞는 쪽만 고르면 됩니다.

## 어느 쪽을 골라야 할까요?

- **CPU** — GPU 가 없거나, 있더라도 일단 한번 돌려보고 싶을 때
- **GPU** — 대량 처리를 해야 하거나, 실시간성이 중요할 때

::: tip 헷갈릴 때 기본값
고민되면 **CPU** 로 시작하세요.

나중에 GPU extras 만 추가로 설치하면 됩니다.
:::

## CPU 설치 (기본)

PyTorch 가 필요 없어 설치가 짧고 이미지도 가벼워집니다.

::: code-group
```bash [pip]
pip install hayakoe
```
```bash [uv]
uv add hayakoe
```
```bash [poetry]
poetry add hayakoe
```
:::

::: tip arm64 도 그대로 동작합니다
Raspberry Pi (4B 이상) 같은 aarch64 Linux 환경에서도 같은 명령 하나로 설치되고 CPU 추론이 돕니다.

실측 수치는 [라즈베리파이 4B 벤치마크](./benchmark#라즈베리파이-4b-에서는-어떨까) 를 참고하세요.
:::

### 확인

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
audio = tts.speakers["jvnv-F1-jp"].generate("テスト、テスト。")
audio.save("test.wav")
print("OK")
```

처음 실행 시 [HuggingFace 공식 repo](https://huggingface.co/lemondouble/hayakoe)에서 BERT·Synthesizer·스타일 벡터가 자동으로 캐시 폴더에 내려받아집니다.

기본 캐시 경로는 현재 디렉터리의 `hayakoe_cache/` 입니다.

## GPU 설치 (CUDA)

### 사전 준비

GPU 모드는 PyTorch CUDA 빌드를 씁니다.

필요한 건 **NVIDIA 드라이버 하나** 뿐입니다.

- CUDA Toolkit 은 별도 설치할 필요가 없습니다 — PyTorch 휠에 필요한 CUDA 런타임이 함께 들어있습니다.
- 다만 본인 드라이버가 설치하려는 CUDA 버전을 지원해야 합니다.

드라이버가 설치되어 있는지 확인:

```bash
nvidia-smi
```

정상적으로 설치되어 있다면 다음과 같은 출력을 보실 수 있습니다.

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:06:00.0 Off |                  N/A |
| 53%   33C    P8             38W /  390W |    1468MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

첫 줄 우측의 `CUDA Version: 13.0` 이 본인 드라이버가 지원하는 **최대 CUDA 버전** 입니다 (위 예시라면 13.0).

::: tip CUDA 버전 고르기
`nvidia-smi` 로 확인한 버전 이하로 PyTorch CUDA 빌드를 고르면 됩니다.

아래 설치 예시의 `cu126` 자리에 본인에게 맞는 버전을 넣으세요 (예: `cu118`, `cu121`, `cu124`, `cu128`).

지원 조합은 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/) 에서 고를 수 있습니다.
:::

### 설치

`hayakoe[gpu]` extras 는 `safetensors` 만 추가할 뿐 `torch` 를 당겨오지 않습니다.

두 줄을 따로 설치하면 되고, 순서는 상관없습니다.

::: code-group
```bash [pip]
pip install hayakoe[gpu]
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
```bash [uv]
uv add hayakoe --extra gpu
uv add torch --index https://download.pytorch.org/whl/cu126
```
```bash [poetry]
poetry add hayakoe -E gpu
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
:::

### 확인

```python
from hayakoe import TTS

tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ完了。").save("gpu_test.wav")
```

::: warning 첫 요청은 느릴 수 있어요
GPU 모드에서는 첫 `generate()` 호출이 평소보다 몇 초 정도 더 걸릴 수 있습니다.

두 번째 호출부터는 정상 속도가 나옵니다.

서버로 띄울 거라면 시작 직후 더미 `generate()` 를 한 번 불러 "워밍업" 해두는 걸 권장합니다.
:::

::: details 왜 첫 호출이 느린가요? (torch.compile 배경)
HayaKoe 는 GPU 모드일 때 `prepare()` 시점에 PyTorch 의 `torch.compile` 을 자동으로 붙입니다.

`torch.compile` 은 PyTorch 2.0 에서 추가된 JIT 컴파일러로, 모델 실행 그래프를 추적해 한 번 컴파일한 뒤 그 결과를 재사용하는 방식입니다.

덕분에 추론 속도가 향상되지만, 대가로 **첫 호출 때 그래프를 트레이싱·컴파일하는 시간** 이 추가로 듭니다.

한 번 컴파일된 그래프는 프로세스가 살아있는 동안 캐시되므로, 두 번째 호출부터는 그 오버헤드 없이 바로 실행됩니다. 따라서 실서비스에서는 컨테이너·프로세스가 뜬 직후에 짧은 문장으로 더미 호출을 돌려 워밍업을 끝내 두는 것이 일반적입니다.

```python
# FastAPI lifespan, Celery worker 초기화 등에서
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ")  # 결과는 버려도 됨
```

CPU (ONNX) 모드에서는 `torch.compile` 을 쓰지 않으므로 이 워밍업 단계가 필요 없습니다.
:::

여기까지 되면 다음 단계로: [첫 음성 만들기 →](./first-voice)
