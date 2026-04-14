# FAQ

자주 묻는 고급 설정 항목을 모았습니다.

## 캐시 경로 바꾸기

기본 캐시 경로(`./hayakoe_cache/`)가 마음에 들지 않는다면 두 가지 방법이 있습니다.

```bash
# 환경변수
export HAYAKOE_CACHE=/var/cache/hayakoe
```

```python
# 코드에서 직접
tts = TTS(cache_dir="/var/cache/hayakoe")
```

HuggingFace · S3 · 로컬 소스 모두 같은 루트 아래 저장됩니다.

## Private HuggingFace 나 S3 에서 모델 받기

private HF repo 의 화자를 쓰거나 S3 버킷에 올려둔 모델을 받으려면 소스 URI 를 지정하면 됩니다.

S3 소스를 쓸 거라면 extras 를 먼저 설치하세요.

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",
        hf_token="hf_...",                     # private HF repo 용
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

S3 호환 엔드포인트 (MinIO, Cloudflare R2 등) 는 `AWS_ENDPOINT_URL_S3` 환경변수로 지정하면 됩니다.

## 화자를 여러 명 올리면 메모리가 얼마나 더 드나요

BERT 는 모든 화자가 공유하기 때문에, 화자당 늘어나는 건 훨씬 가벼운 synthesizer 몫뿐입니다.

궁금해서 제 로컬에서 직접 벤치 스크립트를 돌려 봤는데, 숫자는 하드웨어·OS·torch 버전·ORT 빌드에 따라 달라질 수 있으니 **절대값보다는 증가 경향** 으로만 봐 주세요.

::: info 측정 환경
- GPU — NVIDIA RTX 3090 (24 GB), Driver 580.126.09
- 텍스트 — 일본어 두 문장 (문장 경계 포함, 약 50 자)
- 화자 — `jvnv-F1-jp`, `jvnv-F2-jp`, `jvnv-M1-jp`, `jvnv-M2-jp`
- 각 시나리오는 별도의 Python 프로세스로 실행 (힙 오염 방지)
:::

### 화자 수에 따른 메모리 (로드만 한 상태)

| 화자 수 | CPU (ONNX) RAM | GPU (PyTorch) RAM | GPU VRAM |
| :------ | -------------: | ----------------: | -------: |
| 1 명    | ≈ 1.7 GB       | ≈ 1.3 GB          | ≈ 1.8 GB |
| 4 명    | ≈ 2.8 GB       | ≈ 1.5 GB          | ≈ 2.6 GB |

화자 3 명이 더 붙었을 때 늘어난 양을 3 으로 나누면, 화자 하나당 대략 이 정도입니다.

- **CPU RAM** — 약 +360 MB / 화자
- **GPU VRAM** — 약 +280 MB / 화자

### 4 명을 동시에 돌리면

실제 서비스에서는 여러 화자가 한꺼번에 돌 수도 있어서, **순차 4 회** 와 **스레드 4 개로 동시** 를 따로 측정했습니다 (합성 중 피크 기준).

| 시나리오    | CPU RAM peak | GPU RAM peak | GPU VRAM peak |
| :---------- | -----------: | -----------: | ------------: |
| 1 화자 합성 | ≈ 2.0 GB     | ≈ 2.3 GB     | ≈ 1.7 GB      |
| 4 화자 순차 | ≈ 3.2 GB     | ≈ 2.1 GB     | ≈ 2.6 GB      |
| 4 화자 동시 | ≈ 3.2 GB     | ≈ 2.2 GB     | ≈ 2.8 GB      |

동시 실행이라도 메모리가 4 배가 되는 일은 없습니다.

CPU 쪽은 ORT 가 내부에서 이미 병렬화를 하고 있어서 "순차 vs 동시" 차이가 거의 없고, GPU VRAM 도 동시 실행이 +200 MB 정도 더 드는 선에서 멈춥니다.

### 직접 재현해 보기

리포지토리의 `docs/benchmarks/memory/` 아래에 스크립트가 들어 있습니다.

```bash
# 단일 시나리오
python docs/benchmarks/memory/run_one.py --device cpu --scenario idle4

# 전체 10 시나리오 (CPU/GPU × idle1/idle4/gen1/seq4/conc4) 를 별도 프로세스로
bash docs/benchmarks/memory/run_all.sh
```

- `run_one.py` 는 한 시나리오를 돌고 JSON 한 줄을 찍습니다.
- `run_all.sh` 는 모든 시나리오를 별도의 Python 프로세스로 돌려 결과를 스크립트 옆의 `results_<timestamp>.jsonl` 에 모읍니다.
- RAM 은 `psutil` 로 50 ms 마다 RSS 를 폴링해서 피크를 잡고, VRAM 은 `torch.cuda.max_memory_allocated()` 값을 그대로 가져옵니다.
- `gen*` 시나리오는 워밍업 후 `torch.cuda.reset_peak_memory_stats()` 를 불러, torch.compile 콜드스타트를 피크에서 제외합니다.

측정이 필요하면 본인 환경에서 한 번 돌려 보고 숫자를 비교해 보는 쪽이 가장 정확합니다.
