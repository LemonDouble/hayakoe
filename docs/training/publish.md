# ④ 배포 (HF · S3 · 로컬)

학습이 끝나면 `<dataset>/exports/<model_name>/` 아래에 최종 모델 파일이 모여 있습니다.

이 폴더를 **HuggingFace Hub / S3 / 로컬 폴더** 중 한 곳에 올리고, 다른 머신에서 `TTS().load("내_이름")` 한 줄로 다시 받아쓸 수 있게 만들어 주는 게 `cli publish` 의 역할입니다.

손으로 HF CLI·S3 CLI 를 다루고 레포 구조를 외우고 업로드 검증까지 해야 하는 과정을 하나의 인터랙티브 플로우로 묶어 놓았습니다.

## 실행

```bash
uv run poe cli
```

메인 메뉴에서 **화자 배포** 를 선택하면 아래 순서대로 묻습니다.

1. 배포할 데이터셋 (또는 외부 폴더)
2. 백엔드 — CPU / GPU / CPU + GPU
3. 체크포인트
4. 화자 이름
5. 목적지 + 자격 증명
6. 요약 패널 → 확인
7. 자동 업로드 → 실제 합성 검증

각 단계는 아래에서 다룹니다.

## 1. 배포 대상 고르기

두 종류의 대상이 보입니다.

- **학습 dataset** — `data/dataset/<name>/exports/<model>/` 에 최종 파일이 있는 데이터셋이 자동으로 리스트업됩니다.
- **📁 다른 폴더에서 직접 선택** — 학습은 다른 곳에서 했고 HayaKoe 형식 폴더만 가지고 있을 때, 경로를 직접 입력합니다.

::: details 외부 폴더로 가져갈 때 필요한 파일
```
<my-folder>/
├── config.json                # 필수
├── style_vectors.npy          # 필수
├── *.safetensors              # 필수 (하나 이상)
├── synthesizer.onnx           # 선택 (있으면 재사용)
└── duration_predictor.onnx    # 선택 (있으면 재사용)
```
:::

## 2. 백엔드 선택

```
CPU (ONNX)        — GPU 없는 서버/로컬용
GPU (PyTorch)     — 최저 지연
CPU + GPU (권장)  — 두 환경 모두에 배포
```

`CPU + GPU` 를 고르면 같은 레포에 두 백엔드용 파일이 **함께** 올라갑니다. 런타임에서 `TTS(device="cpu")` 로 만들면 ONNX 쪽만, `TTS(device="cuda")` 로 만들면 PyTorch 쪽만 자동으로 받아갑니다.

**한 번만 올려두면 두 환경에서 같은 이름으로 재사용** 할 수 있으니, 특별한 이유가 없으면 이 옵션을 고르세요.

두 백엔드의 차이는 [백엔드 선택](/deploy/backend) 에서 자세히 다룹니다.

## 3. 체크포인트와 화자 이름

- 체크포인트가 1개면 자동 선택, 여러 개면 고릅니다 (보통 [③ 품질 리포트](/training/quality-check) 에서 고른 것).
- **화자 이름** 은 런타임에서 `TTS().load("내_이름")` 할 때 쓸 식별자입니다. 간결하고 소문자-하이픈 스타일을 권장합니다 (예: `tsukuyomi`).

## 4. 목적지 선택

세 가지 옵션이 있습니다. 처음 한 번만 자격 증명을 입력하면 `dev-tools/.env` 에 `chmod 600` 으로 저장되어, 다음부터는 프롬프트가 건너뛰어집니다.

### HuggingFace Hub

레포 경로(`org/repo` 또는 `hf://org/repo`)와 **write 권한 토큰** 을 입력합니다. `@<revision>` 으로 브랜치/태그를 지정할 수도 있습니다.

::: details 지원 URL 형식 & 저장되는 환경 변수
허용되는 URL 형식:

- `lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices@main`
- `https://huggingface.co/lemondouble/hayakoe-voices`
- `https://huggingface.co/lemondouble/hayakoe-voices/tree/dev`

저장되는 `.env` 예시:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # write 권한 HuggingFace 액세스 토큰
HAYAKOE_HF_REPO=lemondouble/hayakoe-voices       # 화자 파일이 업로드될 HF 레포 (org/repo 형식)
```
:::

### AWS S3

버킷 이름 (+ 선택적 prefix) 과 AWS 자격 증명(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) 을 입력합니다. 엔드포인트 URL 은 비워 두면 됩니다.

### S3-호환 스토리지 (R2, MinIO 등)

Cloudflare R2, MinIO, Wasabi 같은 S3-호환 스토리지를 쓸 때는 **엔드포인트 URL 을 함께 입력** 합니다.

- Cloudflare R2 — `https://<account>.r2.cloudflarestorage.com`
- MinIO — `http://<host>:9000`

버킷·자격 증명 입력은 AWS S3 와 동일합니다.

::: details 저장되는 환경 변수 예시
**AWS S3**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                # 화자 파일이 업로드될 S3 버킷 이름
HAYAKOE_S3_PREFIX=hayakoe-voices               # 버킷 내 경로 prefix (비워 두면 버킷 루트)
AWS_ACCESS_KEY_ID=<your_access_key_here>       # AWS 액세스 키 ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>   # AWS 시크릿 액세스 키
AWS_REGION=ap-northeast-2                      # S3 리전 (예시는 서울)
# AWS_ENDPOINT_URL_S3 는 비워 둠 (AWS S3 는 자동 결정)
```

**S3-호환 (Cloudflare R2)**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                                 # 업로드될 R2 버킷 이름
HAYAKOE_S3_PREFIX=hayakoe-voices                                # 버킷 내 경로 prefix (비워 두면 버킷 루트)
AWS_ACCESS_KEY_ID=<your_access_key_here>                        # R2 대시보드에서 발급한 Access Key ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>                    # R2 Secret Access Key
AWS_REGION=auto                                                 # R2 는 항상 auto
AWS_ENDPOINT_URL_S3=https://abc123def.r2.cloudflarestorage.com  # R2 엔드포인트 (계정별 고유)
```
:::

### 로컬 디렉토리

네트워크 업로드 없이 로컬 경로로만 복사합니다. NFS 공유 볼륨이나 내부 네트워크 드라이브에 두고 팀이 같이 쓰는 시나리오에 적합합니다. 런타임에서는 `file:///...` URI 로 접근합니다.

::: details 저장되는 환경 변수 예시
```env
HAYAKOE_LOCAL_PATH=/srv/hayakoe-voices   # 화자 파일을 복사할 로컬 디렉토리 경로
```
:::

## 5. 레포 구조

`CPU + GPU` 로 배포하면 레포 안에 ONNX 폴더와 PyTorch 폴더가 함께 들어갑니다. 같은 레포에 여러 화자를 두고 같이 운영할 수 있습니다 (`speakers/voice-a/`, `speakers/voice-b/`, ...).

::: details 내부 구조
```
<repo-root>/
├── pytorch/
│   └── speakers/
│       └── <speaker-name>/
│           ├── config.json
│           ├── style_vectors.npy
│           └── *.safetensors
└── onnx/
    └── speakers/
        └── <speaker-name>/
            ├── config.json
            ├── style_vectors.npy
            ├── synthesizer.onnx
            └── duration_predictor.onnx
```

BERT 모델도 `pytorch/bert/` 와 `onnx/bert/` 아래에 공유 위치로 함께 올라갑니다. 런타임은 화자별 파일과 공통 BERT 를 같은 캐시 규칙으로 내려받습니다.
:::

## 6. ONNX export (자동)

CPU 백엔드(`CPU (ONNX)` · `CPU + GPU`) 를 고르면, 업로드 직전에 PyTorch 체크포인트를 ONNX 로 자동 변환합니다. 별도의 `cli export` 커맨드는 없습니다.

변환 결과는 `<dataset>/onnx/` 에 캐시되어, 같은 체크포인트를 다시 publish 할 때는 재사용됩니다. 강제로 재변환하고 싶으면 이 폴더를 지우고 publish 를 다시 돌리세요.

::: details 내부 동작 — 변환되는 모델과 방식
화자 고유의 두 모델이 `dev-tools/cli/export/exporter.py` 를 통해 opset 17 로 export 됩니다.

#### 변환 대상 — 화자 고유의 두 모델

**Synthesizer (VITS 디코더)**

음소 시퀀스 + BERT 임베딩 + 스타일 벡터를 입력받아 실제 파형(waveform) 을 만들어내는 핵심 모델입니다. 화자마다 전부 다르게 학습되므로, 배포 대상의 대부분을 이 모델이 차지합니다.

- 함수: `export_synthesizer`
- 출력: `synthesizer.onnx` (+ 경우에 따라 `synthesizer.onnx.data`)

**Duration Predictor**

각 음소가 얼마나 오래 발음되어야 할지를 예측합니다. 이 예측이 정확하지 않으면 문장 경계의 pause·템포 처리가 어색해집니다.

- 함수: `export_duration_predictor`
- 출력: `duration_predictor.onnx`

#### `synthesizer.onnx.data` 는 뭔가요?

ONNX 는 내부적으로 Protobuf 로 직렬화되는데, Protobuf 에는 **단일 메시지 2GB 제한** 이 있습니다. Synthesizer 의 가중치가 이 임계치를 넘으면, 그래프 구조만 `.onnx` 에 두고 **대형 텐서는 옆의 `.data` 파일로 외부화** 합니다.

- 두 파일은 **항상 같은 폴더에 함께 있어야** 합니다 (분리 이동 금지)
- 모델 크기에 따라 `.data` 가 아예 안 생기는 경우도 있습니다
- 런타임은 `.onnx` 만 지정해 로드해도 같은 폴더의 `.data` 를 자동으로 함께 읽습니다

#### BERT 는 화자마다 만들지 않고 공용

BERT (DeBERTa) 는 화자와 무관한 일본어 언어 모델입니다. 모든 화자가 공용으로 쓰는 **Q8 양자화 ONNX** (`bert_q8.onnx`) 를 HuggingFace 의 공용 위치에서 내려받아 쓰고, publish 단계에서 화자마다 새로 변환하지 않습니다.

- Q8 양자화 덕에 CPU 에서도 실시간에 가까운 지연으로 임베딩을 뽑아낼 수 있음
- 모든 화자가 같은 BERT 를 공유하므로 레포마다 중복 저장할 필요 없음

즉, 이 단계에서 실제로 변환되는 대상은 **화자 고유의 Synthesizer + Duration Predictor 두 개뿐** 입니다.

#### 트레이싱에 시간이 걸리는 이유

ONNX export 는 "실제로 모델을 한 번 통과시키면서 연산 그래프를 기록" 하는 **트레이싱** 방식입니다. Synthesizer 는 구조가 복잡해서 수십 초~수 분이 걸릴 수 있습니다.

같은 체크포인트를 다른 이름·다른 목적지로 여러 번 publish 하는 경우가 많기 때문에, 한 번 변환한 결과는 `<dataset>/onnx/` 에 캐시되어 재사용됩니다.

#### 스크립트로 직접 export 하기

두 export 함수는 공개되어 있어 스크립트로 직접 호출할 수도 있습니다. 하지만 publish 플로우가 같은 일을 자동으로 하므로, 특별한 이유가 없으면 publish 를 쓰는 것을 권장합니다. 직접 호출 경로는 향후 구조가 바뀔 수 있습니다.
:::

## 7. 덮어쓰기 확인

목적지에 이미 같은 이름의 `speakers/<speaker-name>/` 가 있으면 **덮어쓸지 먼저 물어봅니다**. 승인하면 해당 화자 디렉토리만 깨끗하게 지우고 새로 올립니다 — 같은 레포에 있는 다른 화자는 건드리지 않습니다.

README 도 같은 원칙입니다. 레포 루트에 README 가 없으면 4개 국어(ko/en/ja/zh) 템플릿을 자동 생성해 같이 올리고, 이미 있으면 diff 를 보여준 뒤 덮어쓸지 물어봅니다.

## 8. 업로드 후 자동 검증

업로드가 끝나면 **올린 파일로 실제로 합성이 되는지** 를 자동으로 확인합니다.

CPU + GPU 모두 선택했다면 두 백엔드를 각각 검증하고, 결과 wav 는 `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` 에 저장되어 직접 재생해 확인할 수 있습니다.

::: tip 검증이 성공했다는 건
"레포에 올라간 파일로 진짜 합성이 되더라" 를 의미합니다.

이 검증이 통과하면 다른 머신에서 `TTS().load(<speaker>, source="hf://...")` 같은 방식으로 바로 꺼내 쓸 수 있다고 보장할 수 있습니다.
:::

::: details 내부 동작 — 검증 절차
1. 선택한 백엔드로 `TTS(device=...)` 인스턴스 생성
2. 방금 올린 이름으로 `load(<speaker>)` → `prepare()`
3. 고정 문구 `"テスト音声です。"` 합성
4. 결과 wav 를 `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` 에 저장

GPU 검증 직전에는 전역 BERT / dynamo / CUDA 캐시를 리셋하여 서로 영향을 주지 않게 합니다.
:::

## 런타임에서 받아쓰기

업로드가 끝난 화자는 다른 머신·컨테이너에서 이렇게 로드합니다.

```python
from hayakoe import TTS

# HF 에서
tts = TTS(device="cpu").load("tsukuyomi", source="hf://me/my-voices").prepare()

# S3 에서
tts = TTS(device="cuda").load("tsukuyomi", source="s3://my-bucket/hayakoe-voices").prepare()

# 로컬에서
tts = TTS(device="cpu").load("tsukuyomi", source="file:///srv/voices").prepare()

# 합성
audio = tts.speakers["tsukuyomi"].generate("こんにちは。")
```

`device` 만 바꾸면 같은 코드가 자동으로 CPU(ONNX) / GPU(PyTorch) 백엔드를 탑니다 — publish 단계에서 `CPU + GPU` 를 골랐기 때문에 양쪽 파일이 레포에 모두 있어서 가능한 일입니다.

단, 런타임 쪽에도 해당 백엔드용 의존성이 깔려 있어야 합니다. `device="cuda"` 를 쓰려면 실제 돌리는 머신에 **PyTorch CUDA 빌드** 가 설치되어 있어야 하고, `device="cpu"` 는 기본 설치만으로 충분합니다. 자세한 건 [설치 — CPU vs GPU](/quickstart/install) 를 참고하세요.

## 다음 단계

- 받아쓰기: [서버로 배포](/deploy/)
- 런타임에서 어느 백엔드를 쓸지: [백엔드 선택](/deploy/backend)
