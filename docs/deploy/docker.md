# Docker 이미지

빌드 단계에서 `TTS.pre_download()` 로 가중치를 이미지에 구워두면, 런타임에 네트워크를 타지 않고 바로 합성할 수 있습니다.

## 기본 아이디어

HayaKoe 의 권장 운영 패턴은 **Docker 이미지 하나에 모델 가중치까지 전부 담아, 런타임 컨테이너가 외부 네트워크 없이 바로 뜨게 하는 것** 입니다. 오프라인 환경, 방화벽 안쪽, HF · S3 자격 증명을 런타임 컨테이너에 노출하고 싶지 않은 경우에 특히 깔끔합니다.

이를 위해 두 가지를 제공합니다.

1. **`TTS.pre_download(device=...)`** — 초기화 없이 캐시만 채우는 메서드. GPU 가 없는 빌드 환경에서도 성공합니다.
2. **공유 캐시 루트** — HF · S3 · 로컬 소스가 모두 같은 디렉토리(`$HAYAKOE_CACHE`) 를 씁니다. 빌드 스테이지에서 채운 캐시를 런타임 스테이지가 그대로 씁니다.

## Dockerfile 예시 (GPU)

2-stage 구조로, BuildKit secret 을 이용해 HF 토큰을 이미지 레이어에 남기지 않고 주입합니다.

```dockerfile
# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

# ---- builder ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

WORKDIR /server
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

ENV HAYAKOE_CACHE=/server/hayakoe_cache
RUN --mount=type=secret,id=hf_token,env=HUGGINGFACE_TOKEN \
    /server/.venv/bin/python -c "\
import os; from hayakoe import TTS; \
tts = TTS(hf_token=os.environ.get('HUGGINGFACE_TOKEN')); \
[tts.load(n, source='hf://me/my-voices') for n in ('tsukuyomi',)]; \
tts.pre_download(device='cuda')"

# ---- prod ----
FROM python:${PYTHON_VERSION}-slim-bookworm AS prod

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 x264 gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /server
COPY --from=builder /server /server

ENV PATH="/server/.venv/bin:${PATH}" \
    HAYAKOE_CACHE=/server/hayakoe_cache

EXPOSE 80
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

## 빌드 커맨드

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HUGGINGFACE_TOKEN \
  -t hayakoe-server .
```

BuildKit 이 켜져 있어야 `--secret` 플래그가 동작합니다. 토큰은 빌드 단계에서만 환경변수로 노출되고, **최종 이미지 레이어에는 남지 않습니다**.

## 실행

```bash
docker run --gpus all --env-file .env -p 80:80 hayakoe-server
```

`.env` 예시는 [④ 배포](/training/publish#_4-목적지-선택) 페이지 참고.

::: tip `pre_download(device="cuda")` 는 GPU 없이 돌아갑니다
`pre_download` 는 **해당 백엔드용 파일을 캐시에 받는 것만** 합니다. 실제 CUDA 초기화나 `torch.compile` 은 런타임의 `prepare()` 에서 일어납니다.

그래서 GitHub Actions 같은 CI 러너(GPU 없음) 에서도 이미지 빌드가 가능합니다.
:::

::: warning GPU 런타임에 `gcc` 가 필요합니다
`torch.compile` 은 Inductor/Triton 이 런타임에 C++ 래퍼를 JIT 컴파일하므로 컴파일러가 있어야 합니다.

`python:*-slim` 이나 `nvidia/cuda:*-runtime` 베이스 모두 컴파일러를 포함하지 않으니 `gcc` (또는 `build-essential`) 을 반드시 추가하세요.

누락 시 첫 합성 요청 시점에 Inductor 에서 에러가 발생합니다.
:::

## CPU 전용 이미지 (더 가벼움)

GPU 가 필요 없다면 PyTorch 를 아예 빼고 ONNX Runtime 만 쓸 수 있습니다. 이미지 크기가 수백 MB 까지 줄어듭니다.

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

WORKDIR /server
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

ENV HAYAKOE_CACHE=/server/hayakoe_cache
RUN --mount=type=secret,id=hf_token,env=HUGGINGFACE_TOKEN \
    /server/.venv/bin/python -c "\
import os; from hayakoe import TTS; \
tts = TTS(device='cpu', hf_token=os.environ.get('HUGGINGFACE_TOKEN')); \
tts.load('tsukuyomi', source='hf://me/my-voices'); \
tts.pre_download(device='cpu')"

ENV PATH="/server/.venv/bin:${PATH}"
EXPOSE 80
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

`app/tts.py` 의 `build_tts_engine()` 에서도 `TTS(device="cpu")` 로 맞추는 걸 잊지 마세요.

::: tip `pre_download(device="cpu")` vs `"cuda"`
CPU 백엔드는 Q8 양자화 BERT + ONNX Synthesizer 를 받고, GPU 백엔드는 FP32 BERT + PyTorch safetensors 를 받습니다.

두 파일 세트는 서로 다르므로 **런타임에서 쓸 디바이스와 동일한 값** 으로 `pre_download` 해야 합니다.
:::

## GitHub Actions 에서 빌드하기

태그가 푸시될 때 이미지를 자동으로 빌드·레지스트리에 올리고 싶다면 GitHub Actions 워크플로우 하나로 충분합니다. 핵심은 **BuildKit secret 으로 HF 토큰을 주입** 하는 부분.

`.github/workflows/build.yaml`

```yaml
name: build docker image

on:
  push:
    tags:
      - v*

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v5

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v4

      - name: Login to GHCR
        uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract version tag
        run: echo "VERSION=${GITHUB_REF#refs/*/v}" >> $GITHUB_ENV

      - name: Build and push
        uses: docker/build-push-action@v7
        with:
          context: .
          platforms: linux/amd64
          push: true
          secrets: |
            hf_token=${{ secrets.HUGGINGFACE_TOKEN }}
          tags: |
            ghcr.io/${{ github.repository }}:${{ env.VERSION }}
            ghcr.io/${{ github.repository }}:latest
```

필요한 시크릿 두 개만 준비하면 됩니다.

- **`HUGGINGFACE_TOKEN`** — HF private 레포에서 화자 가중치를 받을 때 필요. 워크플로우가 `secrets:` 블록으로 BuildKit 시크릿 `hf_token` 에 넣어 주입하고, Dockerfile 의 `RUN --mount=type=secret,id=hf_token` 이 이를 받아 씁니다.
- **`GITHUB_TOKEN`** — GHCR push 용 (GitHub Actions 가 자동 주입하는 기본 토큰).

::: warning HF 토큰은 `ARG` · `ENV` 로 넘기지 마세요
일반 `ARG` / `ENV` 로 토큰을 전달하면 이미지 레이어 히스토리에 그대로 박힙니다. 반드시 워크플로우의 `secrets:` 블록 → Dockerfile 의 `--mount=type=secret` 경로로만 주입하세요.
:::

::: details GHCR 가 아닌 다른 레지스트리에 올리기
- **Docker Hub** — `docker/login-action@v3` 의 `registry` 를 비우고 `username` / `password` 를 본인 계정으로. 태그는 `<user>/<image>:<tag>` 형식.
- **사설 OCI 레지스트리** — `docker/build-push-action` 대신 `docker buildx build --output type=oci,dest=/tmp/image.tar` 로 OCI archive 를 먼저 뽑고, [`skopeo`](https://github.com/containers/skopeo) 로 최종 목적지에 복사하는 패턴이 깔끔합니다. 멀티 레지스트리 푸시·서명·복제가 자유롭습니다.
:::

## 기타 디테일

::: details 캐시 경로 관련 주의점
- 기본 캐시 루트는 `$CWD/hayakoe_cache` 입니다. 컨테이너 안에서 작업 디렉토리가 바뀌면 캐시를 못 찾을 수 있습니다.
- **반드시 `HAYAKOE_CACHE` env 또는 `TTS(cache_dir=...)` 로 절대 경로를 고정** 하세요.
- HF · S3 · 로컬 소스 모두 같은 루트 아래에 저장되므로 한 경로만 관리하면 됩니다.
:::

::: details 멀티 화자 이미지
여러 화자를 한 이미지에 담으려면 빌드 단계에서 전부 `load` 후 한 번에 `pre_download` 하면 됩니다.

```python
tts = TTS(hf_token=token)
for name in ("tsukuyomi", "another-voice"):
    tts.load(name, source="hf://me/my-voices")
tts.pre_download(device="cuda")
```

이미지 크기는 화자 수에 선형으로 늘어납니다. 큰 화자 목록을 서빙한다면 **공유 볼륨에 캐시를 두고 여러 컨테이너가 같은 경로를 마운트** 하는 전략도 고려하세요.
:::

## 자주 마주치는 문제

::: details `pre_download` 가 실패합니다
- HuggingFace private 레포면 `HUGGINGFACE_TOKEN` 을 `--secret` 으로 넘겼는지 확인하세요. 일반 `ENV` 로 넣으면 이미지 레이어에 토큰이 박힙니다.
- S3 소스라면 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, (S3 호환) `AWS_ENDPOINT_URL_S3` 를 BuildKit secret 으로 주입하세요.
:::

::: details 런타임에서 여전히 네트워크를 탑니다
- `HAYAKOE_CACHE` env 가 빌드 타임과 런타임에 같은 절대 경로인지 확인하세요. 상대 경로로 두면 작업 디렉토리에 따라 다른 경로를 가리킵니다.
- `docker run -v` 로 캐시 볼륨을 마운트했다면, 마운트 경로가 이미지 안의 캐시 경로를 덮어쓰고 있지 않은지 확인하세요.
:::

::: details 이미지가 너무 큽니다
- 2-스테이지 빌드로 빌드 툴체인을 벗기세요.
- CPU 전용이면 `nvidia/cuda` 베이스 대신 `python:3.12-slim-bookworm` 을 쓰세요.
- 필요 없는 화자를 `pre_download` 에서 빼세요.
:::

## 다음 단계

- CPU / GPU 중 어느 쪽으로 `pre_download` 할지: [백엔드 선택](/deploy/backend)
