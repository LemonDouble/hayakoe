# Docker Image

By running `TTS.pre_download()` at build time to bake weights into the image, the runtime can synthesize immediately with no network access.

## Core Idea

HayaKoe's recommended operational pattern is **packing model weights entirely into a single Docker image so runtime containers start immediately with no external network**. This is especially clean for offline environments, behind firewalls, or when you do not want to expose HF/S3 credentials to runtime containers.

Two features make this possible.

1. **`TTS.pre_download(device=...)`** — A method that fills the cache without initialization. It succeeds even in build environments without a GPU.
2. **Shared cache root** — HF, S3, and local sources all use the same directory (`$HAYAKOE_CACHE`). The cache filled at the build stage is used as-is by the runtime stage.

## Dockerfile Example (GPU)

A 2-stage structure that uses BuildKit secrets to inject the HF token without leaving it in the image layer.

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

## Build Command

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HUGGINGFACE_TOKEN \
  -t hayakoe-server .
```

BuildKit must be enabled for the `--secret` flag to work. The token is exposed as an environment variable only during the build step and **does not remain in the final image layer**.

## Running

```bash
docker run --gpus all --env-file .env -p 80:80 hayakoe-server
```

See the [Step 4: Publish](/en/training/publish#_4-destination-selection) page for `.env` examples.

::: tip `pre_download(device="cuda")` works without a GPU
`pre_download` **only downloads the files for that backend to the cache**. Actual CUDA initialization and `torch.compile` happen during `prepare()` at runtime.

This means images can be built on CI runners like GitHub Actions (which have no GPU).
:::

::: warning GPU runtime requires `gcc`
`torch.compile` uses Inductor/Triton to JIT-compile C++ wrappers at runtime, so a compiler must be present.

Both `python:*-slim` and `nvidia/cuda:*-runtime` base images do not include a compiler, so be sure to add `gcc` (or `build-essential`).

If missing, an Inductor error will occur at the first synthesis request.
:::

## CPU-only Image (Lighter)

If you do not need a GPU, you can skip PyTorch entirely and use only ONNX Runtime. Image size shrinks to the hundreds of MB range.

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

Do not forget to also set `TTS(device="cpu")` in `build_tts_engine()` of `app/tts.py`.

::: tip `pre_download(device="cpu")` vs `"cuda"`
The CPU backend downloads Q8 quantized BERT + ONNX Synthesizer, while the GPU backend downloads FP32 BERT + PyTorch safetensors.

The two file sets are different, so you must **use the same device value for `pre_download` as you will use at runtime**.
:::

## Building with GitHub Actions

If you want to automatically build and push images to a registry when a tag is pushed, a single GitHub Actions workflow is sufficient. The key part is **injecting the HF token via BuildKit secret**.

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

Only two secrets need to be set up.

- **`HUGGINGFACE_TOKEN`** — Needed to download speaker weights from a private HF repo. The workflow injects it as BuildKit secret `hf_token` via the `secrets:` block, and the Dockerfile's `RUN --mount=type=secret,id=hf_token` picks it up.
- **`GITHUB_TOKEN`** — For GHCR push (auto-injected by GitHub Actions).

::: warning Do not pass the HF token via `ARG` or `ENV`
Passing the token via regular `ARG` / `ENV` embeds it directly in the image layer history. Always inject only through the workflow's `secrets:` block -> Dockerfile's `--mount=type=secret` path.
:::

::: details Pushing to a registry other than GHCR
- **Docker Hub** — Leave `registry` empty in `docker/login-action@v3` and use your own `username` / `password`. Tags follow `<user>/<image>:<tag>` format.
- **Private OCI registry** — Instead of `docker/build-push-action`, first extract as an OCI archive with `docker buildx build --output type=oci,dest=/tmp/image.tar`, then copy to the final destination with [`skopeo`](https://github.com/containers/skopeo). This gives you clean multi-registry push, signing, and replication.
:::

## Additional Details

::: details Cache path considerations
- The default cache root is `$CWD/hayakoe_cache`. If the working directory changes inside the container, the cache may not be found.
- **Always fix with an absolute path via `HAYAKOE_CACHE` env or `TTS(cache_dir=...)`**.
- HF, S3, and local sources all store under the same root, so only one path needs management.
:::

::: details Multi-speaker image
To pack multiple speakers into one image, `load` them all at build time and `pre_download` once.

```python
tts = TTS(hf_token=token)
for name in ("tsukuyomi", "another-voice"):
    tts.load(name, source="hf://me/my-voices")
tts.pre_download(device="cuda")
```

Image size grows linearly with the number of speakers. If serving a large speaker roster, consider a strategy of **placing the cache on a shared volume and mounting the same path from multiple containers**.
:::

## Common Issues

::: details `pre_download` fails
- For private HuggingFace repos, verify that `HUGGINGFACE_TOKEN` was passed via `--secret`. Using regular `ENV` embeds the token in the image layer.
- For S3 sources, inject `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and (for S3-compatible) `AWS_ENDPOINT_URL_S3` as BuildKit secrets.
:::

::: details Runtime still accesses the network
- Verify that the `HAYAKOE_CACHE` env is the same absolute path at both build time and runtime. Relative paths point to different locations depending on the working directory.
- If you mounted a cache volume with `docker run -v`, check that the mount path is not overwriting the cache path inside the image.
:::

::: details Image is too large
- Use a 2-stage build to strip the build toolchain.
- For CPU-only, use `python:3.12-slim-bookworm` instead of an `nvidia/cuda` base.
- Remove unnecessary speakers from `pre_download`.
:::

## Next Step

- CPU or GPU for `pre_download`: [Backend Selection](/en/deploy/backend)
