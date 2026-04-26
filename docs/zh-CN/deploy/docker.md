# Docker 镜像

在构建阶段用 `TTS.pre_download()` 将权重烘焙到镜像中,运行时无需网络即可直接合成。

## 基本思路

HayaKoe 推荐的运维模式是 **将模型权重全部打包到 Docker 镜像中,运行时容器无需外部网络即可启动**。在离线环境、防火墙内部、不想在运行时容器中暴露 HF · S3 凭证的情况下特别整洁。

为此提供两项支持。

1. **`TTS.pre_download(device=...)`** — 不初始化只填充缓存的方法。在没有 GPU 的构建环境中也能成功。
2. **共享缓存根目录** — HF · S3 · 本地源都使用同一目录(`$HAYAKOE_CACHE`)。构建阶段填充的缓存直接被运行阶段使用。

## Dockerfile 示例 (GPU)

2 阶段结构,使用 BuildKit secret 注入 HF token 而不留在镜像层中。

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

## 构建命令

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HUGGINGFACE_TOKEN \
  -t hayakoe-server .
```

需要启用 BuildKit 才能使用 `--secret` 标志。token 仅在构建阶段作为环境变量暴露,**不会留在最终镜像层中**。

## 运行

```bash
docker run --gpus all --env-file .env -p 80:80 hayakoe-server
```

`.env` 示例参考 [④ 部署](/zh/training/publish#_4-选择目标位置) 页面。

::: tip `pre_download(device="cuda")` 无需 GPU 即可运行
`pre_download` **仅下载对应后端的文件到缓存**。实际 CUDA 初始化和 `torch.compile` 发生在运行时的 `prepare()` 中。

因此在 GitHub Actions 等 CI runner(无 GPU)上也可以构建镜像。
:::

::: warning GPU 运行时需要 `gcc`
`torch.compile` 的 Inductor/Triton 在运行时 JIT 编译 C++ 包装器,因此需要编译器。

`python:*-slim` 和 `nvidia/cuda:*-runtime` 基础镜像都不包含编译器,请务必添加 `gcc`(或 `build-essential`)。

缺少时首次合成请求时 Inductor 会报错。
:::

## CPU 专用镜像(更轻量)

不需要 GPU 的话可以完全去掉 PyTorch,只用 ONNX Runtime。镜像大小可缩减到几百 MB。

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

别忘了在 `app/tts.py` 的 `build_tts_engine()` 中也改为 `TTS(device="cpu")`。

::: tip `pre_download(device="cpu")` vs `"cuda"`
CPU 后端下载 Q8 量化 BERT + ONNX Synthesizer,GPU 后端下载 FP32 BERT + PyTorch safetensors。

两套文件不同,因此 **pre_download 时应使用与运行时相同的 device 值**。
:::

## 在 GitHub Actions 中构建

想在标签推送时自动构建镜像并上传到 registry,一个 GitHub Actions 工作流即可。关键是 **用 BuildKit secret 注入 HF token**。

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

只需准备两个 secret。

- **`HUGGINGFACE_TOKEN`** — 从 HF private 仓库下载说话人权重时需要。工作流通过 `secrets:` 块注入为 BuildKit secret `hf_token`,Dockerfile 的 `RUN --mount=type=secret,id=hf_token` 接收使用。
- **`GITHUB_TOKEN`** — GHCR push 用(GitHub Actions 自动注入的默认 token)。

::: warning HF token 不要用 `ARG` · `ENV` 传递
用普通 `ARG` / `ENV` 传递 token 会原样保留在镜像层历史中。请务必只通过工作流的 `secrets:` 块 → Dockerfile 的 `--mount=type=secret` 路径注入。
:::

::: details 上传到非 GHCR 的其他 registry
- **Docker Hub** — `docker/login-action@v3` 的 `registry` 留空,`username` / `password` 改为自己的账户。标签格式为 `<user>/<image>:<tag>`。
- **私有 OCI registry** — 可以先用 `docker buildx build --output type=oci,dest=/tmp/image.tar` 生成 OCI archive,再用 [`skopeo`](https://github.com/containers/skopeo) 复制到最终目标。多 registry 推送·签名·复制都很灵活。
:::

## 其他细节

::: details 缓存路径相关注意事项
- 默认缓存根目录是 `$CWD/hayakoe_cache`。如果容器内的工作目录变更可能找不到缓存。
- **务必用 `HAYAKOE_CACHE` env 或 `TTS(cache_dir=...)` 固定为绝对路径**。
- HF · S3 · 本地源都存储在同一根目录下,只需管理一个路径。
:::

::: details 多说话人镜像
想在一个镜像中包含多个说话人,在构建阶段全部 `load` 后一次性 `pre_download` 即可。

```python
tts = TTS(hf_token=token)
for name in ("tsukuyomi", "another-voice"):
    tts.load(name, source="hf://me/my-voices")
tts.pre_download(device="cuda")
```

镜像大小随说话人数量线性增长。如果要服务大量说话人,也可以考虑 **将缓存放在共享卷上,多个容器挂载同一路径** 的策略。
:::

## 常见问题

::: details `pre_download` 失败
- 如果是 HuggingFace private 仓库,请确认 `HUGGINGFACE_TOKEN` 是否通过 `--secret` 传递。用普通 `ENV` 传入会将 token 保留在镜像层中。
- 如果是 S3 源,请将 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、(S3 兼容)`AWS_ENDPOINT_URL_S3` 通过 BuildKit secret 注入。
:::

::: details 运行时仍然在访问网络
- 请确认 `HAYAKOE_CACHE` env 在构建时和运行时是否为同一绝对路径。使用相对路径时会因工作目录不同指向不同路径。
- 如果用 `docker run -v` 挂载了缓存卷,请确认挂载路径是否覆盖了镜像内的缓存路径。
:::

::: details 镜像太大
- 使用 2 阶段构建去掉构建工具链。
- CPU 专用的话使用 `python:3.12-slim-bookworm` 而非 `nvidia/cuda` 基础镜像。
- 从 `pre_download` 中去掉不需要的说话人。
:::

## 下一步

- 选择用 CPU 还是 GPU 做 `pre_download`:[后端选择](/zh/deploy/backend)
