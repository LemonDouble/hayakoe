# Docker 映像檔

在建置階段用 `TTS.pre_download()` 將權重烘焙到映像檔中,執行時無需網路即可直接合成。

## 基本思路

HayaKoe 推薦的維運模式是 **將模型權重全部打包到 Docker 映像檔中,執行時容器無需外部網路即可啟動**。

在離線環境、防火牆內部、不想在執行時容器中暴露 HF · S3 憑證的情況下特別整潔。

為此提供兩項支援。

1. **`TTS.pre_download(device=...)`** — 不初始化只填充快取的方法。在沒有 GPU 的建置環境中也能成功。
2. **共享快取根目錄** — HF · S3 · 本地源都使用同一目錄(`$HAYAKOE_CACHE`)。建置階段填充的快取直接被執行階段使用。

## Dockerfile 範例 (GPU)

2 階段結構,使用 BuildKit secret 注入 HF token 而不留在映像檔層中。

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

## 建置指令

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HUGGINGFACE_TOKEN \
  -t hayakoe-server .
```

需要啟用 BuildKit 才能使用 `--secret` 旗標。

token 僅在建置階段作為環境變數暴露,**不會留在最終映像檔層中**。

## 執行

```bash
docker run --gpus all --env-file .env -p 80:80 hayakoe-server
```

`.env` 範例參考 [④ 部署](/zh-TW/training/publish#_4-選擇目標位置) 頁面。

::: tip `pre_download(device="cuda")` 無需 GPU 即可執行
`pre_download` **僅下載對應後端的檔案到快取**。

實際 CUDA 初始化和 `torch.compile` 發生在執行時的 `prepare()` 中。

因此在 GitHub Actions 等 CI runner(無 GPU)上也可以建置映像檔。
:::

::: warning GPU 執行時需要 `gcc`
`torch.compile` 的 Inductor/Triton 在執行時 JIT 編譯 C++ 包裝器,因此需要編譯器。

`python:*-slim` 和 `nvidia/cuda:*-runtime` 基礎映像檔都不包含編譯器,請務必添加 `gcc`(或 `build-essential`)。

缺少時首次合成請求時 Inductor 會報錯。
:::

## CPU 專用映像檔(更輕量)

不需要 GPU 的話可以完全去掉 PyTorch,只用 ONNX Runtime。

映像檔大小可縮減到幾百 MB。

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

別忘了在 `app/tts.py` 的 `build_tts_engine()` 中也改為 `TTS(device="cpu")`。

::: tip `pre_download(device="cpu")` vs `"cuda"`
CPU 後端下載 Q8 量化 BERT + ONNX Synthesizer,GPU 後端下載 FP32 BERT + PyTorch safetensors。

兩套檔案不同,因此 **pre_download 時應使用與執行時相同的 device 值**。
:::

## 在 GitHub Actions 中建置

想在標籤推送時自動建置映像檔並上傳到 registry,一個 GitHub Actions 工作流即可。

關鍵是 **用 BuildKit secret 注入 HF token**。

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

只需準備兩個 secret。

- **`HUGGINGFACE_TOKEN`** — 從 HF private 儲存庫下載說話人權重時需要。工作流透過 `secrets:` 區塊注入為 BuildKit secret `hf_token`,Dockerfile 的 `RUN --mount=type=secret,id=hf_token` 接收使用。
- **`GITHUB_TOKEN`** — GHCR push 用(GitHub Actions 自動注入的預設 token)。

::: warning HF token 不要用 `ARG` · `ENV` 傳遞
用普通 `ARG` / `ENV` 傳遞 token 會原樣保留在映像檔層歷史中。

請務必只透過工作流的 `secrets:` 區塊 → Dockerfile 的 `--mount=type=secret` 路徑注入。
:::

::: details 上傳到非 GHCR 的其他 registry
- **Docker Hub** — `docker/login-action@v3` 的 `registry` 留空,`username` / `password` 改為自己的帳戶。標籤格式為 `<user>/<image>:<tag>`。
- **私有 OCI registry** — 可以先用 `docker buildx build --output type=oci,dest=/tmp/image.tar` 產生 OCI archive,再用 [`skopeo`](https://github.com/containers/skopeo) 複製到最終目標。多 registry 推送·簽署·複製都很靈活。
:::

## 其他細節

::: details 快取路徑相關注意事項
- 預設快取根目錄是 `$CWD/hayakoe_cache`。如果容器內的工作目錄變更可能找不到快取。
- **務必用 `HAYAKOE_CACHE` env 或 `TTS(cache_dir=...)` 固定為絕對路徑**。
- HF · S3 · 本地源都儲存在同一根目錄下,只需管理一個路徑。
:::

::: details 多說話人映像檔
想在一個映像檔中包含多個說話人,在建置階段全部 `load` 後一次性 `pre_download` 即可。

```python
tts = TTS(hf_token=token)
for name in ("tsukuyomi", "another-voice"):
    tts.load(name, source="hf://me/my-voices")
tts.pre_download(device="cuda")
```

映像檔大小隨說話人數量線性增長。

如果要服務大量說話人,也可以考慮 **將快取放在共享卷上,多個容器掛載同一路徑** 的策略。
:::

## 常見問題

::: details `pre_download` 失敗
- 如果是 HuggingFace private 儲存庫,請確認 `HUGGINGFACE_TOKEN` 是否透過 `--secret` 傳遞。用普通 `ENV` 傳入會將 token 保留在映像檔層中。
- 如果是 S3 源,請將 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、(S3 相容)`AWS_ENDPOINT_URL_S3` 透過 BuildKit secret 注入。
:::

::: details 執行時仍然在存取網路
- 請確認 `HAYAKOE_CACHE` env 在建置時和執行時是否為同一絕對路徑。使用相對路徑時會因工作目錄不同指向不同路徑。
- 如果用 `docker run -v` 掛載了快取卷,請確認掛載路徑是否覆蓋了映像檔內的快取路徑。
:::

::: details 映像檔太大
- 使用 2 階段建置去掉建置工具鏈。
- CPU 專用的話使用 `python:3.12-slim-bookworm` 而非 `nvidia/cuda` 基礎映像檔。
- 從 `pre_download` 中去掉不需要的說話人。
:::

## 下一步

- 選擇用 CPU 還是 GPU 做 `pre_download`:[後端選擇](/zh-TW/deploy/backend)
