# Docker イメージ

ビルド段階で `TTS.pre_download()` で重みをイメージに焼き込んでおけば、ランタイムでネットワークを経由せずにすぐ合成できます。

## 基本的なアイデア

HayaKoe の推奨運用パターンは **Docker イメージひとつにモデルの重みまですべて含め、ランタイムコンテナが外部ネットワークなしにすぐ起動できるようにすること** です。オフライン環境、ファイアウォール内側、HF・S3 の認証情報をランタイムコンテナに露出させたくない場合に特にクリーンです。

これのために2つを提供します。

1. **`TTS.pre_download(device=...)`** — 初期化なしにキャッシュだけを埋めるメソッド。GPU がないビルド環境でも成功します。
2. **共有キャッシュルート** — HF・S3・ローカルソースがすべて同じディレクトリ（`$HAYAKOE_CACHE`）を使います。ビルドステージで埋めたキャッシュをランタイムステージがそのまま使います。

## Dockerfile 例（GPU）

2ステージ構造で、BuildKit secret を利用して HF トークンをイメージレイヤーに残さずに注入します。

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

## ビルドコマンド

```bash
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HUGGINGFACE_TOKEN \
  -t hayakoe-server .
```

BuildKit が有効になっていないと `--secret` フラグが動作しません。トークンはビルド段階でのみ環境変数として露出し、**最終イメージレイヤーには残りません**。

## 実行

```bash
docker run --gpus all --env-file .env -p 80:80 hayakoe-server
```

`.env` の例は [④ デプロイ](/ja/training/publish#_4-送信先選択) ページ参照。

::: tip `pre_download(device="cuda")` は GPU なしで動きます
`pre_download` は **該当バックエンド用ファイルをキャッシュにダウンロードするだけ** です。実際の CUDA 初期化や `torch.compile` はランタイムの `prepare()` で行われます。

そのため GitHub Actions のような CI ランナー（GPU なし）でもイメージビルドが可能です。
:::

::: warning GPU ランタイムに `gcc` が必要です
`torch.compile` は Inductor/Triton がランタイムに C++ ラッパーを JIT コンパイルするためコンパイラが必要です。

`python:*-slim` や `nvidia/cuda:*-runtime` ベースはどちらもコンパイラを含まないので `gcc`（または `build-essential`）を必ず追加してください。

漏れると初回合成リクエスト時に Inductor でエラーが発生します。
:::

## CPU 専用イメージ（より軽量）

GPU が不要なら PyTorch を丸ごと外して ONNX Runtime だけ使えます。イメージサイズが数百 MB まで縮まります。

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

`app/tts.py` の `build_tts_engine()` でも `TTS(device="cpu")` に合わせるのを忘れないでください。

::: tip `pre_download(device="cpu")` vs `"cuda"`
CPU バックエンドは Q8 量子化 BERT + ONNX Synthesizer をダウンロードし、GPU バックエンドは FP32 BERT + PyTorch safetensors をダウンロードします。

2つのファイルセットは異なるため、**ランタイムで使うデバイスと同じ値** で `pre_download` する必要があります。
:::

## GitHub Actions でビルド

タグがプッシュされたときにイメージを自動ビルド・レジストリにアップロードしたい場合は GitHub Actions ワークフローひとつで十分です。ポイントは **BuildKit secret で HF トークンを注入** する部分。

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

必要なシークレットは2つだけ用意してください。

- **`HUGGINGFACE_TOKEN`** — HF private リポから話者の重みをダウンロードする際に必要。ワークフローが `secrets:` ブロックで BuildKit シークレット `hf_token` に注入し、Dockerfile の `RUN --mount=type=secret,id=hf_token` がこれを受け取って使います。
- **`GITHUB_TOKEN`** — GHCR push 用（GitHub Actions が自動注入するデフォルトトークン）。

::: warning HF トークンは `ARG` ・ `ENV` で渡さないでください
通常の `ARG` / `ENV` でトークンを渡すとイメージレイヤーの履歴にそのまま残ります。必ずワークフローの `secrets:` ブロック → Dockerfile の `--mount=type=secret` 経路でのみ注入してください。
:::

::: details GHCR 以外のレジストリにアップロードする
- **Docker Hub** — `docker/login-action@v3` の `registry` を空にし `username` / `password` を自分のアカウントに。タグは `<user>/<image>:<tag>` 形式。
- **プライベート OCI レジストリ** — `docker/build-push-action` の代わりに `docker buildx build --output type=oci,dest=/tmp/image.tar` で OCI archive をまず生成し、[`skopeo`](https://github.com/containers/skopeo) で最終送信先にコピーするパターンがクリーンです。マルチレジストリプッシュ・署名・レプリケーションが自由になります。
:::

## その他の細かい点

::: details キャッシュパスに関する注意点
- デフォルトのキャッシュルートは `$CWD/hayakoe_cache` です。コンテナ内で作業ディレクトリが変わるとキャッシュを見つけられない場合があります。
- **必ず `HAYAKOE_CACHE` env または `TTS(cache_dir=...)` で絶対パスを固定** してください。
- HF・S3・ローカルソースすべて同じルート以下に保存されるため、ひとつのパスだけ管理すれば済みます。
:::

::: details 多話者イメージ
複数話者をひとつのイメージに含めるなら、ビルド段階ですべて `load` した後に一括で `pre_download` します。

```python
tts = TTS(hf_token=token)
for name in ("tsukuyomi", "another-voice"):
    tts.load(name, source="hf://me/my-voices")
tts.pre_download(device="cuda")
```

イメージサイズは話者数に線形で増加します。大量の話者をサービングするなら **共有ボリュームにキャッシュを置いて複数コンテナが同じパスをマウント** する戦略も検討してください。
:::

## よく遭遇する問題

::: details `pre_download` が失敗します
- HuggingFace private リポなら `HUGGINGFACE_TOKEN` を `--secret` で渡しているか確認してください。通常の `ENV` に入れるとイメージレイヤーにトークンが残ります。
- S3 ソースなら `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、（S3 互換）`AWS_ENDPOINT_URL_S3` を BuildKit secret で注入してください。
:::

::: details ランタイムで依然としてネットワークにアクセスします
- `HAYAKOE_CACHE` env がビルドタイムとランタイムで同じ絶対パスか確認してください。相対パスにすると作業ディレクトリによって異なるパスを指します。
- `docker run -v` でキャッシュボリュームをマウントした場合、マウントパスがイメージ内のキャッシュパスを上書きしていないか確認してください。
:::

::: details イメージが大きすぎます
- 2ステージビルドでビルドツールチェーンを除外してください。
- CPU 専用なら `nvidia/cuda` ベースの代わりに `python:3.12-slim-bookworm` を使ってください。
- 不要な話者を `pre_download` から外してください。
:::

## 次のステップ

- CPU / GPU のどちらで `pre_download` するか：[バックエンド選択](/ja/deploy/backend)
