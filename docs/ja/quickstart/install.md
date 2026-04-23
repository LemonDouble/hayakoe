# インストール — CPU vs GPU

HayaKoe は **CPU 専用** と **GPU（CUDA）** の2つのインストールプロファイルをサポートしています。

お使いの環境に合った方を選ぶだけです。

## どちらを選ぶべきですか？

- **CPU** — GPU がない場合や、あっても一度試してみたいとき
- **GPU** — 大量処理が必要な場合や、リアルタイム性が重要なとき

::: tip 迷ったときのデフォルト
迷ったら **CPU** から始めてください。

後から GPU extras を追加インストールするだけで済みます。
:::

## CPU インストール（デフォルト）

PyTorch が不要なのでインストールが短く、イメージも軽量になります。

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

::: tip arm64 でもそのまま動きます
Raspberry Pi（4B 以上）のような aarch64 Linux 環境でも同じコマンドひとつでインストールでき、CPU 推論が動作します。

実測値は [ラズベリーパイ 4B ベンチマーク](./benchmark#ラズベリーパイ-4b-ではどうか) を参照してください。
:::

### 確認

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
audio = tts.speakers["jvnv-F1-jp"].generate("テスト、テスト。")
audio.save("test.wav")
print("OK")
```

初回実行時に [HuggingFace 公式 repo](https://huggingface.co/lemondouble/hayakoe) から BERT・Synthesizer・スタイルベクトルが自動的にキャッシュフォルダにダウンロードされます。

デフォルトのキャッシュパスはカレントディレクトリの `hayakoe_cache/` です。

## GPU インストール（CUDA）

### 事前準備

GPU モードは PyTorch CUDA ビルドを使用します。

必要なのは **NVIDIA ドライバひとつ** だけです。

- CUDA Toolkit は別途インストール不要です — PyTorch wheel に必要な CUDA ランタイムが含まれています。
- ただし、お使いのドライバがインストールしようとする CUDA バージョンをサポートしている必要があります。

ドライバがインストールされているか確認：

```bash
nvidia-smi
```

正常にインストールされていれば以下のような出力が表示されます。

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

1行目右側の `CUDA Version: 13.0` がお使いのドライバがサポートする **最大 CUDA バージョン** です（上記の例では 13.0）。

::: tip CUDA バージョンの選び方
`nvidia-smi` で確認したバージョン以下の PyTorch CUDA ビルドを選べば大丈夫です。

以下のインストール例の `cu126` の部分をお使いの環境に合ったバージョンに置き換えてください（例：`cu118`、`cu121`、`cu124`、`cu128`）。

サポートされる組み合わせは [PyTorch 公式インストールページ](https://pytorch.org/get-started/locally/) で選べます。
:::

### インストール

`hayakoe[gpu]` extras は `safetensors` のみを追加し、`torch` は引き込みません。

2行を別々にインストールすれば良く、順序は問いません。

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

### 確認

```python
from hayakoe import TTS

tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ完了。").save("gpu_test.wav")
```

::: warning 初回リクエストは遅い場合があります
GPU モードでは初回の `generate()` 呼び出しが通常より数秒長くかかる場合があります。

2回目の呼び出しからは通常速度が出ます。

サーバーとして起動する場合は、起動直後にダミーの `generate()` を1回呼んで「ウォームアップ」しておくことを推奨します。
:::

::: details なぜ初回呼び出しが遅いのですか？（torch.compile の背景）
HayaKoe は GPU モード時に `prepare()` の時点で PyTorch の `torch.compile` を自動的に適用します。

`torch.compile` は PyTorch 2.0 で追加された JIT コンパイラで、モデル実行グラフをトレースして一度コンパイルした後、その結果を再利用する方式です。

おかげで推論速度が向上しますが、代わりに **初回呼び出し時にグラフのトレーシング・コンパイルにかかる時間** が追加されます。

一度コンパイルされたグラフはプロセスが生きている間キャッシュされるため、2回目の呼び出しからはそのオーバーヘッドなしに即座に実行されます。そのため本番サービスでは、コンテナ・プロセスが起動した直後に短い文章でダミー呼び出しを行い、ウォームアップを済ませておくのが一般的です。

```python
# FastAPI lifespan, Celery worker 初期化などで
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ")  # 結果は捨てて構いません
```

CPU（ONNX）モードでは `torch.compile` を使用しないため、このウォームアップステップは不要です。
:::

ここまでできたら次のステップへ：[初めての音声を作る →](./first-voice)
