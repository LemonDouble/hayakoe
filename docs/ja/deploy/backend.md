# バックエンド選択（CPU vs GPU）

HayaKoe は CPU（ONNX Runtime）と GPU（PyTorch + `torch.compile`）の2つのバックエンドをサポートしています。コードレベルでは `device` パラメータひとつの違いです。

```python
tts_cpu = TTS(device="cpu").load("tsukuyomi").prepare()
tts_gpu = TTS(device="cuda").load("tsukuyomi").prepare()
```

ただし **インストールプロファイルから異なります** — CPU は `pip install hayakoe` だけで済みますが、GPU は `hayakoe[gpu]` + PyTorch CUDA ビルドが追加で必要です。ひとつの環境に両方をインストールして動かすことも可能ですが、実際のデプロイではターゲット環境に合わせて **片方だけ** インストールするのが一般的です（詳細は [インストール — CPU vs GPU](/ja/quickstart/install)）。

その下の構造もまったく異なります。どちらが自分のデプロイ環境に適しているか判断する基準をまとめます。

## CPU（ONNX）が適する場合

- **GPU のないサーバー環境** — 一般的な Web ホスティング、VPS、マネージドコンテナプラットフォームのように CUDA サポートがない環境でそのまま動作します。
- **イメージサイズを最小化する必要がある場合** — PyTorch + CUDA スタックは数 GB 規模ですが、ONNX Runtime のみを含むイメージは数百 MB 台に縮まります。
- **低い同時リクエストを処理するワークロード** — 個人プロジェクトや社内ツールのように同時負荷が大きくない場合、CPU だけでも十分なスループットを確保できます。
- **コールドスタートが短い必要があるとき** — ONNX 経路は `torch.compile` コンパイル段階がないため、プロセスが起動するとすぐに `prepare()` が完了し合成を受け付けられます。GPU 経路は初回の `prepare()` で数十秒のグラフコンパイル時間を負担する必要があるため、オートスケール・サーバーレス環境で体感差が大きくなります。

::: details CPU 経路の構成
- **BERT** — `bert_q8.onnx`（Q8 量子化 DeBERTa）、ONNX Runtime `CPUExecutionProvider`
- **Synthesizer** — `synthesizer.onnx`（ONNX で export された VITS デコーダー）
- **Duration Predictor** — `duration_predictor.onnx`
:::

## GPU（PyTorch）が適する場合

- **低レイテンシが求められるリアルタイムサービス** — ユーザー対面のレスポンス、対話型 UI など、単一リクエストの応答時間が体感品質に直結する場合。
- **高い同時スループットが必要な環境** — ひとつの GPU で複数話者を並列合成でき、CPU 比で同時リクエスト許容幅が大きいです。
- **既に GPU インフラが構築された環境** — 追加投資なしに既存リソースを活用でき、同一コストでより良いレイテンシ・スループットが得られます。
- **長い文を繰り返し合成するワークロード** — `torch.compile` のグラフ最適化の恩恵が合成長に比例して大きくなります。

::: details GPU 経路の構成
- **BERT** — FP32 DeBERTa が GPU VRAM にロードされ埋め込みを計算します。量子化しないため CPU ONNX 経路より精度がやや高くなります。
- **Synthesizer** — PyTorch VITS デコーダー。`torch.compile` が適用されます。
- **Duration Predictor** — Synthesizer と同じ PyTorch 経路で、`torch.compile` 対象に一緒に含まれます。
:::

::: tip GPU バックエンドのコールドスタートを縮める
GPU バックエンドの初回 `prepare()` はモデルダウンロード + `torch.compile` 初期化が重なり数十秒かかる場合があります。実際のサービスでは以下の2つでこのコストを事前に支払っておくことを推奨します。

- **Docker ビルド時に `pre_download()`** — ビルド段階で重みをイメージ内に焼き込んでおくと、ランタイムの `prepare()` は HF・S3 アクセスなしにキャッシュから即座にロードします。イメージが起動するとすぐにネットワーク遅延なしに初期化が進行します。(→ [Docker イメージ](/ja/deploy/docker))
- **`prepare(warmup=True)`** — prepare 時にダミー推論を先行して `torch.compile` コンパイルと CUDA graph キャプチャまで prepare に前倒しします。prepare 自体はやや長くなりますが **最初の実リクエストが warmup コストを負担しなくなります**。(→ [FastAPI 統合](/ja/deploy/fastapi))
:::

## 並列比較

| 項目 | CPU (ONNX) | GPU (PyTorch + compile) |
|---|---|---|
| インストール | `pip install hayakoe` | `pip install hayakoe[gpu]` |
| イメージサイズ | 数百 MB | 数 GB |
| コールドスタート | 速い（秒） | 遅い（数十秒、初回 compile） |
| 単一リクエストレイテンシ | 普通 | 最低 |
| 同時スループット | コア数制限 | GPU 1台で並列 |
| メモリ（話者1名ロード） | ≈ 1.7 GB RAM | ≈ 1.3 GB RAM + 1.8 GB VRAM |
| メモリ（話者あたり増加） | +300~400 MB RAM | +250~300 MB VRAM |
| 必要ハードウェア | 任意の CPU | NVIDIA GPU + CUDA |

::: info 具体的な数値はベンチマークで
倍速・メモリ・レイテンシの数値はハードウェアに強く依存します。

- 倍速測定 — [自分のマシンでベンチマーク](/ja/quickstart/benchmark)
- メモリ測定（実測表と再現スクリプト） — [FAQ — 話者を複数ロードするとメモリはどれくらい増えますか](/ja/faq/#話者を複数ロードするとメモリはどれくらい増えますか)
:::

