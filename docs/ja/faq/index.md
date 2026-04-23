# FAQ

よくある詳細設定の項目をまとめました。

## キャッシュパスの変更

デフォルトのキャッシュパス（`./hayakoe_cache/`）を変更したい場合は2つの方法があります。

```bash
# 環境変数
export HAYAKOE_CACHE=/var/cache/hayakoe
```

```python
# コードで直接
tts = TTS(cache_dir="/var/cache/hayakoe")
```

HuggingFace・S3・ローカルソースすべて同じルート以下に保存されます。

## Private HuggingFace や S3 からモデルを取得

private HF repo の話者を使ったり S3 バケットにアップロードしたモデルを取得するにはソース URI を指定します。

S3 ソースを使う場合は extras を先にインストールしてください。

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",
        hf_token="hf_...",                     # private HF repo 用
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

S3 互換エンドポイント（MinIO、Cloudflare R2 など）は `AWS_ENDPOINT_URL_S3` 環境変数で指定します。

## 話者を複数ロードするとメモリはどれくらい増えますか

BERT は全話者が共有するため、話者あたり増えるのは遥かに軽い synthesizer 分だけです。

気になったのでローカルで直接ベンチスクリプトを実行してみましたが、数値はハードウェア・OS・torch バージョン・ORT ビルドによって変わりうるので **絶対値よりは増加傾向** としてだけ見てください。

::: info 測定環境
- GPU — NVIDIA RTX 3090 (24 GB)、Driver 580.126.09
- テキスト — 日本語2文（文境界含む、約50文字）
- 話者 — `jvnv-F1-jp`、`jvnv-F2-jp`、`jvnv-M1-jp`、`jvnv-M2-jp`
- 各シナリオは別の Python プロセスで実行（ヒープ汚染防止）
:::

### 話者数に応じたメモリ（ロードのみの状態）

| 話者数 | CPU (ONNX) RAM | GPU (PyTorch) RAM | GPU VRAM |
| :------ | -------------: | ----------------: | -------: |
| 1名    | ≈ 1.7 GB       | ≈ 1.3 GB          | ≈ 1.8 GB |
| 4名    | ≈ 2.8 GB       | ≈ 1.5 GB          | ≈ 2.6 GB |

話者3名が追加されたとき増えた量を3で割ると、話者1名あたりおおよそ以下の通りです。

- **CPU RAM** — 約 +360 MB / 話者
- **GPU VRAM** — 約 +280 MB / 話者

### 4名を同時に実行すると

実際のサービスでは複数話者が同時に動くこともあるため、**シーケンシャル4回** と **スレッド4つで同時** を別々に測定しました（合成中のピーク基準）。

| シナリオ    | CPU RAM peak | GPU RAM peak | GPU VRAM peak |
| :---------- | -----------: | -----------: | ------------: |
| 1話者合成 | ≈ 2.0 GB     | ≈ 2.3 GB     | ≈ 1.7 GB      |
| 4話者シーケンシャル | ≈ 3.2 GB     | ≈ 2.1 GB     | ≈ 2.6 GB      |
| 4話者同時 | ≈ 3.2 GB     | ≈ 2.2 GB     | ≈ 2.8 GB      |

同時実行でもメモリが4倍になることはありません。

CPU 側は ORT が内部で既に並列化を行っているため「シーケンシャル vs 同時」の差がほぼなく、GPU VRAM も同時実行が +200 MB 程度多い程度で止まります。

### 自分で再現してみる

リポジトリの `docs/benchmarks/memory/` 以下にスクリプトがあります。

```bash
# 単一シナリオ
python docs/benchmarks/memory/run_one.py --device cpu --scenario idle4

# 全10シナリオ（CPU/GPU × idle1/idle4/gen1/seq4/conc4）を別プロセスで
bash docs/benchmarks/memory/run_all.sh
```

- `run_one.py` はひとつのシナリオを実行して JSON 1行を出力します。
- `run_all.sh` は全シナリオを別の Python プロセスで実行して結果をスクリプト横の `results_<timestamp>.jsonl` に集めます。
- RAM は `psutil` で 50 ms ごとに RSS をポーリングしてピークを捉え、VRAM は `torch.cuda.max_memory_allocated()` の値をそのまま取得します。
- `gen*` シナリオはウォームアップ後に `torch.cuda.reset_peak_memory_stats()` を呼んで、torch.compile のコールドスタートをピークから除外します。

測定が必要なら自分の環境で一度実行してみて数値を比較するのが最も正確です。
