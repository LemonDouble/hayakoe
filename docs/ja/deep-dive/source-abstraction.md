# Source 抽象化（HF・S3・ローカル）

話者モデルと BERT ファイルがどこにあっても **URI を変えるだけで同一の API でロード** できるよう抽象化したレイヤーです。

## なぜ必要か

話者モデルを読み込むソースは状況によって異なります。

- 公開デフォルト話者は **HuggingFace リポ**（`hf://lemondouble/hayakoe`）
- 自前で学習した話者は **private HF リポ・S3・ローカルディレクトリ** など

ソースごとにダウンロードコードを分岐処理するとエンジン本体が肥大化し、キャッシュパスが重複する問題が生じます。

## 実装

### Source インターフェース

すべてのソースは **「prefix 単位でファイルをローカルキャッシュにダウンロードしてパスを返す」** という共通インターフェースを実装します。

```python
class Source(Protocol):
    def fetch(self, prefix: str) -> Path:
        """prefix/ 以下のすべてのファイルをキャッシュにダウンロードしてローカルパスを返す。"""
        ...

    def upload(self, prefix: str, local_dir: Path) -> None:
        """local_dir の内容を prefix/ 以下にアップロード（デプロイ用）。"""
        ...
```

`fetch()` はモデルロード時に、`upload()` は CLI の `publish`（モデルデプロイ）時に使用されます。

### 実装体

| URI スキーム | 実装 | 動作 |
|---|---|---|
| `hf://user/repo[@revision]` | `HFSource` | `huggingface_hub.snapshot_download()` でダウンロード。`HF_TOKEN` 環境変数または `hf_token` パラメータで private リポにアクセス可能 |
| `s3://bucket/prefix` | `S3Source` | `boto3` ベース。`AWS_ENDPOINT_URL_S3` 環境変数で S3 互換エンドポイント（R2・MinIO 等）サポート |
| `file:///abs/path` または `/abs/path` | `LocalSource` | ローカルディレクトリをそのまま使用。ダウンロードなし |

### URI 自動ルーティング

`TTS().load()` に URI を渡すだけで、スキームに該当する Source が自動選択されます。

```python
# HuggingFace（デフォルト）
tts.load("jvnv-F1-jp")

# HuggingFace — private リポ
tts.load("jvnv-F1-jp", source="hf://myorg/my-voices")

# S3
tts.load("jvnv-F1-jp", source="s3://my-bucket/voices")

# ローカル
tts.load("jvnv-F1-jp", source="/data/models")
```

HuggingFace の Web URL（`https://huggingface.co/user/repo`）も自動的に `hf://` 形式に正規化して受け入れます。

### キャッシュ

すべてのソースは同一のキャッシュルート以下に保存されます。

キャッシュパスは `HAYAKOE_CACHE` 環境変数で指定するか、未指定時は `$CWD/hayakoe_cache` がデフォルトです。

キャッシュポリシーはシンプルです — ファイルがあれば再利用、なければ新規ダウンロードします。

### BERT ソース分離

話者モデルと BERT モデルのソースを **別々に指定** できます。

```python
TTS(
    device="cpu",
    bert_source="hf://lemondouble/hayakoe",  # BERT は公式リポから
).load(
    "custom-speaker",
    source="/data/my-models",                 # 話者はローカルから
).prepare()
```

デフォルトは両方とも `hf://lemondouble/hayakoe` です。

## 改善効果

- エンジン本体からストレージ別の分岐コードが除去されました。
- 新規ストレージを追加するには `Source` プロトコルを実装するクラスひとつを書くだけです。
- CLI の `publish` コマンドも同じ抽象化を逆方向（`upload`）に使用します。
