# ④ デプロイ（HF・S3・ローカル）

学習が完了すると `<dataset>/exports/<model_name>/` 以下に最終モデルファイルが集まっています。

このフォルダを **HuggingFace Hub / S3 / ローカルフォルダ** のいずれかにアップロードし、他のマシンから `TTS().load("my_name")` の1行で再ダウンロードして使えるようにするのが `cli publish` の役割です。

手動で HF CLI・S3 CLI を操作しリポジトリ構造を覚えてアップロード検証までする過程を、ひとつのインタラクティブフローにまとめてあります。

## 実行

```bash
uv run poe cli
```

メインメニューで **話者デプロイ** を選択すると以下の順番で尋ねます。

1. デプロイするデータセット（または外部フォルダ）
2. バックエンド — CPU / GPU / CPU + GPU
3. チェックポイント
4. 話者名
5. 送信先 + 認証情報
6. サマリーパネル → 確認
7. 自動アップロード → 実際の合成検証

各ステップは以下で扱います。

## 1. デプロイ対象の選択

2種類の対象が表示されます。

- **学習 dataset** — `data/dataset/<name>/exports/<model>/` に最終ファイルがあるデータセットが自動的にリストアップされます。
- **別フォルダから直接選択** — 学習は別の場所で行い HayaKoe 形式のフォルダだけ持っている場合、パスを直接入力します。

::: details 外部フォルダで必要なファイル
```
<my-folder>/
├── config.json                # 必須
├── style_vectors.npy          # 必須
├── *.safetensors              # 必須（1つ以上）
├── synthesizer.onnx           # 任意（あれば再利用）
└── duration_predictor.onnx    # 任意（あれば再利用）
```
:::

## 2. バックエンド選択

```
CPU (ONNX)        — GPU のないサーバー/ローカル用
GPU (PyTorch)     — 最低レイテンシ
CPU + GPU (推奨)  — 両環境にデプロイ
```

`CPU + GPU` を選ぶと同じリポジトリに2つのバックエンド用ファイルが **一緒に** アップロードされます。ランタイムで `TTS(device="cpu")` にすると ONNX 側だけ、`TTS(device="cuda")` にすると PyTorch 側だけが自動的にダウンロードされます。

**一度だけアップロードしておけば2つの環境で同じ名前で再利用** できるので、特別な理由がなければこのオプションを選んでください。

2つのバックエンドの違いは [バックエンド選択](/ja/deploy/backend) で詳しく扱います。

## 3. チェックポイントと話者名

- チェックポイントが1個なら自動選択、複数あれば選びます（通常 [③ 品質レポート](/ja/training/quality-check) で選んだもの）。
- **話者名** はランタイムで `TTS().load("my_name")` するときに使う識別子です。簡潔で小文字-ハイフンスタイルを推奨します（例：`tsukuyomi`）。

## 4. 送信先選択

3つのオプションがあります。初回のみ認証情報を入力すれば `dev-tools/.env` に `chmod 600` で保存され、次回からはプロンプトがスキップされます。

### HuggingFace Hub

リポジトリパス（`org/repo` または `hf://org/repo`）と **write 権限トークン** を入力します。`@<revision>` でブランチ/タグを指定することもできます。

::: details 対応 URL 形式 & 保存される環境変数
許可される URL 形式：

- `lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices@main`
- `https://huggingface.co/lemondouble/hayakoe-voices`
- `https://huggingface.co/lemondouble/hayakoe-voices/tree/dev`

保存される `.env` 例：

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # write 権限 HuggingFace アクセストークン
HAYAKOE_HF_REPO=lemondouble/hayakoe-voices       # 話者ファイルがアップロードされる HF リポ (org/repo 形式)
```
:::

### AWS S3

バケット名（+ オプションの prefix）と AWS 認証情報（`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`）を入力します。エンドポイント URL は空欄で構いません。

### S3 互換ストレージ（R2、MinIO など）

Cloudflare R2、MinIO、Wasabi のような S3 互換ストレージを使う場合は **エンドポイント URL を一緒に入力** します。

- Cloudflare R2 — `https://<account>.r2.cloudflarestorage.com`
- MinIO — `http://<host>:9000`

バケット・認証情報の入力は AWS S3 と同一です。

::: details 保存される環境変数の例
**AWS S3**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                # 話者ファイルがアップロードされる S3 バケット名
HAYAKOE_S3_PREFIX=hayakoe-voices               # バケット内のパス prefix（空欄ならバケットルート）
AWS_ACCESS_KEY_ID=<your_access_key_here>       # AWS アクセスキー ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>   # AWS シークレットアクセスキー
AWS_REGION=ap-northeast-2                      # S3 リージョン（例はソウル）
# AWS_ENDPOINT_URL_S3 は空欄（AWS S3 は自動決定）
```

**S3 互換（Cloudflare R2）**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                                 # アップロードされる R2 バケット名
HAYAKOE_S3_PREFIX=hayakoe-voices                                # バケット内のパス prefix（空欄ならバケットルート）
AWS_ACCESS_KEY_ID=<your_access_key_here>                        # R2 ダッシュボードで発行した Access Key ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>                    # R2 Secret Access Key
AWS_REGION=auto                                                 # R2 は常に auto
AWS_ENDPOINT_URL_S3=https://abc123def.r2.cloudflarestorage.com  # R2 エンドポイント（アカウント別固有）
```
:::

### ローカルディレクトリ

ネットワークアップロードなしにローカルパスにコピーするだけです。NFS 共有ボリュームや社内ネットワークドライブに置いてチームで共有するシナリオに適しています。ランタイムでは `file:///...` URI でアクセスします。

::: details 保存される環境変数の例
```env
HAYAKOE_LOCAL_PATH=/srv/hayakoe-voices   # 話者ファイルをコピーするローカルディレクトリパス
```
:::

## 5. リポジトリ構造

`CPU + GPU` でデプロイすると、リポジトリ内に ONNX フォルダと PyTorch フォルダが一緒に入ります。同じリポジトリに複数の話者を置いて一緒に運用できます（`speakers/voice-a/`、`speakers/voice-b/`、...）。

::: details 内部構造
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

BERT モデルも `pytorch/bert/` と `onnx/bert/` 以下に共有位置として一緒にアップロードされます。ランタイムは話者別ファイルと共通 BERT を同じキャッシュルールでダウンロードします。
:::

## 6. ONNX export（自動）

CPU バックエンド（`CPU (ONNX)` ・ `CPU + GPU`）を選ぶと、アップロード直前に PyTorch チェックポイントを ONNX に自動変換します。別途の `cli export` コマンドはありません。

変換結果は `<dataset>/onnx/` にキャッシュされ、同じチェックポイントを再度 publish するときは再利用されます。強制的に再変換したい場合はこのフォルダを削除して publish をやり直してください。

::: details 内部動作 — 変換されるモデルと方式
話者固有の2つのモデルが `dev-tools/cli/export/exporter.py` を通じて opset 17 で export されます。

#### 変換対象 — 話者固有の2つのモデル

**Synthesizer（VITS デコーダー）**

音素シーケンス + BERT 埋め込み + スタイルベクトルを入力として実際の波形（waveform）を生成する中核モデルです。話者ごとにすべて異なる学習がされるため、デプロイ対象の大部分をこのモデルが占めます。

- 関数：`export_synthesizer`
- 出力：`synthesizer.onnx`（+ 場合によっては `synthesizer.onnx.data`）

**Duration Predictor**

各音素がどれくらいの長さ発音されるべきかを予測します。この予測が正確でないと文境界の pause・テンポ処理が不自然になります。

- 関数：`export_duration_predictor`
- 出力：`duration_predictor.onnx`

#### `synthesizer.onnx.data` とは？

ONNX は内部的に Protobuf でシリアライズされますが、Protobuf には **単一メッセージ 2GB 制限** があります。Synthesizer の重みがこの閾値を超えると、グラフ構造のみ `.onnx` に置き **大型テンソルは隣の `.data` ファイルに外部化** します。

- 2つのファイルは **常に同じフォルダに一緒に存在する必要があります**（分離移動禁止）
- モデルサイズによっては `.data` がまったく生成されない場合もあります
- ランタイムは `.onnx` のみ指定してロードしても同じフォルダの `.data` を自動的に一緒に読み取ります

#### BERT は話者ごとに作らず共用

BERT（DeBERTa）は話者と無関係な日本語言語モデルです。すべての話者が共用で使う **Q8 量子化 ONNX**（`bert_q8.onnx`）を HuggingFace の共用位置からダウンロードして使い、publish 段階で話者ごとに新たに変換しません。

- Q8 量子化のおかげで CPU でもリアルタイムに近いレイテンシで埋め込みを抽出可能
- すべての話者が同じ BERT を共有するのでリポジトリごとに重複保存する必要なし

つまり、このステップで実際に変換される対象は **話者固有の Synthesizer + Duration Predictor の2つだけ** です。

#### トレーシングに時間がかかる理由

ONNX export は「実際にモデルを一度通過させながら演算グラフを記録する」 **トレーシング** 方式です。Synthesizer は構造が複雑なため数十秒〜数分かかる場合があります。

同じチェックポイントを別名・別の送信先で複数回 publish するケースが多いため、一度変換した結果は `<dataset>/onnx/` にキャッシュされて再利用されます。

#### スクリプトで直接 export する

2つの export 関数は公開されているため、スクリプトで直接呼び出すこともできます。ただし publish フローが同じことを自動で行うため、特別な理由がなければ publish の使用を推奨します。直接呼び出しの経路は今後変更される可能性があります。
:::

## 7. 上書き確認

送信先に既に同じ名前の `speakers/<speaker-name>/` がある場合は **上書きするか先に確認します**。承認するとその話者ディレクトリだけをクリーンに削除して新しくアップロードします — 同じリポジトリにある他の話者には触れません。

README も同じ原則です。リポジトリルートに README がなければ4ヶ国語（ko/en/ja/zh）テンプレートを自動生成して一緒にアップロードし、既にあれば diff を見せた上で上書きするか尋ねます。

## 8. アップロード後の自動検証

アップロードが完了すると **アップロードしたファイルで実際に合成できるか** を自動的に確認します。

CPU + GPU 両方選択した場合は2つのバックエンドをそれぞれ検証し、結果の wav は `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` に保存されて直接再生して確認できます。

::: tip 検証が成功したということは
「リポジトリにアップロードしたファイルで本当に合成できた」を意味します。

この検証が通過すれば、他のマシンから `TTS().load(<speaker>, source="hf://...")` のような方法ですぐに取り出して使えると保証できます。
:::

::: details 内部動作 — 検証手順
1. 選択したバックエンドで `TTS(device=...)` インスタンスを生成
2. アップロードした名前で `load(<speaker>)` → `prepare()`
3. 固定フレーズ `"テスト音声です。"` を合成
4. 結果 wav を `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` に保存

GPU 検証の直前にはグローバル BERT / dynamo / CUDA キャッシュをリセットして互いに影響を与えないようにします。
:::

## ランタイムでの利用

アップロードが完了した話者は他のマシン・コンテナからこのようにロードします。

```python
from hayakoe import TTS

# HF から
tts = TTS(device="cpu").load("tsukuyomi", source="hf://me/my-voices").prepare()

# S3 から
tts = TTS(device="cuda").load("tsukuyomi", source="s3://my-bucket/hayakoe-voices").prepare()

# ローカルから
tts = TTS(device="cpu").load("tsukuyomi", source="file:///srv/voices").prepare()

# 合成
audio = tts.speakers["tsukuyomi"].generate("こんにちは。")
```

`device` を変えるだけで同じコードが自動的に CPU（ONNX）/ GPU（PyTorch）バックエンドに乗ります — publish 段階で `CPU + GPU` を選んだため両方のファイルがリポジトリにすべて揃っているから可能なことです。

ただし、ランタイム側にも該当バックエンド用の依存関係がインストールされている必要があります。`device="cuda"` を使うなら実際に動かすマシンに **PyTorch CUDA ビルド** がインストールされている必要があり、`device="cpu"` はデフォルトインストールだけで十分です。詳細は [インストール — CPU vs GPU](/ja/quickstart/install) を参照してください。

## 次のステップ

- ランタイムでの利用：[サーバーへデプロイ](/ja/deploy/)
- ランタイムでどのバックエンドを使うか：[バックエンド選択](/ja/deploy/backend)
