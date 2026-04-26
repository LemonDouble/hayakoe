# Source 抽象層 (HF · S3 · 本地)

無論說話人模型和 BERT 檔案位於何處,**只需更改 URI 即可用相同 API 載入** 的抽象層。

## 為什麼需要

說話人模型的載入源因情境而異。

- 公開的預設說話人在 **HuggingFace 儲存庫** (`hf://lemondouble/hayakoe`)
- 自行訓練的說話人在 **private HF 儲存庫 · S3 · 本地目錄** 等

如果按源逐一分支處理下載程式碼,引擎本體會變得臃腫,快取路徑也會重複。

## 實作

### Source 介面

所有源實作 **「按 prefix 將檔案下載到本地快取並回傳路徑」** 的公共介面。

```python
class Source(Protocol):
    def fetch(self, prefix: str) -> Path:
        """將 prefix/ 下所有檔案下載到快取並回傳本地路徑。"""
        ...

    def upload(self, prefix: str, local_dir: Path) -> None:
        """將 local_dir 內容上傳到 prefix/ 下(部署用)。"""
        ...
```

`fetch()` 在模型載入時使用,`upload()` 在 CLI 的 `publish`(模型部署)時使用。

### 實作類別

| URI 方案 | 實作 | 行為 |
|---|---|---|
| `hf://user/repo[@revision]` | `HFSource` | 透過 `huggingface_hub.snapshot_download()` 下載。可透過 `HF_TOKEN` 環境變數或 `hf_token` 參數存取 private 儲存庫 |
| `s3://bucket/prefix` | `S3Source` | 基於 `boto3`。透過 `AWS_ENDPOINT_URL_S3` 環境變數支援 S3 相容端點(R2 · MinIO 等) |
| `file:///abs/path` 或 `/abs/path` | `LocalSource` | 直接使用本地目錄。無需下載 |

### URI 自動路由

向 `TTS().load()` 只傳 URI,即可自動選擇對應方案的 Source。

```python
# HuggingFace (預設)
tts.load("jvnv-F1-jp")

# HuggingFace — private 儲存庫
tts.load("jvnv-F1-jp", source="hf://myorg/my-voices")

# S3
tts.load("jvnv-F1-jp", source="s3://my-bucket/voices")

# 本地
tts.load("jvnv-F1-jp", source="/data/models")
```

HuggingFace 網頁 URL (`https://huggingface.co/user/repo`) 也會自動規範化為 `hf://` 格式接受。

### 快取

所有源儲存在同一快取根目錄下。

快取路徑透過 `HAYAKOE_CACHE` 環境變數指定,未指定時預設為 `$CWD/hayakoe_cache`。

快取策略很簡單 — 有檔案就複用,沒有就重新下載。

### BERT 源分離

說話人模型和 BERT 模型的源可以 **分別指定**。

```python
TTS(
    device="cpu",
    bert_source="hf://lemondouble/hayakoe",  # BERT 從官方儲存庫
).load(
    "custom-speaker",
    source="/data/my-models",                 # 說話人從本地
).prepare()
```

預設值均為 `hf://lemondouble/hayakoe`。

## 改善效果

- 引擎本體中消除了按儲存類型的分支程式碼。
- 要新增儲存只需編寫一個實作 `Source` 協定的類別。
- CLI 的 `publish` 指令也使用同一抽象層的反向操作(`upload`)。
