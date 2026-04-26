# ④ 部署 (HF · S3 · 本地)

訓練完成後 `<dataset>/exports/<model_name>/` 下會集中最終模型檔案。

將此資料夾上傳到 **HuggingFace Hub / S3 / 本地資料夾** 中的某一個,然後在其他機器上用 `TTS().load("我的名字")` 一行程式碼即可重新下載使用 — 這就是 `cli publish` 的作用。

它將手動操作 HF CLI、S3 CLI、記憶儲存庫結構、上傳驗證等過程整合為一個互動式流程。

## 執行

```bash
uv run poe cli
```

在主選單中選擇 **發佈說話人** 後會按以下順序詢問。

1. 要部署的資料集(或外部資料夾)
2. 後端 — CPU / GPU / CPU + GPU
3. 檢查點
4. 說話人名稱
5. 目標位置 + 憑證
6. 摘要面板 → 確認
7. 自動上傳 → 實際合成驗證

各步驟在下方說明。

## 1. 選擇部署對象

有兩種對象。

- **訓練 dataset** — `data/dataset/<name>/exports/<model>/` 中有最終檔案的資料集會自動列出。
- **從其他資料夾直接選擇** — 訓練在其他地方完成,只有 HayaKoe 格式資料夾時,直接輸入路徑。

::: details 從外部資料夾匯入時需要的檔案
```
<my-folder>/
├── config.json                # 必需
├── style_vectors.npy          # 必需
├── *.safetensors              # 必需 (至少一個)
├── synthesizer.onnx           # 可選 (有的話複用)
└── duration_predictor.onnx    # 可選 (有的話複用)
```
:::

## 2. 選擇後端

```
CPU (ONNX)        — 無 GPU 的伺服器/本地用
GPU (PyTorch)     — 最低延遲
CPU + GPU (推薦)  — 同時部署到兩種環境
```

選擇 `CPU + GPU` 時兩種後端的檔案會 **一起** 上傳到同一儲存庫。執行時用 `TTS(device="cpu")` 建立時只拉取 ONNX 側,用 `TTS(device="cuda")` 建立時只拉取 PyTorch 側。

**只需上傳一次即可在兩種環境下以相同名稱複用**,所以沒有特殊原因的話請選這個選項。

兩種後端的差異在 [後端選擇](/zh-TW/deploy/backend) 中詳細說明。

## 3. 檢查點與說話人名稱

- 檢查點只有 1 個時自動選擇,多個時手動選擇(通常是在 [③ 品質報告](/zh-TW/training/quality-check) 中選定的)。
- **說話人名稱** 是執行時用 `TTS().load("我的名字")` 時使用的識別符。建議簡潔的小寫加連字號風格(例如:`tsukuyomi`)。

## 4. 選擇目標位置

有三個選項。只需首次輸入一次憑證,之後會以 `chmod 600` 儲存到 `dev-tools/.env`,下次會跳過提示。

### HuggingFace Hub

輸入儲存庫路徑(`org/repo` 或 `hf://org/repo`)和 **write 權限 token**。可以用 `@<revision>` 指定分支/標籤。

::: details 支援的 URL 格式 & 儲存的環境變數
允許的 URL 格式:

- `lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices@main`
- `https://huggingface.co/lemondouble/hayakoe-voices`
- `https://huggingface.co/lemondouble/hayakoe-voices/tree/dev`

儲存的 `.env` 範例:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # write 權限 HuggingFace 存取 token
HAYAKOE_HF_REPO=lemondouble/hayakoe-voices       # 說話人檔案上傳的 HF 儲存庫 (org/repo 格式)
```
:::

### AWS S3

輸入 bucket 名稱(+ 可選 prefix)和 AWS 憑證(`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`)。端點 URL 留空即可。

### S3 相容儲存 (R2, MinIO 等)

使用 Cloudflare R2、MinIO、Wasabi 等 S3 相容儲存時需要 **同時輸入端點 URL**。

- Cloudflare R2 — `https://<account>.r2.cloudflarestorage.com`
- MinIO — `http://<host>:9000`

bucket、憑證輸入與 AWS S3 相同。

::: details 儲存的環境變數範例
**AWS S3**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                # 說話人檔案上傳的 S3 bucket 名稱
HAYAKOE_S3_PREFIX=hayakoe-voices               # bucket 內路徑 prefix (留空則為 bucket 根目錄)
AWS_ACCESS_KEY_ID=<your_access_key_here>       # AWS 存取金鑰 ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>   # AWS 密鑰
AWS_REGION=ap-northeast-2                      # S3 區域 (範例為首爾)
# AWS_ENDPOINT_URL_S3 留空 (AWS S3 自動確定)
```

**S3 相容 (Cloudflare R2)**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                                 # 上傳的 R2 bucket 名稱
HAYAKOE_S3_PREFIX=hayakoe-voices                                # bucket 內路徑 prefix (留空則為 bucket 根目錄)
AWS_ACCESS_KEY_ID=<your_access_key_here>                        # R2 儀表板中頒發的 Access Key ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>                    # R2 Secret Access Key
AWS_REGION=auto                                                 # R2 始終為 auto
AWS_ENDPOINT_URL_S3=https://abc123def.r2.cloudflarestorage.com  # R2 端點 (每個帳號唯一)
```
:::

### 本地目錄

不透過網路上傳,僅複製到本地路徑。適用於將檔案放在 NFS 共享 volume 或內部網路磁碟機上供團隊共用的場景。執行時透過 `file:///...` URI 存取。

::: details 儲存的環境變數範例
```env
HAYAKOE_LOCAL_PATH=/srv/hayakoe-voices   # 說話人檔案複製到的本地目錄路徑
```
:::

## 5. 儲存庫結構

以 `CPU + GPU` 部署時儲存庫中會同時包含 ONNX 資料夾和 PyTorch 資料夾。可以在同一儲存庫中放置多個說話人一起管理(`speakers/voice-a/`、`speakers/voice-b/`、...)。

::: details 內部結構
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

BERT 模型也會一起上傳到 `pytorch/bert/` 和 `onnx/bert/` 下的共享位置。執行時使用相同的快取規則下載說話人檔案與公共 BERT。
:::

## 6. ONNX export (自動)

選擇 CPU 後端(`CPU (ONNX)`、`CPU + GPU`)時,在上傳前會自動將 PyTorch 檢查點轉換為 ONNX。無需單獨的 `cli export` 指令。

轉換結果快取在 `<dataset>/onnx/`,同一檢查點再次 publish 時會複用。想強制重新轉換的話刪除此資料夾後重新 publish。

::: details 內部機制 — 轉換的模型與方式
透過 `dev-tools/cli/export/exporter.py` 以 opset 17 匯出說話人專屬的兩個模型。

#### 轉換對象 — 說話人專屬的兩個模型

**Synthesizer (VITS 解碼器)**

接收音素序列 + BERT 嵌入 + 風格向量作為輸入,生成實際波形(waveform)的核心模型。由於每個說話人訓練不同,部署對象的大部分都是這個模型。

- 函式: `export_synthesizer`
- 輸出: `synthesizer.onnx` (+ 可能的 `synthesizer.onnx.data`)

**Duration Predictor**

預測每個音素應該發音多長時間。如果預測不準確,句子邊界的 pause、節奏處理會不自然。

- 函式: `export_duration_predictor`
- 輸出: `duration_predictor.onnx`

#### `synthesizer.onnx.data` 是什麼?

ONNX 內部使用 Protobuf 序列化,Protobuf 有 **單條訊息 2GB 限制**。當 Synthesizer 的權重超過此閾值時,圖結構保留在 `.onnx` 中而 **大型張量外置到旁邊的 `.data` 檔案**。

- 兩個檔案 **必須始終在同一資料夾中**(禁止分離移動)
- 根據模型大小可能完全不生成 `.data`
- 執行時只指定 `.onnx` 載入也會自動讀取同資料夾的 `.data`

#### BERT 不按說話人製作而是公用

BERT (DeBERTa) 是與說話人無關的日語語言模型。所有說話人共用的 **Q8 量化 ONNX** (`bert_q8.onnx`) 從 HuggingFace 的公用位置下載使用,publish 步驟不會為每個說話人重新轉換。

- 得益於 Q8 量化,CPU 上也能以接近即時的延遲擷取嵌入
- 所有說話人共享同一個 BERT,無需在每個儲存庫中重複儲存

也就是說,此步驟實際轉換的對象 **僅有說話人專屬的 Synthesizer + Duration Predictor 兩個**。

#### 追蹤耗時的原因

ONNX export 是「實際讓模型跑一遍,同時記錄計算圖」的 **追蹤** 方式。Synthesizer 結構複雜,可能需要數十秒到數分鐘。

由於同一檢查點可能會以不同名稱、不同目標多次 publish,轉換結果會快取在 `<dataset>/onnx/` 中複用。

#### 用腳本直接 export

兩個 export 函式是公開的,也可以用腳本直接呼叫。但 publish 流程會自動完成同樣的事,除非有特殊原因否則建議使用 publish。直接呼叫路徑將來可能會變更。
:::

## 7. 覆寫確認

如果目標位置已存在同名的 `speakers/<speaker-name>/`,會 **先詢問是否覆寫**。確認後僅清理該說話人目錄並重新上傳 — 同儲存庫中的其他說話人不受影響。

README 也遵循同樣原則。如果儲存庫根目錄沒有 README 則自動生成四語(ko/en/ja/zh)範本一起上傳,已存在則顯示 diff 後詢問是否覆寫。

## 8. 上傳後自動驗證

上傳完成後會自動確認 **用上傳的檔案是否真的能合成**。

如果選擇了 CPU + GPU 則分別驗證兩種後端,結果 wav 儲存在 `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` 中可以直接播放確認。

::: tip 驗證成功意味著
「用儲存庫中上傳的檔案真的能合成了」。

此驗證通過後可以保證在其他機器上用 `TTS().load(<speaker>, source="hf://...")` 等方式直接取出使用。
:::

::: details 內部機制 — 驗證流程
1. 用選定的後端建立 `TTS(device=...)` 實例
2. 用剛上傳的名稱 `load(<speaker>)` → `prepare()`
3. 合成固定文字 `"テスト音声です。"`
4. 將結果 wav 儲存到 `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav`

GPU 驗證前會重置全域 BERT / dynamo / CUDA 快取以避免相互影響。
:::

## 在執行時下載使用

上傳完成的說話人可以在其他機器、容器上這樣載入。

```python
from hayakoe import TTS

# 從 HF
tts = TTS(device="cpu").load("tsukuyomi", source="hf://me/my-voices").prepare()

# 從 S3
tts = TTS(device="cuda").load("tsukuyomi", source="s3://my-bucket/hayakoe-voices").prepare()

# 從本地
tts = TTS(device="cpu").load("tsukuyomi", source="file:///srv/voices").prepare()

# 合成
audio = tts.speakers["tsukuyomi"].generate("こんにちは。")
```

只需更改 `device` 即可自動使用 CPU(ONNX) / GPU(PyTorch) 後端 — 這是因為 publish 步驟選擇了 `CPU + GPU`,兩側檔案都在儲存庫中。

但執行時側也需要安裝對應後端的相依套件。使用 `device="cuda"` 時實際執行的機器上需要安裝 **PyTorch CUDA 建置**,`device="cpu"` 僅需基本安裝。詳情請參考 [安裝 — CPU vs GPU](/zh-TW/quickstart/install)。

## 下一步

- 下載使用:[伺服器部署](/zh-TW/deploy/)
- 執行時選擇哪個後端:[後端選擇](/zh-TW/deploy/backend)
