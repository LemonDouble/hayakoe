# 自訂說話人訓練

::: warning 部署前請務必確認版權
**如果計劃公開學習好的模型或對外發佈/上傳合成的語音**,請務必先確認原始聲音的版權與肖像權。

使用 YouTube、動畫、遊戲、商業配音等他人聲音訓練的模型公開發佈可能構成版權、肖像權、公開權侵權。

如果考慮部署,請使用本人聲音、已獲得明確授權的聲音或允許發佈的免費語料庫。

---

**可自由使用至部署的日語語料庫範例**

- **つくよみちゃんコーパス** — 標註署名條件下商業、非商業均可使用。
- **あみたろの声素材工房** — 個人、商業使用均可(需確認使用條款)。
- **ずんだもん** 等 VOICEVOX 角色的 ITA/ROHAN 語料庫 — 需確認各角色使用條款。

各語料庫的署名標註、商業使用可否、二次創作範圍各不相同。部署前請務必再次確認官方條款。
:::

HayaKoe 只需準備包含聲音的影片檔案,即可支援到訓練的全流程。

資料準備到部署分為兩個工具。

### `dev-tools/preprocess/` — 瀏覽器 GUI(資料前處理)

上傳影片、音訊檔案後,可以在 GUI 介面上按順序執行直到生成學習用資料集的各個步驟。

- **音訊擷取 (Audio Extraction, 自動)** — 從影片中擷取音軌。
- **背景音去除 (Source Separation, 自動)** — 使用 audio-separator 函式庫去除 BGM、音效等背景聲音,僅保留人聲。
- **句子分割 (VAD, 自動)** — 將長錄音按靜音區間分割為短句。
- **分類 (Classification, `手動`)** — 將分割出的語音按說話人分類,丟棄不可用的部分。
- **轉錄 (Transcription, 自動)** — 使用語音識別模型(Whisper)為每段語音自動生成對應文字。
- **審核 (Review, `手動`, 可跳過)** — 在瀏覽器中手動修正轉錄錯誤。
- **資料集生成 (Dataset Export, 自動)** — 將資料匯出為可訓練的格式。

### `dev-tools/cli/` — 互動式 CLI

接收 GUI 製作的資料集,從訓練到部署繼續推進。

- **前處理 (Preprocessing, 自動)** — 預先計算訓練所需的 G2P、BERT 嵌入、風格向量。
- **訓練 (Training, 自動)** — 基於預訓練資料,用我們準備的資料進行微調。
- **品質報告 (Quality Report, 自動)** — 用訓練期間儲存的各檢查點批次推論語音,確認哪個模型發出最好的聲音。
- **部署 (Publish, 自動)** — 從 ONNX(推論最佳化模型)轉換到 HuggingFace / S3 / 本地模型下載全部完成。

兩個工具共享同一個 `data/` 目錄,因此 GUI 製作的資料集會被 CLI 自動識別。

## 全流程

<PipelineFlow
  :steps="[
    {
      num: '①',
      title: '資料準備',
      tool: 'GUI',
      content: [
        '從想要訓練的說話人的影片中製作學習用語音資料集。',
        '從影片中擷取音訊,去除背景音與音效只留人聲,然後按靜音區間分割成短句。',
        '將分割出的片段按說話人分類,用 Whisper 自動生成文字,必要時手動修正後匯出為可訓練的格式。'
      ],
      chips: ['準備影片', '音訊擷取', '背景音去除', '句子分割', '說話人分類', '轉錄', '審核', '資料集匯出'],
      gpu: '必需'
    },
    {
      num: '②',
      title: '前處理 & 訓練',
      tool: 'CLI',
      content: [
        '用準備好的資料集微調日語 TTS 模型。',
        '預先計算訓練所需的 G2P(發音轉換)、BERT 嵌入、風格向量後,在預訓練的 Style-Bert-VITS2 JP-Extra 基礎上載入我們的資料針對說話人進行訓練。',
        '中間檢查點會按一定間隔持續儲存,在下一步驟中用於比較。'
      ],
      chips: ['G2P·BERT 計算', '風格嵌入', '微調', '檢查點儲存'],
      gpu: '必需'
    },
    {
      num: '③',
      title: '品質報告',
      tool: 'CLI',
      content: [
        '訓練並非跑得越久越好,過了某個點後音質或說話人音色反而可能崩壞。',
        '因此使用訓練期間儲存的多個檢查點對同一句話進行批次推論,比較哪個時間點的模型發出最好的聲音。',
        '結果整理在一張 HTML 中,可在瀏覽器中直接試聽,選擇滿意的檢查點進入下一步。'
      ],
      chips: ['批次推論', 'HTML 報告', '檢查點選擇']
    },
    {
      num: '④',
      title: '部署',
      tool: 'CLI',
      content: [
        '將選定的檢查點轉換為 ONNX 格式。',
        'ONNX 是針對 CPU 推論最佳化的模型格式,即使在沒有 GPU 的普通筆電上也能輕量執行。',
        '轉換後的模型可以上傳到 HuggingFace、S3 等雲端儲存或本地目錄。',
        '上傳一次後,hayakoe 套件可以直接透過說話人名稱載入使用。'
      ],
      chips: ['ONNX 轉換', 'HuggingFace', 'S3', '本地']
    }
  ]"
/>

::: warning 資料準備(①)、訓練(②)需要 GPU
兩個步驟內部都執行模型(背景音去除、Whisper、VITS2),沒有 GPU 實際上無法進行。

品質報告(③)、部署(④)不需要 GPU 也能執行。不建議在 CPU 筆電上進行訓練。
:::

## 準備工作

本指南需要直接 clone hayakoe 儲存庫來進行。

::: info 假設 Linux 環境
訓練工具目前僅保證在 Linux 環境下執行。

Windows 建議在 WSL2 上按照 Linux 指南操作。
:::

### 1. Clone 儲存庫

```bash
git clone https://github.com/LemonDouble/hayakoe.git
cd hayakoe
```

### 2. 安裝 uv

uv 是一個快速的 Python 套件、環境管理器。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

詳細安裝方法請參考 [uv 官方指南](https://docs.astral.sh/uv/getting-started/installation/)。

安裝完成後應該能看到版本輸出。

```bash
uv --version
```

### 3. 安裝開發相依套件

之後的指令全部在第 1 步 clone 的 **儲存庫根目錄(`hayakoe/`)** 下執行。

```bash
uv sync
```

前處理 GUI 與訓練 CLI 所需的函式庫(FastAPI、Whisper、audio-separator、torchaudio 等)會一次性安裝完成。

### 4. 安裝 GPU (CUDA) PyTorch

資料準備(①)、訓練(②)步驟內部執行 ML 模型,因此 NVIDIA GPU 是必需的。

首先確認驅動程式是否正常安裝。

```bash
nvidia-smi
```

正常安裝的話可以看到如下輸出。

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

輸出第一行右側的 `CUDA Version` 是您的驅動程式支援的 **最大 CUDA 版本**(上例中為 13.0)。

選擇不超過該版本的 PyTorch 建置進行安裝(以下範例基於 CUDA 12.6)。

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

使用其他 CUDA 版本時,將 `cu126` 替換為適合您的版本(`cu118`、`cu121`、`cu124`、`cu128` 等)。

安裝驗證:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
# 輸出 True 即可
```

完成後按照下方 [各步驟詳解](#各步驟詳解) 的順序操作。

## 各步驟詳解

準備完成後,請按以下順序逐頁操作。

- [① 資料準備](./data-prep)
- [② 前處理 & 訓練](./training)
- [③ 品質報告](./quality-check)
- [④ 部署 (HF·S3·本地)](./publish)
- [疑難排解](./troubleshooting)
