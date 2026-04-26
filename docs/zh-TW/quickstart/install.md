# 安裝 — CPU vs GPU

HayaKoe 支援 **CPU 專用** 和 **GPU (CUDA)** 兩種安裝設定。

根據自己的環境選擇其中一種即可。

## 應該選擇哪種?

- **CPU** — 沒有 GPU,或者有 GPU 但想先試試看的時候
- **GPU** — 需要批次處理,或者對即時性要求較高的時候

::: tip 猶豫時的預設選擇
猶豫的話就從 **CPU** 開始。

之後只需額外安裝 GPU extras 即可。
:::

## CPU 安裝(預設)

不需要 PyTorch,安裝更快,映像檔也更輕量。

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

::: tip arm64 同樣可以執行
在 Raspberry Pi (4B 及以上) 等 aarch64 Linux 環境下,同樣可以用一條指令安裝並進行 CPU 推論。

實測資料請參考 [樹莓派 4B 基準測試](./benchmark#raspberry-pi-4b-實測)。
:::

### 驗證

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
audio = tts.speakers["jvnv-F1-jp"].generate("テスト、テスト。")
audio.save("test.wav")
print("OK")
```

首次執行時,BERT・Synthesizer・風格向量會從 [HuggingFace 官方儲存庫](https://huggingface.co/lemondouble/hayakoe) 自動下載到快取目錄。

預設快取路徑為當前目錄下的 `hayakoe_cache/`。

## GPU 安裝 (CUDA)

### 前置準備

GPU 模式使用 PyTorch CUDA 建置版本。

只需要 **NVIDIA 驅動** 即可。

- 不需要單獨安裝 CUDA Toolkit — PyTorch wheel 中已包含所需的 CUDA 執行時。
- 但需確保您的驅動支援要安裝的 CUDA 版本。

檢查驅動是否已安裝:

```bash
nvidia-smi
```

如果正常安裝,您將看到如下輸出。

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

第一行右側的 `CUDA Version: 13.0` 是您的驅動支援的 **最大 CUDA 版本**(上例中為 13.0)。

::: tip 選擇 CUDA 版本
選擇不超過 `nvidia-smi` 顯示版本的 PyTorch CUDA 建置即可。

請在下方安裝範例的 `cu126` 位置填入適合您的版本(例如:`cu118`、`cu121`、`cu124`、`cu128`)。

支援的組合可在 [PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/) 選擇。
:::

### 安裝

`hayakoe[gpu]` extras 僅新增 `safetensors`,不會引入 `torch`。

分兩行安裝即可,順序不限。

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

### 驗證

```python
from hayakoe import TTS

tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ完了。").save("gpu_test.wav")
```

::: warning 首次請求可能較慢
在 GPU 模式下,第一次 `generate()` 呼叫可能會比平時多花幾秒。

從第二次呼叫開始就會恢復正常速度。

如果要作為伺服器執行,建議在啟動後立即呼叫一次虛擬 `generate()` 進行「預熱」。
:::

::: details 為什麼首次呼叫會慢?(torch.compile 背景)
HayaKoe 在 GPU 模式下會在 `prepare()` 時自動套用 PyTorch 的 `torch.compile`。

`torch.compile` 是 PyTorch 2.0 中新增的 JIT 編譯器,它追蹤模型執行圖,編譯一次後複用其結果。

因此推論速度會提升,但代價是 **首次呼叫時需要額外的圖追蹤和編譯時間**。

一旦編譯完成的圖在行程存活期間會被快取,因此從第二次呼叫開始沒有此開銷,可以直接執行。所以在實際服務中,通常在容器/行程啟動後用短文本進行一次虛擬呼叫來完成預熱。

```python
# 在 FastAPI lifespan, Celery worker 初始化等場景
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("ウォームアップ")  # 結果可以丟棄
```

CPU (ONNX) 模式不使用 `torch.compile`,因此不需要此預熱步驟。
:::

到這裡就完成了,接下來:[生成第一條語音 →](./first-voice)
