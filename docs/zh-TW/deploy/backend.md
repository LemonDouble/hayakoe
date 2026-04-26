# 後端選擇 (CPU vs GPU)

HayaKoe 支援 CPU (ONNX Runtime) 和 GPU (PyTorch + `torch.compile`) 兩種後端。

在程式碼層面只是 `device` 參數的區別。

```python
tts_cpu = TTS(device="cpu").load("tsukuyomi").prepare()
tts_gpu = TTS(device="cuda").load("tsukuyomi").prepare()
```

但 **從安裝設定就開始不同** — CPU 只需 `pip install hayakoe`,GPU 還需要 `hayakoe[gpu]` + PyTorch CUDA 建置。

在同一環境中同時安裝兩者執行也是可能的,但實際部署中通常根據目標環境 **只安裝一個**(詳見 [安裝 — CPU vs GPU](/zh-TW/quickstart/install))。

底層結構也完全不同。

以下整理幫助您判斷哪種適合自己的部署環境。

## CPU (ONNX) 適用的場景

- **沒有 GPU 的伺服器環境** — 在一般 Web 託管、VPS、託管容器平台等沒有 CUDA 支援的環境中直接執行。
- **需要最小化映像檔大小的場景** — PyTorch + CUDA 堆疊在數 GB 級別,而僅包含 ONNX Runtime 的映像檔可縮減至數百 MB。
- **低並發請求的工作負載** — 個人專案或內部工具等並發負載不大的情況下,僅 CPU 也能確保足夠的處理量。
- **需要短冷啟動時間時** — ONNX 路徑沒有 `torch.compile` 編譯步驟,行程啟動後 `prepare()` 立即完成,可以馬上開始合成。GPU 路徑的首次 `prepare()` 需要承受數十秒的圖編譯時間,在自動擴縮·無伺服器環境中體感差異很大。

::: details CPU 路徑構成
- **BERT** — `bert_q8.onnx` (Q8 量化 DeBERTa), ONNX Runtime `CPUExecutionProvider`
- **Synthesizer** — `synthesizer.onnx` (匯出為 ONNX 的 VITS 解碼器)
- **Duration Predictor** — `duration_predictor.onnx`
:::

## GPU (PyTorch) 適用的場景

- **要求低延遲的即時服務** — 面向使用者的回應、對話式 UI 等單請求回應時間直接影響體驗品質的場景。
- **需要高並發處理量的環境** — 一塊 GPU 上可以並行合成多個說話人,比 CPU 的並發請求容納能力大得多。
- **已有 GPU 基礎設施的環境** — 無需額外投資即可利用現有資源,以相同成本獲得更好的延遲和處理量。
- **反覆合成長句子的工作負載** — `torch.compile` 的圖最佳化收益隨合成長度成比例增長。

::: details GPU 路徑構成
- **BERT** — FP32 DeBERTa 載入到 GPU VRAM 中計算嵌入。因未量化,精度比 CPU ONNX 路徑略高。
- **Synthesizer** — PyTorch VITS 解碼器。套用了 `torch.compile`。
- **Duration Predictor** — 與 Synthesizer 相同的 PyTorch 路徑,一同包含在 `torch.compile` 目標中。
:::

::: tip 縮短 GPU 後端冷啟動
GPU 後端的首次 `prepare()` 可能因模型下載 + `torch.compile` 初始化交織而耗時數十秒。

在實際服務中建議透過以下兩種方式提前支付此成本。

- **Docker 建置時 `pre_download()`** — 在建置階段將權重烘焙到映像檔中,執行時 `prepare()` 無需 HF · S3 存取直接從快取載入。映像檔啟動後立即無網路延遲地進行初始化。(→ [Docker 映像檔](/zh-TW/deploy/docker))
- **`prepare(warmup=True)`** — 在 prepare 時預先執行虛擬推論,將 `torch.compile` 編譯和 CUDA graph 捕獲成本提前到 prepare 階段。prepare 本身更久但 **第一個實際請求不承擔 warmup 成本**。(→ [FastAPI 整合](/zh-TW/deploy/fastapi))
:::

## 並排對比

| 項目 | CPU (ONNX) | GPU (PyTorch + compile) |
|---|---|---|
| 安裝 | `pip install hayakoe` | `pip install hayakoe[gpu]` |
| 映像檔大小 | 數百 MB | 數 GB |
| 冷啟動 | 快 (秒級) | 慢 (數十秒,首次 compile) |
| 單請求延遲 | 一般 | 最低 |
| 並發處理量 | 受核心數限制 | 1 塊 GPU 上並行 |
| 記憶體 (載入 1 個說話人) | ≈ 1.7 GB RAM | ≈ 1.3 GB RAM + 1.8 GB VRAM |
| 記憶體 (每說話人增加) | +300~400 MB RAM | +250~300 MB VRAM |
| 所需硬體 | 任何 CPU | NVIDIA GPU + CUDA |

::: info 具體數值請看基準測試
倍速·記憶體·延遲數值高度依賴硬體。

- 倍速測量 — [在我的機器上做基準測試](/zh-TW/quickstart/benchmark)
- 記憶體測量(實測表和重現腳本) — [FAQ — 載入多個說話人時記憶體增加多少](/zh-TW/faq/#載入多個說話人時記憶體增加多少)
:::

