# 伺服器部署

HayaKoe 採用 **單例 TTS 實例 + 建置時內嵌權重** 模式,為伺服器環境量身設計。

可以用 FastAPI · Docker 組合搭建簡潔的 API 伺服器。

## 設計要點

### 1. 模型只載入一次(單例)

TTS 模型載入一次需要相當長的時間。

在 GPU 環境下包括編譯階段可能需要數十秒。

如果每個請求都建立新實例,實際服務就不可行了,因此需要在 **行程生命週期內只維護一個**,讓所有請求共享。

實際程式碼在 FastAPI 的 lifespan 鉤子中用 `TTS(...).load(...).prepare(warmup=True)` 建構單例並儲存到 `app.state.tts`,處理器複用這一個實例。

並發無需擔心。

`Speaker` 內部持有 `threading.Lock`,同一說話人的並發請求會自動串列化,不同說話人之間則並行執行 — 無需額外的池·佇列實作。

::: details GPU 後端會同時準備 torch.compile
`TTS.prepare()` 在 CUDA 後端不僅載入模型,還會對所有說話人和 BERT 統一套用 `torch.compile`。

`warmup=True` 時會預先執行 1 次虛擬推論,將編譯成本提前到 prepare 階段。

此過程本身可能需要數十秒,因此必須在應用啟動時只做一次。

**每個請求都新建 TTS 會導致每次重新編譯**,服務實際上會癱瘓。

CPU 後端使用 ONNX Runtime 因此沒有單獨的編譯步驟,prepare 快得多。
:::

→ 實作見 [FastAPI 整合](/zh-TW/deploy/fastapi)

### 2. 權重在建置時烘焙到映像檔中

HayaKoe 推薦的維運模式是 **將模型權重全部打包到 Docker 映像檔中,執行時無需外部網路即可啟動**。

為此提供了 `TTS.pre_download(device=...)` — 「不初始化,只填充快取」的方法。

在 Docker 建置階段呼叫將所需的說話人檔案全部烘焙到映像檔中,執行時容器就不需要存取 HF · S3。

在離線環境、防火牆內部、不想在執行時容器中暴露 HF·S3 憑證的情況下特別整潔。

→ 實作見 [Docker 映像檔](/zh-TW/deploy/docker)

## 板塊構成

| 頁面 | 內容 |
|---|---|
| [FastAPI 整合](/zh-TW/deploy/fastapi) | lifespan 中載入單例,`agenerate` / `astream`,並發 |
| [Docker 映像檔](/zh-TW/deploy/docker) | 建置時 `pre_download`,BuildKit secret,多階段建置 |
| [後端選擇](/zh-TW/deploy/backend) | CPU(ONNX) vs GPU(PyTorch) 權衡 |

