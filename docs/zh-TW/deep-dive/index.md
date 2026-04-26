# 深入解讀

HayaKoe 是將 Style-Bert-VITS2 精簡為日語專用,並重新建構為適合 CPU 推論和伺服器運維的 TTS 引擎。

本板塊整理了 **修改了哪些地方、如何修改的、結果有多大改善**,附帶實測資料。

可以根據感興趣的主題選擇性閱讀。

## 概覽

HayaKoe 相比原版 SBV2 的實測改善如下(詳情請參見各頁面)。

| 類別 | 原版 SBV2 | HayaKoe | 差異 |
|---|---|---|---|
| CPU 速度 (短句, 約 2 秒) | 1.13 s | 0.68 s | **1.67x 更快** |
| CPU 速度 (中等句子, 約 8 秒) | 3.35 s | 2.44 s | **1.37x 更快** |
| CPU 速度 (長句, 約 38 秒) | 35.33 s | 10.43 s | **3.39x 更快** |
| CPU 記憶體 | 5,122 MB | 2,346 MB | **減少 54%** |
| GPU VRAM | 3,712 MB | 1,661 MB | **減少 55%** |
| 執行架構 | x86_64 | x86_64 · aarch64 Linux | **支援 ARM 開發板** |

## 各頁面的結構

各頁面以 **為什麼是問題 → 實作 → 改善效果** 的流程為基本結構,根據主題彈性調整。

## 目錄

### 全局視角
- [架構概覽](./architecture) — TTS 引擎的整體構成

### CPU 推論的即時化
- [ONNX 最佳化 / 量化](./onnx-optimization) — Q8 BERT + FP32 Synthesizer, arm64 支援
- [句子邊界 pause — Duration Predictor](./duration-predictor) — 多句合成時自然停頓的恢復

### GPU 推論的額外最佳化
- [BERT GPU 保持 & 批次推論](./bert-gpu) — 消除 PCIe 往返與多句批次化

### 運維便利
- [Source 抽象層 (HF · S3 · 本地)](./source-abstraction) — 將說話人供給源統一為 URI
- [OpenJTalk 字典打包](./openjtalk-dict) — 消除首次 import 延遲與網路依賴
- [arm64 支援](./arm64) — Raspberry Pi 4B 實測

### 其他
- [問題回饋 & 授權](./contributing)

::: info 建議閱讀順序
初次閱讀建議先瀏覽 [架構概覽](./architecture),然後根據興趣選擇性閱讀其他主題。
:::
