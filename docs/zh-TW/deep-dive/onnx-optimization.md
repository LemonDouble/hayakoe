# ONNX 最佳化 / 量化

原版 Style-Bert-VITS2 基於 PyTorch,僅用 CPU 難以即時推論。

HayaKoe 將 **BERT 匯出為 Q8 量化 ONNX,Synthesizer 匯出為 FP32 ONNX** 並在 ONNX Runtime 上執行,將 CPU 推論速度 **按文本長度提升了 1.5x ~ 3.3x**。

同時將 1 說話人載入時的 RAM 使用量從 **5,122 MB 降至 2,346 MB (-54%)**。

得益於同一路徑,**不僅 x86_64,aarch64 (Raspberry Pi 等) 也能用相同程式碼執行**。

## 不足之處

原版 SBV2 (CPU, PyTorch FP32) 有兩個不足。

- **速度** — 文本越長推論時間指數級增長。short (1.7 s 音訊) 倍速 1.52x,而 xlong (38.5 s 音訊) 推論 35.3 秒·倍速 1.09x,勉強跟上即時。
- **記憶體** — 單說話人推論 Peak 記憶體約 5 GB 以上,負擔較重。

## 分析

先看模型參數分布,全部參數中約 **84%** 集中在 BERT ([DeBERTa-v2-Large-Japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm), 約 329 M),Synthesizer (VITS) 為 63 M 約 16%。

BERT 佔模型大部分,因此量化 BERT 預計可以大幅減少記憶體。在 PyTorch 中僅對 BERT 套用 Q8 Dynamic Quantization (`torch.quantization.quantize_dynamic`) 進行驗證。

| 設定 | 平均推論時間 | RAM |
|---|---|---|
| PyTorch BERT FP32 | 4.796 s | +1,698 MB |
| PyTorch BERT Q8 | 4.536 s | **+368 MB** (-78%) |

確認 BERT 量化雖不改善速度,但能大幅減少記憶體使用。

在此基礎上透過 **轉到 ONNX Runtime** 進一步確保速度改善。

ONNX Runtime 在載入模型時自動套用 [圖級最佳化](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)。

- **Kernel fusion** — 將連續多個運算合併為一個。例如將 Conv → BatchNorm → Activation 三步合為一步,消除中間結果的記憶體寫入和再讀取,減少記憶體存取從而加速。
- **Constant folding** — 將不論輸入如何始終產生相同值的運算在載入時預先計算好,推論時使用預計算值加速。
- **刪除不必要節點** — 找出未使用、重複或執行無意義運算的節點並刪除。

總之,重構為數學上等價但推論更最佳化的運算。

Synthesizer 參數量僅 63 M,量化的記憶體收益有限,且 Flow 層(`rational_quadratic_spline`)在 FP16 以下數值不穩定,因此排除在量化對象之外。僅匯出為 ONNX 以獲取圖最佳化收益。

### BERT 最佳化

為確認量化是否影響音質,在同一文本·同一說話人下比較了 FP32 · Q8 · Q4 三種 BERT 設定的輸出(Synthesizer 均為 FP32 固定)。

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。
>
> (旅途中來到了一個神奇的小鎮。稍微繞道走走吧。)

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="BERT dtype"
  :defaultIndex="0"
  :samples='[
    { value: "FP32", caption: "原版 baseline", src: "/hayakoe/deep-dive/quantization/fp32_med_ja.wav" },
    { value: "Q8",   caption: "INT8 dynamic quantization", src: "/hayakoe/deep-dive/quantization/q8_med_ja.wav" },
    { value: "Q4",   caption: "INT4 weight-only (MatMulNBits)", src: "/hayakoe/deep-dive/quantization/q4_med_ja.wav" }
  ]'
/>

FP32 和 Q8 在直接聽辨時難以一致地區分。

Q4 在大部分區間與 FP32 · Q8 相似,但在尾部可以聽到微小差異。

| 設定 | BERT 大小 | RAM (1 說話人) |
|---|---|---|
| FP32 | 1,157 MB | 1,599 MB |
| Q8 | 497 MB | 1,079 MB (-33%) |
| Q4 | 394 MB | 958 MB (-40%) |

判斷進一步 Q4 量化獲得的記憶體收益不足以彌補音質差異,決定 **預設使用 Q8**。

### Synthesizer 最佳化

BERT 佔參數的 84%,按理快速化 BERT 就能加速整體。

但實際分別測量 BERT 和 Synthesizer 的推論時間後發現,**CPU 時間的大部分消耗在 Synthesizer 側**。

PyTorch FP32 CPU 下的實測結果(5 次平均)。

| 文本 | BERT | Synthesizer | BERT 佔比 | Synth 佔比 |
|---|---|---|---|---|
| short (1.7 s) | 0.489 s | 0.885 s | 36% | **64%** |
| medium (5.3 s) | 0.602 s | 2.504 s | 19% | **81%** |
| long (7.8 s) | 0.690 s | 3.714 s | 16% | **84%** |
| xlong (30 s) | 1.074 s | 11.410 s | 9% | **91%** |

文本越長 Synthesizer 佔比越大,因為 BERT 對文本長度相對不敏感,而 Synthesizer 時間與要生成的音訊長度成正比增長。

實際僅量化 BERT Q8 時整體推論時間僅減少約 5%。

也就是說,**要提速必須最佳化 Synthesizer 部分**。

Synthesizer 採用 **僅 ONNX 轉換** 而非量化。

- VITS 的 Flow 層(`rational_quadratic_spline`)在 FP16 以下因浮點誤差導致 assertion error,無法量化。
- 參數量僅 63 M,量化的記憶體收益也有限。

透過轉換到 ONNX Runtime,將前述圖級最佳化(kernel fusion · constant folding · 刪除不必要節點)同樣套用到 Synthesizer。

### ONNX Runtime + `CPUExecutionProvider`

BERT 量化和 Synthesizer 圖最佳化都在 ONNX Runtime 上執行。

此外透過 [intra-op parallelism](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) 將單個運算分散到多個 CPU 核心,即使只有一個請求也能利用全部 CPU。

## 改善效果

### CPU 效能對比(倍速,同一硬體)

倍速 = 音訊時長 / 推論時間(值越大越快)。

| 設定 | short (1.7 s) | medium (7.6 s) | long (10.7 s) | xlong (38.5 s) |
|---|---|---|---|---|
| SBV2 PyTorch FP32 | 1.52x | 2.27x | 2.16x | 1.09x |
| SBV2 ONNX FP32 | 1.76x | 3.09x | 3.26x | 2.75x |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2.50x** | **3.35x** | **3.33x** | **3.60x** |

相比 PyTorch FP32 的速度提升為 **按文本長度 1.5x ~ 3.3x**。

### 記憶體(1 說話人載入基準)

| 設定 | RAM |
|---|---|
| SBV2 PyTorch FP32 | 5,122 MB |
| SBV2 ONNX FP32 | 2,967 MB |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2,346 MB** (-54%) |
