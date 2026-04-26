# 術語表

本頁面整理了 deep-dive 全篇中頻繁出現的 TTS · 推論術語。

面向 TTS 新手,**確保閱讀 architecture 之後的頁面時不會卡住**。

可以按順序通讀,也可以依需求搜尋查閱。

## 流水線全局視角

### TTS (Text-to-Speech)

將文字(文本)轉換為人聲音訊的技術總稱。

輸入通常是句子,輸出是 WAV · MP3 等音訊檔案。

TTS 系統內部經過「將文字解釋為發音」 → 「將發音合成為波形」等多個階段。

HayaKoe 也屬於此類,接收日語輸入生成 WAV 波形。

### 音素 (Phoneme)

區分語義的最小聲音單位。

TTS 模型接收的不是文字而是音素。如果直接接收文字,模型就必須學習「在什麼條件下如何發音」的全部發音規則。先轉換為音素後,模型只需專注於「如何讓這個聲音聽起來對」。

負責這個轉換的模組就是 **G2P**。

### G2P (Grapheme-to-Phoneme)

將文字 (Grapheme) 轉換為音素 (Phoneme) 的過程或模組。

處理各語言特有的發音規則,如日語的漢字讀法·連音規則等。

在 TTS 流水線中屬於將輸入送入模型前的步驟。

HayaKoe 專用於日語,將日語 G2P 委託給 [pyopenjtalk](./openjtalk-dict)。

### 波形 (Waveform)

將空氣壓力變化沿時間軸記錄的數字序列。是喇叭可以播放的「實際聲音」本身。

每個數字表示 **特定瞬間的空氣壓力(振幅,amplitude)**。正值表示空氣被壓縮,負值表示膨脹,絕對值越大聲音越響。0 對應靜音(基準壓力)。

取樣率 (sample rate) 為 22,050 Hz 時 1 秒 = 22,050 個這樣的數字。HayaKoe 的輸出為 44,100 Hz,每秒 44,100 個。

TTS 的最終輸出物就是這個波形。

## 模型組成要素

### VITS

2021 年發表的語音合成模型架構。

核心貢獻是將之前分為兩個階段(Acoustic Model + Vocoder)的 TTS 流水線整合為 **一個端到端模型**。

文本 → 波形的轉換由單一模型直接完成,內部由 Text Encoder · Duration Predictor · Flow · Decoder 構成。

HayaKoe 處於 VITS 譜系的延長線上。

- **VITS (2021)** — End-to-End TTS 的起點。
- **[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)** — Fish Audio 團隊在 VITS 上加入 BERT 以增強 **基於上下文的 prosody** 的開源專案。
- **[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)** — litagin02 fork Bert-VITS2 並添加 **Style Vector**,使同一說話人能表達多種語氣·情感。日語特化變體 **JP-Extra** 展現品質優勢。
- **HayaKoe** — 將 Style-Bert-VITS2 JP-Extra **精簡為日語專用**,重構為適合 CPU 與伺服器運維的形態。

Synthesizer 本身的模型結構沿用 Style-Bert-VITS2,HayaKoe 新增的變更主要集中在外圍(ONNX 路徑、推論流水線、部署·源簡化)。

### Synthesizer

HayaKoe 中對 **VITS 本體(Text Encoder + Duration Predictor + Flow + Decoder)** 的統稱。

接收音素序列和 BERT 特徵作為輸入,生成最終波形的部分。

BERT 存在於 Synthesizer **之外** 並由所有說話人共享。每個說話人不同的是 Synthesizer 的權重(約 250 MB)。

### BERT

Google 於 2018 年發表的基於 Transformer 的預訓練語言模型。讀取句子並為每個 token 生成上下文嵌入。

在 TTS 中用於 **將句子的語義·上下文資訊反映到合成中**。即使是相同的音素序列,BERT 也能根據上下文生成不同的韻律·重音。

HayaKoe 使用日語專用 DeBERTa v2 模型(`ku-nlp/deberta-v2-large-japanese-char-wwm`)。

在 CPU 路徑中將此 BERT 量化為 INT8 並以 ONNX 執行。

### Text Encoder

Synthesizer 內部模組。接收音素序列作為輸入,輸出每個音素對應的 **192 維隱藏向量**。

Transformer encoder 結構,透過 self-attention 使音素參考前後上下文生成合成所需的嵌入。

概念上可以看作 BERT 的縮小版。BERT 在詞·句子層面,Text Encoder 在音素層面。

### Duration Predictor (SDP)

預測每個音素 **發音多少幀** 的模組。例如「安」5 幀、「寧」4 幀。

「SDP」是 **Stochastic Duration Predictor** 的縮寫。由於從機率分布中取樣而非確定性,即使是同一句話每次呼叫韻律·速度也會略有不同。

HayaKoe 在原本用途之外還將此模組 **複用於句子邊界 pause 預測**。詳情請參見 [句子邊界 pause — Duration Predictor](./duration-predictor)。

### Flow

Synthesizer 內部模組。**可逆 (invertible) 變換**,正向·反向都可計算的神經網路。

訓練時在「正確音訊的 latent → 文本嵌入空間」方向對齊,推論時走反向從文本嵌入生成音訊 latent。

正式名稱是 **Normalizing Flow**。

::: warning Flow 與量化
HayaKoe 不將 Synthesizer 降到 FP16 的主要原因在於 Flow。Flow 的 `rational_quadratic_spline` 運算在 FP16 下因浮點誤差導致 assertion error。

Synthesizer INT8 因另外的原因排除 — 以 Conv1d 為中心的結構不適合 PyTorch dynamic quantization 自動套用,static quantization 實作複雜度高。
:::

### Decoder (HiFi-GAN)

Synthesizer 的最後一個模組。接收 Flow 生成的 latent 向量,生成 **實際波形 (waveform)**。

過去作為獨立 Vocoder 使用的 HiFi-GAN 結構被 VITS 整合進了模型內部。

**是 VITS 能端到端工作的核心模組**,同時也是 TTS 推論時間中佔比最大的部分。

### Style Vector

將說話人的「語氣·說話方式」等風格資訊壓縮為一個向量。

即使是同一說話人,也可以切換「平靜」、「開心」、「生氣」等風格進行合成。

這是 Style-Bert-VITS2 系列特有的元件,與每個說話人的 safetensors 一起以 `style_vectors.npy` 提供。

HayaKoe 目前為簡化 **僅使用 Neutral 風格**。多樣風格選擇支援計劃在後續改進中加入。

### Prosody (韻律)

對語音的 **韻律·節奏·重音·停頓** 的統稱。

如果音素回答的是「發什麼音」,那麼 prosody 回答的是「怎麼發音」。

TTS 聽起來「像機器人」最常見的原因就是 prosody 不夠自然。

Bert-VITS2 系列使用 BERT 的主要原因之一就是從句子上下文中獲取 prosody 線索。

## 效能·執行術語

### ONNX · ONNX Runtime

**ONNX (Open Neural Network Exchange)** 是可以 **獨立於框架儲存** 神經網路模型的標準格式。

無論在 PyTorch · TensorFlow 等哪裡訓練,匯出為 ONNX 後都被當作同一個圖處理。

**ONNX Runtime** 是實際執行 ONNX 模型的推論引擎。用 C++ 編寫,Python 開銷小,會分析模型圖並預先執行各種最佳化。

支援 CPU · CUDA · ARM (aarch64) 等多種執行裝置。

HayaKoe 的 CPU 路徑完全在 ONNX Runtime 上執行。同樣的程式碼能在 x86_64 和 aarch64 上通用執行也得益於此。

### 量化 (Quantization)

透過降低模型權重的數字表示精度來節省記憶體和計算的技術。

深度學習模型權重通常以以下精度之一儲存。

- **FP32** — 32 位元浮點。預設。最精確但體積最大。
- **FP16** — 16 位元浮點。FP32 的一半大小。
- **INT8** — 8 位元整數。約 FP32 的 1/4 大小。也常稱為「Q8」。
- **INT4** — 4 位元整數。約 FP32 的 1/8 大小。LLM 領域近期活躍使用。

位元數減少後模型檔案大小和 RAM 使用量也幾乎成比例減少,在某些硬體上運算也會加快。

代價是 **精度下降可能導致輸出品質變差。** 能量化到什麼程度品質仍然可接受,取決於模型和運算類型。

HayaKoe 選擇了 **僅對 BERT 的 MatMul 進行 INT8 動態量化 (Q8 Dynamic Quantization)**,Synthesizer 保持 FP32。詳細原因和實測效果請參見 [ONNX 最佳化](./onnx-optimization)。

### Kernel Launch Overhead

CPU 向 GPU 請求「執行此 kernel」時產生的固定成本。與實際計算時間無關,每次 kernel 呼叫產生數 us ~ 數十 us。

當單個 kernel 執行繁重計算時此成本被淹沒。但 **像 TTS 這樣小型 Conv1d 運算重複數百次的情況**,kernel launch overhead 可能佔整體時間的相當比重。

CUDA Graph · kernel fusion · torch.compile 等都是減少此成本的技術。

### Eager Mode

PyTorch 的預設執行方式。Python 程式碼逐行執行,每次單獨呼叫 GPU kernel。

偵錯方便但每個 kernel 都會累積 Python dispatch 開銷和 kernel launch overhead。

`torch.compile` 是透過圖級最佳化消除此開銷的替代方案。

### torch.compile

PyTorch 2.0 起提供的 **JIT 編譯器**。

首次呼叫時將模型追蹤為圖,融合·重編譯 kernel,後續呼叫更快執行。

HayaKoe 在 GPU 路徑中使用 `torch.compile`。

首次呼叫需要編譯時間,可透過 `prepare(warmup=True)` 將此成本轉移到服務啟動階段。

## 其他

### OpenJTalk

名古屋工業大學開發的開源日語 TTS 前端。

接收日語文本並生成 **音素序列·韻律資訊**。日語特有的漢字讀法·連音等規則都包含在其中。

HayaKoe 透過 Python 綁定 [pyopenjtalk](./openjtalk-dict) 使用此功能。
