# 句子邊界 pause — Duration Predictor

多句分割合成時會產生 **句間停頓 (pause) 遺失** 的副作用。

HayaKoe 複用 Duration Predictor 直接預測各句子邊界的自然 pause 時間。

跳過 Flow · Decoder,**僅執行 TextEncoder + Duration Predictor**,因此額外成本很低。

## 為什麼是問題

### 句子分割的優勢

如 [架構概覽](./architecture#_1-%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC-%E1%84%87%E1%85%AE%E1%86%AB%E1%84%92%E1%85%A1%E1%86%AF) 所述,HayaKoe 將多句輸入按句子分割後逐個合成。

長文本整體合成時韻律容易模糊或不穩定。

按句子分割後每個句子都能保證穩定的 prosody(韻律)。

### 分割的副作用 — pause 遺失

但分割有副作用。

原版 SBV2 在通篇合成中會在 `.`、`!`、`?` 等標點後插入自然的停頓。

按句子分割後每個句子在標點處結束,下一句從頭開始,**標點後的停頓隨之消失**。

初期實作在句間插入固定 80 ms 靜音。

實際上 Duration Predictor 預測的句子邊界 pause 在 0.3 ~ 0.6 秒水平,80 ms 相比之下非常短。

結果產生了「沒有喘息空間」的不自然發話。

## 原理分析

閱讀本節前先回顧一下 Synthesizer 內部流程(詳見 [架構概覽 — Synthesizer](./architecture#_4-synthesizer-—-音素-bert-→-波形))。

<PipelineFlow
  :steps="[
    {
      num: '1',
      title: 'Text Encoder',
      content: 'Transformer 編碼器將音素序列嵌入為 192 維向量。BERT 特徵在此處與音素級嵌入結合,句子上下文首次注入到音素中。'
    },
    {
      num: '2',
      title: 'Duration Predictor',
      content: '預測每個音素發音多少幀。將穩定但單調的 DDP(確定性)和自然但不穩定的 SDP(隨機性)兩個 predictor 的輸出透過 sdp_ratio 混合,平衡穩定性和自然性。此步驟中音素序列在時間軸上被展開。'
    },
    {
      num: '3',
      title: 'Flow',
      content: '經過 Normalizing Flow(可逆神經網路)的反向變換,將 Text Encoder 生成的高斯分布(均值·方差)變形為實際音訊的複雜分布,生成 latent z 向量。訓練時走正方向(音訊 → 文本空間),推論時走反方向(文本 → 音訊空間)。'
    },
    {
      num: '4',
      title: 'Decoder',
      content: 'HiFi-GAN 系列的聲碼器,將 latent z 經過 ConvTranspose 上取樣和殘差塊 (ResBlock) 生成時域的實際波形 (44.1 kHz)。Synthesizer 子模組中運算量最大,CPU 推論時間的大部分在此消耗。'
    }
  ]"
/>

本文件的核心是 **僅執行 1 · 2 步驟 (Text Encoder + Duration Predictor)**。

跳過 3 · 4 步驟 (Flow + Decoder),因此成本非常低。

### 原版模型如何生成 pause

追蹤原版 SBV2 在通篇合成中生成自然 pause 的原理後發現,這是 **Duration Predictor 預測標點音素幀數的副效果**。

Duration Predictor 原本是預測「每個音素發音多少幀」的模組。

如「安」5 幀、「寧」4 幀。

而 `.`、`!`、`?` 等標點也包含在音素序列中。

Duration Predictor 對標點預測的幀數就成為 **該標點位置的停頓時長**。

例如 `.` 預測為 20 幀時 Synthesizer 會在該區間生成靜音或接近靜音的波形。

在分割合成中由於標點位置合成被切斷,這些資訊直接被丟棄。

### Duration Predictor 的內部動作

更詳細地看 Duration Predictor 的預測流程,兩個子模組並行工作。

**DDP (Deterministic Duration Predictor)** 對相同輸入始終輸出相同 duration。

穩定但發話可能聽起來機械性單調。

**SDP (Stochastic Duration Predictor)** 對相同輸入每次輸出略有不同的 duration。

基於機率取樣產生自然變動,但結果不太穩定。

兩個 predictor 的輸出透過 `sdp_ratio` 參數混合。

`sdp_ratio=0.0` 僅用 DDP,`1.0` 僅用 SDP,`0.5` 各半混合。

`length_scale` (= speed 參數) 乘以預測的全部 duration 來調整語速。

最終 `ceil()` 向上取整確定各音素的 **整數幀數**。

### blank token 和標點

計算 pause 時有一個注意點。

原版 SBV2 在音素序列的所有音素之間插入 **blank token(空白標記, ID = 0)**。HayaKoe 沿用此行為。

```
原始:  [は, い, .]
插入後: [0, は, 0, い, 0, ., 0]
```

blank token 也會被預測 duration,因此計算標點 `.` 的 pause 時需要 **將標點本身 + 前後 blank 的 duration 求和**。

例: `.` = 20 幀, 前 blank = 3 幀, 後 blank = 5 幀 → 總計 28 幀

## 實作

### 核心思路

核心很簡單。

**將完整原文文本僅通過 TextEncoder + Duration Predictor,獲取標點位置的幀數**。

跳過 Flow 和 Decoder。

Synthesizer 全程中成本大部分在 Flow 和 Decoder 中產生([ONNX 最佳化 — Synthesizer 佔比](./onnx-optimization#synthesizer-最佳化) 參考),因此僅執行到 Duration Predictor 的成本相對較低。

```
完整文本 (分割前原文)
  │
  ├─ TextEncoder (G2P → 音素序列 → 嵌入)
  │
  ├─ Duration Predictor (各音素幀數預測)
  │     └─ 僅擷取標點位置的幀數
  │
  └─ pause 時間計算
        frames × hop_length / sample_rate = 秒
```

全程合成中將已分割的各句子分別通過 TextEncoder → Duration Predictor → Flow → Decoder。

pause 預測中將 **分割前的原文整體** 僅通過 TextEncoder → Duration Predictor。

使用分割前原文的原因是句子邊界的標點只在原文中完整存在。

分割為各句子後除最後一句的標點外,邊界標點會消失或位置改變。

### pause 時間計算

獲取標點位置的幀數後轉換為秒。

```
pause (秒) = frames × hop_length / sample_rate
```

HayaKoe 預設設定中 `hop_length = 512`、`sample_rate = 44100`,1 幀約 11.6 ms。

例如標點 + 相鄰 blank 的合計幀數為 35:

```
35 × 512 / 44100 ≈ 0.41 秒
```

實際實作(`durations_to_boundary_pauses()`)經過以下過程。

1. 在完整音素序列中 **找到句子邊界標點的位置**(對應 `.`、`!`、`?` 的音素 ID)。
2. 獲取各標點位置該音素的 duration。
3. 如果前方相鄰 token 是 blank (ID = 0) 則加上其 duration。
4. 如果後方相鄰 token 是 blank (ID = 0) 則加上其 duration。
5. 將合計幀數用 `frames × hop_length / sample_rate` 轉換。

如果有 N 個句子則有 N - 1 個邊界,結果是 N - 1 個 pause 時間的清單。

### trailing silence(尾部靜音)補償

還有一個需要考慮的點。

Synthesizer 合成各句子時,句尾可能已經 **包含短暫的靜音**。

如果忽略此 trailing silence 直接插入預測的 pause,實際停頓會過長。

HayaKoe 直接測量合成音訊尾部的靜音區間。

測量方式是從音訊末尾以 10 ms 視窗逐格向前移動,**峰值振幅低於 2% 的區間** 判定為靜音。

之後插入 pause 時從預測的目標 pause 時間中減去 trailing silence,**僅補充不足部分為靜音樣本**。

```
額外靜音 = max(0, 預測 pause - trailing silence)
```

如果模型已經生成了足夠的靜音,額外插入為 0。

目標 pause 時間的下限設為 80 ms,因此即使預測值再短,句間總靜音也始終不低於 80 ms。

### ONNX 支援

PyTorch 路徑中可以單獨呼叫模型內部模組,只需單獨執行 Duration Predictor。

而 `synthesizer.onnx` 是將 Synthesizer 整體匯出為一個端到端圖的形式,無法擷取中間輸出。

為解決此問題,額外匯出了 **僅包含 TextEncoder + Duration Predictor 的獨立 ONNX 模型** (`duration_predictor.onnx`, ~30 MB, FP32)。

## 改善效果

### pause 時間分布

同一文本下自動預測的句子邊界 pause。

| 後端 | pause 範圍 |
|---|---|
| GPU (PyTorch) | 0.41 s ~ 0.55 s |
| CPU (ONNX) | 0.38 s ~ 0.57 s |

兩種後端的差異屬於 SDP 的 stochastic sampling(機率取樣)特性產生的偏差水平。

SDP 基於機率取樣,即使相同輸入每次呼叫結果也略有不同。

GPU 和 CPU 的差異落在此自然變動幅度內,因此 ONNX 轉換帶來的品質損失可以忽略。

### Before / After

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。きっと楽しい発見がありますよ。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pause 方式"
  :defaultIndex="1"
  :samples='[
    { "value": "Before (80 ms 固定)", "caption": "所有句子邊界相同的短停頓", "src": "/hayakoe/samples/duration-predictor/before.wav" },
    { "value": "After (DP 預測)", "caption": "Duration Predictor 自動預測句子邊界 pause", "src": "/hayakoe/samples/duration-predictor/after.wav" }
  ]'
/>

### 成本

額外成本為 TextEncoder + Duration Predictor 1 次執行。

如 [ONNX 最佳化 — Synthesizer 佔比](./onnx-optimization#synthesizer-最佳化) 所示,Synthesizer 佔整體 CPU 推論時間的 64 ~ 91%,其中大部分消耗在 Flow + Decoder。

僅執行到 Duration Predictor 的成本相比之下很低,因此 pause 預測帶來的感知延遲幾乎沒有。

## 相關提交

- `c57e0ad` — 基於 Duration Predictor 的 pause 預測改善多句合成自然度
- `5522db1` — 新增 ONNX `duration_predictor` 使 CPU 後端也支援自然的句子邊界靜音

## 未來課題

- **按情感的 pause 長度分化** — 開心時短、悲傷時長等根據情感風格套用不同的 pause 分布
- **逗號·冒號等細分** — 目前僅針對句末標點(`. ! ?`),對逗號(`,`、`、`)或冒號等需要長呼吸的位置進行追加細分
- **pause 直接控制 API** — 使用者可以明確指定特定句子邊界 pause 長度的介面
