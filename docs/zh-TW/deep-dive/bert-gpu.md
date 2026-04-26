# BERT GPU 保持 & 批次推論

GPU 路徑中有兩個主要最佳化點。

- **消除原版 SBV2 不必要的 CPU 轉換**,將 BERT 輸出保持為 GPU 張量
- **將多句打包為批次僅呼叫 1 次 BERT**,減少 kernel launch overhead(每次向 GPU 請求運算時產生的固定成本)

## 為什麼是問題

原版 SBV2 基本上 **將文本整體一次性合成** (`line_split=False`)。

BERT 也只呼叫 1 次,不需要批次化。

HayaKoe 為了 prosody(韻律)穩定性引入了 **按標點句子分割**,由此產生了 BERT 按句子數重複呼叫的新問題。

### BERT 輸出的不必要 CPU 轉換

原版 SBV2 的 BERT feature 擷取程式碼中有以下部分。

```python
# 原版 SBV2 (style_bert_vits2/nlp/japanese/bert_feature.py)
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```

BERT 在 GPU 上 forward 後,對輸出張量呼叫 `.cpu()` **每次都轉到 CPU**。

此輸出之後傳遞給 Synthesizer,而 Synthesizer 在 GPU 上執行所以又要轉回 GPU。

結果每個句子都產生 **GPU → CPU → GPU 往返**,這個不必要的往返本身就成為瓶頸。

### 逐句單獨呼叫 BERT

多句分割後對每個句子單獨呼叫 BERT 時,GPU kernel launch 按句子數重複。

kernel launch 是每次向 GPU 請求運算時產生的固定成本。

句子短時實際計算時間比 launch overhead 佔比小,隨句子數累積效率低下。

## 實作

### 移除 `.cpu()` — 保持 GPU 張量

移除原版的 `.cpu()` 呼叫,使 BERT 輸出作為 GPU 張量直接傳遞給 Synthesizer。

```python
# 原版 SBV2
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()    # GPU → CPU

# HayaKoe
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].float()  # 保持 GPU
```

BERT 模型本身也在 `prepare()` 時載入到 GPU,直到推論結束保持不變。

BERT 模型作為 **全局單例** 管理,載入多個說話人時 BERT 也只載入一次,所有說話人共享。

### 多句 BERT 批次化

HayaKoe 使用的 BERT (DeBERTa) 是 HuggingFace Transformer 模型,天然支援 batch 輸入。

利用此特性,多句合成時不逐句呼叫 BERT,而是 **將所有句子打包為一個批次 1 次處理**。

將多個句子一起送入 tokenizer 製作 padding 後的批次輸入,BERT **僅 forward 1 次**。

ONNX 路徑中也實作了相同的批次邏輯。

## 改善效果

### GPU 批次推論速度

同一硬體上 **順序 (sequential) vs 批次 (batched)** 對比(5 次平均)。

| 句子數 | 順序 | 批次 | 速度提升 |
|---|---|---|---|
| 2 | 0.447 s | 0.364 s | **1.23x** |
| 4 | 0.812 s | 0.566 s | **1.43x** |
| 8 | 1.598 s | 1.121 s | **1.43x** |
| 16 | 2.972 s | 2.264 s | **1.31x** |

kernel launch overhead 合併為 1 次的效果,帶來 +23% ~ +43% 的速度提升。

### GPU 記憶體

確認批次化是否額外消耗記憶體。

| 句子數 | 順序 peak | 批次 peak | 差異 |
|---|---|---|---|
| 2 | 1,662.2 MB | 1,661.9 MB | -0.3 MB |
| 4 | 1,661.8 MB | 1,662.2 MB | +0.4 MB |
| 8 | 1,697.7 MB | 1,699.0 MB | +1.3 MB |
| 16 | 1,934.3 MB | 1,934.3 MB | 0 MB |

順序和批次間差異 **在 1.3 MB 以內**,實質上相同。

### CPU 上無效果

在 CPU (ONNX) 上重複同一實驗,批次化效果幾乎不出現。

| 句子數 | 順序 | 批次 | 速度差異 |
|---|---|---|---|
| 2 | 2.566 s | 2.564 s | 1.00x |
| 4 | 5.464 s | 4.855 s | 1.13x |
| 8 | 10.647 s | 11.783 s | 0.90x |
| 16 | 24.559 s | 24.195 s | 1.01x |

ONNX Runtime 的圖最佳化已經足夠強,Python 層面的 dispatch overhead 不是瓶頸,批次時的 padding overhead 抵消了收益。

GPU 上保持批次化,CPU 上收益·損失都不大,為了後端間程式碼統一保持相同路徑。
