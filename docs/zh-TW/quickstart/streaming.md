# 句子級串流傳輸

HayaKoe 支援 **不一次性合成全部長文本,而是每合成完一個句子就立即發送** 的串流模式。

在對話式 UI 或需要即時回應的場景中,可以更快地輸出第一條語音。

## 何時使用?

想像一下合成多句文本的場景。

`generate()` 需要等到所有句子合成完畢才能返回 wav。

如果是 15 秒的音訊,使用者在合成完成之前聽不到任何聲音。

而 `stream()` 在第一個句子完成的瞬間就會立即傳遞該 chunk。

在播放第一個句子的同時後續句子繼續合成,因此感知延遲大大降低。

## 基本 — 按句子接收 chunk

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

text = "こんにちは。私はイレイナ。旅の魔女です。"

for i, chunk in enumerate(speaker.stream(text)):
    chunk.save(f"chunk_{i:02d}.wav")
```

`speaker.stream()` 是一個將每個句子作為 `AudioResult` yield 的 Python 生成器。

與 `generate()` 返回的型別相同,`.save()`・`.to_bytes()`・`.data` 全部可以同樣使用。

範例文本在 `。` 處被分成 3 個句子,按順序輸出 3 個 chunk。

## 參數與 `generate()` 相同

可以直接使用 `speed`、`pitch_scale`、`style`、`intonation_scale` 等 [速度・韻律調整](./parameters) 參數。

```python
for chunk in speaker.stream(text, speed=0.95, style="Happy"):
    chunk.save(...)  # 或直接寫入音訊裝置
```

## 句間靜音自動插入

從第二個 chunk 開始,前端會自動包含 **填充前一句間隔的靜音**。

因此只需按順序拼接或播放 chunk,就能獲得與 `generate()` 一次性接收相同的呼吸節奏。

CPU (ONNX) 和 GPU (PyTorch) 兩種後端都使用 Duration Predictor 預先預測句子邊界 pause,並反映到各 chunk 之間的間隔中。

ONNX 後端使用訓練時一同匯出的獨立 `duration_predictor.onnx` 模型在 CPU 會話上執行。

兩種路徑都套用了 80ms 的下限,即使預測值過短也不會拼接得不自然。

::: info Duration Predictor 是什麼
每個說話人在句間停頓多久有其固有的習慣。

HayaKoe 使用按說話人訓練的小型模型來預測此 pause 時長。
:::

## 非同步版本 — Web 伺服器用

在 FastAPI 等 async 執行時中使用 `astream()`。

```python
async for chunk in speaker.astream(text):
    await send(chunk.to_bytes())
```

內部在獨立執行緒中取出 `stream()` 的各 chunk 並 yield,因此不會阻塞事件循環。

::: warning 生成器請消耗到底
`stream()` / `astream()` 內部持有 per-speaker lock。

如果同一說話人可能收到多個請求,請不要在中間丟棄生成器,而是用 `for` 循環遍歷完畢,或用 `try/finally` 保證 close。

在消耗完畢或 close 時 lock 才會釋放。
:::
