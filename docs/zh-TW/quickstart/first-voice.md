# 生成第一條語音

安裝完成後,現在是實際生成一個 wav 檔案的時候了。

本頁面涵蓋最基本的「一行文本 → 一個 wav」流程,以及由此自然延伸的幾種變體。

## 基本 — 將一句話儲存為 wav

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()

speaker = tts.speakers["jvnv-F1-jp"]
speaker.generate("こんにちは、はじめまして。").save("hello.wav")
```

`TTS()` 僅註冊引擎規格。

模型實際下載到硬碟並載入到記憶體是在呼叫 `prepare()` 的時候。

`speakers` 是一個可以透過名稱取出已準備好的說話人的 dict。

::: tip 想用 GPU 執行的話
首先需要安裝 GPU extras([安裝 — GPU 安裝 (CUDA)](./install#gpu-安裝-cuda))。

然後只需將 `TTS(device="cuda")` 改一個字即可。其餘程式碼完全相同。

只需記住首次呼叫可能會稍慢。
:::

## 多句話直接輸入即可

在句子邊界(`。`、`！`、`？`、`!`、`?`、換行)處會自動分割,句子之間會插入自然長度的靜音。

```python
text = """
こんにちは。今日はいい天気ですね。
散歩でもしましょうか？
"""

speaker.generate(text).save("long.wav")
```

句間 pause 的長度由 Duration Predictor 預測並插入,該預測器是 **按說話人分別訓練** 的。

也就是說,有的說話人句間停頓較長,有的較短,每位說話人固有的呼吸節奏都會得到反映。

## 不要 wav 檔案而要 bytes

如果想在 FastAPI 等 Web 伺服器中直接作為回應返回,可以使用 `to_bytes()`。

返回值是 WAV 格式的位元組串流。

```python
audio = speaker.generate("テストです。")
payload: bytes = audio.to_bytes()
```

## 同時載入多個說話人

`load()` 支援鏈式呼叫,因此可以一次註冊多個說話人。

```python
tts = (
    TTS()
    .load("jvnv-F1-jp")
    .load("jvnv-F2-jp")
    .load("jvnv-M1-jp")
    .load("jvnv-M2-jp")
    .prepare()
)

for name, speaker in tts.speakers.items():
    speaker.generate("おはようございます。").save(f"{name}.wav")
```

由於 BERT 是所有說話人共享的,每增加一個說話人記憶體不會翻倍或三倍增長。

每個說話人增加的量不是 BERT 而是小得多的 synthesizer 部分,所以載入 4 個說話人 RAM 也不會變成 4 倍。

實際測量值和重現方法已在 [FAQ — 載入多個說話人時記憶體增加多少?](/zh-TW/faq/#載入多個說話人時記憶體增加多少) 中單獨整理。

## 想稍微調整速度或音高時

`generate()` 接受可以直接調整語速・音高・韻律的參數。

```python
speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。",
    speed=0.95,        # 稍微快一點
    pitch_scale=1.05,  # 稍微高一點
).save("tuned.wav")
```

可用參數和建議範圍在 [速度・韻律調整](./parameters) 中整理。
