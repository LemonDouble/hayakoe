# 生成第一条语音

安装完成后,现在是实际生成一个 wav 文件的时候了。

本页面涵盖最基本的"一行文本 → 一个 wav"流程,以及由此自然延伸的几种变体。

## 基本 — 将一句话保存为 wav

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()

speaker = tts.speakers["jvnv-F1-jp"]
speaker.generate("こんにちは、はじめまして。").save("hello.wav")
```

`TTS()` 仅注册引擎规格。

模型实际下载到磁盘并加载到内存是在调用 `prepare()` 的时候。

`speakers` 是一个可以通过名称取出已准备好的说话人的 dict。

::: tip 想用 GPU 运行的话
首先需要安装 GPU extras([安装 — GPU 安装 (CUDA)](./install#gpu-安装-cuda))。

然后只需将 `TTS(device="cuda")` 改一个字即可。其余代码完全相同。

只需记住首次调用可能会稍慢。
:::

## 多句话直接输入即可

在句子边界(`。`、`！`、`？`、`!`、`?`、换行)处会自动分割,句子之间会插入自然长度的静音。

```python
text = """
こんにちは。今日はいい天気ですね。
散歩でもしましょうか？
"""

speaker.generate(text).save("long.wav")
```

句间 pause 的长度由 Duration Predictor 预测并插入,该预测器是 **按说话人分别训练** 的。

也就是说,有的说话人句间停顿较长,有的较短,每位说话人固有的呼吸节奏都会得到反映。

## 不要 wav 文件而要 bytes

如果想在 FastAPI 等 Web 服务器中直接作为响应返回,可以使用 `to_bytes()`。

返回值是 WAV 格式的字节流。

```python
audio = speaker.generate("テストです。")
payload: bytes = audio.to_bytes()
```

## 同时加载多个说话人

`load()` 支持链式调用,因此可以一次注册多个说话人。

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

由于 BERT 是所有说话人共享的,每增加一个说话人内存不会翻倍或三倍增长。

每个说话人增加的量不是 BERT 而是小得多的 synthesizer 部分,所以加载 4 个说话人 RAM 也不会变成 4 倍。

实际测量值和复现方法已在 [FAQ — 加载多个说话人时内存增加多少?](/zh/faq/#加载多个说话人时内存增加多少) 中单独整理。

## 想稍微调整速度或音高时

`generate()` 接受可以直接调节语速·音高·韵律的参数。

```python
speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。",
    speed=0.95,        # 稍微快一点
    pitch_scale=1.05,  # 稍微高一点
).save("tuned.wav")
```

可用参数和建议范围在 [速度·韵律调节](./parameters) 中整理。
