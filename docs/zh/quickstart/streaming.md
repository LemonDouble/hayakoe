# 句子级流式传输

HayaKoe 支持 **不一次性合成全部长文本,而是每合成完一个句子就立即发送** 的流式模式。

在对话式 UI 或需要实时响应的场景中,可以更快地输出首条语音。

## 何时使用?

想象一下合成多句文本的场景。

`generate()` 需要等到所有句子合成完毕才能返回 wav。

如果是 15 秒的音频,用户在合成完成之前听不到任何声音。

而 `stream()` 在第一个句子完成的瞬间就会立即传递该 chunk。

在播放第一个句子的同时后续句子继续合成,因此感知延迟大大降低。

## 基本 — 按句子接收 chunk

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

text = "こんにちは。私はイレイナ。旅の魔女です。"

for i, chunk in enumerate(speaker.stream(text)):
    chunk.save(f"chunk_{i:02d}.wav")
```

`speaker.stream()` 是一个将每个句子作为 `AudioResult` yield 的 Python 生成器。

与 `generate()` 返回的类型相同,`.save()` · `.to_bytes()` · `.data` 全部可以同样使用。

示例文本在 `。` 处被分成 3 个句子,按顺序输出 3 个 chunk。

## 参数与 `generate()` 相同

可以直接使用 `speed`、`pitch_scale`、`style`、`intonation_scale` 等 [速度·韵律调节](./parameters) 参数。

```python
for chunk in speaker.stream(text, speed=0.95, style="Happy"):
    chunk.save(...)  # 或直接写入音频设备
```

## 句间静音自动插入

从第二个 chunk 开始,前端会自动包含 **填充前一句间隔的静音**。

因此只需按顺序拼接或播放 chunk,就能获得与 `generate()` 一次性接收相同的呼吸节奏。

CPU (ONNX) 和 GPU (PyTorch) 两种后端都使用 Duration Predictor 预先预测句子边界 pause,并反映到各 chunk 之间的间隔中。

ONNX 后端使用训练时一同导出的独立 `duration_predictor.onnx` 模型在 CPU 会话上运行。

两种路径都应用了 80ms 的下限,即使预测值过短也不会拼接得不自然。

::: info Duration Predictor 是什么
每个说话人在句间停顿多久有其固有的习惯。

HayaKoe 使用按说话人训练的小型模型来预测此 pause 时长。
:::

## 异步版本 — Web 服务器用

在 FastAPI 等 async 运行时中使用 `astream()`。

```python
async for chunk in speaker.astream(text):
    await send(chunk.to_bytes())
```

内部在独立线程中取出 `stream()` 的各 chunk 并 yield,因此不会阻塞事件循环。

::: warning 生成器请消耗到底
`stream()` / `astream()` 内部持有 per-speaker lock。

如果同一说话人可能收到多个请求,请不要在中间丢弃生成器,而是用 `for` 循环遍历完毕,或用 `try/finally` 保证 close。

在消耗完毕或 close 时 lock 才会释放。
:::
