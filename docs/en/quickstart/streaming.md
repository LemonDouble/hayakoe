# Sentence-level Streaming

HayaKoe supports a streaming mode that **synthesizes and delivers audio sentence by sentence, rather than waiting for the entire text to finish**.

This lets you send the first audio chunk quickly in conversational UIs or real-time response scenarios.

## When Should I Use This?

Imagine you are synthesizing multi-sentence text.

`generate()` requires waiting until all sentences are fully synthesized before returning the wav.

For a 15-second audio clip, the user hears nothing until synthesis completes.

In contrast, `stream()` delivers each chunk as soon as the first sentence is ready.

While the first sentence plays, the following sentences continue synthesizing in the background, significantly reducing perceived latency.

## Basics — Receiving Chunks Per Sentence

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

text = "こんにちは。私はイレイナ。旅の魔女です。"

for i, chunk in enumerate(speaker.stream(text)):
    chunk.save(f"chunk_{i:02d}.wav")
```

`speaker.stream()` is a Python generator that yields each sentence as an `AudioResult`.

It is the same type returned by `generate()`, so `.save()`, `.to_bytes()`, and `.data` all work identically.

The example text is split into 3 sentences at `。`, yielding 3 chunks in order.

## Parameters Are Identical to `generate()`

You can use the same [Speed & Prosody Controls](./parameters) parameters like `speed`, `pitch_scale`, `style`, and `intonation_scale`.

```python
for chunk in speaker.stream(text, speed=0.95, style="Happy"):
    chunk.save(...)  # or write directly to an audio device
```

## Silence Between Sentences Is Automatic

Starting from the second chunk, **silence to fill the gap with the previous sentence** is automatically included at the beginning.

This means simply concatenating or playing chunks in order produces the same breathing rhythm as a single `generate()` call.

Both CPU (ONNX) and GPU (PyTorch) use the Duration Predictor to predict sentence boundary pauses in advance and reflect them in each chunk gap.

The ONNX backend runs a separate `duration_predictor.onnx` model exported during training as a CPU session.

Both paths apply a minimum floor of 80ms, so even if the predicted value is very short, sentences never sound unnaturally glued together.

::: info What is the Duration Predictor?
Each speaker has their own habits for how long they pause between sentences.

HayaKoe predicts this pause length using a small model trained per speaker.
:::

## Async Version — For Web Servers

For use in async runtimes like FastAPI, use `astream()`.

```python
async for chunk in speaker.astream(text):
    await send(chunk.to_bytes())
```

Internally, it pulls each chunk from `stream()` in a separate thread and yields them, so it does not block the event loop.

::: warning Fully consume the generator
`stream()` / `astream()` hold a per-speaker lock internally.

If multiple requests may target the same speaker, do not abandon the generator midway. Either iterate with a `for` loop until the end, or guarantee `close` with `try/finally`.

The lock is released when the generator is exhausted or closed.
:::
