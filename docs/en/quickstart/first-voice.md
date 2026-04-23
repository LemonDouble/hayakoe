# First Voice

Now that installation is complete, it is time to actually generate a wav file.

This page covers the most basic "one line of text to one wav file" workflow and a few natural variations from there.

## Basics — Save a Single Sentence as wav

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()

speaker = tts.speakers["jvnv-F1-jp"]
speaker.generate("こんにちは、はじめまして。").save("hello.wav")
```

`TTS()` only registers the engine specification.

The actual model download to disk and loading into memory happens when `prepare()` is called.

`speakers` is a dict that lets you access prepared speakers by name.

::: tip Running on GPU
First, make sure the GPU extras are installed ([Installation — GPU Installation (CUDA)](./install#gpu-installation-cuda)).

Then just change one thing: `TTS(device="cuda")`. The rest of the code stays the same.

Just remember that the first call may be slightly slow.
:::

## Multiple Sentences Work as-is

Sentences are automatically split at boundaries (`。`, `！`, `？`, `!`, `?`, newlines), and natural-length silence is inserted between sentences.

```python
text = """
こんにちは。今日はいい天気ですね。
散歩でもしましょうか？
"""

speaker.generate(text).save("long.wav")
```

The pause length between sentences is predicted by the Duration Predictor, which is **trained separately for each speaker**.

This means some speakers pause longer between sentences and some shorter — the speaker's own breathing rhythm is faithfully reflected.

## Getting bytes Instead of a wav File

To return audio directly as a response from a web server like FastAPI, use `to_bytes()`.

The return value is a WAV-format byte stream.

```python
audio = speaker.generate("テストです。")
payload: bytes = audio.to_bytes()
```

## Loading Multiple Speakers at Once

`load()` supports chaining, so you can register multiple speakers at once.

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

Because BERT is shared across all speakers, adding one more speaker does not double or triple the memory.

The per-speaker increase comes from the much smaller synthesizer, not BERT, so loading 4 speakers does not mean 4x the RAM.

Actual measurements and how to reproduce them are documented separately in [FAQ — How much more memory does loading multiple speakers use?](/en/faq/#how-much-more-memory-does-loading-multiple-speakers-use).

## Tweaking Speed or Pitch

`generate()` accepts parameters for directly adjusting speech speed, pitch, and intonation.

```python
speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。",
    speed=0.95,        # slightly slower
    pitch_scale=1.05,  # slightly higher
).save("tuned.wav")
```

Available parameters and recommended ranges are covered in [Speed & Prosody Controls](./parameters).
