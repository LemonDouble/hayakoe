---
layout: home

hero:
  name: HayaKoe
  text: "Near-real-time TTS with your favorite voice,<br>on CPU alone."
  tagline: Just bring a video or recording — we handle data prep, training, benchmarking, and deployment for you.
  actions:
    - theme: brand
      text: Get Started in 10 Minutes
      link: /en/quickstart/
    - theme: alt
      text: Training
      link: /en/training/
    - theme: alt
      text: Deployment
      link: /en/deploy/
    - theme: alt
      text: Deep Dive
      link: /en/deep-dive/

features:
  - title: Real-time CPU Inference
    details: "With ONNX optimization, 1.5x faster for short texts and 3.3x faster for long texts compared to Style-Bert-VITS2 on CPU alone.<br>On GPU, torch.compile makes it even faster."
    link: /en/deep-dive/onnx-optimization
    linkText: How we did it
  - title: AMD64 · ARM64 Everywhere
    details: "Install on both x86_64 and aarch64 Linux with a single command.<br>CPU inference works seamlessly even on ARM boards like the Raspberry Pi."
    link: /en/quickstart/benchmark#raspberry-pi-4b
    linkText: Raspberry Pi Benchmark
  - title: 47% Memory Savings
    details: "BERT Q8 quantization reduces RAM by 47% compared to PyTorch.<br>Approx. 2.0 GB RAM in CPU mode, approx. 1.7 GB VRAM in GPU mode."
    link: /en/deep-dive/onnx-optimization
    linkText: How we did it
  - title: Lightweight Multi-speaker
    details: "BERT is shared across all speakers.<br>Adding one more speaker costs only ~300 MB of additional RAM."
    link: /en/deploy/fastapi
    linkText: Multi-speaker Serving
  - title: Sentence-level Streaming
    details: "Use astream() to stream audio as each sentence is synthesized.<br>Receive the first audio chunk much sooner than waiting for the full synthesis."
    link: /en/deploy/fastapi
    linkText: Streaming Example
  - title: Your Voice, Your Way
    details: "Just prepare a video with the voice you love.<br>We handle preprocessing, training, quality comparison, optimization, and deployment."
    link: /en/training/
    linkText: Training Guide
  - title: HF · S3-compatible · Local Pluggable
    details: "CLI deployment supports HuggingFace, S3-compatible, and local destinations.<br>Runtime loading supports the same three sources identically."
    link: /en/deep-dive/source-abstraction
    linkText: Source Abstraction
---

## Sample Voices

Here are samples of the built-in speakers saying the same sentence ("こんにちは、はじめまして。").

<SpeakerSample badge="JVNV" name="jvnv-F1-jp  —  Female Speaker 1" src="/hayakoe/samples/hello_jvnv-F1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-F2-jp  —  Female Speaker 2" src="/hayakoe/samples/hello_jvnv-F2-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M1-jp  —  Male Speaker 1" src="/hayakoe/samples/hello_jvnv-M1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M2-jp  —  Male Speaker 2" src="/hayakoe/samples/hello_jvnv-M2-jp.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="tsukuyomi_chan  —  Anime-style" src="/hayakoe/samples/hello_tsukuyomi_chan.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_normal  —  Normal" src="/hayakoe/samples/hello_amitaro_normal.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_runrun  —  Excited" src="/hayakoe/samples/hello_amitaro_runrun.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_yofukashi  —  Calm" src="/hayakoe/samples/hello_amitaro_yofukashi.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_punsuka  —  Angry" src="/hayakoe/samples/hello_amitaro_punsuka.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_a  —  Whisper A" src="/hayakoe/samples/hello_amitaro_sasayaki_a.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_b  —  Whisper B" src="/hayakoe/samples/hello_amitaro_sasayaki_b.wav" />

Want to generate the above samples yourself on your laptop, using only the CPU? Head to [Get Started in 10 Minutes](/en/quickstart/).

## Quick Overview

### Installation

::: code-group
```bash [CPU]
pip install hayakoe
```
```bash [GPU (CUDA)]
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
:::

The CPU profile does not require PyTorch, keeping installation fast and image size small.

The GPU profile installs additional dependencies but provides faster inference.

### Inference

```python
from hayakoe import TTS

text = "こんにちは、はじめまして。"

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate(text).save("hello.wav")
```

Go ahead and listen to `hello.wav`!

There are 11 built-in speakers.

- `jvnv-F1-jp` / `jvnv-F2-jp` / `jvnv-M1-jp` / `jvnv-M2-jp` — Based on the JVNV corpus
- `tsukuyomi_chan` — Based on the Tsukuyomi-chan Corpus
- `amitaro_normal` / `amitaro_runrun` / `amitaro_yofukashi` / `amitaro_punsuka` / `amitaro_sasayaki_a` / `amitaro_sasayaki_b` — Based on the Amitaro ITA Corpus

Simply replace `"jvnv-F1-jp"` in the code above to try a different voice.

If you installed the GPU profile, just add `TTS(device="cuda")` to run inference on GPU.

## Which Docs Should I Read?

1. **Start with the [Quickstart](/en/quickstart/)**. Follow along from installation to first synthesis and benchmarking to see how fast and how good the TTS sounds.
2. **Ready for more? Try [Custom Speaker Training](/en/training/)**. Use a single video with the voice you like to go from data preparation to deployment.
3. **Want to share it? Head to [Server Deployment](/en/deploy/)**. Learn how to expose an API on FastAPI and Docker.
4. **Want to dig deeper? Read the [Deep Dive](/en/deep-dive/)**. A detailed walkthrough of every optimization point that achieved these speed and memory improvements.
5. **Stuck on something? Check the [FAQ](/en/faq/)**. Advanced settings like cache paths, private HF repos, S3, and multi-speaker memory are covered here.

## Voice Data Credits

This project uses the following voice data for speech synthesis.

- **Tsukuyomi-chan Corpus** (CV. Rei Yumesaki, (C) Rei Yumesaki) — https://tyc.rei-yumesaki.net/material/corpus/
- **Amitaro no Koe Sozai Koubou** ITA Corpus Recordings — https://amitaro.net/
