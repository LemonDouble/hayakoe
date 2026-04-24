# Glossary

This page defines TTS and inference terms frequently used throughout the deep-dive section.

It is designed so that **readers new to TTS can follow the pages after the architecture overview without getting stuck**.

You can read through in order, or look up specific terms as needed.

## The Big Picture of the Pipeline

### TTS (Text-to-Speech)

The general term for technology that converts text into human-sounding audio.

Input is typically a sentence, and output is an audio file like WAV or MP3.

Internally, a TTS system goes through several stages: "interpret characters as pronunciation" -> "synthesize that pronunciation into a waveform."

HayaKoe falls into this category, taking Japanese input and generating WAV waveforms.

### Phoneme

The smallest unit of sound that distinguishes meaning in speech.

In Korean, "bal" and "dal" differ by only the first sound (b vs d) yet have completely different meanings. These **meaning-distinguishing sound units** are phonemes.

In Korean, written characters and actual pronunciation often differ. For example, "같이" (gat-i) is two characters but pronounced "gachi," so the number of characters and phonemes do not always match.

TTS models receive phonemes, not characters, as input. If characters were used directly, the model would also need to learn all the pronunciation rules about "how to read in which context." Converting to phonemes first lets the model focus solely on "how to make this sound audible."

The module responsible for this conversion is the **G2P**.

### G2P (Grapheme-to-Phoneme)

The process, or module, that converts characters (Graphemes) to phonemes (Phonemes).

This handles all language-specific pronunciation rules: Korean palatalization ("같이 -> 가치"), nasalization ("독립 -> 동닙"), Japanese kanji readings, liaison rules, etc.

It sits at the stage just before feeding input to the model in the TTS pipeline.

HayaKoe is Japanese-only, so it delegates Japanese G2P to [pyopenjtalk](./openjtalk-dict).

### Waveform

A sequence of numbers recording air pressure changes over time. This is the "actual sound" that speakers can play.

Each number represents **air pressure (amplitude) at a specific moment**. Positive values mean air is compressed relative to the baseline, negative means expanded, and larger absolute values mean louder sound. 0 corresponds to silence (baseline pressure).

At a sample rate of 22,050 Hz, 1 second = 22,050 such numbers. HayaKoe outputs at 44,100 Hz, so 1 second = 44,100 numbers.

The final output of TTS is precisely this waveform.

## Model Components

### VITS

A speech synthesis model architecture published in 2021.

Its key contribution was unifying what had previously been a two-stage TTS pipeline (Acoustic Model + Vocoder) into a **single End-to-End model**.

It performs text-to-waveform conversion directly in a single model, internally composed of Text Encoder, Duration Predictor, Flow, and Decoder.

HayaKoe is an extension of the VITS lineage.

- **VITS (2021)** — Starting point of End-to-End TTS.
- **[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)** — Fish Audio team added BERT to VITS for **context-based prosody** enhancement, as an open-source project.
- **[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)** — litagin02 forked Bert-VITS2 and added **Style Vectors**, enabling diverse tones and emotions for the same speaker. The Japanese-specialized variant **JP-Extra** showed quality advantages.
- **HayaKoe** — Trimmed Style-Bert-VITS2 JP-Extra to **Japanese-only** and restructured it for practical CPU and server operations.

The Synthesizer model architecture itself is unchanged from Style-Bert-VITS2; HayaKoe's additions are concentrated outside of it (ONNX path, inference pipeline, deployment, and source simplification).

### Synthesizer

In HayaKoe, the collective name for the **VITS core (Text Encoder + Duration Predictor + Flow + Decoder)**.

It takes phoneme sequences and BERT features as input and produces the final waveform.

BERT exists **outside** the Synthesizer as a separate module shared by all speakers. What changes per speaker is the Synthesizer weights (~300 MB).

### BERT

A Transformer-based pretrained language model published by Google in 2018. It reads sentences and creates contextual embeddings for each token.

In TTS, it is used to **reflect sentence meaning and context in synthesis**. Even with the same phoneme sequence, BERT enables more natural intonation and stress.

HayaKoe uses a Japanese-specific DeBERTa v2 model (`ku-nlp/deberta-v2-large-japanese-char-wwm`).

On the CPU path, this BERT is quantized to INT8 and run via ONNX.

### Text Encoder

An internal module of the Synthesizer. It takes a phoneme sequence as input and outputs a **192-dimensional hidden vector** for each phoneme.

It uses a Transformer encoder structure, where self-attention lets each phoneme reference its surrounding context to create the embeddings needed for synthesis.

Conceptually, it can be thought of as a miniature BERT. The difference is that BERT operates at the word/sentence level, while the Text Encoder operates at the phoneme level.

### Duration Predictor (SDP)

A module that predicts **how many frames** each phoneme should be voiced. Like "a" for 5 frames, "n" for 4 frames.

"SDP" stands for **Stochastic Duration Predictor**. Because it samples from a probability distribution rather than being deterministic, intonation and speed vary slightly with each call even for the same sentence.

HayaKoe reuses this module beyond its original purpose for **sentence boundary pause prediction**. Details are in [Sentence Boundary Pause — Duration Predictor](./duration-predictor).

### Flow

An internal module of the Synthesizer. An **invertible** neural network that can compute both forward and backward.

During training, it maps "ground truth audio latent -> text embedding space," and during inference, it runs in reverse to generate audio latent from text embeddings.

Its formal name is **Normalizing Flow**.

::: warning Flow and quantization
The main reason HayaKoe does not lower the Synthesizer to FP16 is the Flow. Flow's `rational_quadratic_spline` operation causes floating-point assertion errors at FP16.

Synthesizer INT8 was excluded for a separate reason — the Conv1d-centric structure means PyTorch dynamic quantization does not apply automatically, and static quantization has high implementation complexity.
:::

### Decoder (HiFi-GAN)

The final module of the Synthesizer. It takes the latent vector from Flow and generates the **actual waveform**.

It is the HiFi-GAN architecture that was previously used as a standalone Vocoder, which VITS integrated into the model.

**The key module enabling VITS's End-to-End operation**, and simultaneously the part that accounts for a significant portion of TTS inference time.

### Style Vector

A vector that compresses a speaker's "tone and speaking style" information into a single representation.

Even for the same speaker, you can switch between styles like "calm," "happy," and "angry" for synthesis.

This is a component unique to the Style-Bert-VITS2 lineage, provided as `style_vectors.npy` alongside the per-speaker safetensors.

HayaKoe currently uses **only the Neutral style** for simplification. Support for diverse style selection is planned for future improvement.

### Prosody

A collective term for **intonation, rhythm, stress, and pauses** in speech.

If phonemes are "what is being pronounced," prosody is "how it is pronounced."

"Really?" (rising intonation — question) and "Really." (falling intonation — statement) have the same phonemes but different prosody.

The most common reason TTS sounds "robotic" is when prosody is not natural.

One of the main reasons the Bert-VITS2 lineage uses BERT is to obtain prosody hints from sentence context.

## Performance and Execution Terms

### ONNX and ONNX Runtime

**ONNX (Open Neural Network Exchange)** is a standard format for saving neural network models **independently of the framework**.

Whether trained in PyTorch, TensorFlow, or elsewhere, exporting to ONNX treats them as the same graph.

**ONNX Runtime** is the inference engine that actually executes ONNX models. Written in C++, it has low Python overhead and pre-applies various optimizations by analyzing the model graph.

It supports diverse execution devices including CPU, CUDA, and ARM (aarch64).

HayaKoe's entire CPU path runs on ONNX Runtime. This is also why the same code works on both x86_64 and aarch64.

### Quantization

A technique that saves memory and computation by reducing the precision of model weight representations.

Deep learning model weights are typically stored at one of these precisions.

- **FP32** — 32-bit floating point. Default. Most precise but largest.
- **FP16** — 16-bit floating point. Half the size of FP32.
- **INT8** — 8-bit integer. About 1/4 the size of FP32. Often called "Q8."
- **INT4** — 4-bit integer. About 1/8 the size of FP32. Actively used in recent LLM applications.

Lower bit counts reduce model file size and RAM usage roughly proportionally, and on certain hardware, computations also speed up.

However, **reduced precision can degrade output quality.** How far you can quantize without quality impact varies by model and operation type.

HayaKoe chose to apply **INT8 dynamic quantization to BERT's MatMul only (Q8 Dynamic Quantization)**, while keeping the Synthesizer at FP32. Detailed reasoning and measured effects are in [ONNX Optimization](./onnx-optimization).

### Kernel Launch Overhead

The fixed cost incurred when the CPU tells the GPU to "execute this kernel." Separate from actual computation time, each kernel call costs microseconds to tens of microseconds.

In workloads where a single kernel does heavy computation, this cost is negligible. But **in cases like TTS where small Conv1d operations repeat hundreds of times**, kernel launch overhead can account for a significant portion of total time.

CUDA Graph, kernel fusion, and torch.compile are techniques for reducing this cost.

### Eager Mode

PyTorch's default execution mode. Python code executes line by line, dispatching individual GPU kernels on the fly.

The advantage is easy debugging, but Python dispatch overhead and kernel launch overhead accumulate with every kernel.

`torch.compile` is the alternative that eliminates this overhead through graph-level optimization.

### torch.compile

A **JIT compiler** available since PyTorch 2.0.

On the first call, it traces the model as a graph, fuses and recompiles kernels, and runs faster on subsequent calls.

HayaKoe uses `torch.compile` on the GPU path.

The first call incurs compilation time, so `prepare(warmup=True)` can shift this cost to the serving startup phase.

## Other

### OpenJTalk

An open-source Japanese TTS frontend developed at Nagoya Institute of Technology.

It takes Japanese text and generates **phoneme sequences and accent information**. Japanese-specific rules like kanji readings and liaison are included here.

HayaKoe uses this functionality through the Python binding [pyopenjtalk](./openjtalk-dict).
