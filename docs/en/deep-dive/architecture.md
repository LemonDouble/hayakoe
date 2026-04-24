# Architecture Overview

HayaKoe is an engine that takes Japanese text and produces WAV waveforms. Internally, it is not a single "big model" processing everything at once, but rather **a multi-stage pipeline** with divided responsibilities.

This page walks through **what happens at each stage** as one input text becomes an audio waveform, in order.

If the terminology is unfamiliar, we recommend lightly skimming the [Glossary](./glossary) first and coming back.

## Full Pipeline

```
Input text (Japanese)
  |
  +- 1. Sentence splitting (by punctuation)
  |
  +- For each sentence:
  |    +- 2. G2P              -- Text -> phoneme sequence + accent
  |    +- 3. BERT             -- Sentence-level context embedding (shared across speakers)
  |    +- 4. Synthesizer      -- Phonemes + BERT -> sentence-level waveform
  |         +- 4-1. Text Encoder
  |         +- 4-2. Duration Predictor
  |         +- 4-3. Flow
  |         +- 4-4. Decoder
  |
  +- 5. Sentence boundary pause -- Duration Predictor reuse
  |
  +- 6. Waveform concatenation -> final WAV
```

Let us walk through each step below.

## Step-by-step Walkthrough

### 1. Sentence Splitting

The input text is first **split into sentences.** It is divided at Japanese punctuation (`。`, `！`, `？`) and ASCII punctuation (`.`, `!`, `?`), and all subsequent processing is performed **individually** per sentence.

The reason HayaKoe splits rather than feeding everything at once is **quality**. Synthesizing long texts whole tends to blur intonation (prosody instability), and the context captured by BERT becomes excessively long, causing certain syllables to be exaggerated. Splitting by sentence ensures stable prosody for each sentence.

There are two trade-offs.

First, the **inter-sentence pauses** that the original model naturally created during whole-text synthesis are lost due to splitting. This is resolved in the later [step 5: sentence boundary pause](#_5-sentence-boundary-pause-prediction) by reusing the Duration Predictor.

Second, **on the GPU path, splitting incurs a speed penalty**. GPU synthesis time for a single sentence is already short. Splitting forces this short synthesis to repeat for each sentence count, plus the additional step of computing inter-sentence pause lengths. The result is more total time than whole-text synthesis.

HayaKoe maintains splitting as the default behavior because the quality benefit justifies this cost. On the CPU path, the cost of synthesizing long text whole is much greater, so splitting actually **benefits speed as well**.

### 2. G2P — Text to Phoneme Sequence + Accent

After receiving each sentence, the first task is to **convert characters to pronunciation**. HayaKoe delegates Japanese G2P to `pyopenjtalk`.

G2P handles more than just one or two things. Kanji readings (`天気` -> `てんき`), long vowel rules, liaison (sandhi), accent type (pitch accent) determination — all Japanese-specific rules are resolved here. The output is not a simple list of phonemes but a **phoneme sequence + accent information** pair.

This step runs entirely on CPU/Python and is unrelated to model inference. pyopenjtalk's internal dictionary files are bundled inside the wheel, so it works without network access (-> [OpenJTalk Dictionary Bundling](./openjtalk-dict)).

### 3. BERT — Context Embedding

Separately from G2P, the **raw text itself** is fed into the DeBERTa BERT to obtain sentence context embeddings. Each token gets a vector capturing "what role this word plays in this sentence."

BERT features directly impact the Synthesizer's synthesis quality. Even with the same phoneme sequence, the intonation and stress vary depending on the context BERT observes.

For example, the same response `そうですね` naturally sounds like a confident affirmation if preceded by a strong assertion, but an empathetic lingering tone if preceded by hesitation. G2P only sees the phoneme sequence and cannot distinguish between the two cases. BERT compresses the surrounding context and passes it to the Synthesizer so these differences are reflected in synthesis.

This is the core mechanism by which the Bert-VITS2 lineage produces more natural speech compared to the original VITS.

A key structural feature is that **BERT is shared across all speakers**. BERT accounts for approximately 84% of total model parameters (~329M, ~1.2 GB at FP32) — it is the **largest single module**. In contrast, the per-speaker Synthesizer is 63M (~251 MB).

This asymmetry makes the structure of loading BERT once and sharing it across all speakers decisive. Even when serving N speakers simultaneously, BERT is loaded only once and each speaker adds only 251 MB. This is why memory does not explode linearly in multi-speaker serving.

Additionally, when processing multiple sentences, they are **batched together for a single BERT call**. On GPU, this reduces kernel launch overhead with benefits proportional to sentence count. On CPU, the gain and loss are nearly zero, so the same path is maintained for code consistency across backends (-> [BERT GPU Retention & Batch Inference](./bert-gpu)).

### 4. Synthesizer — Phonemes + BERT to Waveform

This is where the actual audio is produced. The input is `(phoneme sequence, accent, BERT embedding, style vector)`, and the output is **the waveform for that sentence**.

The **style vector** is loaded from the speaker's `style_vectors.npy` at `.load(speaker)` time — a representation that compresses the speaker's speaking style and tone characteristics into a single vector. HayaKoe currently uses only the Neutral style for simplification (-> [Glossary — Style Vector](./glossary#style-vector)), and this value is injected as-is with every synthesis call.

The Synthesizer is internally divided into four sub-modules, and text information flows through them in this order to become a waveform.

#### 4-1. Text Encoder — Phonemes to Vector Space

A Transformer encoder structure that embeds each phoneme into a **192-dimensional hidden vector**. BERT features are combined with phoneme-level embeddings here, marking the first point where sentence context is injected into phoneme-level information.

Output shape is `(phoneme count, 192)`, shared as input to the next two stages.

#### 4-2. Duration Predictor — How Many Frames for Each Phoneme

The Stochastic Duration Predictor (SDP) samples from a probability distribution to determine "how many frames each phoneme should sound." Like "a" for 5 frames, "n" for 4 frames. Because it uses probability sampling, intonation and speed vary slightly with each call even for the same sentence.

At this step, the phoneme sequence **expands along the time axis.** Think of each phoneme being copied and concatenated for its predicted duration. The resulting length directly corresponds to the number of frames in the final audio.

#### 4-3. Flow — Text Embedding to Audio Latent

The reverse transform of Normalizing Flow converts the embeddings from the Text Encoder into **z vectors in the audio latent space**. This z is a low-dimensional representation of "what the sound should be," which the next Decoder uses to produce the actual waveform.

Because Flow is an invertible neural network, it maps in the forward direction (ground truth audio latent -> text embedding space) during training, and runs in reverse (text -> audio space) during inference.

#### 4-4. Decoder — Latent z to Waveform

A HiFi-GAN-based Decoder that takes latent z and generates the **actual time-domain waveform**. It consists of ConvTranspose upsampling and residual blocks (ResBlock).

This has the highest computational cost among Synthesizer sub-modules, consuming most of the inference time. It is also the primary optimization target when exporting to ONNX for the CPU path.

After all four stages, the 44.1 kHz waveform for that single sentence is complete.

### 5. Sentence Boundary Pause Prediction

This step resolves the "inter-sentence pause loss" problem mentioned in [step 1](#_1-sentence-splitting).

HayaKoe **reuses the Duration Predictor beyond its original purpose**. The full original text is passed through only Text Encoder + Duration Predictor, extracting frame counts at `.`, `!`, `?` punctuation positions and converting them to pause durations in seconds. Flow and Decoder are skipped, keeping the additional cost low.

As a result, instead of a fixed 80ms silence, **pauses that naturally vary with sentence length and structure** are generated. Details are in [Sentence Boundary Pause — Duration Predictor](./duration-predictor).

### 6. Waveform Concatenation -> Final Output

Finally, each sentence's waveform is concatenated in order, with silence samples of the length predicted in [step 5](#_5-sentence-boundary-pause-prediction) inserted between sentences.

The final output is a 44.1 kHz, single-channel, float32 waveform (NumPy array). Calling `.save()` writes it as a WAV file.

When using the Streaming API (`astream()`), each completed sentence is yielded immediately, so playback can start before full synthesis is done. This **significantly reduces time-to-first-audio** for long texts (-> [Streaming Example](/en/quickstart/streaming)).

## Further Reading

- **Backend selection (CPU vs GPU) and `load`, `prepare`, `generate` lifecycle** — [Backend Selection](/en/deploy/backend)
- **Optimization details for each stage** — [ONNX Optimization](./onnx-optimization) / [BERT GPU Retention & Batch Inference](./bert-gpu)
- **Sentence boundary pause implementation details** — [Sentence Boundary Pause](./duration-predictor)
