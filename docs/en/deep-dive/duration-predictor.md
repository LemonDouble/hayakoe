# Sentence Boundary Pause — Duration Predictor

When synthesizing multi-sentence text by splitting, **the pauses between sentences are lost** as a side effect.

HayaKoe reuses the Duration Predictor to directly predict the natural pause time at each sentence boundary.

By skipping Flow and Decoder and **running only TextEncoder + Duration Predictor**, the additional cost is low.

## Why It Matters

### Benefits of Sentence Splitting

As explained in [Architecture Overview](./architecture#_1-sentence-splitting), HayaKoe splits multi-sentence input and synthesizes each sentence individually.

Feeding long text whole tends to blur or destabilize intonation.

Splitting by sentence ensures stable prosody for each sentence.

### Side Effect of Splitting — Pause Loss

However, splitting has a side effect.

The original SBV2 naturally inserts pauses after punctuation like `.`, `!`, `?` during whole-text synthesis.

With per-sentence splitting, each sentence ends at punctuation and the next starts from scratch, so **the pause after punctuation is lost along with it.**

The initial implementation inserted a fixed 80ms silence between sentences.

In reality, the Duration Predictor's predicted sentence boundary pauses are in the 0.3 to 0.6 second range, so 80ms is very short by comparison.

The result was unnatural speech with "no room to breathe."

## Mechanism Analysis

Before diving into this section, let us recap the internal flow of the Synthesizer (see [Architecture Overview — Synthesizer](./architecture#_4-synthesizer-phonemes-bert-to-waveform) for details).

<PipelineFlow
  :steps="[
    {
      num: '1',
      title: 'Text Encoder',
      content: 'A Transformer encoder embeds the phoneme sequence into 192-dimensional vectors. BERT features are combined with phoneme-level embeddings here, injecting sentence context into phonemes for the first time.'
    },
    {
      num: '2',
      title: 'Duration Predictor',
      content: 'Predicts how many frames each phoneme should be voiced. Blends outputs from the stable but monotone DDP (deterministic) and the natural but less stable SDP (stochastic) predictors using sdp_ratio, balancing stability and naturalness. The phoneme sequence expands along the time axis at this step.'
    },
    {
      num: '3',
      title: 'Flow',
      content: 'Through the reverse transform of Normalizing Flow (invertible neural network), transforms the Gaussian distribution (mean and variance) from the Text Encoder into the complex distribution of actual speech to generate the latent z vector. Forward during training (speech -> text space), reverse during inference (text -> speech space).'
    },
    {
      num: '4',
      title: 'Decoder',
      content: 'A HiFi-GAN-based vocoder that generates the actual time-domain waveform (44.1 kHz) from latent z through ConvTranspose upsampling and residual blocks (ResBlock). The most compute-intensive Synthesizer sub-module, consuming the majority of CPU inference time.'
    }
  ]"
/>

The key point of this document is **running only stages 1 and 2 (Text Encoder + Duration Predictor) separately**.

Stages 3 and 4 (Flow + Decoder) are skipped, keeping the cost very low.

### How the Original Model Created Pauses

Tracing how the original SBV2 generates natural pauses in whole-text synthesis revealed that it was a **side effect of the Duration Predictor predicting frame counts for punctuation phonemes**.

The Duration Predictor's original purpose is predicting "how many frames each phoneme should be voiced."

Like "a" for 5 frames, "n" for 4 frames.

However, punctuation like `.`, `!`, `?` is also included in the phoneme sequence.

The frame count predicted by the Duration Predictor for punctuation becomes the **pause length at that punctuation position**.

For example, if 20 frames are predicted for `.`, the Synthesizer generates a silent or near-silent waveform for that duration.

In split synthesis, synthesis cuts off at punctuation positions, so this information was being discarded.

### Internal Operation of the Duration Predictor

Looking more closely at the Duration Predictor's prediction flow, two sub-modules operate in parallel.

**DDP (Deterministic Duration Predictor)** always outputs the same duration for the same input.

Stable but can make speech sound mechanically monotone.

**SDP (Stochastic Duration Predictor)** outputs slightly different durations each time for the same input.

Being probability-sampling-based, it creates natural variation but is less stable.

The two predictors' outputs are blended by the `sdp_ratio` parameter.

`sdp_ratio=0.0` uses DDP only, `1.0` uses SDP only, and `0.5` uses a half-and-half mix.

`length_scale` (= the speed parameter) multiplies the entire predicted duration to adjust speech speed.

Finally, `ceil()` rounds up to determine the **integer frame count** for each phoneme.

### Blank Token and Punctuation

There is one additional consideration for pause calculation.

The original SBV2 inserts a **blank token (ID = 0)** between every phoneme in the sequence. HayaKoe follows this behavior exactly.

```
Original:  [は, い, .]
After insertion: [0, は, 0, い, 0, ., 0]
```

Since blank tokens also get predicted durations, when calculating the pause for punctuation `.`, you must **sum the durations of the punctuation itself plus adjacent blanks**.

Example: `.` = 20 frames, preceding blank = 3 frames, following blank = 5 frames -> total 28 frames

## Implementation

### Core Idea

The core is simple.

**Pass the full original text through TextEncoder + Duration Predictor only, and extract frame counts at punctuation positions.**

Flow and Decoder are skipped.

Since most of the cost in a full Synthesizer pass comes from Flow and Decoder ([ONNX Optimization — Synthesizer share](./onnx-optimization#synthesizer-optimization)), running only up to the Duration Predictor is relatively cheap.

```
Full text (original, before splitting)
  |
  +- TextEncoder (G2P -> phoneme sequence -> embedding)
  |
  +- Duration Predictor (predict frame count per phoneme)
  |     +- Extract frame counts at punctuation positions only
  |
  +- Pause time calculation
        frames x hop_length / sample_rate = seconds
```

During full synthesis, already-split individual sentences each pass through TextEncoder -> Duration Predictor -> Flow -> Decoder.

For pause prediction, the **unsplit original text** passes through only TextEncoder -> Duration Predictor.

The unsplit original is used because sentence boundary punctuation exists intact only in the original text.

After splitting into individual sentences, boundary punctuation (except the last sentence's) disappears or shifts position.

### Pause Time Calculation

Once frame counts at punctuation positions are obtained, convert to seconds.

```
pause (seconds) = frames x hop_length / sample_rate
```

With HayaKoe's default settings of `hop_length = 512` and `sample_rate = 44100`, 1 frame is approximately 11.6 ms.

For example, if the summed frame count for punctuation + adjacent blanks is 35:

```
35 x 512 / 44100 = approx. 0.41 seconds
```

The actual implementation (`durations_to_boundary_pauses()`) follows these steps.

1. Find **sentence boundary punctuation positions** in the full phoneme sequence (phoneme IDs for `.`, `!`, `?`).
2. Get the duration for each punctuation position's phoneme.
3. If the preceding adjacent token is blank (ID = 0), add its duration too.
4. If the following adjacent token is blank (ID = 0), add its duration too.
5. Convert the summed frame count using `frames x hop_length / sample_rate`.

With N sentences, there are N - 1 boundaries, so the result is a list of N - 1 pause times.

### Trailing Silence Compensation

One more thing to consider.

When the Synthesizer synthesizes each sentence, **short silence may already be included** at the end of the output.

Ignoring this trailing silence and inserting the predicted pause as-is makes the actual gap excessively long.

HayaKoe directly measures the silent region at the end of synthesized audio.

The measurement slides a 10ms window backward from the audio end, marking regions where **peak amplitude is 2% or below** as silence.

When inserting pauses, it then subtracts the trailing silence from the predicted target pause time, **adding only the shortfall as silence samples**.

```
additional silence = max(0, predicted pause - trailing silence)
```

If the model already generated sufficient silence, no additional insertion occurs.

A minimum floor of 80ms is applied to the target pause time itself, so total silence between sentences is always at least 80ms regardless of how short the prediction is.

### ONNX Support

On the PyTorch path, model internal modules can be called individually, so running just the Duration Predictor is straightforward.

However, `synthesizer.onnx` exports the entire Synthesizer as a single end-to-end graph, making it impossible to extract intermediate outputs.

To solve this, a **separate ONNX model containing only TextEncoder + Duration Predictor** (`duration_predictor.onnx`, ~30 MB, FP32) was additionally exported.

## Improvement Results

### Pause Time Distribution

Auto-predicted sentence boundary pauses for the same text.

| Backend | Pause Range |
|---|---|
| GPU (PyTorch) | 0.41 s ~ 0.55 s |
| CPU (ONNX) | 0.38 s ~ 0.57 s |

The difference between backends is within the variance expected from SDP's stochastic sampling characteristics.

SDP is probability-sampling-based, so results vary slightly between calls even for the same input.

Since the GPU-CPU difference falls within this natural variation, quality loss from ONNX conversion is negligible.

### Before / After

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。きっと楽しい発見がありますよ。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="Pause method"
  :defaultIndex="1"
  :samples='[
    { "value": "Before (fixed 80 ms)", "caption": "All sentence boundaries have identical short pauses", "src": "/hayakoe/samples/duration-predictor/before.wav" },
    { "value": "After (DP predicted)", "caption": "Duration Predictor auto-predicts sentence boundary pauses", "src": "/hayakoe/samples/duration-predictor/after.wav" }
  ]'
/>

### Cost

The additional cost is one run of TextEncoder + Duration Predictor.

As confirmed in [ONNX Optimization — Synthesizer share](./onnx-optimization#synthesizer-optimization), the Synthesizer accounts for 64-91% of total CPU inference time, with most of that in Flow + Decoder.

Running only up to the Duration Predictor is cheap by comparison, so perceived latency from pause prediction is virtually zero.

## Related Commits

- `c57e0ad` — Improved multi-sentence synthesis naturalness with Duration Predictor-based pause prediction
- `5522db1` — Added ONNX `duration_predictor` for natural sentence boundary pauses on CPU backend

## Future Work

- **Emotion-specific pause differentiation** — Shorter pauses for happy, longer for sad, etc., varying pause distribution by emotional style
- **Comma and colon granularity** — Currently only sentence-ending punctuation (`. ! ?`) is targeted, but further granularity for commas (`,`, `、`) and colons at positions requiring long breaths
- **Direct pause control API** — An interface allowing users to explicitly specify pause length at specific sentence boundaries
