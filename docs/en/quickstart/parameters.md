# Speed & Prosody Controls

::: tip Quickstart recommendation: just use `speed`
In most cases, **leaving parameters other than `speed` at their defaults sounds most natural**.

`pitch_scale` and `intonation_scale` can introduce slight quality degradation when moved away from 1.0.

If you just want to make things a bit faster or slower, try `speed` alone, and come back to the [Advanced Parameters](#advanced-parameters) section when you need more.
:::

`generate()` accepts six keyword arguments for adjusting speech speed, pitch, intonation, and variability.

All are optional, and if none are passed, the trained defaults are used for synthesis.

The samples below all use the same sentence ("今日はどんな国に辿り着くのでしょうか。新しい出会いが楽しみです。") with the `tsukuyomi_chan` speaker, varying only the parameter in question.

Parameter behavior is identical to the `infer()` function in [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2), which HayaKoe forked from.

## `speed` — Speech Speed

Based on the default of `1.0`, **smaller values are faster and larger values are slower**.

Internally, this multiplies the phoneme durations predicted by the Duration Predictor directly by `speed`, so pronunciation itself is well preserved.

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="speed"
  :defaultIndex="2"
  :samples='[
    { value: "0.5", caption: "2x faster — extreme", src: "/hayakoe/samples/params/speed_0.5.wav" },
    { value: "0.75", caption: "Faster", src: "/hayakoe/samples/params/speed_0.75.wav" },
    { value: "1.0", caption: "Default", src: "/hayakoe/samples/params/speed_1.0.wav" },
    { value: "1.25", caption: "Slower", src: "/hayakoe/samples/params/speed_1.25.wav" },
    { value: "1.5", caption: "Very slow — extreme", src: "/hayakoe/samples/params/speed_1.5.wav" }
  ]'
/>

Below 0.8, pronunciation starts to blur, and above 1.3, it sounds more "dragged out" than simply "slow".

In practice, 0.9 to 1.1 sounds most natural.

```python
speaker.generate(text, speed=0.9)   # slightly faster
speaker.generate(text, speed=1.1)   # slightly slower
```

## Advanced Parameters

The settings below are already natural at their defaults, but can be adjusted when fine-tuning is needed.

### Summary

| Parameter          | Default | Recommended Range | Effect                                                        |
| ------------------ | ------: | ----------------- | ------------------------------------------------------------- |
| `pitch_scale`      |   `1.0` | 0.95 ~ 1.05      | Pitch multiplier. Slight quality loss away from 1.0           |
| `intonation_scale` |   `1.0` | 0.8 ~ 1.3        | Intonation range. Slight quality loss away from 1.0           |
| `sdp_ratio`        |   `0.2` | 0.0 ~ 0.5        | Blend ratio of deterministic DP and stochastic SDP            |
| `noise`            |   `0.6` | 0.3 ~ 0.9        | Noise fed into the DP side                                    |
| `noise_w`          |   `0.8` | 0.5 ~ 1.2        | Noise fed into the SDP side                                   |

We recommend moving one parameter at a time.

In the samples below, we intentionally pushed values beyond the recommended range so you can hear the differences.

### `pitch_scale` — Pitch

A simple multiplier that raises or lowers the overall pitch.

Moving away from `1.0` introduces slight quality degradation, so **it is recommended to adjust this more narrowly than other parameters**.

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pitch_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.8", caption: "Much lower — extreme", src: "/hayakoe/samples/params/pitch_0.8.wav" },
    { value: "0.9", caption: "Lower", src: "/hayakoe/samples/params/pitch_0.9.wav" },
    { value: "1.0", caption: "Default", src: "/hayakoe/samples/params/pitch_1.0.wav" },
    { value: "1.1", caption: "Higher", src: "/hayakoe/samples/params/pitch_1.1.wav" },
    { value: "1.2", caption: "Much higher — extreme", src: "/hayakoe/samples/params/pitch_1.2.wav" }
  ]'
/>

In the 0.95 to 1.05 range, speaker identity is mostly preserved, but at extreme values the voice sounds like a different person or quality noticeably drops.

```python
speaker.generate(text, pitch_scale=1.05)
```

### `intonation_scale` — Intonation Range

Controls the "width" of intonation variation.

`0.0` is a near-completely monotone robotic tone, while `2.0` is an exaggerated reading tone.

Like `pitch_scale`, moving away from 1.0 introduces slight quality degradation.

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="intonation_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.0", caption: "Completely flat — extreme", src: "/hayakoe/samples/params/intonation_0.0.wav" },
    { value: "0.5", caption: "Flat", src: "/hayakoe/samples/params/intonation_0.5.wav" },
    { value: "1.0", caption: "Default", src: "/hayakoe/samples/params/intonation_1.0.wav" },
    { value: "1.5", caption: "More dynamic", src: "/hayakoe/samples/params/intonation_1.5.wav" },
    { value: "2.0", caption: "Exaggerated — extreme", src: "/hayakoe/samples/params/intonation_2.0.wav" }
  ]'
/>

In practice, 0.85 to 1.3 sounds natural.

```python
speaker.generate(text, intonation_scale=1.2)
```

### `sdp_ratio` — Deterministic/Stochastic Duration Blend

HayaKoe (and Style-Bert-VITS2) uses two types of duration predictors together.

- **DP (Deterministic Duration Predictor)** — Always produces the same duration for the same text
- **SDP (Stochastic Duration Predictor)** — Produces slightly different durations each time

`sdp_ratio` is the blend ratio between the two, where **`0.0` uses DP only and `1.0` uses SDP only**.

Higher values increase rhythm variation within sentences, and results differ with each run for the same text.

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="sdp_ratio"
  :defaultIndex="1"
  :samples='[
    { value: "0.0", caption: "DP only — always same duration", src: "/hayakoe/samples/params/sdp_0.0.wav" },
    { value: "0.25", caption: "DP-dominant", src: "/hayakoe/samples/params/sdp_0.25.wav" },
    { value: "0.5", caption: "Half and half", src: "/hayakoe/samples/params/sdp_0.5.wav" },
    { value: "0.75", caption: "SDP-dominant", src: "/hayakoe/samples/params/sdp_0.75.wav" },
    { value: "1.0", caption: "SDP only — different every time", src: "/hayakoe/samples/params/sdp_1.0.wav" }
  ]'
/>

For services where reproducibility matters (e.g., fixed subtitle timing), set it to `0.0`; for one-off generation, `0.2 ~ 0.4` sounds natural.

```python
speaker.generate(text, sdp_ratio=0.0)   # always identical
```

### `noise` / `noise_w` — Noise for the Two Predictors

Both control the **magnitude of noise fed into the duration predictors** (not the phoneme audio itself).

- `noise` — Noise fed into the DP (deterministic predictor)
- `noise_w` — Noise fed into the SDP (stochastic predictor)

In other words, if `sdp_ratio` is set to 0, `noise_w` has almost no effect, and conversely if set to 1, `noise` has almost no effect.

The samples below were generated with all other parameters at their defaults, changing only the respective noise value.

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise"
  :defaultIndex="1"
  :samples='[
    { value: "0.3", caption: "Low", src: "/hayakoe/samples/params/noise_0.3.wav" },
    { value: "0.6", caption: "Default", src: "/hayakoe/samples/params/noise_0.6.wav" },
    { value: "0.9", caption: "High", src: "/hayakoe/samples/params/noise_0.9.wav" }
  ]'
/>

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise_w"
  :defaultIndex="1"
  :samples='[
    { value: "0.5", caption: "Low", src: "/hayakoe/samples/params/noise_w_0.5.wav" },
    { value: "0.8", caption: "Default", src: "/hayakoe/samples/params/noise_w_0.8.wav" },
    { value: "1.2", caption: "High", src: "/hayakoe/samples/params/noise_w_1.2.wav" }
  ]'
/>

In most cases, leaving the defaults (`0.6`, `0.8`) sounds most natural.

If you feel the output is "wobbling too much", try lowering the corresponding noise slightly; if it sounds "too mechanical", try raising it a bit.
