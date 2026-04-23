# 速度·韵律调节

::: tip 快速开始建议:先只调 `speed`
大多数情况下,**除 `speed` 之外的参数保持默认听起来最自然**。

`pitch_scale` 或 `intonation_scale` 偏离 1.0 时会伴随轻微的音质下降。

"只是想稍微快一点/慢一点"的话,先只调 `speed`,其余的等到需要时再前往 [高级参数](#高级参数) 部分。
:::

`generate()` 接受六个可调节语速·音高·韵律·变化度的关键字参数。

全部可省略,不传任何参数时使用训练时的默认值进行合成。

以下示例均使用 `tsukuyomi_chan` 说话人,朗读同一句话("今日はどんな国に辿り着くのでしょうか。新しい出会いが楽しみです。"),仅改变该参数。

参数行为与 HayaKoe 所 fork 的 [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 的 `infer()` 相同。

## `speed` — 语速

以默认值 `1.0` 为基准,**值越小越快,值越大越慢**。

内部机制是将 Duration Predictor 预测的音素时长直接乘以 `speed`,因此发音本身能保持良好。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="speed"
  :defaultIndex="2"
  :samples='[
    { value: "0.5", caption: "2 倍速 — 极端", src: "/hayakoe/samples/params/speed_0.5.wav" },
    { value: "0.75", caption: "快速", src: "/hayakoe/samples/params/speed_0.75.wav" },
    { value: "1.0", caption: "默认", src: "/hayakoe/samples/params/speed_1.0.wav" },
    { value: "1.25", caption: "慢速", src: "/hayakoe/samples/params/speed_1.25.wav" },
    { value: "1.5", caption: "非常慢 — 极端", src: "/hayakoe/samples/params/speed_1.5.wav" }
  ]'
/>

低于 0.8 时发音会模糊,超过 1.3 时更接近"拖沓"而非"慢"。

实际使用中 0.9 ~ 1.1 左右最自然。

```python
speaker.generate(text, speed=0.9)   # 稍快
speaker.generate(text, speed=1.1)   # 稍慢
```

## 高级参数

从这里开始的设置在默认值下已经足够自然,但当需要精细调节时可以使用。

### 总览

| 参数               | 默认值 | 建议范围 | 效果                                                |
| ------------------ | -----: | --------- | --------------------------------------------------- |
| `pitch_scale`      |  `1.0` | 0.95 ~ 1.05 | 音高倍率。偏离 1.0 时会有轻微音质下降              |
| `intonation_scale` |  `1.0` | 0.8 ~ 1.3   | 韵律起伏。偏离 1.0 时会有轻微音质下降              |
| `sdp_ratio`        |  `0.2` | 0.0 ~ 0.5   | 确定性 DP 与随机性 SDP 的混合比例                   |
| `noise`            |  `0.6` | 0.3 ~ 0.9   | DP 侧噪声                                          |
| `noise_w`          |  `0.8` | 0.5 ~ 1.2   | SDP 侧噪声                                         |

建议每次只调整一个参数。

下面的示例为了让差异可以被听出,刻意将值推到了建议范围之外。

### `pitch_scale` — 音高

整体提高或降低音高的简单倍率。

偏离 `1.0` 时会伴随轻微的音质下降,因此 **建议调整幅度比其他参数更小**。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pitch_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.8", caption: "大幅降低 — 极端", src: "/hayakoe/samples/params/pitch_0.8.wav" },
    { value: "0.9", caption: "降低", src: "/hayakoe/samples/params/pitch_0.9.wav" },
    { value: "1.0", caption: "默认", src: "/hayakoe/samples/params/pitch_1.0.wav" },
    { value: "1.1", caption: "升高", src: "/hayakoe/samples/params/pitch_1.1.wav" },
    { value: "1.2", caption: "大幅升高 — 极端", src: "/hayakoe/samples/params/pitch_1.2.wav" }
  ]'
/>

在 0.95 ~ 1.05 范围内说话人特征基本保持不变,但在极端值下会变成"另一个人"或音质明显下降。

```python
speaker.generate(text, pitch_scale=1.05)
```

### `intonation_scale` — 韵律起伏

调节韵律变化的"幅度"。

`0.0` 几乎是完全单调的机器人音,`2.0` 是夸张的朗读音。

与 `pitch_scale` 一样,偏离 1.0 时会有轻微的音质下降。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="intonation_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.0", caption: "完全单调 — 极端", src: "/hayakoe/samples/params/intonation_0.0.wav" },
    { value: "0.5", caption: "平坦", src: "/hayakoe/samples/params/intonation_0.5.wav" },
    { value: "1.0", caption: "默认", src: "/hayakoe/samples/params/intonation_1.0.wav" },
    { value: "1.5", caption: "起伏大", src: "/hayakoe/samples/params/intonation_1.5.wav" },
    { value: "2.0", caption: "夸张 — 极端", src: "/hayakoe/samples/params/intonation_2.0.wav" }
  ]'
/>

实际使用中 0.85 ~ 1.3 左右自然。

```python
speaker.generate(text, intonation_scale=1.2)
```

### `sdp_ratio` — 确定性/随机性时长预测混合

HayaKoe(和 Style-Bert-VITS2)同时使用两种时长预测器。

- **DP (Deterministic Duration Predictor)** — 对相同文本始终给出相同时长
- **SDP (Stochastic Duration Predictor)** — 每次调用给出略有不同的时长

`sdp_ratio` 是两个预测器的混合比例,**`0.0` 表示仅用 DP,`1.0` 表示仅用 SDP**。

值越高,句子内的节奏起伏越大,同一文本多次运行时结果每次都不同。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="sdp_ratio"
  :defaultIndex="1"
  :samples='[
    { value: "0.0", caption: "仅 DP — 时长恒定", src: "/hayakoe/samples/params/sdp_0.0.wav" },
    { value: "0.25", caption: "DP 主导", src: "/hayakoe/samples/params/sdp_0.25.wav" },
    { value: "0.5", caption: "各半", src: "/hayakoe/samples/params/sdp_0.5.wav" },
    { value: "0.75", caption: "SDP 主导", src: "/hayakoe/samples/params/sdp_0.75.wav" },
    { value: "1.0", caption: "仅 SDP — 每次不同", src: "/hayakoe/samples/params/sdp_1.0.wav" }
  ]'
/>

对可复现性要求较高的服务(例如:字幕时间轴固定)中设为 `0.0`,一次性生成则 `0.2 ~ 0.4` 比较自然。

```python
speaker.generate(text, sdp_ratio=0.0)   # 总是相同
```

### `noise` / `noise_w` — 两个预测器的噪声

两者都是 **时长预测器** 侧的噪声大小(不是音素本身的噪声)。

- `noise` — DP(确定性预测器)的噪声
- `noise_w` — SDP(随机性预测器)的噪声

也就是说,当 `sdp_ratio` 设为 0 时 `noise_w` 几乎没有影响,反之设为 1 时 `noise` 几乎没有影响。

以下示例保持其余参数为默认值,仅改变该噪声。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise"
  :defaultIndex="1"
  :samples='[
    { value: "0.3", caption: "小", src: "/hayakoe/samples/params/noise_0.3.wav" },
    { value: "0.6", caption: "默认", src: "/hayakoe/samples/params/noise_0.6.wav" },
    { value: "0.9", caption: "大", src: "/hayakoe/samples/params/noise_0.9.wav" }
  ]'
/>

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise_w"
  :defaultIndex="1"
  :samples='[
    { value: "0.5", caption: "小", src: "/hayakoe/samples/params/noise_w_0.5.wav" },
    { value: "0.8", caption: "默认", src: "/hayakoe/samples/params/noise_w_0.8.wav" },
    { value: "1.2", caption: "大", src: "/hayakoe/samples/params/noise_w_1.2.wav" }
  ]'
/>

大多数情况下保持默认值(`0.6`、`0.8`)最自然。

如果觉得"总是在抖动",试着稍微降低该侧噪声;如果觉得"太机械化",试着稍微提高。
