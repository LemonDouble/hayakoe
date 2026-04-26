# 速度・韻律調整

::: tip 快速入門建議:先只調 `speed`
大多數情況下,**除 `speed` 之外的參數保持預設聽起來最自然**。

`pitch_scale` 或 `intonation_scale` 偏離 1.0 時會伴隨輕微的音質下降。

「只是想稍微快一點/慢一點」的話,先只調 `speed`,其餘的等到需要時再前往 [進階參數](#進階參數) 部分。
:::

`generate()` 接受六個可調整語速・音高・韻律・變化度的關鍵字參數。

全部可省略,不傳任何參數時使用訓練時的預設值進行合成。

以下範例均使用 `tsukuyomi_chan` 說話人,朗讀同一句話(「今日はどんな国に辿り着くのでしょうか。新しい出会いが楽しみです。」),僅改變該參數。

參數行為與 HayaKoe 所 fork 的 [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 的 `infer()` 相同。

## `speed` — 語速

以預設值 `1.0` 為基準,**值越小越快,值越大越慢**。

內部機制是將 Duration Predictor 預測的音素時長直接乘以 `speed`,因此發音本身能保持良好。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="speed"
  :defaultIndex="2"
  :samples='[
    { value: "0.5", caption: "2 倍速 — 極端", src: "/hayakoe/samples/params/speed_0.5.wav" },
    { value: "0.75", caption: "快速", src: "/hayakoe/samples/params/speed_0.75.wav" },
    { value: "1.0", caption: "預設", src: "/hayakoe/samples/params/speed_1.0.wav" },
    { value: "1.25", caption: "慢速", src: "/hayakoe/samples/params/speed_1.25.wav" },
    { value: "1.5", caption: "非常慢 — 極端", src: "/hayakoe/samples/params/speed_1.5.wav" }
  ]'
/>

低於 0.8 時發音會模糊,超過 1.3 時更接近「拖沓」而非「慢」。

實際使用中 0.9 ~ 1.1 左右最自然。

```python
speaker.generate(text, speed=0.9)   # 稍快
speaker.generate(text, speed=1.1)   # 稍慢
```

## 進階參數

從這裡開始的設定在預設值下已經足夠自然,但當需要精細調整時可以使用。

### 總覽

| 參數               | 預設值 | 建議範圍 | 效果                                                |
| ------------------ | -----: | --------- | --------------------------------------------------- |
| `pitch_scale`      |  `1.0` | 0.95 ~ 1.05 | 音高倍率。偏離 1.0 時會有輕微音質下降              |
| `intonation_scale` |  `1.0` | 0.8 ~ 1.3   | 韻律起伏。偏離 1.0 時會有輕微音質下降              |
| `sdp_ratio`        |  `0.2` | 0.0 ~ 0.5   | 確定性 DP 與隨機性 SDP 的混合比例                   |
| `noise`            |  `0.6` | 0.3 ~ 0.9   | 語音變動性（音色隨機性）                             |
| `noise_w`          |  `0.8` | 0.5 ~ 1.2   | 發話節奏變動性（SDP 噪聲）                           |

建議每次只調整一個參數。

下面的範例為了讓差異可以被聽出,刻意將值推到了建議範圍之外。

### `pitch_scale` — 音高

整體提高或降低音高的簡單倍率。

偏離 `1.0` 時會伴隨輕微的音質下降,因此 **建議調整幅度比其他參數更小**。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pitch_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.8", caption: "大幅降低 — 極端", src: "/hayakoe/samples/params/pitch_0.8.wav" },
    { value: "0.9", caption: "降低", src: "/hayakoe/samples/params/pitch_0.9.wav" },
    { value: "1.0", caption: "預設", src: "/hayakoe/samples/params/pitch_1.0.wav" },
    { value: "1.1", caption: "升高", src: "/hayakoe/samples/params/pitch_1.1.wav" },
    { value: "1.2", caption: "大幅升高 — 極端", src: "/hayakoe/samples/params/pitch_1.2.wav" }
  ]'
/>

在 0.95 ~ 1.05 範圍內說話人特徵基本保持不變,但在極端值下會變成「另一個人」或音質明顯下降。

```python
speaker.generate(text, pitch_scale=1.05)
```

### `intonation_scale` — 韻律起伏

調整韻律變化的「幅度」。

`0.0` 幾乎是完全單調的機器人音,`2.0` 是誇張的朗讀音。

與 `pitch_scale` 一樣,偏離 1.0 時會有輕微的音質下降。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="intonation_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.0", caption: "完全單調 — 極端", src: "/hayakoe/samples/params/intonation_0.0.wav" },
    { value: "0.5", caption: "平坦", src: "/hayakoe/samples/params/intonation_0.5.wav" },
    { value: "1.0", caption: "預設", src: "/hayakoe/samples/params/intonation_1.0.wav" },
    { value: "1.5", caption: "起伏大", src: "/hayakoe/samples/params/intonation_1.5.wav" },
    { value: "2.0", caption: "誇張 — 極端", src: "/hayakoe/samples/params/intonation_2.0.wav" }
  ]'
/>

實際使用中 0.85 ~ 1.3 左右自然。

```python
speaker.generate(text, intonation_scale=1.2)
```

### `sdp_ratio` — 確定性/隨機性時長預測混合

HayaKoe(和 Style-Bert-VITS2)同時使用兩種時長預測器。

- **DP (Deterministic Duration Predictor)** — 對相同文本始終給出相同時長
- **SDP (Stochastic Duration Predictor)** — 每次呼叫給出略有不同的時長

`sdp_ratio` 是兩個預測器的混合比例,**`0.0` 表示僅用 DP,`1.0` 表示僅用 SDP**。

值越高,句子內的節奏起伏越大,同一文本多次執行時結果每次都不同。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="sdp_ratio"
  :defaultIndex="1"
  :samples='[
    { value: "0.0", caption: "僅 DP — 時長恆定", src: "/hayakoe/samples/params/sdp_0.0.wav" },
    { value: "0.25", caption: "DP 主導", src: "/hayakoe/samples/params/sdp_0.25.wav" },
    { value: "0.5", caption: "各半", src: "/hayakoe/samples/params/sdp_0.5.wav" },
    { value: "0.75", caption: "SDP 主導", src: "/hayakoe/samples/params/sdp_0.75.wav" },
    { value: "1.0", caption: "僅 SDP — 每次不同", src: "/hayakoe/samples/params/sdp_1.0.wav" }
  ]'
/>

對可重現性要求較高的服務(例如:字幕時間軸固定)中設為 `0.0`,一次性生成則 `0.2 ~ 0.4` 比較自然。

```python
speaker.generate(text, sdp_ratio=0.0)   # 總是相同
```

### `noise` / `noise_w` — 語音・節奏變動性

各自作用於不同階段的噪聲（不是音素本身的噪聲）。

- `noise` — 語音變動性。在 Flow 階段控制音色的整體隨機性。與 `sdp_ratio` 無關,始終有效。
- `noise_w` — 發話節奏變動性。輸入 SDP（隨機性預測器）的噪聲。`sdp_ratio` 為 0 時無效。

以下範例保持其餘參數為預設值,僅改變該噪聲。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise"
  :defaultIndex="1"
  :samples='[
    { value: "0.3", caption: "小", src: "/hayakoe/samples/params/noise_0.3.wav" },
    { value: "0.6", caption: "預設", src: "/hayakoe/samples/params/noise_0.6.wav" },
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
    { value: "0.8", caption: "預設", src: "/hayakoe/samples/params/noise_w_0.8.wav" },
    { value: "1.2", caption: "大", src: "/hayakoe/samples/params/noise_w_1.2.wav" }
  ]'
/>

大多數情況下保持預設值(`0.6`、`0.8`)最自然。

如果覺得「總是在抖動」,試著稍微降低該側噪聲;如果覺得「太機械化」,試著稍微提高。
