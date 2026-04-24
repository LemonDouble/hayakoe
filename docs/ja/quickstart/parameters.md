# 速度・韻律の調整

::: tip クイックスタート推奨：まずは `speed` だけ
ほとんどの場合 **`speed` 以外のパラメータはデフォルトのままが最も自然に聞こえます**。

`pitch_scale` や `intonation_scale` は 1.0 から外れると若干の音質劣化を伴います。

「少し速く/遅くしたいだけ」なら `speed` ひとつだけ触ってみて、残りは必要になったとき [詳細パラメータ](#詳細パラメータ) セクションへ進んでください。
:::

`generate()` は話速・ピッチ・抑揚・変動性を調整できるキーワード引数を6つ受け付けます。

すべて省略可能で、何も渡さなければ学習済みのデフォルト値で合成されます。

以下のサンプルはすべて同じ文章（「今日はどんな国に辿り着くのでしょうか。新しい出会いが楽しみです。」）を `tsukuyomi_chan` 話者で、該当パラメータのみを変えて生成した結果です。

パラメータの動作は HayaKoe がフォークした [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) の `infer()` と同一です。

## `speed` — 話速

デフォルト値 `1.0` を基準に、**小さいほど速く、大きいほど遅くなります**。

内部的に Duration Predictor が予測した音素長に `speed` をそのまま掛ける方式なので、発音自体はよく維持されます。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="speed"
  :defaultIndex="2"
  :samples='[
    { value: "0.5", caption: "2倍速 — 極端", src: "/hayakoe/samples/params/speed_0.5.wav" },
    { value: "0.75", caption: "速め", src: "/hayakoe/samples/params/speed_0.75.wav" },
    { value: "1.0", caption: "デフォルト", src: "/hayakoe/samples/params/speed_1.0.wav" },
    { value: "1.25", caption: "ゆっくり", src: "/hayakoe/samples/params/speed_1.25.wav" },
    { value: "1.5", caption: "非常にゆっくり — 極端", src: "/hayakoe/samples/params/speed_1.5.wav" }
  ]'
/>

0.8 より下げると発音が潰れ、1.3 を超えると「遅い」より「間延びする」方に近づきます。

実用上は 0.9 ~ 1.1 程度が最も自然です。

```python
speaker.generate(text, speed=0.9)   # やや速め
speaker.generate(text, speed=1.1)   # ややゆっくり
```

## 詳細パラメータ

ここからはデフォルトでも十分自然ですが、細かいチューニングが必要なときに触る設定です。

### 全体まとめ

| パラメータ         | デフォルト | 推奨範囲 | 効果                                                |
| ------------------ | ---------: | --------- | --------------------------------------------------- |
| `pitch_scale`      |      `1.0` | 0.95 ~ 1.05 | 音の高さの倍率。1.0 から外れると若干音質劣化       |
| `intonation_scale` |      `1.0` | 0.8 ~ 1.3   | 抑揚の起伏。1.0 から外れると若干音質劣化       |
| `sdp_ratio`        |      `0.2` | 0.0 ~ 0.5   | 決定的 DP と確率的 SDP の混合比率                |
| `noise`            |      `0.6` | 0.3 ~ 0.9   | 音声の変動性（音色のランダム性）              |
| `noise_w`          |      `0.8` | 0.5 ~ 1.2   | 発話リズムの変動性（SDP ノイズ）              |

一度に1つずつ動かしてみることをお勧めします。

以下のサンプルでは違いを耳で確認できるよう、意図的に推奨範囲の外まで振っています。

### `pitch_scale` — 音の高さ

音の高さを全体的に上げたり下げたりする単純な倍率です。

`1.0` から外れると若干の音質劣化を伴うため、**他のパラメータより狭い範囲で動かすことをお勧めします**。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pitch_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.8", caption: "かなり低く — 極端", src: "/hayakoe/samples/params/pitch_0.8.wav" },
    { value: "0.9", caption: "低く", src: "/hayakoe/samples/params/pitch_0.9.wav" },
    { value: "1.0", caption: "デフォルト", src: "/hayakoe/samples/params/pitch_1.0.wav" },
    { value: "1.1", caption: "高く", src: "/hayakoe/samples/params/pitch_1.1.wav" },
    { value: "1.2", caption: "かなり高く — 極端", src: "/hayakoe/samples/params/pitch_1.2.wav" }
  ]'
/>

0.95 ~ 1.05 の範囲では話者のアイデンティティがほぼ維持されますが、極端な値では「別人」になったり音質が目に見えて落ちたりします。

```python
speaker.generate(text, pitch_scale=1.05)
```

### `intonation_scale` — 抑揚の起伏

抑揚変化の「幅」を調整します。

`0.0` はほぼ完全に単調なロボットトーン、`2.0` は大げさな朗読トーンです。

`pitch_scale` と同様に 1.0 から外れると若干の音質劣化があります。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="intonation_scale"
  :defaultIndex="2"
  :samples='[
    { value: "0.0", caption: "完全に単調 — 極端", src: "/hayakoe/samples/params/intonation_0.0.wav" },
    { value: "0.5", caption: "平坦", src: "/hayakoe/samples/params/intonation_0.5.wav" },
    { value: "1.0", caption: "デフォルト", src: "/hayakoe/samples/params/intonation_1.0.wav" },
    { value: "1.5", caption: "起伏大きめ", src: "/hayakoe/samples/params/intonation_1.5.wav" },
    { value: "2.0", caption: "大げさ — 極端", src: "/hayakoe/samples/params/intonation_2.0.wav" }
  ]'
/>

実用上は 0.85 ~ 1.3 程度が自然です。

```python
speaker.generate(text, intonation_scale=1.2)
```

### `sdp_ratio` — 決定的/確率的な長さ予測の混合

HayaKoe（および Style-Bert-VITS2）は2種類の長さ予測器を併用します。

- **DP (Deterministic Duration Predictor)** — 同じテキストに対して常に同じ長さを出力します
- **SDP (Stochastic Duration Predictor)** — 呼び出すたびに少しずつ異なる長さを出力します

`sdp_ratio` は2つの予測器の混合比率で、**`0.0` は DP のみ、`1.0` は SDP のみ** を意味します。

高いほど文内のリズム起伏が大きくなり、同じテキストを複数回実行したときに毎回結果が変わります。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="sdp_ratio"
  :defaultIndex="1"
  :samples='[
    { value: "0.0", caption: "DP のみ — 常に同じ長さ", src: "/hayakoe/samples/params/sdp_0.0.wav" },
    { value: "0.25", caption: "DP 主導", src: "/hayakoe/samples/params/sdp_0.25.wav" },
    { value: "0.5", caption: "半々", src: "/hayakoe/samples/params/sdp_0.5.wav" },
    { value: "0.75", caption: "SDP 主導", src: "/hayakoe/samples/params/sdp_0.75.wav" },
    { value: "1.0", caption: "SDP のみ — 毎回変わる", src: "/hayakoe/samples/params/sdp_1.0.wav" }
  ]'
/>

再現性が重要なサービス（例：字幕タイミング固定）では `0.0` にし、一回限りの生成なら `0.2 ~ 0.4` が自然です。

```python
speaker.generate(text, sdp_ratio=0.0)   # 常に同じ
```

### `noise` / `noise_w` — 音声・リズムの変動性

それぞれ異なるステージに作用するノイズです（音素自体のノイズではありません）。

- `noise` — 音声の変動性。Flow ステージで音色の全体的なランダム性を調節します。`sdp_ratio` に関係なく常に影響します。
- `noise_w` — 発話リズムの変動性。SDP（確率的予測器）に入るノイズです。`sdp_ratio` が 0 の場合は影響がありません。

以下のサンプルは残りのパラメータをすべてデフォルトのまま、該当ノイズのみを変えて生成した結果です。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise"
  :defaultIndex="1"
  :samples='[
    { value: "0.3", caption: "小さめ", src: "/hayakoe/samples/params/noise_0.3.wav" },
    { value: "0.6", caption: "デフォルト", src: "/hayakoe/samples/params/noise_0.6.wav" },
    { value: "0.9", caption: "大きめ", src: "/hayakoe/samples/params/noise_0.9.wav" }
  ]'
/>

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="noise_w"
  :defaultIndex="1"
  :samples='[
    { value: "0.5", caption: "小さめ", src: "/hayakoe/samples/params/noise_w_0.5.wav" },
    { value: "0.8", caption: "デフォルト", src: "/hayakoe/samples/params/noise_w_0.8.wav" },
    { value: "1.2", caption: "大きめ", src: "/hayakoe/samples/params/noise_w_1.2.wav" }
  ]'
/>

ほとんどの場合はデフォルト値（`0.6`、`0.8`）のままが最も自然です。

「揺れが気になる」と思ったら該当側のノイズを少し下げてみて、「機械的すぎる」と思ったら少し上げてみる、という使い方をすればよいでしょう。
