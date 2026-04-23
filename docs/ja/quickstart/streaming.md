# 文単位ストリーミング

HayaKoe は長いテキストを **一度にすべて合成せず、文ができ次第ひとつずつ流し出す** ストリーミングモードをサポートしています。

対話型 UI やリアルタイムレスポンスが必要な場面で最初の音声を素早く送り出せます。

## いつ使うと良いですか？

複数の文からなるテキストを合成する場合を考えてみてください。

`generate()` はすべての文の合成が完了するまで待ってから wav を返します。

15秒分のオーディオなら、ユーザーはその合成が終わるまで何も聞こえません。

一方 `stream()` は最初の文が完成した瞬間にその chunk をすぐに渡します。

最初の文を再生している間に後続の文が続けて合成されるため、体感遅延が大きく縮まります。

## 基本 — 文ごとに chunk を受け取る

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

text = "こんにちは。私はイレイナ。旅の魔女です。"

for i, chunk in enumerate(speaker.stream(text)):
    chunk.save(f"chunk_{i:02d}.wav")
```

`speaker.stream()` は各文を `AudioResult` ひとつとして yield する Python ジェネレータです。

`generate()` が返すものと同じ型なので、`.save()` ・ `.to_bytes()` ・ `.data` すべて同様に使えます。

例のテキストは `。` で3文に分割され、3つの chunk が順番に出力されます。

## パラメータは `generate()` と同一

`speed`、`pitch_scale`、`style`、`intonation_scale` などの [速度・韻律の調整](./parameters) パラメータをそのまま使用できます。

```python
for chunk in speaker.stream(text, speed=0.95, style="Happy"):
    chunk.save(...)  # または直接オーディオデバイスに write
```

## 文間の無音は自動で挿入されます

2つ目の chunk からは先頭に **前の文との間を埋める無音** が自動的に含まれて出力されます。

そのため chunk を順番につなげるか再生するだけで、`generate()` で一度に受け取った結果と同じ呼吸が得られます。

CPU（ONNX）と GPU（PyTorch）どちらも Duration Predictor で文の境界の pause を事前に予測して、各 chunk 間のギャップに反映します。

ONNX バックエンドは学習時に一緒に export しておいた別途の `duration_predictor.onnx` モデルを CPU セッションで実行します。

両方の経路とも最低 80ms のフロアが適用されるため、予測値が短すぎても不自然にくっつくことはありません。

::: info Duration Predictor とは
話者ごとに文と文の間をどれくらい休むかは固有の癖があります。

HayaKoe はこの pause の長さを話者ごとに学習された小さなモデルで予測します。
:::

## 非同期版 — Webサーバー用

FastAPI のように async ランタイムで使うなら `astream()` を使ってください。

```python
async for chunk in speaker.astream(text):
    await send(chunk.to_bytes())
```

内部的に別スレッドで `stream()` の各 chunk を取り出して yield するため、イベントループをブロッキングしません。

::: warning ジェネレータは最後まで消費してください
`stream()` / `astream()` は内部的に per-speaker lock を取得します。

同じ話者に他のリクエストが集中する可能性がある環境では、ジェネレータを途中で捨てずに `for` 文で最後まで回すか `try/finally` で close を保証してください。

消費されるか close された時点で lock が解放されます。
:::
