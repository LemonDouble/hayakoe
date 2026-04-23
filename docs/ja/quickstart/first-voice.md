# 初めての音声を作る

インストールが完了したら、実際に wav ファイルを1つ作ってみましょう。

このページでは最も基本的な「テキスト1行 → wav 1個」の流れと、そこから自然につながるいくつかのバリエーションを扱います。

## 基本 — 1文を wav に保存する

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()

speaker = tts.speakers["jvnv-F1-jp"]
speaker.generate("こんにちは、はじめまして。").save("hello.wav")
```

`TTS()` はエンジンスペックの登録のみを行います。

実際にモデルがディスクにダウンロードされメモリにロードされるのは `prepare()` が呼ばれた時点です。

`speakers` は準備が完了した話者を名前で取得できる dict です。

::: tip GPU で動かしたい場合
まず GPU extras がインストールされている必要があります（[インストール — GPU インストール（CUDA）](./install#gpu-インストール-cuda)）。

その後は `TTS(device="cuda")` と1つ変えるだけです。残りのコードは同一です。

初回呼び出しが少し遅い場合があることだけ覚えておいてください。
:::

## 複数の文はそのまま入れれば大丈夫です

文の境界（`。`、`！`、`？`、`!`、`?`、改行）で自動的に分割され、文と文の間には自然な長さの無音が挿入されます。

```python
text = """
こんにちは。今日はいい天気ですね。
散歩でもしましょうか？
"""

speaker.generate(text).save("long.wav")
```

文間の pause の長さは Duration Predictor が予測して付加しますが、この予測器は **話者ごとに別々に** 学習されています。

つまり、ある話者は文と文の間を長く休み、ある話者は短く休むという、その話者固有の呼吸がそのまま反映されます。

## wav ファイルではなく bytes で受け取る

FastAPI のようなWebサーバーでそのままレスポンスとして返すには `to_bytes()` を使います。

戻り値は WAV フォーマットのバイトストリームです。

```python
audio = speaker.generate("テストです。")
payload: bytes = audio.to_bytes()
```

## 複数話者を一度にロードする

`load()` はチェーンできるため、複数の話者を一度に登録できます。

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

BERT は全話者が共有するため、話者を1人追加してもメモリが2倍・3倍に跳ね上がることはありません。

話者あたり増えるのは BERT ではなく遥かに小さい synthesizer 分だけなので、4名ロードしても RAM が4倍にはなりません。

実際の測定値と再現方法は [FAQ — 話者を複数ロードするとメモリはどれくらい増えますか？](/ja/faq/#話者を複数ロードするとメモリはどれくらい増えますか) に別途まとめてあります。

## 速度やピッチだけ少し変えたいとき

`generate()` は話速・ピッチ・抑揚を直接調整できるパラメータを受け付けます。

```python
speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。",
    speed=0.95,        # 少しゆっくり
    pitch_scale=1.05,  # 少し高く
).save("tuned.wav")
```

利用可能なパラメータと推奨範囲は [速度・韻律の調整](./parameters) でまとめます。
