# ONNX 最適化 / 量子化

オリジナル Style-Bert-VITS2 は PyTorch ベースのため CPU 単独でのリアルタイム推論が困難でした。

HayaKoe は **BERT を Q8 量子化 ONNX に、Synthesizer を FP32 ONNX に** エクスポートし ONNX Runtime 上で実行することで、CPU 推論速度を **テキスト長に応じて 1.5x ~ 3.3x まで向上** させました。

同時に1話者ロード時の RAM 使用量を **5,122 MB → 2,346 MB (-54%)** に削減しました。

同一経路のおかげで **x86_64 だけでなく aarch64（Raspberry Pi 等）でも同じコードで動作** します。

## 課題

オリジナル SBV2（CPU, PyTorch FP32）には2つの課題がありました。

- **速度** — テキストが長くなるほど推論時間が急激に増加します。short（1.7 s 分量）では倍速 1.52x 水準ですが、xlong（38.5 s 分量）では推論 35.3 秒・倍速 1.09x とリアルタイムをかろうじて追いかける水準まで落ちます。
- **メモリ** — 1話者推論時の Peak メモリ約 5 GB 以上は負担の大きい規模です。

## 分析

モデルパラメータ分布をまず見ると、全体の約 **84%** が BERT（[DeBERTa-v2-Large-Japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm)、約 329 M）に集中しており、Synthesizer（VITS）は 63 M で約 16% の水準です。

BERT がモデルの大部分を占めるため、BERT を量子化すればメモリを大幅に削減できると予想しました。PyTorch で BERT のみ Q8 Dynamic Quantization（`torch.quantization.quantize_dynamic`）を適用して検証しました。

| 構成 | 平均推論時間 | RAM |
|---|---|---|
| PyTorch BERT FP32 | 4.796 s | +1,698 MB |
| PyTorch BERT Q8 | 4.536 s | **+368 MB**（-78%） |

BERT 量子化は速度を改善してはくれませんが、メモリ使用量を大幅に削減できることを確認しました。

ここに加えて **ONNX Runtime への移行** で速度改善まで確保する方向で進めました。

ONNX Runtime はモデルをロードする際に [グラフレベル最適化](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) を自動的に適用します。

- **Kernel fusion** — 連続する複数の演算をひとつの演算にまとめます。例えば Conv → BatchNorm → Activation の3段階をひとつにまとめると、中間結果をメモリに書いて再び読む過程がなくなりメモリアクセスが減って高速化されます。
- **Constant folding** — 入力に関係なく常に同じ値を出す演算をロード時に事前計算しておき、推論時にはその事前計算値を使って速度を上げます。
- **不要ノードの除去** — 使われていない、重複している、意味のない演算を行うノードを見つけて除去します。

結論として、推論に最適化された数学的に同一の演算を再構成してより高速に推論できるようにしてくれます。

Synthesizer はパラメータが 63 M と小さく量子化のメモリ利得が限定的で、Flow レイヤー（`rational_quadratic_spline`）が FP16 以下で数値的に不安定なため量子化対象から除外しました。代わりに ONNX にエクスポートしてグラフ最適化の利得を確保しました。

### BERT 最適化

量子化が音質に影響を与えるか確認するため、同一テキスト・同一話者で FP32・Q8・Q4 BERT の3構成の出力を比較しました（Synthesizer はすべて FP32 固定）。

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="BERT dtype"
  :defaultIndex="0"
  :samples='[
    { value: "FP32", caption: "オリジナル baseline", src: "/hayakoe/deep-dive/quantization/fp32_med_ja.wav" },
    { value: "Q8",   caption: "INT8 dynamic quantization", src: "/hayakoe/deep-dive/quantization/q8_med_ja.wav" },
    { value: "Q4",   caption: "INT4 weight-only (MatMulNBits)", src: "/hayakoe/deep-dive/quantization/q4_med_ja.wav" }
  ]'
/>

FP32 と Q8 は直接聴取時に一貫して区別することが困難なレベルでした。

Q4 はほとんどの区間で FP32・Q8 と類似していますが、末尾で微細な違いが聞き取れます。

| 構成 | BERT サイズ | RAM（1話者） |
|---|---|---|
| FP32 | 1,157 MB | 1,599 MB |
| Q8 | 497 MB | 1,079 MB（-33%） |
| Q4 | 394 MB | 958 MB（-40%） |

Q4 で追加量子化しても得られるメモリ利得が音質より大きくないと判断し、**デフォルトとして Q8 を使用** することに決定しました。

### Synthesizer 最適化

BERT がパラメータの 96% を占めるので、BERT を高速化すれば全体が速くなりそうです。

しかし実際に BERT と Synthesizer の推論時間を別々に測定してみると、**CPU 時間の大部分は Synthesizer 側で消費** されています。

PyTorch FP32 CPU での実測結果です（5回平均）。

| テキスト | BERT | Synthesizer | BERT 比率 | Synth 比率 |
|---|---|---|---|---|
| short (1.7 s) | 0.489 s | 0.885 s | 36% | **64%** |
| medium (5.3 s) | 0.602 s | 2.504 s | 19% | **81%** |
| long (7.8 s) | 0.690 s | 3.714 s | 16% | **84%** |
| xlong (30 s) | 1.074 s | 11.410 s | 9% | **91%** |

テキストが長くなるほど Synthesizer の比率が大きくなりますが、BERT はテキスト長に比較的鈍感な一方、Synthesizer は生成するオーディオ長に比例して時間が増えるためです。

実際に BERT のみ Q8 量子化したとき全体の推論時間は約 5% しか縮まりませんでした。

つまり、**速度を上げるには Synthesizer 区間を最適化する必要があります**。

Synthesizer は量子化の代わりに **ONNX 変換のみ** を適用しました。

- VITS の Flow レイヤー（`rational_quadratic_spline`）が FP16 以下で浮動小数点誤差による assertion error を引き起こし量子化が不可能です。
- パラメータ数が 63 M と小さく量子化のメモリ利得も限定的です。

代わりに ONNX Runtime に変換して先述のグラフレベル最適化（kernel fusion・constant folding・不要ノード除去）を Synthesizer にも同様に適用しました。

### ONNX Runtime + `CPUExecutionProvider`

BERT 量子化と Synthesizer グラフ最適化はすべて ONNX Runtime 上で動作します。

また [intra-op parallelism](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) で単一演算を複数 CPU コアに分散し、リクエストがひとつだけでも CPU 全体を活用できます。

## 改善効果

### CPU パフォーマンス比較（倍速、同一ハードウェア）

倍速 はオーディオ長 / 推論時間（値が大きいほど速い）。

| 構成 | short (1.7 s) | medium (7.6 s) | long (10.7 s) | xlong (38.5 s) |
|---|---|---|---|---|
| SBV2 PyTorch FP32 | 1.52x | 2.27x | 2.16x | 1.09x |
| SBV2 ONNX FP32 | 1.76x | 3.09x | 3.26x | 2.75x |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2.50x** | **3.35x** | **3.33x** | **3.60x** |

PyTorch FP32 比での速度向上は **テキスト長に応じて 1.5x ~ 3.3x** です。

### メモリ（1話者ロード基準）

| 構成 | RAM |
|---|---|
| SBV2 PyTorch FP32 | 5,122 MB |
| SBV2 ONNX FP32 | 2,967 MB |
| **HayaKoe (Q8 BERT + FP32 ONNX)** | **2,346 MB**（-54%） |
