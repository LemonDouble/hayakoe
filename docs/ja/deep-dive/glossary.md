# 用語集

このページは deep-dive 全般で頻出する TTS・推論用語を整理します。

TTS が初めての方が **architecture 以降のページを読む際につまずかないよう** 構成しました。

順番に目を通しても良いですし、必要な用語だけ検索して読んでも構いません。

## パイプラインの全体像

### TTS (Text-to-Speech)

テキストを人の声のオーディオに変換する技術の総称です。

入力は通常文章、出力は WAV・MP3 などの音声ファイルです。

TTS システムは内部的に「文字を発音として解釈」→「その発音を波形として合成」する複数の段階を経ます。

HayaKoe もこの範疇に属し、日本語入力を受けて WAV 波形を生成します。

### 音素 (Phoneme)

言葉の意味を区別する最小の音の単位です。

「ば」と「だ」は最初の音（b vs d）だけ違っても完全に異なる意味になります。このように **意味を変える音の単位** が音素です。

TTS モデルは文字ではなく音素を入力として受け取ります。文字をそのまま受け取ると「どの条件でどう読まれるか」という発音規則までモデルがすべて学習しなければならないためです。音素に事前変換して入力すれば、モデルは「この音をどう聞こえさせるか」にだけ集中できます。

この変換を担当するモジュールが **G2P** です。

### G2P (Grapheme-to-Phoneme)

文字 (Grapheme) を音素 (Phoneme) に変換するプロセス、またはそのモジュールです。

日本語の漢字の読み方・連音規則など言語ごとの発音規則をすべてここで処理します。

TTS パイプラインでモデルに入力を渡す直前の段階に該当します。

HayaKoe は日本語専用なので日本語 G2P を [pyopenjtalk](./openjtalk-dict) に委譲します。

### 波形 (Waveform)

空気の圧力変化を時間軸に沿って記録した数列です。スピーカーが再生できる「実際の音」そのものを意味します。

各数値は **特定の瞬間の空気圧（振幅、amplitude）** を表します。値が正なら基準より空気が圧縮された状態、負なら膨張した状態を意味し、絶対値が大きいほど音が大きく聞こえます。0 は無音（基準圧力）に該当します。

サンプリングレート (sample rate) が 22,050 Hz なら1秒 = 22,050 個のこのような数値で表現されます。HayaKoe の出力は 44,100 Hz なので1秒あたり 44,100 個です。

TTS の最終出力物がまさにこの波形です。

## モデル構成要素

### VITS

2021年に発表された音声合成モデル構造です。

それまで2段階（Acoustic Model + Vocoder）に分かれていた TTS パイプラインを **ひとつの End-to-End モデル** に統合したことが核心的な貢献です。

テキスト → 波形変換を単一モデルが直接行い、内部的に Text Encoder・Duration Predictor・Flow・Decoder で構成されます。

HayaKoe は VITS 系譜の延長線上にあるモデルです。

- **VITS (2021)** — End-to-End TTS の出発点。
- **[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)** — Fish Audio チームが VITS に BERT を付けて **文脈ベース prosody** を補強したオープンソースプロジェクト。
- **[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)** — litagin02 が Bert-VITS2 をフォークして **Style Vector** を追加、同じ話者の多様なトーン・感情を表現可能に拡張。日本語特化バリアントの **JP-Extra** が品質で優位。
- **HayaKoe** — Style-Bert-VITS2 JP-Extra を **日本語専用に縮小** し、CPU・サーバー運用に実用的な形に再構成。

Synthesizer 自体のモデル構造は Style-Bert-VITS2 をそのまま使用しており、HayaKoe が追加した変更は主にその外側（ONNX 経路、推論パイプライン、デプロイ・ソースの簡素化）に集中しています。

### Synthesizer

HayaKoe で **VITS 本体（Text Encoder + Duration Predictor + Flow + Decoder）** を総称する名前です。

音素シーケンスと BERT 特徴を入力として受け取り、最終的な波形を生成する部分です。

BERT は Synthesizer の **外** に別途存在し全話者が共有します。話者ごとに変わるのは Synthesizer の重み（約 300 MB）です。

### BERT

Google が 2018 年に発表した Transformer ベースの事前学習言語モデルです。文章を読んで各トークンの文脈埋め込みを生成します。

TTS では **文章の意味・文脈情報を合成に反映** するために使用されます。同じ音素列でも BERT のおかげでより自然な抑揚・強勢が生成されます。

HayaKoe は日本語専用 DeBERTa v2 モデル（`ku-nlp/deberta-v2-large-japanese-char-wwm`）を使用します。

CPU 経路ではこの BERT を INT8 で量子化して ONNX で実行します。

### Text Encoder

Synthesizer 内部のモジュールです。音素シーケンスを入力として各音素に対応する **192次元隠れベクトル** を出力します。

Transformer encoder 構造で、self-attention によって音素が前後の文脈を参照しながら合成に必要な埋め込みを生成します。

概念的に BERT のミニチュアと見ることができます。BERT は単語・文レベル、Text Encoder は音素レベルという違いがあります。

### Duration Predictor (SDP)

各音素を **何フレーム分** 発音するかを予測するモジュールです。「あ」は5フレーム、「い」は4フレームという具合です。

「SDP」は **Stochastic Duration Predictor** の略です。決定的 (deterministic) ではなく確率分布からサンプリングするため、同じ文章でも呼び出すたびに抑揚・速度が少しずつ異なります。

HayaKoe はこのモジュールを本来の用途以外に **文境界 pause 予測** にも再利用しています。詳細は [文境界 pause — Duration Predictor](./duration-predictor) で扱います。

### Flow

Synthesizer 内部のモジュールです。**可逆 (invertible) 変換** のため順方向・逆方向の両方を計算できる神経回路網です。

学習時には「正解音声の latent → テキスト埋め込み空間」に合わせられ、推論時にはその逆方向を経てテキスト埋め込みから音声 latent を生成します。

正式名称は **Normalizing Flow** です。

::: warning Flow と量子化
HayaKoe が Synthesizer を FP16 に落とさない主な理由が Flow にあります。Flow の `rational_quadratic_spline` 演算が FP16 で浮動小数点誤差による assertion error を引き起こします。

Synthesizer INT8 は別の理由で除外されました — Conv1d 中心の構造のため PyTorch dynamic quantization が自動適用されず、static quantization は実装の複雑度が高いためです。
:::

### Decoder (HiFi-GAN)

Synthesizer の最後のモジュールです。Flow が生成した latent ベクトルを入力として **実際の波形 (waveform)** を生成します。

かつて独立した Vocoder として使われていた HiFi-GAN 構造を VITS がモデル内に統合したものです。

**VITS が End-to-End で動作できる核心モジュール** であり、同時に TTS 推論時間の相当部分を占める部分でもあります。

### Style Vector

話者の「トーン・話し方」などのスタイル情報をひとつのベクトルに圧縮したものです。

同じ話者でも「穏やか」「喜び」「怒り」などスタイルを切り替えて合成できます。

Style-Bert-VITS2 系列特有の構成要素で、話者別 safetensors とともに `style_vectors.npy` で提供されます。

HayaKoe は現在簡素化のため **Neutral スタイルのみ** を使用します。多様なスタイル選択のサポートは今後改善予定です。

### Prosody（韻律）

発話の **抑揚・リズム・強勢・休止** を総称する言葉です。

音素が「何を発音するか」なら、prosody は「どのように発音するか」に該当します。

TTS が「ロボットのよう」に聞こえる最も一般的な原因が、まさに prosody が自然でないときです。

Bert-VITS2 系列が BERT を使用する主な理由のひとつが、文脈から prosody のヒントを得るためです。

## パフォーマンス・実行用語

### ONNX・ONNX Runtime

**ONNX (Open Neural Network Exchange)** はニューラルネットワークモデルを **フレームワーク非依存で保存** できる標準フォーマットです。

PyTorch・TensorFlow などどこで学習しても ONNX で export すれば同一のグラフとして扱えます。

**ONNX Runtime** は ONNX モデルを実際に実行する推論エンジンです。C++ で書かれているため Python オーバーヘッドが少なく、モデルグラフを分析してさまざまな最適化を事前に実行します。

CPU・CUDA・ARM（aarch64）など多様な実行デバイスをサポートします。

HayaKoe の CPU 経路は全体が ONNX Runtime 上で動作します。同一コードが x86_64 と aarch64 で共通に動作するのもこのおかげです。

### 量子化 (Quantization)

モデルの重みの数値表現精度を下げて、メモリと演算を節約する手法です。

ディープラーニングモデルの重みは通常以下の精度のいずれかで保存されます。

- **FP32** — 32ビット浮動小数。デフォルト。最も正確だがサイズが大きい。
- **FP16** — 16ビット浮動小数。FP32 比で半分のサイズ。
- **INT8** — 8ビット整数。FP32 の約 1/4 サイズ。よく「Q8」とも呼ばれる。
- **INT4** — 4ビット整数。FP32 の約 1/8 サイズ。LLM 分野で最近活発に使用。

ビット数が減るとモデルファイルサイズと RAM 使用量もほぼ比例して減り、特定のハードウェアでは演算も速くなります。

代わりに **精度が落ちるため出力品質が悪化する可能性があります。** どこまで量子化しても品質が許容範囲かはモデルごとに、また演算の種類ごとに異なります。

HayaKoe は **BERT の MatMul のみ INT8 で動的量子化（Q8 Dynamic Quantization）**、Synthesizer は FP32 を維持する選択をしました。詳しい理由と実測効果は [ONNX 最適化](./onnx-optimization) で扱います。

### Kernel Launch Overhead

CPU から GPU に「このカーネルを実行せよ」と要求する際にかかる固定コストです。実際の計算時間とは別に、カーネル呼び出し1件あたり数 us ~ 数十 us 程度発生します。

カーネルひとつが重い計算を行うワークロードではこのコストは埋もれます。しかし **TTS のように小さな Conv1d 演算が数百回繰り返される場合**、kernel launch overhead が全体時間の相当部分を占めることがあります。

CUDA Graph・kernel fusion・torch.compile などがこのコストを削減するための手法です。

### Eager Mode

PyTorch のデフォルト実行方式です。Python コードが1行ずつ実行されながらその都度 GPU カーネルを個別呼び出しします。

デバッグが容易という利点がありますが、カーネルごとに Python ディスパッチオーバーヘッドと kernel launch overhead が累積します。

`torch.compile` はこのオーバーヘッドをグラフレベル最適化で除去するための代替手段です。

### torch.compile

PyTorch 2.0 から提供される **JIT コンパイラ** です。

モデルを初回呼び出し時にグラフとしてトレースし、カーネルを融合・再コンパイルして以降の呼び出しからより高速に実行します。

HayaKoe は GPU 経路で `torch.compile` を使用します。

初回呼び出しにはコンパイル時間がかかるため、`prepare(warmup=True)` でこのコストをサービング開始段階に移すことができます。

## その他

### OpenJTalk

名古屋工業大学が開発したオープンソース日本語 TTS フロントエンドです。

日本語テキストを受け取り **音素列・アクセント情報** を生成します。漢字の読み方・連音など日本語特有の規則がここに含まれます。

HayaKoe は Python バインディングの [pyopenjtalk](./openjtalk-dict) を通じてこの機能を使用します。
