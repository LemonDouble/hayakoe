# 文境界 pause — Duration Predictor

多文を分割合成すると **文間の休止（pause）が消失する** 副作用が発生します。

HayaKoe は Duration Predictor を再利用して各文境界の自然な pause 時間を直接予測します。

Flow・Decoder をスキップし **TextEncoder + Duration Predictor のみを実行** するため、追加コストが低いです。

## なぜ問題か

### 文分割の利点

[アーキテクチャ一覧](./architecture#_1-%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC-%E1%84%87%E1%85%AE%E1%86%AB%E1%84%92%E1%85%A1%E1%86%AF) で説明したように、HayaKoe は多文入力を文単位で分割して個別合成します。

長いテキストを丸ごと入れると抑揚が崩れたり不安定になる傾向があります。

文単位で区切ると各文ごとに安定した prosody（韻律）が保証されます。

### 分割の副作用 — pause の消失

しかし分割には副作用があります。

オリジナル SBV2 は通文合成で `.`、`!`、`?` のような句読点の後に自然な休止を入れてくれます。

文単位の分割をすると各文が句読点で終わり次の文が最初から始まるため、**句読点後の休止が一緒に消えます。**

初期実装では文間に固定 80 ms の無音を挿入していました。

実際に Duration Predictor が予測する文境界 pause は 0.3 ~ 0.6 秒水準なので、80 ms はそれに比べて非常に短いです。

結果的に「息つく暇のない」不自然な発話が生成されていました。

## 原理分析

この節を読む前に Synthesizer の内部フローをまず押さえておきます（詳細は [アーキテクチャ一覧 — Synthesizer](./architecture#_4-synthesizer-—-音素-bert-→-波形) 参照）。

<PipelineFlow
  :steps="[
    {
      num: '1',
      title: 'Text Encoder',
      content: 'Transformer エンコーダーが音素シーケンスを192次元ベクトルに埋め込みます。BERT 特徴がここで音素レベルの埋め込みと結合され、文脈が初めて音素に注入されます。'
    },
    {
      num: '2',
      title: 'Duration Predictor',
      content: '各音素を何フレーム発音するかを予測します。安定的だが単調な DDP（決定的）と、自然だが不安定な SDP（確率的）の2つの predictor の出力を sdp_ratio でブレンドして安定性と自然さのバランスをとります。この段階で音素シーケンスが時間軸方向に伸長されます。'
    },
    {
      num: '3',
      title: 'Flow',
      content: 'Normalizing Flow（可逆神経回路網）の逆変換を経て、Text Encoder が生成したガウス分布（平均・分散）を実際の音声の複雑な分布に変形して latent z ベクトルを生成します。学習時は順方向（音声 → テキスト空間）、推論時は逆方向（テキスト → 音声空間）で動作します。'
    },
    {
      num: '4',
      title: 'Decoder',
      content: 'HiFi-GAN 系列のボコーダーで、latent z を ConvTranspose アップサンプリングと残差ブロック（ResBlock）を経て時間ドメインの実際の波形（44.1 kHz）に生成します。Synthesizer サブモジュール中で計算量が最大で、CPU 推論時間の大部分がここで消費されます。'
    }
  ]"
/>

この文書で扱う核心は **1・2段階（Text Encoder + Duration Predictor）までのみ別途実行** することです。

3・4段階（Flow + Decoder）をスキップするためコストが非常に低いです。

### オリジナルモデルは pause をどう生成していたか

オリジナル SBV2 が通文合成で自然な pause を生成する原理を追跡した結果、**Duration Predictor が句読点音素のフレーム数を予測する副次効果** でした。

Duration Predictor は元々「各音素を何フレーム発音するか」を予測するモジュールです。

「あ」は5フレーム、「い」は4フレームという具合です。

ところが `.`、`!`、`?` のような句読点も音素シーケンスに含まれます。

Duration Predictor が句読点に対して予測したフレーム数がそのまま **その句読点位置での休止長** になります。

例えば `.` に20フレームが予測されれば Synthesizer はその区間の間無音または無声に近い波形を生成します。

分割合成では句読点位置で合成が途切れるためこの情報がそのまま破棄されていました。

### Duration Predictor の内部動作

Duration Predictor の予測フローをもう少し詳しく見ると、2つのサブモジュールが並列に動作します。

**DDP (Deterministic Duration Predictor)** は同じ入力に対して常に同じ duration を出力します。

安定的ですが発話が機械的に単調に聞こえる場合があります。

**SDP (Stochastic Duration Predictor)** は同じ入力に対して毎回わずかに異なる duration を出力します。

確率サンプリングベースのため自然な変動が生まれますが、その分結果がやや不安定です。

2つの predictor の出力は `sdp_ratio` パラメータでブレンドされます。

`sdp_ratio=0.0` なら DDP のみ、`1.0` なら SDP のみ、`0.5` なら半々で混ぜた結果を使用します。

`length_scale`（= speed パラメータ）は予測された duration 全体に掛けられて話速を調整します。

最終的に `ceil()` で切り上げると各音素の **整数フレーム数** が決定されます。

### blank token と句読点

pause 計算時にひとつ注意すべき点があります。

オリジナル SBV2 は音素シーケンスのすべての音素間に **blank token（空白トークン、ID = 0）** を挿入する構造です。HayaKoe もこの動作をそのまま踏襲しています。

```
元：  [は, い, .]
挿入後: [0, は, 0, い, 0, ., 0]
```

blank token にも duration が予測されるため、句読点 `.` の pause を求める際は **句読点自体 + 前後 blank の duration を合算** する必要があります。

例：`.` = 20フレーム、前 blank = 3フレーム、後 blank = 5フレーム → 合計 28フレーム

## 実装

### 核心アイデア

核心は単純です。

**全体の原文テキストを TextEncoder + Duration Predictor までのみ通過させて、句読点位置のフレーム数を得ること** です。

Flow と Decoder はスキップします。

Synthesizer 全体パスでコストの大部分は Flow と Decoder で発生するため（[ONNX 最適化](./onnx-optimization#synthesizer-最適化) 参照）、Duration Predictor までのみ実行するコストは相対的に低いです。

```
全体テキスト（分割前の原文）
  │
  ├─ TextEncoder（G2P → 音素列 → 埋め込み）
  │
  ├─ Duration Predictor（音素別フレーム数予測）
  │     └─ 句読点位置のフレーム数のみ抽出
  │
  └─ pause 時間計算
        frames × hop_length / sample_rate = 秒
```

全体合成では既に分割された個別の文をそれぞれ TextEncoder → Duration Predictor → Flow → Decoder で通過させます。

pause 予測では **分割前の原文をまるごと** TextEncoder → Duration Predictor のみ通過させます。

分割前の原文を使用する理由は、文境界の句読点が原文でのみ完全に存在するためです。

個別の文に分割した後は最後の文の句読点以外は境界句読点が消えたり位置が変わったりします。

### pause 時間計算

句読点位置のフレーム数を得たら秒単位に変換します。

```
pause（秒） = frames × hop_length / sample_rate
```

HayaKoe のデフォルト設定では `hop_length = 512`、`sample_rate = 44100` なので、1フレームは約 11.6 ms に該当します。

例えば句読点 + 隣接 blank の合算フレーム数が 35 なら：

```
35 × 512 / 44100 ≈ 0.41 秒
```

実際の実装（`durations_to_boundary_pauses()`）では以下のプロセスを経ます。

1. 全音素シーケンスで **文境界句読点の位置** を探します（`.`、`!`、`?` に該当する音素 ID）。
2. 各句読点位置でその音素の duration を取得します。
3. 前方の隣接トークンが blank（ID = 0）ならその duration も加算します。
4. 後方の隣接トークンが blank（ID = 0）ならその duration も加算します。
5. 合算されたフレーム数を `frames × hop_length / sample_rate` で変換します。

文が N 個なら境界は N - 1 個なので、結果は N - 1 個の pause 時間リストです。

### trailing silence（末尾無音）の補償

もうひとつ考慮すべき点があります。

Synthesizer が各文を合成する際、文末に既に **短い無音が含まれる** 場合があります。

この trailing silence を無視して予測された pause をそのまま挿入すると、実際の休止が過度に長くなります。

HayaKoe は合成されたオーディオの末尾で無音区間を直接測定します。

測定方式はオーディオ末尾から 10 ms ウィンドウを1つずつ前方に移動しながら、**ピーク振幅の 2% 以下の区間** を無音と判定します。

以降 pause 挿入時には予測された目標 pause 時間から trailing silence を引いて、**不足分のみ無音サンプルで補充** します。

```
追加無音 = max(0, 予測 pause - trailing silence)
```

モデルが既に十分な無音を生成していれば追加挿入は 0 になります。

目標 pause 時間自体に最低 80 ms の下限を設けているため、予測値がいくら短くても文間の総無音は常に 80 ms 以上になります。

### ONNX 対応

PyTorch 経路ではモデル内部モジュールを個別に呼び出せるため Duration Predictor だけ別途実行すれば済みます。

一方 `synthesizer.onnx` は Synthesizer 全体をひとつのエンドツーエンドグラフとしてエクスポートした形態なので中間出力を取り出せません。

これを解決するために **TextEncoder + Duration Predictor のみを含む別途の ONNX モデル**（`duration_predictor.onnx`、~30 MB、FP32）を追加でエクスポートしました。

## 改善効果

### pause 時間分布

同一テキストに対して自動予測された文境界 pause です。

| バックエンド | pause 範囲 |
|---|---|
| GPU (PyTorch) | 0.41 s ~ 0.55 s |
| CPU (ONNX) | 0.38 s ~ 0.57 s |

2つのバックエンドの差は SDP の stochastic sampling（確率的サンプリング）の特性上生じる偏差レベルです。

SDP は確率サンプリングベースのため同じ入力でも呼び出しごとに結果がわずかに異なります。

GPU と CPU の差がこの自然な変動幅内に収まるため、ONNX 変換による品質損失は無視できます。

### Before / After

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。きっと楽しい発見がありますよ。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pause 方式"
  :defaultIndex="1"
  :samples='[
    { "value": "Before (80 ms 固定)", "caption": "すべての文境界が同一の短い休止", "src": "/hayakoe/samples/duration-predictor/before.wav" },
    { "value": "After (DP 予測)", "caption": "Duration Predictor が文境界 pause を自動予測", "src": "/hayakoe/samples/duration-predictor/after.wav" }
  ]'
/>

### コスト

追加コストは TextEncoder + Duration Predictor 1回の実行です。

[ONNX 最適化 — Synthesizer 比率](./onnx-optimization#synthesizer-最適化) で確認できるように、Synthesizer が全 CPU 推論時間の 64 ~ 91% を占め、その大部分は Flow + Decoder で消費されます。

Duration Predictor までのみ実行するコストはこれに比べて低いため、pause 予測による体感遅延はほぼありません。

## 関連コミット

- `c57e0ad` — Duration Predictor ベース pause 予測で多文合成を自然に改善
- `5522db1` — ONNX `duration_predictor` 追加で CPU バックエンドも自然な文境界無音をサポート

## 今後の課題

- **感情別 pause 長の分化** — 喜びは短く、悲しみは長くなど感情スタイルに応じて pause 分布を異なるように適用
- **読点・コロン等の細分化** — 現在は文末句読点（`. ! ?`）のみ対象だが、読点（`,`、`、`）やコロンなど長い呼吸が必要な位置への追加細分化
- **pause 直接制御 API** — ユーザーが特定の文境界の pause 長を明示的に指定できるインターフェース
