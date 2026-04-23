---
layout: home

hero:
  name: HayaKoe
  text: "好きな声で作ったTTSを、<br>CPUだけで準リアルタイムに。"
  tagline: 動画や録音さえあれば、データ準備・学習・ベンチマーク・デプロイまですべてお任せください。
  actions:
    - theme: brand
      text: 10分で試す
      link: /ja/quickstart/
    - theme: alt
      text: 学習
      link: /ja/training/
    - theme: alt
      text: デプロイ
      link: /ja/deploy/
    - theme: alt
      text: 深掘り
      link: /ja/deep-dive/

features:
  - title: CPU リアルタイム推論
    details: "ONNX 最適化により Style-Bert-VITS2 比で短いテキストは1.5倍、長いテキストは3.3倍速い CPU 単独推論。<br>GPU では torch.compile でさらに高速化されます。"
    link: /ja/deep-dive/onnx-optimization
    linkText: どうやったのか
  - title: AMD64・ARM64 どこでも
    details: "x86_64・aarch64 Linux どちらも同じコマンドひとつでインストール。<br>Raspberry Pi のような ARM ボードでも CPU 推論がそのまま動きます。"
    link: /ja/quickstart/benchmark#ラズベリーパイ-4b-ではどうか
    linkText: ラズベリーパイ ベンチマーク
  - title: メモリ 47% 削減
    details: "BERT Q8 量子化により PyTorch 比で RAM 47% 削減。<br>CPU モード約 2.4 GB RAM、GPU モード約 1.7 GB VRAM。"
    link: /ja/deep-dive/onnx-optimization
    linkText: どうやったのか
  - title: 多話者でも軽量
    details: "BERT を全話者で共有する構造。<br>話者を1人追加しても RAM は ~300 MB 増えるだけです。"
    link: /ja/deploy/fastapi
    linkText: 多話者サービング
  - title: 文単位ストリーミング
    details: "astream() で文が合成され次第流し出します。<br>全体の合成を待つより最初の音声をより早く受け取れます。"
    link: /ja/deploy/fastapi
    linkText: ストリーミング例
  - title: 好きな声で
    details: "好きな声が入った動画を用意するだけ。<br>前処理・学習・品質比較・最適化・デプロイまで、すべてお任せください。"
    link: /ja/training/
    linkText: 学習ガイド
  - title: HF・S3 互換・ローカル プラグ式
    details: "CLI デプロイは HuggingFace・S3 互換・ローカルのどこへでも。<br>ランタイムロードも同じ3つの経路を同様にサポートします。"
    link: /ja/deep-dive/source-abstraction
    linkText: Source 抽象化
---

## このような声を作れます

デフォルト提供の話者が同じ文章（「こんにちは、はじめまして。」）を話すサンプルです。

<SpeakerSample badge="JVNV" name="jvnv-F1-jp  —  女性話者 1" src="/hayakoe/samples/hello_jvnv-F1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-F2-jp  —  女性話者 2" src="/hayakoe/samples/hello_jvnv-F2-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M1-jp  —  男性話者 1" src="/hayakoe/samples/hello_jvnv-M1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M2-jp  —  男性話者 2" src="/hayakoe/samples/hello_jvnv-M2-jp.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="tsukuyomi_chan  —  アニメ風" src="/hayakoe/samples/hello_tsukuyomi_chan.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_normal  —  ノーマル" src="/hayakoe/samples/hello_amitaro_normal.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_runrun  —  ワクワク" src="/hayakoe/samples/hello_amitaro_runrun.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_yofukashi  —  落ち着き" src="/hayakoe/samples/hello_amitaro_yofukashi.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_punsuka  —  怒り" src="/hayakoe/samples/hello_amitaro_punsuka.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_a  —  ささやきA" src="/hayakoe/samples/hello_amitaro_sasayaki_a.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_b  —  ささやきB" src="/hayakoe/samples/hello_amitaro_sasayaki_b.wav" />

自分のノートPCで、CPUだけで、上記のサンプルを実際に作ってみたい方は [10分で試す](/ja/quickstart/) へ。

## 簡単に試すと

### インストール

::: code-group
```bash [CPU]
pip install hayakoe
```
```bash [GPU (CUDA)]
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
:::

CPU プロファイルは PyTorch が不要なのでインストールが短く、イメージも軽量になります。

GPU プロファイルは追加の依存関係をインストールする代わりに、より高速に推論します。

### 推論

```python
from hayakoe import TTS

text = "こんにちは、はじめまして。"

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate(text).save("hello.wav")
```

すぐに `hello.wav` を聴いてみてください！

デフォルトで提供される話者は11名です。

- `jvnv-F1-jp` / `jvnv-F2-jp` / `jvnv-M1-jp` / `jvnv-M2-jp` — JVNV コーパスベース
- `tsukuyomi_chan` — つくよみちゃんコーパスベース
- `amitaro_normal` / `amitaro_runrun` / `amitaro_yofukashi` / `amitaro_punsuka` / `amitaro_sasayaki_a` / `amitaro_sasayaki_b` — あみたろ ITAコーパスベース

上記コードの `"jvnv-F1-jp"` の部分を変えるだけで他の声もすぐに聴けます。

GPU プロファイルでインストールした場合は `TTS(device="cuda")` パラメータを追加するだけで GPU 推論ができます。

## どのドキュメントを読めばいいですか？

1. **まず [クイックスタート](/ja/quickstart/)** に沿って進めてみてください。インストールから初回合成、ベンチマークまで、この TTS がどれほど速いか・音質はどうか直接確認できます。
2. **もっと試したくなったら [自前話者の学習](/ja/training/)** へ。好きな声が入った動画1つでデータ準備からデプロイまで全工程を案内します。
3. **自分だけで使うのはもったいないなら [サーバーへデプロイ](/ja/deploy/)** へ。FastAPI・Docker 上で API として公開する方法をまとめました。
4. **技術的に深く掘り下げたいなら [深掘り](/ja/deep-dive/)** へ。どこをどう手を入れてこれだけの速度・メモリ改善を得たのか、改善ポイントを一つずつ解説します。
5. **つまずいた部分があれば [FAQ](/ja/faq/)** へ。キャッシュパス・Private HF・S3・多話者メモリなどの詳細設定をまとめています。

## 音声データクレジット

本プロジェクトの音声合成には以下の音声データを使用しています。

- **つくよみちゃんコーパス** (CV.夢前黎, © Rei Yumesaki) — https://tyc.rei-yumesaki.net/material/corpus/
- **あみたろの声素材工房** ITAコーパス読み上げ音声 — https://amitaro.net/
