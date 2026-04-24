# クイックスタート

「とりあえず動くか確認してみたい」という方のためのガイドです。

既成の公式話者で初回音声まで約10分、ベンチマークまで約15分で完了します。

## 読む順番

1. [インストール — CPU vs GPU](./install) — 自分の環境に合ったインストールプロファイルを選ぶ
2. [初めての音声を作る](./first-voice) — 公式話者で wav 保存まで
3. [速度・韻律の調整](./parameters) — 速度/ピッチ/韻律パラメータを理解する
4. [カスタム単語の登録](./custom-words) — 誤読される単語を手動で固定する
5. [文単位ストリーミング](./streaming) — 長いテキストの最初の音声を早く送り出す
6. [自分のマシンでベンチマーク](./benchmark) — 実際の自分のハードウェアでどれくらい速いか測定

## このセクションが終わるとできること

- 事前に用意された話者11名を自由に呼び出して使う
- 速度・ピッチ・韻律パラメータの調整
- 自分のハードウェアで「1秒の音声を作るのに何秒かかるか」を直接測定

## このような音声を自由に作れます

クイックスタートが終われば、以下の話者が手元に揃います。

同じ文章（「こんにちは、はじめまして。」）を各話者が話したサンプルです。

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

::: info 自前話者の学習は準備中
録音を用意して自前の話者を学習させるガイドは別途整理中です。

準備が整い次第 [自前話者の学習](/ja/training/) セクションに掲載します。
:::
