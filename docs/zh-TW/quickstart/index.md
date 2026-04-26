# 快速入門

本指南面向「想先跑起來看看」的讀者。

使用內建的官方說話人,到生成第一條語音約 10 分鐘,到基準測試約 15 分鐘即可完成。

## 閱讀順序

1. [安裝 — CPU vs GPU](./install) — 選擇適合自己環境的安裝設定
2. [生成第一條語音](./first-voice) — 用官方說話人儲存 wav 檔案
3. [速度・韻律調整](./parameters) — 了解速度/音高/韻律參數
4. [註冊自訂詞彙](./custom-words) — 手動修正讀錯的詞彙
5. [句子級串流傳輸](./streaming) — 盡快發送長文本的第一條語音
6. [在我的機器上做基準測試](./benchmark) — 測量在自己硬體上實際有多快

## 完成本節後能做到的事

- 自由使用 11 位預置說話人
- 調整速度・音高・韻律參數
- 在自己的硬體上直接測量「生成 1 秒語音需要幾秒」

## 可以隨意製作這樣的語音

完成快速入門後,以下說話人就到你手中了。

以下是各說話人朗讀同一句話(「こんにちは、はじめまして。」)的範例。

<SpeakerSample badge="JVNV" name="jvnv-F1-jp  —  女性說話人 1" src="/hayakoe/samples/hello_jvnv-F1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-F2-jp  —  女性說話人 2" src="/hayakoe/samples/hello_jvnv-F2-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M1-jp  —  男性說話人 1" src="/hayakoe/samples/hello_jvnv-M1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M2-jp  —  男性說話人 2" src="/hayakoe/samples/hello_jvnv-M2-jp.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="tsukuyomi_chan  —  動畫風" src="/hayakoe/samples/hello_tsukuyomi_chan.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_normal  —  普通" src="/hayakoe/samples/hello_amitaro_normal.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_runrun  —  興奮" src="/hayakoe/samples/hello_amitaro_runrun.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_yofukashi  —  沉穩" src="/hayakoe/samples/hello_amitaro_yofukashi.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_punsuka  —  生氣" src="/hayakoe/samples/hello_amitaro_punsuka.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_a  —  低語A" src="/hayakoe/samples/hello_amitaro_sasayaki_a.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_b  —  低語B" src="/hayakoe/samples/hello_amitaro_sasayaki_b.wav" />

::: info 自訂說話人訓練正在準備中
準備錄音並訓練自訂說話人的指南正在整理中。

準備完成後將發佈在 [自訂說話人訓練](/zh-TW/training/) 板塊。
:::
