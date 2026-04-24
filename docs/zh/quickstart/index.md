# 快速开始

本指南面向"想先跑起来看看"的读者。

使用内置官方说话人,到生成首条语音约 10 分钟,到基准测试约 15 分钟即可完成。

## 阅读顺序

1. [安装 — CPU vs GPU](./install) — 选择适合自己环境的安装配置
2. [生成第一条语音](./first-voice) — 用官方说话人保存 wav 文件
3. [速度·韵律调节](./parameters) — 了解速度/音高/韵律参数
4. [注册自定义词语](./custom-words) — 手动修正读错的词语
5. [句子级流式传输](./streaming) — 尽快发送长文本的首条语音
6. [在我的机器上做基准测试](./benchmark) — 测量在自己硬件上实际有多快

## 完成本节后能做到的事

- 自由使用 4 位预置说话人(`jvnv-F1/F2/M1/M2-jp`)
- 调节速度·音高·韵律参数
- 在自己的硬件上直接测量"生成 1 秒语音需要多少秒"

## 可以随意制作这样的语音

完成快速开始后,以下说话人就到你手中了。

以下是各说话人朗读同一句话("こんにちは、はじめまして。")的示例。

<SpeakerSample badge="JVNV" name="jvnv-F1-jp  —  女性说话人 1" src="/hayakoe/samples/hello_jvnv-F1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-F2-jp  —  女性说话人 2" src="/hayakoe/samples/hello_jvnv-F2-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M1-jp  —  男性说话人 1" src="/hayakoe/samples/hello_jvnv-M1-jp.wav" />
<SpeakerSample badge="JVNV" name="jvnv-M2-jp  —  男性说话人 2" src="/hayakoe/samples/hello_jvnv-M2-jp.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="tsukuyomi_chan  —  动画风" src="/hayakoe/samples/hello_tsukuyomi_chan.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_normal  —  普通" src="/hayakoe/samples/hello_amitaro_normal.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_runrun  —  兴奋" src="/hayakoe/samples/hello_amitaro_runrun.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_yofukashi  —  沉稳" src="/hayakoe/samples/hello_amitaro_yofukashi.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_punsuka  —  生气" src="/hayakoe/samples/hello_amitaro_punsuka.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_a  —  低语A" src="/hayakoe/samples/hello_amitaro_sasayaki_a.wav" />
<SpeakerSample badge="あみたろ" badgeIcon="/hayakoe/images/speakers/amitaro.png" name="amitaro_sasayaki_b  —  低语B" src="/hayakoe/samples/hello_amitaro_sasayaki_b.wav" />

::: info 自定义说话人训练正在准备中
准备录音并训练自定义说话人的指南正在整理中。

准备完成后将发布在 [自定义说话人训练](/zh/training/) 板块。
:::
