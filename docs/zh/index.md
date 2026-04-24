---
layout: home

hero:
  name: HayaKoe
  text: "用我喜欢的声音制作 TTS,<br>仅用 CPU 也能准实时合成。"
  tagline: 只要有视频或录音,从数据准备、训练、基准测试到部署,我们全部搞定。
  actions:
    - theme: brand
      text: 10 分钟快速体验
      link: /zh/quickstart/
    - theme: alt
      text: 训练
      link: /zh/training/
    - theme: alt
      text: 部署
      link: /zh/deploy/
    - theme: alt
      text: 深入解读
      link: /zh/deep-dive/

features:
  - title: CPU 实时推理
    details: "通过 ONNX 优化,相比 Style-Bert-VITS2,短文本快 1.5 倍,长文本快 3.3 倍,仅用 CPU 即可推理。<br>在 GPU 上还可通过 torch.compile 进一步加速。"
    link: /zh/deep-dive/onnx-optimization
    linkText: 是怎么做到的
  - title: AMD64 · ARM64 全平台
    details: "x86_64 · aarch64 Linux 均可用同一条命令安装。<br>在 Raspberry Pi 等 ARM 开发板上同样可以进行 CPU 推理。"
    link: /zh/quickstart/benchmark#拉-raspberry-pi-4b-实测
    linkText: 树莓派基准测试
  - title: 内存减少 47%
    details: "通过 BERT Q8 量化,相比 PyTorch 减少 47% 的 RAM 占用。<br>CPU 模式约 2.4 GB RAM,GPU 模式约 1.7 GB VRAM。"
    link: /zh/deep-dive/onnx-optimization
    linkText: 是怎么做到的
  - title: 多说话人也很轻量
    details: "BERT 由所有说话人共享的架构。<br>每新增一个说话人,RAM 仅增加约 300 MB。"
    link: /zh/deploy/fastapi
    linkText: 多说话人服务
  - title: 句子级流式传输
    details: "通过 astream() 在句子合成完毕后立即发送。<br>比等待全部合成完毕更快地获取首条语音。"
    link: /zh/deploy/fastapi
    linkText: 流式传输示例
  - title: 用我想要的声音
    details: "只需准备包含喜欢声音的视频。<br>从预处理、训练、质量对比、优化到部署,我们全部搞定。"
    link: /zh/training/
    linkText: 训练指南
  - title: HF · S3 兼容 · 本地可插拔
    details: "CLI 部署可发布到 HuggingFace · S3 兼容存储 · 本地任意位置。<br>运行时加载同样支持相同的三种路径。"
    link: /zh/deep-dive/source-abstraction
    linkText: Source 抽象层
---

## 可以制作这样的声音

以下是内置说话人朗读同一句话("こんにちは、はじめまして。")的示例。

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

想在自己的笔记本上,仅用 CPU 亲自制作上述示例的话,请前往 [10 分钟快速体验](/zh/quickstart/)。

## 快速试用

### 安装

::: code-group
```bash [CPU]
pip install hayakoe
```
```bash [GPU (CUDA)]
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
:::

CPU 配置不需要 PyTorch,安装更快,镜像也更轻量。

GPU 配置会安装额外依赖,换来更快的推理速度。

### 推理

```python
from hayakoe import TTS

text = "こんにちは、はじめまして。"

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate(text).save("hello.wav")
```

马上听听 `hello.wav` 吧!

内置提供 11 位说话人。

- `jvnv-F1-jp` / `jvnv-F2-jp` / `jvnv-M1-jp` / `jvnv-M2-jp` — 基于 JVNV 语料库
- `tsukuyomi_chan` — 基于 つくよみちゃんコーパス
- `amitaro_normal` / `amitaro_runrun` / `amitaro_yofukashi` / `amitaro_punsuka` / `amitaro_sasayaki_a` / `amitaro_sasayaki_b` — 基于 あみたろ ITAコーパス

只需替换上述代码中的 `"jvnv-F1-jp"`,即可立即听到其他声音。

如果已安装 GPU 配置,只需添加 `TTS(device="cuda")` 参数即可使用 GPU 推理。

## 应该阅读哪些文档?

1. **首先跟着 [快速开始](/zh/quickstart/)** 走一遍。从安装到首次合成、基准测试,亲自体验这个 TTS 有多快、音质如何。
2. **想要更多的话,前往 [自定义说话人训练](/zh/training/)**。只需一段包含喜欢声音的视频,从数据准备到部署全程指引。
3. **想分享给更多人的话,前往 [服务器部署](/zh/deploy/)**。整理了在 FastAPI · Docker 上以 API 形式发布的方法。
4. **想深入技术细节的话,前往 [深入解读](/zh/deep-dive/)**。逐一解析是如何做到这样的速度和内存优化的。
5. **遇到问题的话,前往 [FAQ](/zh/faq/)**。汇集了缓存路径、Private HF、S3、多说话人内存等高级设置。

## 语音数据致谢

本项目的语音合成使用了以下语音数据。

- **つくよみちゃんコーパス** (CV.夢前黎, © Rei Yumesaki) — https://tyc.rei-yumesaki.net/material/corpus/
- **あみたろの声素材工房** ITAコーパス読み上げ音声 — https://amitaro.net/
