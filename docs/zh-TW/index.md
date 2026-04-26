---
layout: home

hero:
  name: HayaKoe
  text: "用我喜歡的聲音製作 TTS,<br>僅用 CPU 也能準即時合成。"
  tagline: 只要有影片或錄音,從資料準備、訓練、基準測試到部署,我們全部搞定。
  actions:
    - theme: brand
      text: 10 分鐘快速體驗
      link: /zh-TW/quickstart/
    - theme: alt
      text: 訓練
      link: /zh-TW/training/
    - theme: alt
      text: 部署
      link: /zh-TW/deploy/
    - theme: alt
      text: 深入解讀
      link: /zh-TW/deep-dive/

features:
  - title: CPU 即時推論
    details: "透過 ONNX 最佳化,相比 Style-Bert-VITS2,短文本快 1.5 倍,長文本快 3.3 倍,僅用 CPU 即可推論。<br>在 GPU 上還可透過 torch.compile 進一步加速。"
    link: /zh-TW/deep-dive/onnx-optimization
    linkText: 是怎麼做到的
  - title: AMD64 · ARM64 全平台
    details: "x86_64 · aarch64 Linux 均可用同一條指令安裝。<br>在 Raspberry Pi 等 ARM 開發板上同樣可以進行 CPU 推論。"
    link: /zh-TW/quickstart/benchmark#raspberry-pi-4b-實測
    linkText: 樹莓派基準測試
  - title: 記憶體減少 47%
    details: "透過 BERT Q8 量化,相比 PyTorch 減少 47% 的 RAM 佔用。<br>CPU 模式約 2.0 GB RAM,GPU 模式約 1.7 GB VRAM。"
    link: /zh-TW/deep-dive/onnx-optimization
    linkText: 是怎麼做到的
  - title: 多說話人也很輕量
    details: "BERT 由所有說話人共享的架構。<br>每新增一個說話人,RAM 僅增加約 300 MB。"
    link: /zh-TW/deploy/fastapi
    linkText: 多說話人服務
  - title: 句子級串流傳輸
    details: "透過 astream() 在句子合成完畢後立即發送。<br>比等待全部合成完畢更快地獲取首條語音。"
    link: /zh-TW/deploy/fastapi
    linkText: 串流傳輸範例
  - title: 用我想要的聲音
    details: "只需準備包含喜歡聲音的影片。<br>從前處理、訓練、品質對比、最佳化到部署,我們全部搞定。"
    link: /zh-TW/training/
    linkText: 訓練指南
  - title: HF · S3 相容 · 本地可插拔
    details: "CLI 部署可發佈到 HuggingFace · S3 相容儲存 · 本地任意位置。<br>執行時載入同樣支援相同的三種路徑。"
    link: /zh-TW/deep-dive/source-abstraction
    linkText: Source 抽象層
---

## 可以製作這樣的聲音

以下是內建說話人朗讀同一句話(「こんにちは、はじめまして。」)的範例。

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

想在自己的筆電上,僅用 CPU 親自製作上述範例的話,請前往 [10 分鐘快速體驗](/zh-TW/quickstart/)。

## 快速試用

### 安裝

::: code-group
```bash [CPU]
pip install hayakoe
```
```bash [GPU (CUDA)]
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
:::

CPU 設定不需要 PyTorch,安裝更快,映像檔也更輕量。

GPU 設定會安裝額外依賴,換來更快的推論速度。

### 推論

```python
from hayakoe import TTS

text = "こんにちは、はじめまして。"

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate(text).save("hello.wav")
```

馬上聽聽 `hello.wav` 吧!

內建提供 11 位說話人。

- `jvnv-F1-jp` / `jvnv-F2-jp` / `jvnv-M1-jp` / `jvnv-M2-jp` — 基於 JVNV 語料庫
- `tsukuyomi_chan` — 基於 つくよみちゃんコーパス
- `amitaro_normal` / `amitaro_runrun` / `amitaro_yofukashi` / `amitaro_punsuka` / `amitaro_sasayaki_a` / `amitaro_sasayaki_b` — 基於 あみたろ ITAコーパス

只需替換上述程式碼中的 `"jvnv-F1-jp"`,即可立即聽到其他聲音。

如果已安裝 GPU 設定,只需添加 `TTS(device="cuda")` 參數即可使用 GPU 推論。

## 應該閱讀哪些文件?

1. **首先跟著 [快速開始](/zh-TW/quickstart/)** 走一遍。從安裝到首次合成、基準測試,親自體驗這個 TTS 有多快、音質如何。
2. **想要更多的話,前往 [自訂說話人訓練](/zh-TW/training/)**。只需一段包含喜歡聲音的影片,從資料準備到部署全程指引。
3. **想分享給更多人的話,前往 [伺服器部署](/zh-TW/deploy/)**。整理了在 FastAPI · Docker 上以 API 形式發佈的方法。
4. **想深入技術細節的話,前往 [深入解讀](/zh-TW/deep-dive/)**。逐一解析是如何做到這樣的速度和記憶體最佳化的。
5. **遇到問題的話,前往 [FAQ](/zh-TW/faq/)**。彙集了快取路徑、Private HF、S3、多說話人記憶體等進階設定。

## 語音資料致謝

本專案的語音合成使用了以下語音資料。

- **つくよみちゃんコーパス** (CV.夢前黎, © Rei Yumesaki) — https://tyc.rei-yumesaki.net/material/corpus/
- **あみたろの声素材工房** ITAコーパス読み上げ音声 — https://amitaro.net/
