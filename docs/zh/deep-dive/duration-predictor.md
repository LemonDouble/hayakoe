# 句子边界 pause — Duration Predictor

多句分割合成时会产生 **句间停顿 (pause) 丢失** 的副作用。

HayaKoe 复用 Duration Predictor 直接预测各句子边界的自然 pause 时间。

跳过 Flow · Decoder,**仅执行 TextEncoder + Duration Predictor**,因此额外成本很低。

## 为什么是问题

### 句子分割的优势

如 [架构概览](./architecture#_1-%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC-%E1%84%87%E1%85%AE%E1%86%AB%E1%84%92%E1%85%A1%E1%86%AF) 所述,HayaKoe 将多句输入按句子分割后逐个合成。

长文本整体合成时韵律容易模糊或不稳定。

按句子分割后每个句子都能保证稳定的 prosody(韵律)。

### 分割的副作用 — pause 丢失

但分割有副作用。

原版 SBV2 在通篇合成中会在 `.`、`!`、`?` 等标点后插入自然的停顿。

按句子分割后每个句子在标点处结束,下一句从头开始,**标点后的停顿随之消失**。

初期实现在句间插入固定 80 ms 静音。

实际上 Duration Predictor 预测的句子边界 pause 在 0.3 ~ 0.6 秒水平,80 ms 相比之下非常短。

结果产生了"没有喘息空间"的不自然发话。

## 原理分析

阅读本节前先回顾一下 Synthesizer 内部流程(详见 [架构概览 — Synthesizer](./architecture#_4-synthesizer-—-音素-bert-→-波形))。

<PipelineFlow
  :steps="[
    {
      num: '1',
      title: 'Text Encoder',
      content: 'Transformer 编码器将音素序列嵌入为 192 维向量。BERT 特征在此处与音素级嵌入结合,句子上下文首次注入到音素中。'
    },
    {
      num: '2',
      title: 'Duration Predictor',
      content: '预测每个音素发音多少帧。将稳定但单调的 DDP(确定性)和自然但不稳定的 SDP(随机性)两个 predictor 的输出通过 sdp_ratio 混合,平衡稳定性和自然性。此步骤中音素序列在时间轴上被展开。'
    },
    {
      num: '3',
      title: 'Flow',
      content: '经过 Normalizing Flow(可逆神经网络)的反向变换,将 Text Encoder 生成的高斯分布(均值·方差)变形为实际音频的复杂分布,生成 latent z 向量。训练时走正方向(音频 → 文本空间),推理时走反方向(文本 → 音频空间)。'
    },
    {
      num: '4',
      title: 'Decoder',
      content: 'HiFi-GAN 系列的声码器,将 latent z 经过 ConvTranspose 上采样和残差块 (ResBlock) 生成时域的实际波形 (44.1 kHz)。Synthesizer 子模块中运算量最大,CPU 推理时间的大部分在此消耗。'
    }
  ]"
/>

本文档的核心是 **仅执行 1 · 2 步骤 (Text Encoder + Duration Predictor)**。

跳过 3 · 4 步骤 (Flow + Decoder),因此成本非常低。

### 原版模型如何生成 pause

追踪原版 SBV2 在通篇合成中生成自然 pause 的原理后发现,这是 **Duration Predictor 预测标点音素帧数的副效果**。

Duration Predictor 原本是预测"每个音素发音多少帧"的模块。

如"安"5 帧、"宁"4 帧。

而 `.`、`!`、`?` 等标点也包含在音素序列中。

Duration Predictor 对标点预测的帧数就成为 **该标点位置的停顿时长**。

例如 `.` 预测为 20 帧时 Synthesizer 会在该区间生成静音或接近静音的波形。

在分割合成中由于标点位置合成被切断,这些信息直接被丢弃。

### Duration Predictor 的内部动作

更详细地看 Duration Predictor 的预测流程,两个子模块并行工作。

**DDP (Deterministic Duration Predictor)** 对相同输入始终输出相同 duration。

稳定但发话可能听起来机械性单调。

**SDP (Stochastic Duration Predictor)** 对相同输入每次输出略有不同的 duration。

基于概率采样产生自然变动,但结果不太稳定。

两个 predictor 的输出通过 `sdp_ratio` 参数混合。

`sdp_ratio=0.0` 仅用 DDP,`1.0` 仅用 SDP,`0.5` 各半混合。

`length_scale` (= speed 参数) 乘以预测的全部 duration 来调节语速。

最终 `ceil()` 向上取整确定各音素的 **整数帧数**。

### blank token 和标点

计算 pause 时有一个注意点。

原版 SBV2 在音素序列的所有音素之间插入 **blank token(空白标记, ID = 0)**。HayaKoe 沿用此行为。

```
原始:  [は, い, .]
插入后: [0, は, 0, い, 0, ., 0]
```

blank token 也会被预测 duration,因此计算标点 `.` 的 pause 时需要 **将标点本身 + 前后 blank 的 duration 求和**。

例: `.` = 20 帧, 前 blank = 3 帧, 后 blank = 5 帧 → 总计 28 帧

## 实现

### 核心思路

核心很简单。

**将完整原文文本仅通过 TextEncoder + Duration Predictor,获取标点位置的帧数**。

跳过 Flow 和 Decoder。

Synthesizer 全程中成本大部分在 Flow 和 Decoder 中产生([ONNX 优化 — Synthesizer 占比](./onnx-optimization#synthesizer-优化) 参考),因此仅执行到 Duration Predictor 的成本相对较低。

```
完整文本 (分割前原文)
  │
  ├─ TextEncoder (G2P → 音素序列 → 嵌入)
  │
  ├─ Duration Predictor (各音素帧数预测)
  │     └─ 仅提取标点位置的帧数
  │
  └─ pause 时间计算
        frames × hop_length / sample_rate = 秒
```

全程合成中将已分割的各句子分别通过 TextEncoder → Duration Predictor → Flow → Decoder。

pause 预测中将 **分割前的原文整体** 仅通过 TextEncoder → Duration Predictor。

使用分割前原文的原因是句子边界的标点只在原文中完整存在。

分割为各句子后除最后一句的标点外,边界标点会消失或位置改变。

### pause 时间计算

获取标点位置的帧数后转换为秒。

```
pause (秒) = frames × hop_length / sample_rate
```

HayaKoe 默认设置中 `hop_length = 512`、`sample_rate = 44100`,1 帧约 11.6 ms。

例如标点 + 相邻 blank 的合计帧数为 35:

```
35 × 512 / 44100 ≈ 0.41 秒
```

实际实现(`durations_to_boundary_pauses()`)经过以下过程。

1. 在完整音素序列中 **找到句子边界标点的位置**(对应 `.`、`!`、`?` 的音素 ID)。
2. 获取各标点位置该音素的 duration。
3. 如果前方相邻 token 是 blank (ID = 0) 则加上其 duration。
4. 如果后方相邻 token 是 blank (ID = 0) 则加上其 duration。
5. 将合计帧数用 `frames × hop_length / sample_rate` 转换。

如果有 N 个句子则有 N - 1 个边界,结果是 N - 1 个 pause 时间的列表。

### trailing silence(尾部静音)补偿

还有一个需要考虑的点。

Synthesizer 合成各句子时,句尾可能已经 **包含短暂的静音**。

如果忽略此 trailing silence 直接插入预测的 pause,实际停顿会过长。

HayaKoe 直接测量合成音频尾部的静音区间。

测量方式是从音频末尾以 10 ms 窗口逐格向前移动,**峰值振幅低于 2% 的区间** 判定为静音。

之后插入 pause 时从预测的目标 pause 时间中减去 trailing silence,**仅补充不足部分为静音样本**。

```
额外静音 = max(0, 预测 pause - trailing silence)
```

如果模型已经生成了足够的静音,额外插入为 0。

目标 pause 时间的下限设为 80 ms,因此即使预测值再短,句间总静音也始终不低于 80 ms。

### ONNX 支持

PyTorch 路径中可以单独调用模型内部模块,只需单独运行 Duration Predictor。

而 `synthesizer.onnx` 是将 Synthesizer 整体导出为一个端到端图的形式,无法提取中间输出。

为解决此问题,额外导出了 **仅包含 TextEncoder + Duration Predictor 的独立 ONNX 模型** (`duration_predictor.onnx`, ~30 MB, FP32)。

## 改善效果

### pause 时间分布

同一文本下自动预测的句子边界 pause。

| 后端 | pause 范围 |
|---|---|
| GPU (PyTorch) | 0.41 s ~ 0.55 s |
| CPU (ONNX) | 0.38 s ~ 0.57 s |

两种后端的差异属于 SDP 的 stochastic sampling(概率采样)特性产生的偏差水平。

SDP 基于概率采样,即使相同输入每次调用结果也略有不同。

GPU 和 CPU 的差异落在此自然变动幅度内,因此 ONNX 转换带来的品质损失可以忽略。

### Before / After

> 旅の途中で不思議な街に辿り着きました。少し寄り道していきましょう。きっと楽しい発見がありますよ。

<SpeakerSampleGroup
  badge="つくよみちゃん"
  badgeIcon="/hayakoe/images/speakers/tsukuyomi.png"
  label="pause 方式"
  :defaultIndex="1"
  :samples='[
    { "value": "Before (80 ms 固定)", "caption": "所有句子边界相同的短停顿", "src": "/hayakoe/samples/duration-predictor/before.wav" },
    { "value": "After (DP 预测)", "caption": "Duration Predictor 自动预测句子边界 pause", "src": "/hayakoe/samples/duration-predictor/after.wav" }
  ]'
/>

### 成本

额外成本为 TextEncoder + Duration Predictor 1 次执行。

如 [ONNX 优化 — Synthesizer 占比](./onnx-optimization#synthesizer-优化) 所示,Synthesizer 占整体 CPU 推理时间的 64 ~ 91%,其中大部分消耗在 Flow + Decoder。

仅执行到 Duration Predictor 的成本相比之下很低,因此 pause 预测带来的感知延迟几乎没有。

## 相关提交

- `c57e0ad` — 基于 Duration Predictor 的 pause 预测改善多句合成自然度
- `5522db1` — 新增 ONNX `duration_predictor` 使 CPU 后端也支持自然的句子边界静音

## 未来课题

- **按情感的 pause 长度分化** — 开心时短、悲伤时长等根据情感风格应用不同的 pause 分布
- **逗号·冒号等细分** — 目前仅针对句末标点(`. ! ?`),对逗号(`,`、`、`)或冒号等需要长呼吸的位置进行追加细分
- **pause 直接控制 API** — 用户可以明确指定特定句子边界 pause 长度的接口
