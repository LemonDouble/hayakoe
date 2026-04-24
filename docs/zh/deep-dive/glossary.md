# 术语表

本页面整理了 deep-dive 全篇中频繁出现的 TTS · 推理术语。

面向 TTS 新手,**确保阅读 architecture 之后的页面时不会卡住**。

可以按顺序通读,也可以按需搜索查阅。

## 流水线全局视角

### TTS (Text-to-Speech)

将文字(文本)转换为人声音频的技术总称。

输入通常是句子,输出是 WAV · MP3 等音频文件。

TTS 系统内部经过"将文字解释为发音" → "将发音合成为波形"等多个阶段。

HayaKoe 也属于此类,接收日语输入生成 WAV 波形。

### 音素 (Phoneme)

区分语义的最小声音单位。

TTS 模型接收的不是文字而是音素。如果直接接收文字,模型就必须学习"在什么条件下如何发音"的全部发音规则。先转换为音素后,模型只需专注于"如何让这个声音听起来对"。

负责这个转换的模块就是 **G2P**。

### G2P (Grapheme-to-Phoneme)

将文字 (Grapheme) 转换为音素 (Phoneme) 的过程或模块。

处理各语言特有的发音规则,如日语的汉字读法·连音规则等。

在 TTS 流水线中属于将输入送入模型前的步骤。

HayaKoe 专用于日语,将日语 G2P 委托给 [pyopenjtalk](./openjtalk-dict)。

### 波形 (Waveform)

将空气压力变化沿时间轴记录的数字序列。是扬声器可以播放的"实际声音"本身。

每个数字表示 **特定瞬间的空气压力(振幅,amplitude)**。正值表示空气被压缩,负值表示膨胀,绝对值越大声音越响。0 对应静音(基准压力)。

采样率 (sample rate) 为 22,050 Hz 时 1 秒 = 22,050 个这样的数字。HayaKoe 的输出为 44,100 Hz,每秒 44,100 个。

TTS 的最终输出物就是这个波形。

## 模型组成要素

### VITS

2021 年发表的语音合成模型架构。

核心贡献是将之前分为两个阶段(Acoustic Model + Vocoder)的 TTS 流水线整合为 **一个端到端模型**。

文本 → 波形的转换由单一模型直接完成,内部由 Text Encoder · Duration Predictor · Flow · Decoder 构成。

HayaKoe 处于 VITS 谱系的延长线上。

- **VITS (2021)** — End-to-End TTS 的起点。
- **[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)** — Fish Audio 团队在 VITS 上加入 BERT 以增强 **基于上下文的 prosody** 的开源项目。
- **[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)** — litagin02 fork Bert-VITS2 并添加 **Style Vector**,使同一说话人能表达多种语气·情感。日语特化变体 **JP-Extra** 展现品质优势。
- **HayaKoe** — 将 Style-Bert-VITS2 JP-Extra **精简为日语专用**,重构为适合 CPU 和服务器运维的形态。

Synthesizer 本身的模型结构沿用 Style-Bert-VITS2,HayaKoe 新增的变更主要集中在外围(ONNX 路径、推理流水线、部署·源简化)。

### Synthesizer

HayaKoe 中对 **VITS 本体(Text Encoder + Duration Predictor + Flow + Decoder)** 的统称。

接收音素序列和 BERT 特征作为输入,生成最终波形的部分。

BERT 存在于 Synthesizer **之外** 并由所有说话人共享。每个说话人不同的是 Synthesizer 的权重(约 300 MB)。

### BERT

Google 于 2018 年发表的基于 Transformer 的预训练语言模型。读取句子并为每个 token 生成上下文嵌入。

在 TTS 中用于 **将句子的语义·上下文信息反映到合成中**。即使是相同的音素序列,BERT 也能根据上下文生成不同的韵律·重音。

HayaKoe 使用日语专用 DeBERTa v2 模型(`ku-nlp/deberta-v2-large-japanese-char-wwm`)。

在 CPU 路径中将此 BERT 量化为 INT8 并以 ONNX 运行。

### Text Encoder

Synthesizer 内部模块。接收音素序列作为输入,输出每个音素对应的 **192 维隐藏向量**。

Transformer encoder 结构,通过 self-attention 使音素参考前后上下文生成合成所需的嵌入。

概念上可以看作 BERT 的缩小版。BERT 在词·句子层面,Text Encoder 在音素层面。

### Duration Predictor (SDP)

预测每个音素 **发音多少帧** 的模块。例如"安"5 帧、"宁"4 帧。

"SDP" 是 **Stochastic Duration Predictor** 的缩写。由于从概率分布中采样而非确定性,即使是同一句话每次调用韵律·速度也会略有不同。

HayaKoe 在原本用途之外还将此模块 **复用于句子边界 pause 预测**。详情请参见 [句子边界 pause — Duration Predictor](./duration-predictor)。

### Flow

Synthesizer 内部模块。**可逆 (invertible) 变换**,正向·反向都可计算的神经网络。

训练时在"正确音频的 latent → 文本嵌入空间"方向对齐,推理时走反向从文本嵌入生成音频 latent。

正式名称是 **Normalizing Flow**。

::: warning Flow 与量化
HayaKoe 不将 Synthesizer 降到 FP16 的主要原因在于 Flow。Flow 的 `rational_quadratic_spline` 运算在 FP16 下因浮点误差导致 assertion error。

Synthesizer INT8 因另外的原因排除 — 以 Conv1d 为中心的结构不适合 PyTorch dynamic quantization 自动应用,static quantization 实现复杂度高。
:::

### Decoder (HiFi-GAN)

Synthesizer 的最后一个模块。接收 Flow 生成的 latent 向量,生成 **实际波形 (waveform)**。

过去作为独立 Vocoder 使用的 HiFi-GAN 结构被 VITS 整合进了模型内部。

**是 VITS 能端到端工作的核心模块**,同时也是 TTS 推理时间中占比最大的部分。

### Style Vector

将说话人的"语气·说话方式"等风格信息压缩为一个向量。

即使是同一说话人,也可以切换"平静"、"开心"、"生气"等风格进行合成。

这是 Style-Bert-VITS2 系列特有的组件,与每个说话人的 safetensors 一起以 `style_vectors.npy` 提供。

HayaKoe 目前为简化 **仅使用 Neutral 风格**。多样风格选择支持计划在后续改进中加入。

### Prosody (韵律)

对语音的 **韵律·节奏·重音·停顿** 的统称。

如果音素回答的是"发什么音",那么 prosody 回答的是"怎么发音"。

TTS 听起来"像机器人"最常见的原因就是 prosody 不够自然。

Bert-VITS2 系列使用 BERT 的主要原因之一就是从句子上下文中获取 prosody 线索。

## 性能·执行术语

### ONNX · ONNX Runtime

**ONNX (Open Neural Network Exchange)** 是可以 **独立于框架保存** 神经网络模型的标准格式。

无论在 PyTorch · TensorFlow 等哪里训练,导出为 ONNX 后都被当作同一个图处理。

**ONNX Runtime** 是实际执行 ONNX 模型的推理引擎。用 C++ 编写,Python 开销小,会分析模型图并预先执行各种优化。

支持 CPU · CUDA · ARM (aarch64) 等多种执行设备。

HayaKoe 的 CPU 路径完全在 ONNX Runtime 上运行。同样的代码能在 x86_64 和 aarch64 上通用运行也得益于此。

### 量化 (Quantization)

通过降低模型权重的数字表示精度来节省内存和计算的技术。

深度学习模型权重通常以以下精度之一存储。

- **FP32** — 32 位浮点。默认。最精确但体积最大。
- **FP16** — 16 位浮点。FP32 的一半大小。
- **INT8** — 8 位整数。约 FP32 的 1/4 大小。也常称为"Q8"。
- **INT4** — 4 位整数。约 FP32 的 1/8 大小。LLM 领域近期活跃使用。

比特数减少后模型文件大小和 RAM 使用量也几乎成比例减少,在某些硬件上运算也会加快。

代价是 **精度下降可能导致输出质量变差。** 能量化到什么程度质量仍然可接受,取决于模型和运算类型。

HayaKoe 选择了 **仅对 BERT 的 MatMul 进行 INT8 动态量化 (Q8 Dynamic Quantization)**,Synthesizer 保持 FP32。详细原因和实测效果请参见 [ONNX 优化](./onnx-optimization)。

### Kernel Launch Overhead

CPU 向 GPU 请求"执行此 kernel"时产生的固定成本。与实际计算时间无关,每次 kernel 调用产生数 us ~ 数十 us。

当单个 kernel 执行繁重计算时此成本被淹没。但 **像 TTS 这样小型 Conv1d 运算重复数百次的情况**,kernel launch overhead 可能占整体时间的相当比重。

CUDA Graph · kernel fusion · torch.compile 等都是减少此成本的技术。

### Eager Mode

PyTorch 的默认执行方式。Python 代码逐行执行,每次单独调用 GPU kernel。

调试方便但每个 kernel 都会累积 Python dispatch 开销和 kernel launch overhead。

`torch.compile` 是通过图级优化消除此开销的替代方案。

### torch.compile

PyTorch 2.0 起提供的 **JIT 编译器**。

首次调用时将模型追踪为图,融合·重编译 kernel,后续调用更快执行。

HayaKoe 在 GPU 路径中使用 `torch.compile`。

首次调用需要编译时间,可通过 `prepare(warmup=True)` 将此成本转移到服务启动阶段。

## 其他

### OpenJTalk

名古屋工业大学开发的开源日语 TTS 前端。

接收日语文本并生成 **音素序列·韵律信息**。日语特有的汉字读法·连音等规则都包含在其中。

HayaKoe 通过 Python 绑定 [pyopenjtalk](./openjtalk-dict) 使用此功能。
