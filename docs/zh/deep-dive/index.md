# 深入解读

HayaKoe 是将 Style-Bert-VITS2 精简为日语专用,并重新构造为适合 CPU 推理和服务器运维的 TTS 引擎。

本板块整理了 **修改了哪些地方、如何修改的、结果有多大改善**,附带实测数据。

可以根据感兴趣的主题选择性阅读。

## 概览

HayaKoe 相比原版 SBV2 的实测改善如下(详情请参见各页面)。

| 类别 | 原版 SBV2 | HayaKoe | 差异 |
|---|---|---|---|
| CPU 速度 (短句, 约 2 秒) | 1.13 s | 0.68 s | **1.67x 更快** |
| CPU 速度 (中等句子, 约 8 秒) | 3.35 s | 2.44 s | **1.37x 更快** |
| CPU 速度 (长句, 约 38 秒) | 35.33 s | 10.43 s | **3.39x 更快** |
| CPU 内存 | 5,122 MB | 2,346 MB | **减少 54%** |
| GPU VRAM | 3,712 MB | 1,661 MB | **减少 55%** |
| 运行架构 | x86_64 | x86_64 · aarch64 Linux | **支持 ARM 开发板** |

## 各页面的结构

各页面以 **为什么是问题 → 实现 → 改善效果** 的流程为基本结构,根据主题灵活调整。

## 目录

### 全局视角
- [架构概览](./architecture) — TTS 引擎的整体构成

### CPU 推理的实时化
- [ONNX 优化 / 量化](./onnx-optimization) — Q8 BERT + FP32 Synthesizer, arm64 支持
- [句子边界 pause — Duration Predictor](./duration-predictor) — 多句合成时自然停顿的恢复

### GPU 推理的额外优化
- [BERT GPU 保持 & 批量推理](./bert-gpu) — 消除 PCIe 往返和多句批量化

### 运维便利
- [Source 抽象层 (HF · S3 · 本地)](./source-abstraction) — 将说话人供给源统一为 URI
- [OpenJTalk 词典打包](./openjtalk-dict) — 消除首次 import 延迟和网络依赖
- [arm64 支持](./arm64) — Raspberry Pi 4B 实测

### 其他
- [问题反馈 & 许可](./contributing)

::: info 建议阅读顺序
初次阅读建议先浏览 [架构概览](./architecture),然后根据兴趣选择性阅读其他主题。
:::
