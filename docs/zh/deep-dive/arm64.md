# arm64 支持 — Raspberry Pi 4B

HayaKoe 不仅在 x86_64 上,**在 aarch64 (ARM64) Linux 上也能用相同代码运行**。

## 为什么可行

因为具备两个条件。

- **ONNX Runtime** 官方提供 aarch64 构建。
- **pyopenjtalk** 的自有 fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) 构建了 aarch64 wheel(→ [OpenJTalk 词典打包](./openjtalk-dict))。

因此预计在 Apple Silicon (macOS) 上也能运行,但由于没有测试设备未能确认。

已确认在 Raspberry Pi 4B 上可以运行。

## Raspberry Pi 4B 实测

在 Raspberry Pi 4B (Linux 6.8, aarch64, ONNX Runtime 1.23.2) 上的测量结果。

| 文本 | 推理耗时 | 倍速 |
|---|---|---|
| 短 | 3.169 s | 0.3x |
| 中 | 13.042 s | 0.3x |
| 长 | 35.119 s | 0.3x |

约为实时的 1/3 水平,不足以用于对话场景,但能在 ARM 开发板上运行本身就有意义。

实测脚本和复现方法请参考 [基准测试 — 树莓派 4B](/zh/quickstart/benchmark#raspberry-pi-4b-实测)。
