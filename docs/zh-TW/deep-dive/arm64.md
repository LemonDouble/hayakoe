# arm64 支援 — Raspberry Pi 4B

HayaKoe 不僅在 x86_64 上,**在 aarch64 (ARM64) Linux 上也能用相同程式碼執行**。

## 為什麼可行

因為具備兩個條件。

- **ONNX Runtime** 官方提供 aarch64 建置。
- **pyopenjtalk** 的自有 fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) 建置了 aarch64 wheel(→ [OpenJTalk 字典打包](./openjtalk-dict))。

因此預計在 Apple Silicon (macOS) 上也能執行,但由於沒有測試裝置未能確認。

已確認在 Raspberry Pi 4B 上可以執行。

## Raspberry Pi 4B 實測

在 Raspberry Pi 4B (Linux 6.8, aarch64, ONNX Runtime 1.23.2) 上的測量結果。

| 文本 | 推論耗時 | 倍速 |
|---|---|---|
| 短 | 3.169 s | 0.3x |
| 中 | 13.042 s | 0.3x |
| 長 | 35.119 s | 0.3x |

約為即時的 1/3 水平,不足以用於對話情境,但能在 ARM 開發板上執行本身就有意義。

實測腳本和重現方法請參考 [基準測試 — 樹莓派 4B](/zh-TW/quickstart/benchmark#raspberry-pi-4b-實測)。
