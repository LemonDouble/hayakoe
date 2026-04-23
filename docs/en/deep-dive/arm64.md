# arm64 Support — Raspberry Pi 4B

HayaKoe works on **aarch64 (ARM64) Linux with the same code** as x86_64.

## Why It Works

Two conditions are met.

- **ONNX Runtime** officially provides aarch64 builds.
- **pyopenjtalk**'s custom fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) builds aarch64 wheels (-> [OpenJTalk Dictionary Bundling](./openjtalk-dict)).

This suggests it should also work on Apple Silicon (macOS), but we have not been able to verify due to lack of test hardware.

Instead, we have confirmed it works on Raspberry Pi 4B.

## Raspberry Pi 4B Measurements

Results measured on Raspberry Pi 4B (Linux 6.8, aarch64, ONNX Runtime 1.23.2).

| Text | Inference Time | Speed Factor |
|---|---|---|
| Short | 3.169 s | 0.3x |
| Medium | 13.042 s | 0.3x |
| Long | 35.119 s | 0.3x |

At approximately 1/3 real-time, it is not sufficient for interactive use, but we believe there is value in the fact that it runs on an ARM board at all.

For measurement scripts and reproduction instructions, see [Benchmark — Raspberry Pi 4B](/en/quickstart/benchmark#raspberry-pi-4b).
