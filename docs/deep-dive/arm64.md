# arm64 지원 — Raspberry Pi 4B

HayaKoe 는 x86_64 뿐 아니라 **aarch64 (ARM64) Linux 에서도 동일한 코드로 동작** 합니다.

## 왜 가능한가

두 가지 조건이 갖춰져 있기 때문입니다.

- **ONNX Runtime** 이 aarch64 빌드를 공식 제공합니다.
- **pyopenjtalk** 의 자체 fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) 가 aarch64 wheel 을 빌드합니다 (→ [OpenJTalk 사전 번들링](./openjtalk-dict)).

이로 인해 Apple Silicon (macOS) 에서도 동작할 것으로 예상하지만, 테스트 장비가 없어 확인하지 못했습니다.

대신 Raspberry Pi 4B 에서 동작하는 것은 확인했습니다.

## Raspberry Pi 4B 실측

Raspberry Pi 4B (Linux 6.8, aarch64, ONNX Runtime 1.23.2) 에서 측정한 결과입니다.

| 텍스트 | 추론 시간 | 배속 |
|---|---|---|
| 짧음 | 3.169 s | 0.3× |
| 중간 | 13.042 s | 0.3× |
| 김 | 35.119 s | 0.3× |

실시간의 약 1/3 수준으로 대화형 용도로는 부족하지만, ARM 보드에서 돌아간다는 것 자체에 의미가 있다고 생각합니다.

실측 스크립트 및 재현 방법은 [벤치마크 — 라즈베리파이 4B](/quickstart/benchmark#라즈베리파이-4b-에서는-어떨까) 를 참고하세요.
