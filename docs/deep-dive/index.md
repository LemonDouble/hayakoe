# 깊이 읽기

HayaKoe 는 Style-Bert-VITS2 를 일본어 전용으로 축소하고, CPU 추론과 서버 운영에 실용적인 형태로 재구성한 TTS 엔진입니다.

이 섹션은 **어느 지점을 어떻게 수정했고, 결과가 얼마나 달라졌는지** 를 실측 수치와 함께 정리합니다.

관심 있는 주제부터 선택해 읽을 수 있습니다.

## 한눈에 요약

HayaKoe 가 원본 SBV2 대비 확보한 실측 개선은 다음과 같습니다 (상세는 각 페이지 참고).

| 구분 | 원본 SBV2 | HayaKoe | 차이 |
|---|---|---|---|
| CPU 속도 (짧은 문장, 약 2초) | 1.13 s | 0.68 s | **1.67× 빠름** |
| CPU 속도 (중간 문장, 약 8초) | 3.35 s | 2.44 s | **1.37× 빠름** |
| CPU 속도 (긴 문장, 약 38초) | 35.33 s | 10.43 s | **3.39× 빠름** |
| CPU 메모리 | 5,122 MB | 2,346 MB | **54 % 절감** |
| GPU VRAM | 3,712 MB | 1,661 MB | **55 % 절감** |
| 실행 아키텍처 | x86_64 | x86_64 · aarch64 Linux | **ARM 보드 지원** |

## 각 페이지의 구성

각 페이지는 **왜 문제인가 → 구현 → 개선 효과** 흐름을 기본으로 하되, 주제에 따라 유연하게 구성됩니다.

## 목차

### 큰 그림
- [아키텍처 한눈에](./architecture) — TTS 엔진의 전체 구성

### CPU 추론의 실시간화
- [ONNX 최적화 / 양자화](./onnx-optimization) — Q8 BERT + FP32 Synthesizer, arm64 지원
- [문장 경계 pause — Duration Predictor](./duration-predictor) — 다문장 합성 시 자연스러운 쉼 복원

### GPU 추론의 추가 최적화
- [BERT GPU 유지 & 배치 추론](./bert-gpu) — PCIe 왕복 제거와 다문장 배치화

### 운영 편의
- [Source 추상화 (HF · S3 · 로컬)](./source-abstraction) — 화자 공급처를 URI 로 통합
- [OpenJTalk 사전 번들링](./openjtalk-dict) — 첫 import 지연과 네트워크 의존 제거
- [arm64 지원](./arm64) — Raspberry Pi 4B 실측

### 기타
- [이슈 제보 & 라이선스](./contributing)

::: info 읽는 순서 권장
처음이라면 [아키텍처 한눈에](./architecture) 를 먼저 훑어보고, 이후 관심 주제를 선택적으로 읽는 방식을 권장합니다.
:::
