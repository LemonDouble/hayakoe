# 퀵스타트

"일단 돌아가는지 확인해보고 싶다" 는 분을 위한 가이드입니다.

기성 공식 화자로 첫 음성까지 약 10분, 벤치마크까지 약 15분이면 끝납니다.

## 읽는 순서

1. [설치 — CPU vs GPU](./install) — 내 환경에 맞는 설치 프로파일 고르기
2. [첫 음성 만들기](./first-voice) — 공식 화자로 wav 저장까지
3. [속도·운율 조절](./parameters) — 속도/피치/운율 파라미터 이해하기
4. [커스텀 단어 등록](./custom-words) — 잘못 읽히는 단어를 직접 고정하기
5. [문장 단위 스트리밍](./streaming) — 긴 텍스트의 첫 음성을 빨리 내보내기
6. [내 머신에서 벤치마크](./benchmark) — 실제 내 하드웨어에서 얼마나 빠른지 측정

## 이 섹션이 끝나면 할 수 있는 것

- 미리 준비된 화자 4명 (`jvnv-F1/F2/M1/M2-jp`) 을 자유롭게 불러 쓰기
- 속도·피치·운율 파라미터 조절
- 내 하드웨어에서 "1초 음성을 만드는 데 몇 초 걸리는지" 직접 측정

## 이런 음성을 마음껏 만들 수 있어요

퀵스타트가 끝나면, 아래 네 화자가 내 손에 쥐어집니다.

같은 문장 ("こんにちは、はじめまして。") 을 네 명이 각자 말한 샘플입니다.

<SpeakerSample name="jvnv-F1-jp  —  여성 화자 1" src="/hayakoe/samples/hello_jvnv-F1-jp.wav" />
<SpeakerSample name="jvnv-F2-jp  —  여성 화자 2" src="/hayakoe/samples/hello_jvnv-F2-jp.wav" />
<SpeakerSample name="jvnv-M1-jp  —  남성 화자 1" src="/hayakoe/samples/hello_jvnv-M1-jp.wav" />
<SpeakerSample name="jvnv-M2-jp  —  남성 화자 2" src="/hayakoe/samples/hello_jvnv-M2-jp.wav" />

::: info 자체 화자 학습은 준비 중
녹음본을 준비해서 자체 화자를 학습시키는 가이드는 따로 정리 중입니다.

준비가 끝나는 대로 [자체 화자 학습](/training/) 섹션에 올려두겠습니다.
:::
