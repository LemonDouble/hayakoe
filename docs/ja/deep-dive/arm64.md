# arm64 対応 — Raspberry Pi 4B

HayaKoe は x86_64 だけでなく **aarch64（ARM64）Linux でも同一コードで動作** します。

## なぜ可能か

2つの条件が揃っているためです。

- **ONNX Runtime** が aarch64 ビルドを公式提供しています。
- **pyopenjtalk** の自前 fork（[lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)）が aarch64 wheel をビルドしています（→ [OpenJTalk 辞書バンドル](./openjtalk-dict)）。

これにより Apple Silicon（macOS）でも動作すると予想されますが、テスト機材がなく確認できていません。

その代わり Raspberry Pi 4B での動作は確認済みです。

## Raspberry Pi 4B 実測

Raspberry Pi 4B（Linux 6.8、aarch64、ONNX Runtime 1.23.2）での測定結果です。

| テキスト | 推論時間 | 倍速 |
|---|---|---|
| 短い | 3.169 s | 0.3x |
| 中程度 | 13.042 s | 0.3x |
| 長い | 35.119 s | 0.3x |

リアルタイムの約 1/3 の水準で対話用途には不足ですが、ARM ボードで動くということ自体に意義があると考えています。

実測スクリプトおよび再現方法は [ベンチマーク — ラズベリーパイ 4B](/ja/quickstart/benchmark#ラズベリーパイ-4b-ではどうか) を参照してください。
