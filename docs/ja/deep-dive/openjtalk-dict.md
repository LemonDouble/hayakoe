# OpenJTalk 辞書バンドル

HayaKoe の日本語 G2P（発音変換）は [pyopenjtalk](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt) に依存しています。

オリジナルの pyopenjtalk は初回 import 時にネットワークから辞書をダウンロードしますが、HayaKoe は **辞書を wheel 内にバンドルした fork** を使用してこの遅延を除去しました。

## 問題点

オリジナルの pyopenjtalk は `import pyopenjtalk` 時点で OpenJTalk 日本語辞書（`open_jtalk_dic_utf_8-1.11`、約 23 MB）がローカルになければ **HTTPS で自動ダウンロード** します。

ダウンロード後は `~/.local/share/pyopenjtalk/` にキャッシュされ以降は再ダウンロードしません。

しかし Docker コンテナは毎回空のファイルシステムで起動するため **コンテナを立ち上げるたびにダウンロードが繰り返されます**。

ネットワークが遮断された環境では import 自体が失敗します。

## 実装

自前の fork（[lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)）で **wheel ビルド時に辞書ファイルをパッケージ内部に含める** よう修正しました。

- ビルドワークフローが `open_jtalk_dic_utf_8-1.11.tar.gz` をダウンロード・展開して `pyopenjtalk/` パッケージ内に配置
- `pyproject.toml` の `package-data`、`MANIFEST.in` に辞書の包含を明示

HayaKoe 側は `pyproject.toml` の依存関係を `lemon-pyopenjtalk-prebuilt` に指定するだけで適用完了です。

インストール後にパッケージ内部に辞書が含まれていることを確認できます。

```
site-packages/pyopenjtalk/
  ├─ open_jtalk_dic_utf_8-1.11/   ← バンドルされた辞書
  ├─ openjtalk.cpython-310-x86_64-linux-gnu.so
  └─ __init__.py
```

## 改善効果

- 初回 import 時のネットワーク呼び出しを完全に除去
- オフライン・閉域環境でインストール直後に即座に動作
- Docker イメージビルド結果がネットワーク状態に関係なく同一
