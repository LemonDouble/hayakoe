# OpenJTalk 字典打包

HayaKoe 的日語 G2P(發音轉換)依賴 [pyopenjtalk](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)。

原版 pyopenjtalk 在首次 import 時從網路下載字典,HayaKoe 使用 **將字典打包在 wheel 內的 fork** 消除了此延遲。

## 問題

原版 pyopenjtalk 在 `import pyopenjtalk` 時如果本地沒有 OpenJTalk 日語字典(`open_jtalk_dic_utf_8-1.11`,約 23 MB),會 **透過 HTTPS 自動下載**。

下載後快取在 `~/.local/share/pyopenjtalk/`,之後不會再次下載。

但 Docker 容器每次都從空檔案系統開始,因此 **每次啟動容器都會重複下載**。

在網路被阻斷的環境中 import 本身會失敗。

## 實作

在自有 fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) 中修改為 **wheel 建置時將字典檔案包含在套件內部**。

- 建置工作流下載並擷取 `open_jtalk_dic_utf_8-1.11.tar.gz`,放置在 `pyopenjtalk/` 套件內部
- 在 `pyproject.toml` 的 `package-data`、`MANIFEST.in` 中明確包含字典

HayaKoe 側只需在 `pyproject.toml` 的依賴中指定 `lemon-pyopenjtalk-prebuilt` 即可完成套用。

安裝後可以確認字典包含在套件內部。

```
site-packages/pyopenjtalk/
  ├─ open_jtalk_dic_utf_8-1.11/   ← 打包的字典
  ├─ openjtalk.cpython-310-x86_64-linux-gnu.so
  └─ __init__.py
```

## 改善效果

- 首次 import 時完全消除網路呼叫
- 在離線·封閉網路環境中安裝後立即可用
- Docker 映像檔建置結果與網路狀態無關保持一致
