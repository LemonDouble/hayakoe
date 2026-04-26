# OpenJTalk 词典打包

HayaKoe 的日语 G2P(发音转换)依赖 [pyopenjtalk](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)。

原版 pyopenjtalk 在首次 import 时从网络下载词典,HayaKoe 使用 **将词典打包在 wheel 内的 fork** 消除了此延迟。

## 问题

原版 pyopenjtalk 在 `import pyopenjtalk` 时如果本地没有 OpenJTalk 日语词典(`open_jtalk_dic_utf_8-1.11`,约 23 MB),会 **通过 HTTPS 自动下载**。

下载后缓存在 `~/.local/share/pyopenjtalk/`,之后不会再次下载。

但 Docker 容器每次都从空文件系统开始,因此 **每次启动容器都会重复下载**。

在网络被阻断的环境中 import 本身会失败。

## 实现

在自有 fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) 中修改为 **wheel 构建时将词典文件包含在包内部**。

- 构建工作流下载并提取 `open_jtalk_dic_utf_8-1.11.tar.gz`,放置在 `pyopenjtalk/` 包内部
- 在 `pyproject.toml` 的 `package-data`、`MANIFEST.in` 中明确包含词典

HayaKoe 侧只需在 `pyproject.toml` 的依赖中指定 `lemon-pyopenjtalk-prebuilt` 即可完成应用。

安装后可以确认词典包含在包内部。

```
site-packages/pyopenjtalk/
  ├─ open_jtalk_dic_utf_8-1.11/   ← 打包的词典
  ├─ openjtalk.cpython-310-x86_64-linux-gnu.so
  └─ __init__.py
```

## 改善效果

- 首次 import 时完全消除网络调用
- 在离线·封闭网络环境中安装后立即可用
- Docker 镜像构建结果与网络状态无关保持一致
