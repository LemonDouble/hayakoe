# Source 抽象层 (HF · S3 · 本地)

无论说话人模型和 BERT 文件位于何处,**只需更改 URI 即可用相同 API 加载** 的抽象层。

## 为什么需要

说话人模型的加载源因场景而异。

- 公开的默认说话人在 **HuggingFace 仓库** (`hf://lemondouble/hayakoe`)
- 自行训练的说话人在 **private HF 仓库 · S3 · 本地目录** 等

如果按源逐一分支处理下载代码,引擎本体会变得臃肿,缓存路径也会重复。

## 实现

### Source 接口

所有源实现 **"按 prefix 将文件下载到本地缓存并返回路径"** 的公共接口。

```python
class Source(Protocol):
    def fetch(self, prefix: str) -> Path:
        """将 prefix/ 下所有文件下载到缓存并返回本地路径。"""
        ...

    def upload(self, prefix: str, local_dir: Path) -> None:
        """将 local_dir 内容上传到 prefix/ 下(部署用)。"""
        ...
```

`fetch()` 在模型加载时使用,`upload()` 在 CLI 的 `publish`(模型部署)时使用。

### 实现类

| URI 方案 | 实现 | 行为 |
|---|---|---|
| `hf://user/repo[@revision]` | `HFSource` | 通过 `huggingface_hub.snapshot_download()` 下载。可通过 `HF_TOKEN` 环境变量或 `hf_token` 参数访问 private 仓库 |
| `s3://bucket/prefix` | `S3Source` | 基于 `boto3`。通过 `AWS_ENDPOINT_URL_S3` 环境变量支持 S3 兼容端点(R2 · MinIO 等) |
| `file:///abs/path` 或 `/abs/path` | `LocalSource` | 直接使用本地目录。无需下载 |

### URI 自动路由

向 `TTS().load()` 只传 URI,即可自动选择对应方案的 Source。

```python
# HuggingFace (默认)
tts.load("jvnv-F1-jp")

# HuggingFace — private 仓库
tts.load("jvnv-F1-jp", source="hf://myorg/my-voices")

# S3
tts.load("jvnv-F1-jp", source="s3://my-bucket/voices")

# 本地
tts.load("jvnv-F1-jp", source="/data/models")
```

HuggingFace 网页 URL (`https://huggingface.co/user/repo`) 也会自动规范化为 `hf://` 格式接受。

### 缓存

所有源存储在同一缓存根目录下。

缓存路径通过 `HAYAKOE_CACHE` 环境变量指定,未指定时默认为 `$CWD/hayakoe_cache`。

缓存策略很简单 — 有文件就复用,没有就重新下载。

### BERT 源分离

说话人模型和 BERT 模型的源可以 **分别指定**。

```python
TTS(
    device="cpu",
    bert_source="hf://lemondouble/hayakoe",  # BERT 从官方仓库
).load(
    "custom-speaker",
    source="/data/my-models",                 # 说话人从本地
).prepare()
```

默认值均为 `hf://lemondouble/hayakoe`。

## 改善效果

- 引擎本体中消除了按存储类型的分支代码。
- 要添加新存储只需编写一个实现 `Source` 协议的类。
- CLI 的 `publish` 命令也使用同一抽象层的反向操作(`upload`)。
