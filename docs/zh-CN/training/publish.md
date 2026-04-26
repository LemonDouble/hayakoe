# ④ 部署 (HF · S3 · 本地)

训练完成后 `<dataset>/exports/<model_name>/` 下会集中最终模型文件。

将此文件夹上传到 **HuggingFace Hub / S3 / 本地文件夹** 中的某一个,然后在其他机器上用 `TTS().load("我的名字")` 一行代码即可重新下载使用 — 这就是 `cli publish` 的作用。

它将手动操作 HF CLI·S3 CLI、记忆仓库结构、上传验证等过程整合为一个交互式流程。

## 运行

```bash
uv run poe cli
```

在主菜单中选择 **发布说话人** 后会按以下顺序询问。

1. 要部署的数据集(或外部文件夹)
2. 后端 — CPU / GPU / CPU + GPU
3. 检查点
4. 说话人名称
5. 目标位置 + 凭证
6. 摘要面板 → 确认
7. 自动上传 → 实际合成验证

各步骤在下方说明。

## 1. 选择部署对象

有两种对象。

- **训练 dataset** — `data/dataset/<name>/exports/<model>/` 中有最终文件的数据集会自动列出。
- **从其他文件夹直接选择** — 训练在其他地方完成,只有 HayaKoe 格式文件夹时,直接输入路径。

::: details 从外部文件夹导入时需要的文件
```
<my-folder>/
├── config.json                # 必需
├── style_vectors.npy          # 必需
├── *.safetensors              # 必需 (至少一个)
├── synthesizer.onnx           # 可选 (有的话复用)
└── duration_predictor.onnx    # 可选 (有的话复用)
```
:::

## 2. 选择后端

```
CPU (ONNX)        — 无 GPU 的服务器/本地用
GPU (PyTorch)     — 最低延迟
CPU + GPU (推荐)  — 同时部署到两种环境
```

选择 `CPU + GPU` 时两种后端的文件会 **一起** 上传到同一仓库。运行时用 `TTS(device="cpu")` 创建时只拉取 ONNX 侧,用 `TTS(device="cuda")` 创建时只拉取 PyTorch 侧。

**只需上传一次即可在两种环境下以相同名称复用**,所以没有特殊原因的话请选这个选项。

两种后端的差异在 [后端选择](/zh-CN/deploy/backend) 中详细说明。

## 3. 检查点与说话人名称

- 检查点只有 1 个时自动选择,多个时手动选择(通常是在 [③ 质量报告](/zh-CN/training/quality-check) 中选定的)。
- **说话人名称** 是运行时用 `TTS().load("我的名字")` 时使用的标识符。建议简洁的小写加连字符风格(例如:`tsukuyomi`)。

## 4. 选择目标位置

有三个选项。只需首次输入一次凭证,之后会以 `chmod 600` 保存到 `dev-tools/.env`,下次会跳过提示。

### HuggingFace Hub

输入仓库路径(`org/repo` 或 `hf://org/repo`)和 **write 权限 token**。可以用 `@<revision>` 指定分支/标签。

::: details 支持的 URL 格式 & 保存的环境变量
允许的 URL 格式:

- `lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices@main`
- `https://huggingface.co/lemondouble/hayakoe-voices`
- `https://huggingface.co/lemondouble/hayakoe-voices/tree/dev`

保存的 `.env` 示例:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # write 权限 HuggingFace 访问 token
HAYAKOE_HF_REPO=lemondouble/hayakoe-voices       # 说话人文件上传的 HF 仓库 (org/repo 格式)
```
:::

### AWS S3

输入桶名称(+ 可选 prefix)和 AWS 凭证(`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`)。端点 URL 留空即可。

### S3 兼容存储 (R2, MinIO 等)

使用 Cloudflare R2、MinIO、Wasabi 等 S3 兼容存储时需要 **同时输入端点 URL**。

- Cloudflare R2 — `https://<account>.r2.cloudflarestorage.com`
- MinIO — `http://<host>:9000`

桶·凭证输入与 AWS S3 相同。

::: details 保存的环境变量示例
**AWS S3**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                # 说话人文件上传的 S3 桶名称
HAYAKOE_S3_PREFIX=hayakoe-voices               # 桶内路径 prefix (留空则为桶根目录)
AWS_ACCESS_KEY_ID=<your_access_key_here>       # AWS 访问密钥 ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>   # AWS 密钥
AWS_REGION=ap-northeast-2                      # S3 区域 (示例为首尔)
# AWS_ENDPOINT_URL_S3 留空 (AWS S3 自动确定)
```

**S3 兼容 (Cloudflare R2)**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                                 # 上传的 R2 桶名称
HAYAKOE_S3_PREFIX=hayakoe-voices                                # 桶内路径 prefix (留空则为桶根目录)
AWS_ACCESS_KEY_ID=<your_access_key_here>                        # R2 仪表盘中颁发的 Access Key ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>                    # R2 Secret Access Key
AWS_REGION=auto                                                 # R2 始终为 auto
AWS_ENDPOINT_URL_S3=https://abc123def.r2.cloudflarestorage.com  # R2 端点 (每个账户唯一)
```
:::

### 本地目录

不通过网络上传,仅复制到本地路径。适用于将文件放在 NFS 共享卷或内部网络驱动器上供团队共用的场景。运行时通过 `file:///...` URI 访问。

::: details 保存的环境变量示例
```env
HAYAKOE_LOCAL_PATH=/srv/hayakoe-voices   # 说话人文件复制到的本地目录路径
```
:::

## 5. 仓库结构

以 `CPU + GPU` 部署时仓库中会同时包含 ONNX 文件夹和 PyTorch 文件夹。可以在同一仓库中放置多个说话人一起管理(`speakers/voice-a/`、`speakers/voice-b/`、...)。

::: details 内部结构
```
<repo-root>/
├── pytorch/
│   └── speakers/
│       └── <speaker-name>/
│           ├── config.json
│           ├── style_vectors.npy
│           └── *.safetensors
└── onnx/
    └── speakers/
        └── <speaker-name>/
            ├── config.json
            ├── style_vectors.npy
            ├── synthesizer.onnx
            └── duration_predictor.onnx
```

BERT 模型也会一起上传到 `pytorch/bert/` 和 `onnx/bert/` 下的共享位置。运行时使用相同的缓存规则下载说话人文件和公共 BERT。
:::

## 6. ONNX export (自动)

选择 CPU 后端(`CPU (ONNX)` · `CPU + GPU`)时,在上传前会自动将 PyTorch 检查点转换为 ONNX。无需单独的 `cli export` 命令。

转换结果缓存在 `<dataset>/onnx/`,同一检查点再次 publish 时会复用。想强制重新转换的话删除此文件夹后重新 publish。

::: details 内部机制 — 转换的模型和方式
通过 `dev-tools/cli/export/exporter.py` 以 opset 17 导出说话人专属的两个模型。

#### 转换对象 — 说话人专属的两个模型

**Synthesizer (VITS 解码器)**

接收音素序列 + BERT 嵌入 + 风格向量作为输入,生成实际波形(waveform)的核心模型。由于每个说话人训练不同,部署对象的大部分都是这个模型。

- 函数: `export_synthesizer`
- 输出: `synthesizer.onnx` (+ 可能的 `synthesizer.onnx.data`)

**Duration Predictor**

预测每个音素应该发音多长时间。如果预测不准确,句子边界的 pause·节奏处理会不自然。

- 函数: `export_duration_predictor`
- 输出: `duration_predictor.onnx`

#### `synthesizer.onnx.data` 是什么?

ONNX 内部使用 Protobuf 序列化,Protobuf 有 **单条消息 2GB 限制**。当 Synthesizer 的权重超过此阈值时,图结构保留在 `.onnx` 中而 **大型张量外置到旁边的 `.data` 文件**。

- 两个文件 **必须始终在同一文件夹中**(禁止分离移动)
- 根据模型大小可能完全不生成 `.data`
- 运行时只指定 `.onnx` 加载也会自动读取同文件夹的 `.data`

#### BERT 不按说话人制作而是公用

BERT (DeBERTa) 是与说话人无关的日语语言模型。所有说话人共用的 **Q8 量化 ONNX** (`bert_q8.onnx`) 从 HuggingFace 的公用位置下载使用,publish 步骤不会为每个说话人重新转换。

- 得益于 Q8 量化,CPU 上也能以接近实时的延迟提取嵌入
- 所有说话人共享同一个 BERT,无需在每个仓库中重复存储

也就是说,此步骤实际转换的对象 **仅有说话人专属的 Synthesizer + Duration Predictor 两个**。

#### 追踪耗时的原因

ONNX export 是"实际让模型跑一遍,同时记录计算图"的 **追踪** 方式。Synthesizer 结构复杂,可能需要数十秒到数分钟。

由于同一检查点可能会以不同名称·不同目标多次 publish,转换结果会缓存在 `<dataset>/onnx/` 中复用。

#### 用脚本直接 export

两个 export 函数是公开的,也可以用脚本直接调用。但 publish 流程会自动完成同样的事,除非有特殊原因否则建议使用 publish。直接调用路径将来可能会变更。
:::

## 7. 覆盖确认

如果目标位置已存在同名的 `speakers/<speaker-name>/`,会 **先询问是否覆盖**。确认后仅清理该说话人目录并重新上传 — 同仓库中的其他说话人不受影响。

README 也遵循同样原则。如果仓库根目录没有 README 则自动生成四语(ko/en/ja/zh)模板一起上传,已存在则显示 diff 后询问是否覆盖。

## 8. 上传后自动验证

上传完成后会自动确认 **用上传的文件是否真的能合成**。

如果选择了 CPU + GPU 则分别验证两种后端,结果 wav 保存在 `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` 中可以直接播放确认。

::: tip 验证成功意味着
"用仓库中上传的文件真的能合成了"。

此验证通过后可以保证在其他机器上用 `TTS().load(<speaker>, source="hf://...")` 等方式直接取出使用。
:::

::: details 内部机制 — 验证流程
1. 用选定的后端创建 `TTS(device=...)` 实例
2. 用刚上传的名称 `load(<speaker>)` → `prepare()`
3. 合成固定文本 `"テスト音声です。"`
4. 将结果 wav 保存到 `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav`

GPU 验证前会重置全局 BERT / dynamo / CUDA 缓存以避免相互影响。
:::

## 在运行时下载使用

上传完成的说话人可以在其他机器·容器上这样加载。

```python
from hayakoe import TTS

# 从 HF
tts = TTS(device="cpu").load("tsukuyomi", source="hf://me/my-voices").prepare()

# 从 S3
tts = TTS(device="cuda").load("tsukuyomi", source="s3://my-bucket/hayakoe-voices").prepare()

# 从本地
tts = TTS(device="cpu").load("tsukuyomi", source="file:///srv/voices").prepare()

# 合成
audio = tts.speakers["tsukuyomi"].generate("こんにちは。")
```

只需更改 `device` 即可自动使用 CPU(ONNX) / GPU(PyTorch) 后端 — 这是因为 publish 步骤选择了 `CPU + GPU`,两侧文件都在仓库中。

但运行时侧也需要安装对应后端的依赖。使用 `device="cuda"` 时实际运行的机器上需要安装 **PyTorch CUDA 构建**,`device="cpu"` 仅需基本安装。详情请参考 [安装 — CPU vs GPU](/zh-CN/quickstart/install)。

## 下一步

- 下载使用:[服务器部署](/zh-CN/deploy/)
- 运行时选择哪个后端:[后端选择](/zh-CN/deploy/backend)
