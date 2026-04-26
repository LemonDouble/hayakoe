# 自定义说话人训练

::: warning 部署前请务必确认版权
**如果计划公开学习好的模型或对外发布/上传合成的语音**,请务必先确认原始声音的版权和肖像权。

使用 YouTube·动画·游戏·商业配音等他人声音训练的模型公开发布可能构成版权、肖像权、公开权侵权。

如果考虑部署,请使用本人声音、已获得明确授权的声音或允许发布的免费语料库。

---

**可自由使用至部署的日语语料库示例**

- **つくよみちゃんコーパス** — 标注署名条件下商业·非商业均可使用。
- **あみたろの声素材工房** — 个人·商业使用均可(需确认使用条款)。
- **ずんだもん** 等 VOICEVOX 角色的 ITA/ROHAN 语料库 — 需确认各角色使用条款。

各语料库的署名标注、商业使用可否、二次创作范围各不相同。部署前请务必再次确认官方条款。
:::

HayaKoe 只需准备包含声音的视频文件,即可支持到训练的全流程。

数据准备到部署分为两个工具。

### `dev-tools/preprocess/` — 浏览器 GUI(数据预处理)

上传视频·音频文件后,可以在 GUI 界面上按顺序执行直到生成学习用数据集的各个步骤。

- **音频提取 (Audio Extraction, 自动)** — 从视频中提取音轨。
- **背景音去除 (Source Separation, 自动)** — 使用 audio-separator 库去除 BGM·音效等背景声音,仅保留人声。
- **句子分割 (VAD, 自动)** — 将长录音按静音区间分割为短句。
- **分类 (Classification, `手动`)** — 将分割出的语音按说话人分类,丢弃不可用的部分。
- **转写 (Transcription, 自动)** — 使用语音识别模型(Whisper)为每段语音自动生成对应文本。
- **审核 (Review, `手动`, 可跳过)** — 在浏览器中手动修正转写错误。
- **数据集生成 (Dataset Export, 自动)** — 将数据导出为可训练的格式。

### `dev-tools/cli/` — 交互式 CLI

接收 GUI 制作的数据集,从训练到部署继续推进。

- **预处理 (Preprocessing, 自动)** — 预先计算训练所需的 G2P、BERT 嵌入、风格向量。
- **训练 (Training, 自动)** — 基于预训练数据,用我们准备的数据进行微调。
- **质量报告 (Quality Report, 自动)** — 用训练期间保存的各检查点批量推理语音,确认哪个模型发出最好的声音。
- **部署 (Publish, 自动)** — 从 ONNX(推理优化模型)转换到 HuggingFace / S3 / 本地模型下载全部完成。

两个工具共享同一个 `data/` 目录,因此 GUI 制作的数据集会被 CLI 自动识别。

## 全流程

<PipelineFlow
  :steps="[
    {
      num: '①',
      title: '数据准备',
      tool: 'GUI',
      content: [
        '从想要训练的说话人的视频中制作学习用语音数据集。',
        '从视频中提取音频,去除背景音和音效只留人声,然后按静音区间分割成短句。',
        '将分割出的片段按说话人分类,用 Whisper 自动生成文本,必要时手动修正后导出为可训练的格式。'
      ],
      chips: ['准备视频', '音频提取', '背景音去除', '句子分割', '说话人分类', '转写', '审核', '数据集导出'],
      gpu: '必需'
    },
    {
      num: '②',
      title: '预处理 & 训练',
      tool: 'CLI',
      content: [
        '用准备好的数据集微调日语 TTS 模型。',
        '预先计算训练所需的 G2P(发音转换)、BERT 嵌入、风格向量后,在预训练的 Style-Bert-VITS2 JP-Extra 基础上加载我们的数据针对说话人进行训练。',
        '中间检查点会按一定间隔持续保存,在下一步骤中用于比较。'
      ],
      chips: ['G2P·BERT 计算', '风格嵌入', '微调', '检查点保存'],
      gpu: '必需'
    },
    {
      num: '③',
      title: '质量报告',
      tool: 'CLI',
      content: [
        '训练并非跑得越久越好,过了某个点后音质或说话人音色反而可能崩坏。',
        '因此使用训练期间保存的多个检查点对同一句话进行批量推理,比较哪个时间点的模型发出最好的声音。',
        '结果整理在一张 HTML 中,可在浏览器中直接试听,选择满意的检查点进入下一步。'
      ],
      chips: ['批量推理', 'HTML 报告', '检查点选择']
    },
    {
      num: '④',
      title: '部署',
      tool: 'CLI',
      content: [
        '将选定的检查点转换为 ONNX 格式。',
        'ONNX 是针对 CPU 推理优化的模型格式,即使在没有 GPU 的普通笔记本上也能轻量运行。',
        '转换后的模型可以上传到 HuggingFace·S3 等云存储或本地目录。',
        '上传一次后,hayakoe 包可以直接通过说话人名称加载使用。'
      ],
      chips: ['ONNX 转换', 'HuggingFace', 'S3', '本地']
    }
  ]"
/>

::: warning 数据准备(①)·训练(②)需要 GPU
两个步骤内部都运行模型(背景音去除·Whisper·VITS2),没有 GPU 实际上无法进行。

质量报告(③)·部署(④)不需要 GPU 也能运行。不建议在 CPU 笔记本上进行训练。
:::

## 准备工作

本指南需要直接克隆 hayakoe 仓库来进行。

::: info 假设 Linux 环境
训练工具目前仅保证在 Linux 环境下运行。

Windows 建议在 WSL2 上按照 Linux 指南操作。
:::

### 1. 克隆仓库

```bash
git clone https://github.com/LemonDouble/hayakoe.git
cd hayakoe
```

### 2. 安装 uv

uv 是一个快速的 Python 包·环境管理器。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

详细安装方法请参考 [uv 官方指南](https://docs.astral.sh/uv/getting-started/installation/)。

安装完成后应该能看到版本输出。

```bash
uv --version
```

### 3. 安装开发依赖

之后的命令全部在第 1 步克隆的 **仓库根目录(`hayakoe/`)** 下执行。

```bash
uv sync
```

预处理 GUI 和训练 CLI 所需的库(FastAPI、Whisper、audio-separator、torchaudio 等)会一次性安装完成。

### 4. 安装 GPU (CUDA) PyTorch

数据准备(①)·训练(②)步骤内部运行 ML 模型,因此 NVIDIA GPU 是必需的。

首先确认驱动是否正常安装。

```bash
nvidia-smi
```

正常安装的话可以看到如下输出。

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:06:00.0 Off |                  N/A |
| 53%   33C    P8             38W /  390W |    1468MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

输出第一行右侧的 `CUDA Version` 是您的驱动支持的 **最大 CUDA 版本**(上例中为 13.0)。

选择不超过该版本的 PyTorch 构建进行安装(以下示例基于 CUDA 12.6)。

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

使用其他 CUDA 版本时,将 `cu126` 替换为适合您的版本(`cu118`、`cu121`、`cu124`、`cu128` 等)。

安装验证:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
# 输出 True 即可
```

完成后按照下方 [各步骤详情](#各步骤详情) 的顺序操作。

## 各步骤详情

准备完成后,请按以下顺序逐页操作。

- [① 数据准备](./data-prep)
- [② 预处理 & 训练](./training)
- [③ 质量报告](./quality-check)
- [④ 部署 (HF·S3·本地)](./publish)
- [故障排查](./troubleshooting)
