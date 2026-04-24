# ① 数据准备

SBV2 训练需要 `(wav 文件, 对应文本)` 对和说话人元数据。

仅凭录音手动制作这些非常耗时,HayaKoe 将此过程封装为基于浏览器的 GUI。

位置在 `dev-tools/preprocess/`。

## 准备

此步骤使用 FFmpeg 从视频中提取音频。

ML 依赖已包含在 [全流程页面的准备工作](./#准备工作) 中的 `uv sync` 里,但 FFmpeg 是系统包,需要单独安装。

### 安装 FFmpeg (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

安装完成后确认版本能正常输出。

```bash
ffmpeg -version
```

## 运行

### 启动预处理工具

```bash
# 在仓库根目录
uv run poe preprocess
```

在浏览器中访问 `http://localhost:8000` 即可看到仪表盘。

::: tip 运行后出现 "address already in use" 错误
表示 8000 端口已被其他程序占用。

错误信息通常如下所示。

```text
ERROR:    [Errno 98] error while attempting to bind on address
('0.0.0.0', 8000): address already in use
```

使用 `--port` 选项指定其他空闲端口重新运行即可(例如:8123)。

```bash
uv run poe preprocess --port 8123
```

此时访问地址也变为 `http://localhost:8123`。
:::

## 基本工作流程

首次进入仪表盘时会看到如下界面。

![HayaKoe 预处理仪表盘主界面 — 顶部工作流引导和各步骤卡片](/images/preprocess/dashboard-zh.png)

顶部的 **WORKFLOW** 区域展示全流程概览,下方以卡片形式展开相同的步骤。

从上到下按顺序操作即可,前一步骤完成后下一步骤的卡片才会激活。

### 1. 注册说话人

注册训练目标说话人的名称。

说话人名称只要是您能识别的标识符就行(例如:`tsukuyomi`)。

### 2. 上传视频

上传作为训练素材的视频。

不仅是视频,mp3·wav·flac 等音频文件也可以直接上传 — 内部由 FFmpeg 统一处理。

上传完成后视频卡片会添加到列表中,点击卡片即可进入该视频的预处理流水线页面。

## 每个视频的 6 步流水线

进入视频详情页面后,顶部显示 6 步进度条,通过下方 **NEXT STEP** 卡片的按钮逐步执行当前步骤。

每步完成后会自动变为"已完成",下一步随即开启。

中途中断再次进入时会从剩余步骤继续。

### 1. 提取

![音频提取步骤 — 顶部进度条第 1 步激活,NEXT STEP 卡片上有"执行提取"按钮](/images/preprocess/step1-extract-zh.png)

从原始数据中提取音频并保存。

提取完成后继续下一步即可。

::: details 内部机制
内部使用 FFmpeg 提取 `extracted.wav` 文件。

如果上传的文件已是 mp3·wav·flac 等音频文件,内容保持不变仅转换格式为 wav。
:::

### 2. 背景音去除

![背景音去除步骤 — 第 2 步激活,NEXT STEP 卡片上有"执行背景音去除"按钮](/images/preprocess/step2-separate-zh.png)

去除 BGM·音效等背景声音,仅保留人声。

所需时间与文件长度成正比,可能需要数分钟,请耐心等待。

::: details 内部机制
使用 `audio-separator` 库分离人声并保存为 `vocals.wav`。
:::

### 3. VAD 分割

![VAD 分割步骤 — 快速设置和详细参数输入,"执行 VAD 分割"按钮](/images/preprocess/step3-vad-zh.png)

将长录音按静音区间分割为短句。

先用默认值运行,如果分割结果不满意,可以调整四个参数后从同一视频重新提取。

- **片段最短时长(秒)** — 短于此值的语音会被丢弃。TTS 训练推荐 1-2 秒。
- **片段最长时长(秒)** — 超过此值的台词会被自动拆分。5-15 秒比较合适。
- **语音检测阈值** — 建议从较低值(0.2~0.3)开始,如果噪声太多则逐渐提高。
- **台词间最短静音(ms)** — 先用默认值开始。如果多个说话人连续说话导致一个片段中混杂多人声音则减小该值,如果单个台词被切得太短则增大该值。

::: details 内部机制
使用 Silero VAD 检测语音活动区间,按上述参数分割后将结果保存为 `vad.json` 和 `segments/unclassified/*.wav`。

重新运行时 `segments/unclassified/` 会被覆盖。
:::

### 4. 分类

![分类步骤 — 片段自动播放,通过说话人编号键或按钮进行分配](/images/preprocess/step4-classify-zh.png)

分割出的片段会逐个自动播放。

按下听到的声音对应的说话人编号键(`1-9`)或按钮进行分配。

噪声·音乐·未注册人的声音用 **丢弃(`D`)** 排除。

| 键 | 操作 |
|---|---|
| `1-9` | 分配给对应编号的说话人 |
| `D` | 丢弃 |
| `R` | 重听 |
| `Z` | 撤销 |

在顶部进度条可以确认剩余片段数量,全部处理完后点击 **分类完成** 按钮进入下一步。

::: details 内部机制
分类结果保存为 `segments/<说话人>/` 目录结构。
:::

### 5. 转写

![转写步骤 — NEXT STEP 卡片上有"执行转写"按钮](/images/preprocess/step5-transcribe-zh.png)

听取各片段的语音并自动转换为日语文本。

转换结果可以在下一步中手动修正,这里只需点击执行按钮。

::: details 内部机制
使用 Whisper 模型转写的结果保存在 `transcription.json` 中。
:::

### 6. 审核

![审核步骤 — 片段列表和日语文本编辑 UI,顶部有"审核完成"按钮](/images/preprocess/step6-review-zh.png)

确认并修正自动转写结果。

如果不懂日语可以先跳过。训练后如果感觉质量不佳,可以返回修正。

- 点击 **播放按钮** 聆听实际发音并与文本对比。
- **点击文本即可直接修改**(`Enter` 保存,`Esc` 取消)。
- 无意义的区间或错误的片段用 `×` 按钮删除。
- 全部确认后点击右上角 **审核完成** 按钮进入下一步。

::: details 内部机制
审核完成标记保存在 `review_done.json` 中。
:::

::: tip 用多个视频收集数据
可以为一个说话人上传多个视频。

每个视频重复上述 6 步积累数据越多训练质量越高。按处理完的数据计算,**最少 10 分钟**,**30 分钟以上通常足够**。

在视频详情页面左上角的 **← 列表** 按钮返回仪表盘上传下一个视频。所有视频审核完成后进入下方数据集生成步骤。
:::

## 数据集生成

所有视频审核完成后,仪表盘上的 **数据集生成** 按钮会被激活。

只需指定 `val_ratio` 一个值即可自动生成训练数据集(默认 0.1)。

::: tip 什么是 val_ratio?
从全部数据中 **不用于训练而是用于中间检验训练效果的比例**。

仅用训练数据的话模型可能只是背下了那些句子,对新句子合成效果不佳。因此刻意留出一部分数据,训练过程中用这些数据合成结果检查是否自然。

默认值 0.1(10%)在大多数情况下足够。
:::

生成的数据集会被 [② 预处理 & 训练](/training/training) CLI 自动识别,可以直接进入下一步。

::: details 内部机制 — 数据集结构和默认设置
生成的目录结构:

```
data/dataset/<speaker>/
├── audio/                          # 所有视频的片段复制到同一位置
│   └── <video_id>_<orig_seg>.wav
├── esd.list                        # <abspath>|<speaker>|JP|<text>
├── train.list                      # esd.list 的 (1 - val_ratio) 随机分割 (seed 42)
├── val.list                        # esd.list 的 val_ratio 随机分割
└── sbv2_data/
    └── config.json                 # SBV2 JP-Extra 默认配置
```

`config.json` 的主要默认值:

- `model_name: "hayakoe_<speaker>"`
- `version: "2.7.0-JP-Extra"`
- `train.epochs: 500`, `batch_size: 2`, `learning_rate: 0.0001`
- `train.eval_interval: 1000`, `log_interval: 200`
- `data.sampling_rate: 44100`, `num_styles: 7`
- `style2id`: Neutral / Happy / Sad / Angry / Fear / Surprise / Disgust

这些值可以在 ② 预处理 & 训练步骤的 `训练设置编辑` 中修改。
:::

::: details 内部机制 — `data/` 根目录完整结构
基于 `--data-dir ./data` 的最终结构:

```
data/
├── speakers.json         # 已注册的说话人列表
├── videos/               # 各视频的预处理工作空间
│   └── <001, 002, ...>/
│       ├── source.<ext>
│       ├── meta.json
│       ├── extracted.wav
│       ├── vocals.wav
│       ├── vad.json
│       ├── segments/
│       ├── classification.json
│       ├── transcription.json
│       └── review_done.json
└── dataset/              # 训练步骤的输入
    └── <speaker>/        # ← CLI 自动识别此路径
```

CLI 会自动列出 `data/dataset/` 下包含 `esd.list` 或 `sbv2_data/esd.list` 的目录。
:::

## 下一步

- 将数据集送入训练:[② 预处理 & 训练](/training/training)
