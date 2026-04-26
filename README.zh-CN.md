**[한국어](./README.md)** | **[日本語](./README.ja.md)** | **简体中文** | **[繁體中文](./README.zh-TW.md)** | **[English](./README.en.md)**

# HayaKoe

基于 [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) 的高速日语 TTS 库。

**[文档站点](https://lemondouble.github.io/hayakoe/zh-CN/)** · **[试听语音样本](https://lemondouble.github.io/hayakoe/zh-CN/quickstart/#可以随意制作这样的语音)** · **[深入阅读](https://lemondouble.github.io/hayakoe/zh-CN/deep-dive/)**

> **📖 请先阅读文档站点！** 从安装到参数调节、话者训练、服务器部署、架构详解，所有内容都已整理完毕。
>
> [한국어](https://lemondouble.github.io/hayakoe/) · [日本語](https://lemondouble.github.io/hayakoe/ja/) · [简体中文](https://lemondouble.github.io/hayakoe/zh-CN/) · [繁體中文](https://lemondouble.github.io/hayakoe/zh-TW/) · [English](https://lemondouble.github.io/hayakoe/en/)

> Based on [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02 (AGPL-3.0)

## 特点

- **ONNX 优化** — CPU 实时推理（相比 PyTorch 速度提升 1.6 倍，RAM 节省 47%）
- **无需 torch** — CPU 推理时无需 PyTorch 即可运行（轻量安装）
- **简洁 API** — 链式调用一行搞定 `TTS().load(...).prepare()`
- **数据源可插拔** — HuggingFace / S3 / 本地路径混合使用
- **Thread-safe** — 单例部署（FastAPI 等）支持同步/异步两种方式
- **JP-Extra 模型** — Style-Bert-VITS2 JP-Extra (v2.7.0), DeBERTa JP
- **英语→片假名自动转换** — 22 万条外来语词典查找（无额外依赖）

## 安装

### CPU（默认，无需 torch）

<details open>
<summary>pip</summary>

```bash
pip install hayakoe
```
</details>

<details>
<summary>uv</summary>

```bash
uv add hayakoe
```
</details>

<details>
<summary>Poetry</summary>

```bash
poetry add hayakoe
```
</details>

### GPU（需另行安装 PyTorch CUDA）

<details open>
<summary>pip</summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install hayakoe[gpu]
```
</details>

<details>
<summary>uv</summary>

```bash
uv add torch --index https://download.pytorch.org/whl/cu126
uv add hayakoe --extra gpu
```
</details>

<details>
<summary>Poetry</summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
poetry add hayakoe -E gpu
```
</details>

默认模型会从 [HuggingFace](https://huggingface.co/lemondouble/hayakoe) 自动下载。
自行训练的话者可以放在 private HF repo / S3 / 本地路径的任意位置。

## 使用方法

### 基本用法

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

GPU 推理（CUDA 下 `prepare()` 会自动应用 `torch.compile`）：

```python
tts = TTS(device="cuda").load("jvnv-F1-jp").prepare()
tts.speakers["jvnv-F1-jp"].generate("こんにちは").save("output.wav")
```

多话者 + 自定义数据源混合：

```python
tts = (
    TTS(device="cuda")
    .load("jvnv-F1-jp")                                 # 官方 repo
    .load("my-voice", source="hf://me/private-voices")  # private HF
    .load("client-a", source="s3://tts-prod/voices")    # S3
    .load("dev-voice", source="file:///mnt/experiments") # 本地
    .prepare()
)
```

参数调节：

```python
speaker = tts.speakers["jvnv-F1-jp"]
audio = speaker.generate(
    "今日はどんな国に辿り着くのでしょうか。楽しみですね。",
    style="Happy",
    speed=0.9,
    sdp_ratio=0.2,
    noise=0.6,
    noise_w=0.8,
    pitch_scale=1.0,
    intonation_scale=1.0,
    style_weight=1.0,
)
```

### 可用的官方话者

| 名称 | 说明 | 风格 |
|------|------|------|
| `jvnv-F1-jp` | 女性话者 1 | Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust |
| `jvnv-F2-jp` | 女性话者 2 | 〃 |
| `jvnv-M1-jp` | 男性话者 1 | 〃 |
| `jvnv-M2-jp` | 男性话者 2 | 〃 |
| `tsukuyomi_chan` | つくよみちゃん — 动漫风 | Neutral |
| `amitaro_normal` | あみたろ — 普通 | Neutral |
| `amitaro_runrun` | あみたろ — 兴奋 | Neutral |
| `amitaro_yofukashi` | あみたろ — 沉稳 | Neutral |
| `amitaro_punsuka` | あみたろ — 生气 | Neutral |
| `amitaro_sasayaki_a` | あみたろ — 耳语A | Neutral |
| `amitaro_sasayaki_b` | あみたろ — 耳语B | Neutral |

各话者的语音样本可以在 **[文档站点直接试听](https://lemondouble.github.io/hayakoe/zh-CN/quickstart/#可以随意制作这样的语音)**。

### FastAPI 单例部署

`Speaker` 内部使用 `threading.Lock` 对并发调用进行串行化，因此可以将一个
`TTS` 实例放在 `app.state` 中让所有请求共享，完全安全。
同步处理器使用 `generate()` / `stream()`，异步处理器使用 `agenerate()` /
`astream()`（异步版本会自动在独立线程中执行）。

```python
from enum import Enum
from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from hayakoe import TTS, Speaker

SPEAKERS = ["jvnv-F1-jp", "jvnv-M1-jp"]

class SpeakerName(str, Enum):
    F1 = "jvnv-F1-jp"
    M1 = "jvnv-M1-jp"

app = FastAPI()

@app.on_event("startup")
def _load_tts() -> None:
    tts = TTS(device="cuda")
    for name in SPEAKERS:
        tts.load(name)
    tts.prepare(warmup=True)  # 话者 materialize + torch.compile + Triton 预热
    app.state.tts = tts

def get_speaker(name: SpeakerName, request: Request) -> Speaker:
    return request.app.state.tts.speakers[name.value]

@app.post("/tts/{name}")
async def tts_async(text: str, speaker: Speaker = Depends(get_speaker)):
    audio = await speaker.agenerate(text)
    return Response(audio.to_bytes(), media_type="audio/wav")

@app.post("/tts/{name}/stream")
async def tts_stream(text: str, speaker: Speaker = Depends(get_speaker)):
    async def body():
        async for chunk in speaker.astream(text):
            yield chunk.to_bytes()
    return StreamingResponse(body(), media_type="audio/wav")
```

### Docker / 服务器环境

构建阶段无需 GPU，仅将模型下载到缓存中；运行时镜像使用相同的
`cache_dir` 即可立即加载：

```dockerfile
# 构建阶段 — 将模型包含在镜像中（无需 GPU）
RUN python -c "\
from hayakoe import TTS; \
TTS().load('jvnv-F1-jp').pre_download(device='cuda')"

# 运行阶段 — 从缓存立即加载
CMD ["python", "server.py"]
```

缓存根目录默认为 `$CWD/hayakoe_cache`，可通过 `HAYAKOE_CACHE` 环境变量或
`TTS(cache_dir=...)` 覆盖。HuggingFace / S3 / 本地数据源全部存储在同一根目录下。

| 方法 | 作用 | 需要 GPU | 用途 |
|------|------|----------|------|
| `TTS(device=...).load(...)` | 注册话者规格（不下载） | 否 | 声明 |
| `tts.pre_download(device=...)` | 仅下载到缓存 | 否 | Docker 构建, CI |
| `tts.prepare()` | 加载模型 +（CUDA 时）torch.compile | 可选 | 运行时初始化 |

### Private / 内部数据源

安装 `hayakoe[s3]` extra 后即可使用 `s3://` 协议。
S3 兼容端点（MinIO, R2 等）通过 `AWS_ENDPOINT_URL_S3` 环境变量指定。

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",          # BERT 也从内部镜像获取
        hf_token="hf_...",                        # 用于 private HF
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

## 用户词典

可以为 pyopenjtalk 不认识的专有名词注册读音。

```python
tts = TTS().load("jvnv-F1-jp").prepare()

# 仅注册读音（重音为平板型）
tts.add_word(surface="担々麺", reading="タンタンメン")

# 同时指定重音位置（在第 3 个音拍处下降）
tts.add_word(surface="担々麺", reading="タンタンメン", accent=3)
```

## 架构

```
TTS (引擎)
├── BERT DeBERTa Q8 (ONNX)  ← 自动下载
│
├── speakers["jvnv-F1-jp"]  → Synthesizer ONNX + style vectors
├── speakers["jvnv-F2-jp"]  → ...
└── ...
```

- **CPU**: ONNX Runtime (BERT Q8 + Synthesizer FP32)
- **GPU**: PyTorch FP32 + `torch.compile` — `prepare()` 自动应用

## 开发工具 (Dev Tools)

从模型训练到部署准备的交互式 CLI。

```bash
uv run poe cli
```

| 阶段 | 功能 | 说明 |
|------|------|------|
| ① 训练 | 数据预处理 + 模型训练 | 使用语音数据训练 TTS 模型 |
| ② 质量报告 | 按检查点比较语音 | 比较试听已训练检查点的语音（HTML） |
| ③ ONNX 导出 | 转换为 CPU 推理模型 | 在无 GPU 环境下推理时需要此步骤。如果仅使用 GPU 推理则可跳过 |
| ④ 基准测试 | 测量 CPU/GPU 推理速度 | 测量相对于实时的倍速（HTML 报告） |
| ⑤ 发布 (Publish) | 上传模型至 HF / S3 / 本地 | 将训练好的话者上传到 private repo 或存储桶，以便通过 `TTS(...).load(source=...)` 获取 |

## 许可证

- 代码: AGPL-3.0（原版 Style-Bert-VITS2）
- JVNV 语音模型: CC BY-SA 4.0 ([JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus))
- 预训练模型 (DeBERTa): MIT
- 英语→片假名词典数据: GPL-3.0 ([loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo))

## Credits

- [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) by litagin02
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) by Fish Audio
- [JVNV Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) — 日语情感语音语料库
- [loanwords_gairaigo](https://github.com/jamesohortle/loanwords_gairaigo) by James O'Hortle — 英语→片假名词典数据
