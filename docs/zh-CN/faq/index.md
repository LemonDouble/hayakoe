# FAQ

汇集了常见的高级设置项目。

## 更改缓存路径

如果不满意默认缓存路径(`./hayakoe_cache/`),有两种方法。

```bash
# 环境变量
export HAYAKOE_CACHE=/var/cache/hayakoe
```

```python
# 在代码中直接指定
tts = TTS(cache_dir="/var/cache/hayakoe")
```

HuggingFace · S3 · 本地源全部存储在同一根目录下。

## 从 Private HuggingFace 或 S3 获取模型

使用 private HF repo 的说话人或从 S3 桶中下载模型时需要指定源 URI。

如果要使用 S3 源,请先安装 extras。

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",
        hf_token="hf_...",                     # private HF repo 用
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

S3 兼容端点(MinIO, Cloudflare R2 等)通过 `AWS_ENDPOINT_URL_S3` 环境变量指定。

## 加载多个说话人时内存增加多少

BERT 由所有说话人共享,每个说话人增加的仅是轻量得多的 synthesizer 部分。

出于好奇在本地实际运行了基准脚本,数字可能因硬件·OS·torch 版本·ORT 构建而异,请 **仅看增长趋势而非绝对值**。

::: info 测量环境
- GPU — NVIDIA RTX 3090 (24 GB), Driver 580.126.09
- 文本 — 日语两句(含句子边界,约 50 字)
- 说话人 — `jvnv-F1-jp`, `jvnv-F2-jp`, `jvnv-M1-jp`, `jvnv-M2-jp`
- 每个场景在独立 Python 进程中运行(防止堆污染)
:::

### 按说话人数量的内存(仅加载状态)

| 说话人数 | CPU (ONNX) RAM | GPU (PyTorch) RAM | GPU VRAM |
| :------ | -------------: | ----------------: | -------: |
| 1 人    | ≈ 1.7 GB       | ≈ 1.3 GB          | ≈ 1.8 GB |
| 4 人    | ≈ 2.8 GB       | ≈ 1.5 GB          | ≈ 2.6 GB |

新增 3 个说话人后增加的量除以 3,每个说话人大约如下。

- **CPU RAM** — 约 +360 MB / 说话人
- **GPU VRAM** — 约 +280 MB / 说话人

### 4 人同时运行时

实际服务中多个说话人可能同时运行,因此分别测量了 **顺序 4 次** 和 **4 个线程同时**(合成中峰值基准)。

| 场景        | CPU RAM peak | GPU RAM peak | GPU VRAM peak |
| :---------- | -----------: | -----------: | ------------: |
| 1 说话人合成 | ≈ 2.0 GB     | ≈ 2.3 GB     | ≈ 1.7 GB      |
| 4 说话人顺序 | ≈ 3.2 GB     | ≈ 2.1 GB     | ≈ 2.6 GB      |
| 4 说话人同时 | ≈ 3.2 GB     | ≈ 2.2 GB     | ≈ 2.8 GB      |

即使同时运行内存也不会变成 4 倍。

CPU 侧 ORT 内部已经在做并行化所以"顺序 vs 同时"差异几乎没有,GPU VRAM 同时运行也只多约 +200 MB。

### 自行复现

脚本在仓库的 `docs/benchmarks/memory/` 下。

```bash
# 单一场景
python docs/benchmarks/memory/run_one.py --device cpu --scenario idle4

# 全部 10 场景 (CPU/GPU × idle1/idle4/gen1/seq4/conc4) 在独立进程中
bash docs/benchmarks/memory/run_all.sh
```

- `run_one.py` 运行一个场景并打印一行 JSON。
- `run_all.sh` 将所有场景在独立 Python 进程中运行,结果汇总到脚本旁的 `results_<timestamp>.jsonl`。
- RAM 通过 `psutil` 每 50 ms 轮询 RSS 捕获峰值,VRAM 直接取 `torch.cuda.max_memory_allocated()` 值。
- `gen*` 场景在预热后调用 `torch.cuda.reset_peak_memory_stats()`,将 torch.compile 冷启动排除在峰值之外。

如果需要测量,在自己的环境中运行一次比较数字最为准确。
