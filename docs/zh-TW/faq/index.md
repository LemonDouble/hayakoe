# FAQ

彙集了常見的進階設定項目。

## 變更快取路徑

如果不滿意預設快取路徑(`./hayakoe_cache/`),有兩種方法。

```bash
# 環境變數
export HAYAKOE_CACHE=/var/cache/hayakoe
```

```python
# 在程式碼中直接指定
tts = TTS(cache_dir="/var/cache/hayakoe")
```

HuggingFace · S3 · 本地源全部儲存在同一根目錄下。

## 從 Private HuggingFace 或 S3 取得模型

使用 private HF repo 的說話人或從 S3 bucket 中下載模型時需要指定源 URI。

如果要使用 S3 源,請先安裝 extras。

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

S3 相容端點(MinIO, Cloudflare R2 等)透過 `AWS_ENDPOINT_URL_S3` 環境變數指定。

## 載入多個說話人時記憶體增加多少

BERT 由所有說話人共享,每個說話人增加的僅是輕量得多的 synthesizer 部分。

出於好奇在本地實際執行了基準腳本,數字可能因硬體·OS·torch 版本·ORT 建置而異,請 **僅看增長趨勢而非絕對值**。

::: info 測量環境
- GPU — NVIDIA RTX 3090 (24 GB), Driver 580.126.09
- 文本 — 日語兩句(含句子邊界,約 50 字)
- 說話人 — `jvnv-F1-jp`, `jvnv-F2-jp`, `jvnv-M1-jp`, `jvnv-M2-jp`
- 每個場景在獨立 Python 行程中執行(防止 heap 污染)
:::

### 按說話人數量的記憶體(僅載入狀態)

| 說話人數 | CPU (ONNX) RAM | GPU (PyTorch) RAM | GPU VRAM |
| :------ | -------------: | ----------------: | -------: |
| 1 人    | ≈ 1.7 GB       | ≈ 1.3 GB          | ≈ 1.8 GB |
| 4 人    | ≈ 2.8 GB       | ≈ 1.5 GB          | ≈ 2.6 GB |

新增 3 個說話人後增加的量除以 3,每個說話人大約如下。

- **CPU RAM** — 約 +360 MB / 說話人
- **GPU VRAM** — 約 +280 MB / 說話人

### 4 人同時執行時

實際服務中多個說話人可能同時執行,因此分別測量了 **順序 4 次** 和 **4 個執行緒同時**(合成中峰值基準)。

| 場景        | CPU RAM peak | GPU RAM peak | GPU VRAM peak |
| :---------- | -----------: | -----------: | ------------: |
| 1 說話人合成 | ≈ 2.0 GB     | ≈ 2.3 GB     | ≈ 1.7 GB      |
| 4 說話人順序 | ≈ 3.2 GB     | ≈ 2.1 GB     | ≈ 2.6 GB      |
| 4 說話人同時 | ≈ 3.2 GB     | ≈ 2.2 GB     | ≈ 2.8 GB      |

即使同時執行記憶體也不會變成 4 倍。

CPU 側 ORT 內部已經在做平行化所以「順序 vs 同時」差異幾乎沒有,GPU VRAM 同時執行也只多約 +200 MB。

### 自行重現

腳本在儲存庫的 `docs/benchmarks/memory/` 下。

```bash
# 單一場景
python docs/benchmarks/memory/run_one.py --device cpu --scenario idle4

# 全部 10 場景 (CPU/GPU × idle1/idle4/gen1/seq4/conc4) 在獨立行程中
bash docs/benchmarks/memory/run_all.sh
```

- `run_one.py` 執行一個場景並輸出一行 JSON。
- `run_all.sh` 將所有場景在獨立 Python 行程中執行,結果彙總到腳本旁的 `results_<timestamp>.jsonl`。
- RAM 透過 `psutil` 每 50 ms 輪詢 RSS 捕獲峰值,VRAM 直接取 `torch.cuda.max_memory_allocated()` 值。
- `gen*` 場景在預熱後呼叫 `torch.cuda.reset_peak_memory_stats()`,將 torch.compile 冷啟動排除在峰值之外。

如果需要測量,在自己的環境中執行一次比較數字最為準確。
