# FAQ

A collection of frequently asked advanced configuration topics.

## Changing the Cache Path

If the default cache path (`./hayakoe_cache/`) does not suit your needs, there are two options.

```bash
# Environment variable
export HAYAKOE_CACHE=/var/cache/hayakoe
```

```python
# Directly in code
tts = TTS(cache_dir="/var/cache/hayakoe")
```

HuggingFace, S3, and local sources all store under the same root.

## Loading Models from Private HuggingFace or S3

To use speakers from a private HF repo or models stored in an S3 bucket, specify the source URI.

If using an S3 source, install the extras first.

```bash
pip install hayakoe[s3]
```

```python
tts = (
    TTS(
        device="cuda",
        bert_source="s3://models/bert",
        hf_token="hf_...",                     # for private HF repo
        cache_dir="/var/cache/hayakoe",
    )
    .load("my-voice", source="s3://models/voices")
    .prepare()
)
```

S3-compatible endpoints (MinIO, Cloudflare R2, etc.) can be specified via the `AWS_ENDPOINT_URL_S3` environment variable.

## How Much More Memory Does Loading Multiple Speakers Use

Since BERT is shared across all speakers, the per-speaker increase is only the much lighter synthesizer portion.

I ran bench scripts on my local machine out of curiosity, but note that numbers vary by hardware, OS, torch version, and ORT build — **focus on the increase trend rather than absolute values**.

::: info Measurement Environment
- GPU — NVIDIA RTX 3090 (24 GB), Driver 580.126.09
- Text — Two Japanese sentences (including sentence boundary, ~50 characters)
- Speakers — `jvnv-F1-jp`, `jvnv-F2-jp`, `jvnv-M1-jp`, `jvnv-M2-jp`
- Each scenario was run in a separate Python process (to prevent heap contamination)
:::

### Memory by Speaker Count (Loaded but Idle)

| Speakers | CPU (ONNX) RAM | GPU (PyTorch) RAM | GPU VRAM |
| :------- | -------------: | ----------------: | -------: |
| 1        | ~1.7 GB        | ~1.3 GB           | ~1.8 GB  |
| 4        | ~2.8 GB        | ~1.5 GB           | ~2.6 GB  |

Dividing the increase from 3 additional speakers by 3, the approximate per-speaker cost is:

- **CPU RAM** — ~+360 MB / speaker
- **GPU VRAM** — ~+280 MB / speaker

### Running 4 Speakers Simultaneously

In real services, multiple speakers might run concurrently, so we measured **sequential 4 runs** and **concurrent 4 threads** separately (peak during synthesis).

| Scenario         | CPU RAM peak | GPU RAM peak | GPU VRAM peak |
| :--------------- | -----------: | -----------: | ------------: |
| 1 speaker synth  | ~2.0 GB      | ~2.3 GB      | ~1.7 GB       |
| 4 speakers seq   | ~3.2 GB      | ~2.1 GB      | ~2.6 GB       |
| 4 speakers conc  | ~3.2 GB      | ~2.2 GB      | ~2.8 GB       |

Even with concurrent execution, memory does not quadruple.

On the CPU side, ORT already parallelizes internally so "sequential vs concurrent" makes almost no difference. GPU VRAM stops at about +200 MB more for concurrent execution.

### Reproducing Locally

Scripts are included under `docs/benchmarks/memory/` in the repository.

```bash
# Single scenario
python docs/benchmarks/memory/run_one.py --device cpu --scenario idle4

# All 10 scenarios (CPU/GPU x idle1/idle4/gen1/seq4/conc4) in separate processes
bash docs/benchmarks/memory/run_all.sh
```

- `run_one.py` runs one scenario and prints a single JSON line.
- `run_all.sh` runs all scenarios in separate Python processes, collecting results into `results_<timestamp>.jsonl` beside the script.
- RAM is measured by polling RSS with `psutil` every 50ms for the peak, and VRAM uses the `torch.cuda.max_memory_allocated()` value directly.
- `gen*` scenarios call `torch.cuda.reset_peak_memory_stats()` after warmup to exclude torch.compile cold start from the peak.

For the most accurate numbers, run the measurement on your own environment and compare.
