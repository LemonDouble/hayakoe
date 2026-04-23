# Step 4: Publish (HF / S3 / Local)

After training, the final model files are gathered under `<dataset>/exports/<model_name>/`.

The role of `cli publish` is to upload this folder to **HuggingFace Hub / S3 / a local folder** and make it loadable from another machine with a single line: `TTS().load("my_name")`.

It wraps the entire process — using HF CLI, S3 CLI, memorizing repo structure, and verifying uploads — into one interactive flow.

## Running

```bash
uv run poe cli
```

Select **Publish** from the main menu. It asks in this order:

1. Dataset to publish (or external folder)
2. Backend — CPU / GPU / CPU + GPU
3. Checkpoint
4. Speaker name
5. Destination + credentials
6. Summary panel -> Confirm
7. Auto upload -> Live synthesis verification

Each step is detailed below.

## 1. Choosing What to Publish

Two types of targets are shown.

- **Training dataset** — Datasets with final files in `data/dataset/<name>/exports/<model>/` are automatically listed.
- **Select from another folder** — When training was done elsewhere and you only have a HayaKoe-format folder, enter the path manually.

::: details Required files for an external folder
```
<my-folder>/
├── config.json                # Required
├── style_vectors.npy          # Required
├── *.safetensors              # Required (at least one)
├── synthesizer.onnx           # Optional (reused if present)
└── duration_predictor.onnx    # Optional (reused if present)
```
:::

## 2. Backend Selection

```
CPU (ONNX)        — For servers/local without GPU
GPU (PyTorch)     — Lowest latency
CPU + GPU (recommended)  — Deploy to both environments
```

Choosing `CPU + GPU` uploads files for **both** backends into the same repo. At runtime, `TTS(device="cpu")` automatically picks the ONNX files, while `TTS(device="cuda")` picks the PyTorch files.

**Upload once, reuse from both environments with the same name** — unless you have a specific reason, choose this option.

The differences between the two backends are detailed in [Backend Selection](/en/deploy/backend).

## 3. Checkpoint and Speaker Name

- If there is only 1 checkpoint, it is auto-selected; if multiple, you choose (typically the one picked in [Step 3: Quality Report](/en/training/quality-check)).
- **Speaker name** is the identifier used at runtime with `TTS().load("my_name")`. We recommend a concise, lowercase-hyphen style (e.g., `tsukuyomi`).

## 4. Destination Selection

Three options are available. Enter credentials once and they are saved to `dev-tools/.env` with `chmod 600`, so subsequent runs skip the prompt.

### HuggingFace Hub

Enter the repo path (`org/repo` or `hf://org/repo`) and a **write-access token**. You can also specify a branch/tag with `@<revision>`.

::: details Supported URL formats & stored environment variables
Accepted URL formats:

- `lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices`
- `hf://lemondouble/hayakoe-voices@main`
- `https://huggingface.co/lemondouble/hayakoe-voices`
- `https://huggingface.co/lemondouble/hayakoe-voices/tree/dev`

Example `.env` entries:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # Write-access HuggingFace access token
HAYAKOE_HF_REPO=lemondouble/hayakoe-voices       # HF repo for speaker file uploads (org/repo format)
```
:::

### AWS S3

Enter the bucket name (+ optional prefix) and AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Leave the endpoint URL empty.

### S3-compatible Storage (R2, MinIO, etc.)

For S3-compatible storage like Cloudflare R2, MinIO, or Wasabi, **enter the endpoint URL as well**.

- Cloudflare R2 — `https://<account>.r2.cloudflarestorage.com`
- MinIO — `http://<host>:9000`

Bucket and credential entry is the same as AWS S3.

::: details Stored environment variable examples
**AWS S3**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                # S3 bucket for speaker file uploads
HAYAKOE_S3_PREFIX=hayakoe-voices               # Path prefix within bucket (empty = bucket root)
AWS_ACCESS_KEY_ID=<your_access_key_here>       # AWS access key ID
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>   # AWS secret access key
AWS_REGION=ap-northeast-2                      # S3 region (example: Seoul)
# AWS_ENDPOINT_URL_S3 left empty (auto-determined for AWS S3)
```

**S3-compatible (Cloudflare R2)**

```env
HAYAKOE_S3_BUCKET=my-tts-bucket                                 # R2 bucket name for uploads
HAYAKOE_S3_PREFIX=hayakoe-voices                                # Path prefix within bucket (empty = bucket root)
AWS_ACCESS_KEY_ID=<your_access_key_here>                        # Access Key ID from R2 dashboard
AWS_SECRET_ACCESS_KEY=<your_secret_key_here>                    # R2 Secret Access Key
AWS_REGION=auto                                                 # Always auto for R2
AWS_ENDPOINT_URL_S3=https://abc123def.r2.cloudflarestorage.com  # R2 endpoint (unique per account)
```
:::

### Local Directory

Copies to a local path only, with no network upload. Suitable for scenarios where a team shares an NFS volume or internal network drive. At runtime, access via `file:///...` URI.

::: details Stored environment variable example
```env
HAYAKOE_LOCAL_PATH=/srv/hayakoe-voices   # Local directory to copy speaker files to
```
:::

## 5. Repo Structure

When publishing with `CPU + GPU`, the repo contains both an ONNX folder and a PyTorch folder. You can host multiple speakers in the same repo (`speakers/voice-a/`, `speakers/voice-b/`, ...).

::: details Internal structure
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

The BERT model is also uploaded to shared locations under `pytorch/bert/` and `onnx/bert/`. The runtime downloads speaker files and the shared BERT with the same caching rules.
:::

## 6. ONNX Export (Automatic)

When a CPU backend is selected (`CPU (ONNX)` or `CPU + GPU`), PyTorch checkpoints are automatically converted to ONNX just before upload. There is no separate `cli export` command.

Conversion results are cached in `<dataset>/onnx/`, so re-publishing the same checkpoint reuses them. To force reconversion, delete this folder and run publish again.

::: details Internal details — What gets converted and how
Two speaker-specific models are exported via `dev-tools/cli/export/exporter.py` at opset 17.

#### Targets — Two speaker-specific models

**Synthesizer (VITS Decoder)**

The core model that takes phoneme sequences + BERT embeddings + style vectors as input and produces the actual waveform. Since it is trained differently for each speaker, this model makes up the bulk of the deployment target.

- Function: `export_synthesizer`
- Output: `synthesizer.onnx` (+ possibly `synthesizer.onnx.data`)

**Duration Predictor**

Predicts how long each phoneme should be pronounced. If this prediction is inaccurate, sentence boundary pauses and tempo handling will sound unnatural.

- Function: `export_duration_predictor`
- Output: `duration_predictor.onnx`

#### What is `synthesizer.onnx.data`?

ONNX is internally serialized as Protobuf, which has a **2GB single message limit**. When the Synthesizer weights exceed this threshold, only the graph structure stays in `.onnx` while **large tensors are externalized to an adjacent `.data` file**.

- The two files **must always remain in the same folder** (do not move them separately)
- Depending on model size, `.data` may not be generated at all
- At runtime, loading just `.onnx` automatically reads `.data` from the same folder

#### BERT is shared, not per-speaker

BERT (DeBERTa) is a language-agnostic Japanese language model. All speakers share a common **Q8 quantized ONNX** (`bert_q8.onnx`) downloaded from a shared location on HuggingFace, and it is not reconverted per speaker during publish.

- Q8 quantization enables near-real-time embedding extraction on CPU
- All speakers share the same BERT, eliminating redundant storage per repo

In other words, the only models actually converted at this step are **the speaker-specific Synthesizer + Duration Predictor**.

#### Why tracing takes time

ONNX export uses a **tracing** approach that "runs the model once while recording the computation graph." The Synthesizer has a complex structure, so this can take tens of seconds to several minutes.

Since it is common to publish the same checkpoint multiple times under different names or destinations, the conversion result is cached in `<dataset>/onnx/` for reuse.

#### Exporting directly via script

The two export functions are publicly available and can be called from scripts. However, since the publish flow does the same thing automatically, we recommend using publish unless you have a specific reason. The direct call path may change in the future.
:::

## 7. Overwrite Confirmation

If the destination already has a `speakers/<speaker-name>/` with the same name, **you are asked whether to overwrite**. Approving cleanly deletes only that speaker directory and uploads fresh — other speakers in the same repo are untouched.

The same principle applies to README. If no README exists at the repo root, a 4-language template (ko/en/ja/zh) is auto-generated and uploaded together; if one already exists, a diff is shown and you are asked whether to overwrite.

## 8. Post-upload Auto-verification

After the upload completes, **it automatically verifies that the uploaded files can actually synthesize**.

If both CPU + GPU were selected, each backend is verified separately, and the result wavs are saved to `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav` for direct playback.

::: tip What a passing verification means
It means "the files uploaded to the repo actually produce synthesis."

Once this verification passes, you can be confident that other machines can immediately use `TTS().load(<speaker>, source="hf://...")` or similar to load and synthesize.
:::

::: details Internal details — Verification procedure
1. Create a `TTS(device=...)` instance with the selected backend
2. `load(<speaker>)` -> `prepare()` with the just-uploaded name
3. Synthesize the fixed phrase `"テスト音声です。"`
4. Save the result wav to `dev-tools/.verify_audio/<name>_<cpu|cuda>.wav`

Before GPU verification, global BERT / dynamo / CUDA caches are reset to ensure no cross-contamination.
:::

## Loading at Runtime

Once uploaded, load the speaker from another machine or container like this.

```python
from hayakoe import TTS

# From HF
tts = TTS(device="cpu").load("tsukuyomi", source="hf://me/my-voices").prepare()

# From S3
tts = TTS(device="cuda").load("tsukuyomi", source="s3://my-bucket/hayakoe-voices").prepare()

# From local
tts = TTS(device="cpu").load("tsukuyomi", source="file:///srv/voices").prepare()

# Synthesize
audio = tts.speakers["tsukuyomi"].generate("こんにちは。")
```

Simply changing `device` makes the same code automatically use the CPU (ONNX) or GPU (PyTorch) backend — this is possible because the publish step with `CPU + GPU` placed both file sets in the repo.

However, the runtime environment must also have the corresponding backend dependencies installed. Using `device="cuda"` requires the machine to have the **PyTorch CUDA build** installed, while `device="cpu"` works with the default installation alone. See [Installation — CPU vs GPU](/en/quickstart/install) for details.

## Next Steps

- Loading at runtime: [Server Deployment](/en/deploy/)
- Which backend to use at runtime: [Backend Selection](/en/deploy/backend)
