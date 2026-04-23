# Custom Speaker Training

::: warning Check copyright before distributing
**If you plan to publish a trained model or distribute synthesized audio externally**, be sure to verify the copyright and publicity rights of the original voice first.

Publishing models made from someone else's voice (YouTube, anime, games, commercial voice actors, etc.) may constitute infringement of copyright, right of publicity, and personality rights.

If you intend to distribute, use your own voice, a voice with explicit consent, or a free corpus that allows distribution.

---

**Example Japanese corpora that can be freely used and distributed**

- **Tsukuyomi-chan Corpus** — Commercial and non-commercial use allowed with credit.
- **Amitaro no Koe Sozai Koubou** — Personal and commercial use allowed (check the terms).
- **Zundamon** and other VOICEVOX character ITA/ROHAN corpora — Check each character's usage terms.

Each corpus has different requirements for credit, commercial use, and derivative work scope. Always recheck the official terms before distribution.
:::

HayaKoe supports the entire workflow up to training as long as you have a video file containing a voice.

Data preparation through deployment is divided into two tools.

### `dev-tools/preprocess/` — Browser GUI (Data Preprocessing)

Upload a video or audio file and follow the steps in order on the GUI to produce a training-ready dataset.

- **Audio Extraction (automatic)** — Extracts the audio track from the video.
- **Source Separation (automatic)** — Removes background sounds like BGM and sound effects using the audio-separator library, leaving only the voice.
- **VAD Segmentation (automatic)** — Splits long recordings into short spoken segments based on silence boundaries.
- **Classification (`manual`)** — Classify each extracted segment by speaker and discard unusable portions.
- **Transcription (automatic)** — Automatically generates matching text data for each segment using a speech recognition model (Whisper).
- **Review (`manual`, optional)** — Correct mistranscribed parts directly in the browser.
- **Dataset Export (automatic)** — Exports the data in a training-ready format.

### `dev-tools/cli/` — Interactive CLI

Takes the dataset created by the GUI and continues from training through deployment.

- **Preprocessing (automatic)** — Pre-computes G2P, BERT embeddings, and style vectors needed for training.
- **Training (automatic)** — Fine-tunes on prepared data starting from a pretrained model.
- **Quality Report (automatic)** — Batch-synthesizes audio from checkpoints saved during training to find which model sounds best.
- **Publish (automatic)** — Handles ONNX conversion (inference-optimized model), plus upload to HuggingFace / S3 / local.

Both tools share the same `data/` folder, so datasets created in the GUI are automatically recognized by the CLI.

## Full Workflow

<PipelineFlow
  :steps="[
    {
      num: '①',
      title: 'Data Preparation',
      tool: 'GUI',
      content: [
        'Create a training audio dataset from a video of the speaker you want to train.',
        'Extract audio from the video, remove background sounds and effects to leave only the voice, then split into short sentence-level segments based on silence.',
        'Classify the segments by speaker, auto-generate text with Whisper, review as needed, and export into a training-ready format.'
      ],
      chips: ['Prepare Video', 'Audio Extraction', 'Source Separation', 'VAD Segmentation', 'Speaker Classification', 'Transcription', 'Review', 'Dataset Export'],
      gpu: 'Required'
    },
    {
      num: '②',
      title: 'Preprocessing & Training',
      tool: 'CLI',
      content: [
        'Fine-tune a Japanese TTS model with the prepared dataset.',
        'Pre-compute G2P (pronunciation conversion), BERT embeddings, and style vectors needed for training, then fine-tune on top of a pretrained Style-Bert-VITS2 JP-Extra model.',
        'Intermediate checkpoints are saved at regular intervals for comparison in the next step.'
      ],
      chips: ['G2P & BERT Computation', 'Style Embedding', 'Fine-tuning', 'Checkpoint Saving'],
      gpu: 'Required'
    },
    {
      num: '③',
      title: 'Quality Report',
      tool: 'CLI',
      content: [
        'Training longer does not always mean better — past a certain point, audio quality or speaker tone can actually degrade.',
        'So we batch-synthesize the same sentences from multiple checkpoints saved during training to compare which point in time produces the best sound.',
        'Results are compiled into a single HTML page where you can listen directly in the browser, then pick the checkpoint you like best for the next step.'
      ],
      chips: ['Batch Synthesis', 'HTML Report', 'Checkpoint Selection']
    },
    {
      num: '④',
      title: 'Publish',
      tool: 'CLI',
      content: [
        'Convert the selected checkpoint to ONNX format.',
        'ONNX is a model format optimized for CPU inference, so it runs smoothly even on a regular laptop without a GPU.',
        'Upload the converted model to your choice of cloud storage like HuggingFace or S3, or a local directory.',
        'Once uploaded, the hayakoe package can load and use it by speaker name alone.'
      ],
      chips: ['ONNX Conversion', 'HuggingFace', 'S3', 'Local']
    }
  ]"
/>

::: warning Data Preparation (Step 1) and Training (Step 2) require a GPU
Both steps run ML models internally (source separation, Whisper, VITS2), making them virtually impossible without a GPU.

Quality Report (Step 3) and Publish (Step 4) work without a GPU. We do not recommend running training on a CPU laptop.
:::

## Getting Ready

This guide works directly from a cloned hayakoe repo.

::: info Linux environment assumed
The training tools currently only guarantee operation on Linux.

For Windows, we recommend following the Linux guide on WSL2.
:::

### 1. Clone the Repo

```bash
git clone https://github.com/LemonDouble/hayakoe.git
cd hayakoe
```

### 2. Install uv

uv is a fast Python package and environment manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For detailed installation instructions, see the [uv official guide](https://docs.astral.sh/uv/getting-started/installation/).

After installation, the version should print:

```bash
uv --version
```

### 3. Install Dev Dependencies

All subsequent commands are run from the **repo root (`hayakoe/`)** cloned in step 1.

```bash
uv sync
```

This installs all libraries needed by the preprocessing GUI and training CLI (FastAPI, Whisper, audio-separator, torchaudio, etc.) in one go.

### 4. Install GPU (CUDA) PyTorch

Data Preparation (Step 1) and Training (Step 2) run ML models internally, so an NVIDIA GPU is required.

First, verify that the driver is properly installed.

```bash
nvidia-smi
```

If properly installed, you should see output like this:

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

The `CUDA Version` on the top right of the output is the **maximum CUDA version** your driver supports (13.0 in the example above).

Choose and install a PyTorch build at or below that version (the example below targets CUDA 12.6).

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

When using a different CUDA version, replace `cu126` with the version matching your setup (`cu118`, `cu121`, `cu124`, `cu128`, etc.).

Verify the installation:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
# Should print True
```

Once complete, follow the [Step-by-step Details](#step-by-step-details) in order below.

## Step-by-step Details

Once preparations are complete, follow these pages one at a time in order.

- [Step 1: Data Preparation](./data-prep)
- [Step 2: Preprocessing & Training](./training)
- [Step 3: Quality Report](./quality-check)
- [Step 4: Publish (HF / S3 / Local)](./publish)
- [Troubleshooting](./troubleshooting)
