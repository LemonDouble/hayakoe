# Step 2: Preprocessing & Training

This step takes the dataset created in [Step 1: Data Preparation](/en/training/data-prep), processes it for training, and trains the voice.

Training is a long-running task that consumes significant computing resources. Since the GUI also uses resources, everything after the data preprocessing pipeline (which requires a UI) is separated into the CLI, which uses virtually no resources.

You do not need to memorize any commands. Just launch the CLI and follow the menus.

## Launching the CLI

Run without arguments from the repo root to see the main menu.

```bash
uv run poe cli
```

```text
  ╦ ╦╔═╗╦ ╦╔═╗╦╔═╔═╗╔═╗
  ╠═╣╠═╣╚╦╝╠═╣╠╩╗║ ║║╣
  ╩ ╩╩ ╩ ╩ ╩ ╩╩ ╩╚═╝╚═╝ Dev Tools

Guides you from training to deployment of HayaKoe TTS models.
Follow the steps below in order.

  1. Training         Train a TTS model from voice data.
  2. Quality Report   Compare audio from trained checkpoints.
                       Repeat steps 1-2 until satisfied.
  3. Benchmark        Measure CPU/GPU inference speed.
  4. Publish          Upload the trained speaker to HF / S3 / local
                       so it can be downloaded at runtime.
                       CPU deployment automatically runs ONNX export.

? What would you like to do?
  > Training Pipeline — Data preprocessing + model training
    Quality Report — Compare audio across checkpoints
    Benchmark — CPU/GPU inference speed measurement
    Publish — Deploy speaker to HF / S3 / local
    Exit
```

This page only uses the **Training Pipeline**. Selecting it automatically lists datasets under `data/dataset/`.

```text
Workflow:  1. Select Dataset -> 2. Preprocess -> 3. Training Settings -> 4. Start Training

+------------------------------- Datasets --------------------------------+
|  +-------------------+-----------+-------------+---------+--------------+|
|  | Speaker           |  Samples  |       Train |     Val |    Status    ||
|  +-------------------+-----------+-------------+---------+--------------+|
|  | tsukuyomi         |        72 |          65 |       7 | Needs Preproc||
|  +-------------------+-----------+-------------+---------+--------------+|
+-------------------------------------------------------------------------+

? Select Dataset
  > tsukuyomi
    Enter path manually
    Back
```

After choosing a dataset, a submenu appears. Follow in order: **Preprocess -> Edit Settings -> Start Training**.

Each menu is detailed in the sections below.

## Preprocessing

The `(audio, text)` pairs from data preparation are not yet in a format the model can directly train on. This step **converts the raw data into numeric representations the model understands and caches them**.

When you select a dataset, the current preprocessing status panel appears first.

```text
+--------------------------- tsukuyomi — Preprocessing Status ---------------------------+
|    OK  Audio Files        72                                                           |
|    -   Text Preprocessing Not done                                                     |
|    -   BERT Embeddings    0/72                                                         |
|    -   Style Vectors      0/72                                                         |
|    -   Default Style      Not done                                                     |
+----------------------------------------------------------------------------------------+
Next: Run preprocessing to prepare the data.

? tsukuyomi — What would you like to do?
  > Run Preprocessing (remaining steps)
    Re-run All Preprocessing
    Edit Training Settings
    Back
```

Selecting **Run Preprocessing (remaining steps)** runs the 4-step preprocessing sequence.

Completion is detected automatically, so if interrupted and resumed, completed steps are skipped. Use **Re-run All Preprocessing** to redo everything.

::: tip If there is a G2P error
If `text_error.log` is generated during preprocessing step 1, some sentences failed phoneme conversion.

Those lines are automatically excluded from training, but if data is scarce, check the log for causes. Special characters or unreadable kanji are common culprits.
:::

::: details Internal details — 4-step breakdown
| Step | What it does | Output |
|---|---|---|
| 1. G2P Text Preprocessing | Converts Japanese text to pronunciation (phonemes + accent tones). The model learns **pronunciation**, not characters. | `train.list` / `val.list` expanded to 7 fields, `text_error.log` |
| 2. BERT Embeddings | Pre-computes sentence context as DeBERTa embeddings and caches them. These values help generate natural intonation, but running BERT every training step would be slow, so they are computed once. | `<wav>.bert.pt` |
| 3. Style Vectors | Extracts voice characteristics (tone, speaking style) from each audio as vectors (pyannote embeddings). | `<wav>.npy` (NaN files auto-removed) |
| 4. Default Style | Averages the above vectors as the speaker's "standard style". | `exports/<model>/style_vectors.npy`, `config.json` |

Completion is determined by file count matching.
:::

## Edit Training Settings

Once preprocessing is done, **Edit Training Settings** lets you adjust key training parameters.

Instead of exposing all dozens of fields, only the **11 that matter in practice** are included.

```text
+----------------------------------- Training Settings -----------------------------------+
|  +---------------------------+-----------------------------------------+------------+   |
|  |  Parameter                |  Description                            |      Value |   |
|  +---------------------------+-----------------------------------------+------------+   |
|  |  Basic Settings           |                                         |            |   |
|  |    epochs                 |  Number of training epochs              |       500  |   |
|  |    batch_size             |  Batch size (per GPU)                   |         2  |   |
|  |    nproc_per_node         |  Number of GPUs                         |         1  |   |
|  |    bf16_run               |  bfloat16 mixed precision               |       OFF  |   |
|  |    eval_interval          |  Checkpoint save interval (steps)       |      1000  |   |
|  |    log_interval           |  TensorBoard logging interval (steps)   |       200  |   |
|  |                           |                                         |            |   |
|  |  Advanced Settings        |                                         |            |   |
|  |    learning_rate          |  Learning rate                          |    0.0001  |   |
|  |    warmup_epochs          |  Warmup epochs                          |         0  |   |
|  |    freeze_JP_bert         |  Freeze JP BERT encoder                 |       OFF  |   |
|  |    freeze_style           |  Freeze style encoder                   |       OFF  |   |
|  |    freeze_decoder         |  Freeze decoder                         |       OFF  |   |
|  +---------------------------+-----------------------------------------+------------+   |
+----------------------------------------------------------------------------------------+
```

::: tip 2000-4000 total steps is a typical target
In our experience, **2000-4000 total steps** often produces usable audio. Since epoch count depends on dataset size and batch size, adjust `epochs` so the briefing panel's `total steps` falls within this range.

Setting `eval_interval` to 500-1000 within this range ensures multiple checkpoints are generated, making it easier to pick the best one in the [Quality Report](/en/training/quality-check).
:::

Descriptions and recommendations for each parameter (what happens when increased / decreased / recommended value) are collapsed below. Expand when you need to adjust.

::: details Parameter details — Basic Settings (6)
#### `train.epochs` — Number of Training Epochs
Sets how many times the entire training dataset is iterated from start to finish.

- **Higher**: Model trains longer, potentially improving quality. But too high and the model memorizes training data, producing unnatural results on new sentences (overfitting).
- **Lower**: Training finishes quickly, but the model may not learn enough, resulting in low quality.
- **Recommended**: Aim for 2000-4000 total steps. Calculate as `epochs = target steps x batch size / training samples`.

#### `train.batch_size` — Batch Size (per GPU)
Sets how many audio samples are grouped together for one training step.

- **Higher**: Faster and more stable training, but uses more GPU memory (VRAM). If memory is insufficient, you get `CUDA out of memory` errors.
- **Lower**: Works with less memory, but slower.
- **Recommended**: RTX 3060 (12GB) -> 2-4, RTX 3090/4090 (24GB) -> 4-8.

#### `train.nproc_per_node` — Number of GPUs
Number of GPUs to use for training. Multiple GPUs split data for parallel training, scaling speed proportionally (DistributedDataParallel).

- **Default 1**: Single GPU training. Do not change this if you have only one GPU.
- Exceeding the actual number causes errors. Check with `nvidia-smi`.

#### `train.bf16_run` — bfloat16 Mixed Precision
Reduces computation precision from 32-bit to 16-bit to speed up training.

- **On**: ~1.5x training speed, reduced VRAM usage. Requires Ampere or newer GPU (RTX 30xx, 40xx, A100, etc.).
- **Off**: Works on all GPUs but slower.
- **Caution**: Enabling on RTX 20xx or older may cause errors or quality degradation.

#### `train.eval_interval` — Checkpoint Save Interval (steps)
Sets how often to save checkpoints. Checkpoints are snapshots of the model during training, letting you pick the best point later.

- **Save frequently**: Finer comparison possible, but higher disk usage (~240MB per checkpoint).
- **Save rarely**: Saves disk but may miss the optimal point.
- **Recommended**: 500-1000 steps. Aim for 4-8 checkpoints within the 2000-4000 total step range for convenient quality report comparisons.

#### `train.log_interval` — TensorBoard Logging Interval (steps)
Sets how often metrics like loss are recorded to TensorBoard.

- **Recommended**: The default (200) is sufficient in most cases.

:::

::: details Parameter details — Advanced Settings (5)
#### `train.learning_rate` — Learning Rate
Determines how much the model is adjusted per step. Think of it like steering sensitivity on a bicycle.

- **Higher**: Learns faster but values may spike, causing unstable or divergent training.
- **Lower**: Stable but very slow learning that may never reach the optimum.
- **Recommended**: If unsure, keep the default (0.0001).

#### `train.warmup_epochs` — Warmup Epochs
A period at the start of training where the learning rate gradually increases from 0. Similar to gently accelerating a car instead of flooring it.

- **Recommended**: 0-3 epochs. If training is unstable at the start, try 1-2.

#### `train.freeze_JP_bert` — Freeze JP BERT Encoder
Freezes the weights of the text understanding component (BERT).

- **On**: Preserves BERT's existing Japanese language knowledge. Effective for preventing overfitting with small datasets (~200 sentences or fewer), and improves speed and memory usage.
- **Off**: BERT is also fine-tuned together, allowing more natural pronunciation and intonation when data is sufficient.
- **Recommended**: 200 sentences or fewer -> ON, 500 sentences or more -> OFF.

#### `train.freeze_style` — Freeze Style Encoder
Freezes the weights of the style encoder that represents speaking style and emotion.

- **On**: Preserves existing style expressiveness. Useful when data is limited or you want to maintain existing emotional expression.
- **Off**: Freely learns the new speaker's style characteristics.
- **Recommended**: Generally OFF. If training results are monotone, try ON.

#### `train.freeze_decoder` — Freeze Decoder
Freezes the weights of the decoder that generates the final audio.

- **On**: Only the text interpretation part is trained, keeping the audio generation part unchanged. Training is faster but quality improvement is limited.
- **Off**: The decoder is also trained, allowing audio quality tailored to the speaker.
- **Recommended**: Generally OFF. No reason to freeze unless you have a special case.
:::

## Starting Training

When you click **Start Training**, a **briefing panel** appears first. It shows epochs, batch size, total steps, checkpoint interval, and estimated disk usage in a single table, and lets you adjust checkpoint interval on the spot.

```text
+--------------------------- tsukuyomi — Training Briefing ----------------------------+
|    Hyperparameters                                                                    |
|      Dataset                            65 train / 7 val                              |
|      Epochs                                           500                             |
|      Batch Size                                         2                             |
|      Learning Rate                                 0.0001                             |
|      bfloat16                                         OFF                             |
|                                                                                       |
|    Steps & Checkpoints                                                                |
|      Steps per Epoch                                   33                             |
|      Total Steps                                   16,500                             |
|      Log Interval                              200 steps                              |
|      Checkpoint Interval                      1000 steps                              |
|      Estimated Saves                               16                                 |
|      Estimated Disk Usage             ~3.8GB (16 x 240MB)                             |
|                                                                                       |
|    Save Paths                                                                         |
|      Checkpoints            data/dataset/tsukuyomi/training                           |
|      Export Models           data/dataset/tsukuyomi/exports                           |
+---------------------------------------------------------------------------------------+

? Proceed
  > Start Training
    Change Checkpoint Interval (current: 1000 steps)
    Cancel
```

After confirming and proceeding, training starts and TensorBoard launches automatically. Open the URL printed in the console to watch loss changes in real time.

::: tip TensorBoard starts automatically
No need to manually run `tensorboard --logdir ...`.

It closes automatically when training finishes or is stopped with `Ctrl-C`.
:::

::: details Internal details — What happens when training starts
1. **Auto-download pretrained model** — Downloads `G_0.safetensors`, `D_0.safetensors`, `WD_0.safetensors` from `hf://lemondouble/hayakoe/pretrained/<name>` to cache. If `G_*.pth` checkpoints already exist, it enters "resume mode" and skips this.
2. **Launch training process** — Starts `train_ms_jp_extra.py` (SBV2 original port) as a `torch.distributed.run` subprocess. The `nproc_per_node` value is passed through directly.
3. **Auto-start TensorBoard** — TensorBoard launches pointing at the training directory.

Estimated disk usage is a fixed estimate of 240MB per checkpoint.
:::

## Training Output

When training finishes, two paths under the dataset directory are populated.

- **`training/`** — Training snapshots (checkpoints and TensorBoard logs). These consume significant disk space, so feel free to delete unnecessary checkpoints after training.
- **`exports/<model_name>/`** — The final model used as the reference for deployment and evaluation. The next steps ([Step 3: Quality Report](/en/training/quality-check), [Step 4: Publish](/en/training/publish)) read from this directory.

::: details Internal details — Directory structure
```
<dataset_root>/
├── training/                          # Training artifacts
│   ├── G_*.pth, D_*.pth, WD_*.pth    # Checkpoints (every eval_interval)
│   └── events.out.tfevents.*          # TensorBoard logs
└── exports/<model_name>/              # Final model
    ├── *.safetensors
    ├── config.json
    └── style_vectors.npy
```
:::

## Interrupting and Resuming

Even if you stop training with `Ctrl-C`, checkpoints remain in `training/`.

Pressing **Start Training** again with the same dataset automatically detects remaining checkpoints and enters resume mode. Pretrained model download is skipped in this case.

## Next Steps

- Pick the best checkpoint: [Step 3: Quality Report](/en/training/quality-check)
- Upload the finished model to HF/S3/local: [Step 4: Publish](/en/training/publish)
