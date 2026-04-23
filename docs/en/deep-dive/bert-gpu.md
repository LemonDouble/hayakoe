# BERT GPU Retention & Batch Inference

On the GPU path, two key optimization points stand out.

- **Removing the unnecessary CPU transfer from the original SBV2** to keep BERT output as a GPU tensor
- **Batching multi-sentence BERT calls into a single invocation** to reduce kernel launch overhead (the fixed cost incurred each time an operation is dispatched to the GPU)

## Why It Matters

The original SBV2 fundamentally **synthesizes entire text in one pass** (`line_split=False`).

Since BERT is called only once, batching was not needed.

HayaKoe introduced **punctuation-based sentence splitting** for prosody stability, which created the new problem of BERT being called once per sentence.

### Unnecessary CPU Transfer of BERT Output

The original SBV2's BERT feature extraction code contains this:

```python
# Original SBV2 (style_bert_vits2/nlp/japanese/bert_feature.py)
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```

After running BERT forward on GPU, it calls `.cpu()` on the output tensor, **transferring it to CPU every time.**

This output is then passed to the Synthesizer, which runs on GPU, requiring another transfer back to GPU.

The result is a **GPU -> CPU -> GPU round-trip** per sentence, and this unnecessary round-trip itself becomes a bottleneck.

### Per-sentence Individual BERT Calls

When calling BERT separately for each sentence after splitting, GPU kernel launches repeat once per sentence.

A kernel launch is the fixed cost incurred each time an operation is dispatched to the GPU.

For short sentences, the overhead proportion exceeds the actual computation time, accumulating inefficiency proportional to sentence count.

## Implementation

### Removing `.cpu()` — Keeping GPU Tensors

The original `.cpu()` call was removed so BERT output passes to the Synthesizer as a GPU tensor directly.

```python
# Original SBV2
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()    # GPU -> CPU

# HayaKoe
res = torch.cat(res["hidden_states"][-3:-2], -1)[0].float()  # stays on GPU
```

The BERT model itself is loaded to GPU at `prepare()` time and stays there through inference.

The BERT model is managed as a **global singleton**, so loading multiple speakers still loads BERT only once, shared by all.

### Multi-sentence BERT Batching

The BERT (DeBERTa) used by HayaKoe is a HuggingFace Transformer model that natively supports batch input.

Leveraging this, instead of calling BERT individually for each sentence in multi-sentence synthesis, **all sentences are grouped into a single batch for one call**.

Multiple sentences are fed to the tokenizer at once to create a padded batch input, and BERT is called **only once**.

The same batch logic is implemented on the ONNX path as well.

## Improvement Results

### GPU Batch Inference Speed

**Sequential vs batched** comparison on the same hardware (5-run average).

| Sentences | Sequential | Batched | Speedup |
|---|---|---|---|
| 2 | 0.447 s | 0.364 s | **1.23x** |
| 4 | 0.812 s | 0.566 s | **1.43x** |
| 8 | 1.598 s | 1.121 s | **1.43x** |
| 16 | 2.972 s | 2.264 s | **1.31x** |

With kernel launch overhead consolidated into a single call, speedups of +23% to +43% are observed.

### GPU Memory

We verified that batching does not consume additional memory.

| Sentences | Sequential peak | Batched peak | Difference |
|---|---|---|---|
| 2 | 1,662.2 MB | 1,661.9 MB | −0.3 MB |
| 4 | 1,661.8 MB | 1,662.2 MB | +0.4 MB |
| 8 | 1,697.7 MB | 1,699.0 MB | +1.3 MB |
| 16 | 1,934.3 MB | 1,934.3 MB | 0 MB |

The difference between sequential and batched is **within 1.3 MB**, essentially identical.

### No Effect on CPU

Repeating the same experiment on CPU (ONNX) shows virtually no batching benefit.

| Sentences | Sequential | Batched | Speed Difference |
|---|---|---|---|
| 2 | 2.566 s | 2.564 s | 1.00x |
| 4 | 5.464 s | 4.855 s | 1.13x |
| 8 | 10.647 s | 11.783 s | 0.90x |
| 16 | 24.559 s | 24.195 s | 1.01x |

ONNX Runtime's graph optimization is already strong enough that Python-level dispatch overhead is not the bottleneck, and padding overhead in batching offsets the gains.

Batching is maintained on GPU, and since neither gains nor losses are significant on CPU, the same path is kept for code consistency across backends.
