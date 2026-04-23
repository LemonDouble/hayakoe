# Source Abstraction (HF / S3 / Local)

An abstraction layer that lets you **load speaker models and BERT files with the same API regardless of where they are located**, just by changing the URI.

## Why It Is Needed

The source for loading speaker models varies by situation.

- Public default speakers come from the **HuggingFace repo** (`hf://lemondouble/hayakoe`)
- Personally trained speakers may be in a **private HF repo, S3, or local directory**

Branching download code per source bloats the engine core and creates cache path duplication issues.

## Implementation

### Source Interface

All sources implement a common interface: **"download all files under a prefix to local cache and return the path."**

```python
class Source(Protocol):
    def fetch(self, prefix: str) -> Path:
        """Download all files under prefix/ to cache and return local path."""
        ...

    def upload(self, prefix: str, local_dir: Path) -> None:
        """Upload local_dir contents under prefix/ (for deployment)."""
        ...
```

`fetch()` is used during model loading, and `upload()` during the CLI's `publish` (model deployment).

### Implementations

| URI Scheme | Implementation | Behavior |
|---|---|---|
| `hf://user/repo[@revision]` | `HFSource` | Downloads via `huggingface_hub.snapshot_download()`. Private repo access via `HF_TOKEN` env or `hf_token` parameter |
| `s3://bucket/prefix` | `S3Source` | `boto3`-based. S3-compatible endpoints (R2, MinIO, etc.) via `AWS_ENDPOINT_URL_S3` env |
| `file:///abs/path` or `/abs/path` | `LocalSource` | Uses local directory directly. No download |

### URI Auto-routing

Just pass a URI to `TTS().load()` and the matching Source is automatically selected.

```python
# HuggingFace (default)
tts.load("jvnv-F1-jp")

# HuggingFace — private repo
tts.load("jvnv-F1-jp", source="hf://myorg/my-voices")

# S3
tts.load("jvnv-F1-jp", source="s3://my-bucket/voices")

# Local
tts.load("jvnv-F1-jp", source="/data/models")
```

HuggingFace web URLs (`https://huggingface.co/user/repo`) are also auto-normalized to `hf://` format and accepted.

### Cache

All sources store under the same cache root.

The cache path is set via the `HAYAKOE_CACHE` environment variable, or defaults to `$CWD/hayakoe_cache` if unset.

Cache policy is simple — reuse if the file exists, download fresh if not.

### BERT Source Separation

Speaker model and BERT model sources can be **specified separately**.

```python
TTS(
    device="cpu",
    bert_source="hf://lemondouble/hayakoe",  # BERT from official repo
).load(
    "custom-speaker",
    source="/data/my-models",                 # speaker from local
).prepare()
```

The default for both is `hf://lemondouble/hayakoe`.

## Improvement Results

- Storage-specific branching code was removed from the engine core.
- Adding a new storage backend only requires writing a single class implementing the `Source` protocol.
- The CLI's `publish` command also uses the same abstraction in the reverse direction (`upload`).
