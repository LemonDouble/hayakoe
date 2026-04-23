# Server Deployment

HayaKoe is designed for server environments with a **singleton TTS instance + build-time weight baking** pattern. You can spin up a clean API server with a FastAPI and Docker combination.

## Design Principles

### 1. Load the Model Only Once (Singleton)

Loading a TTS model takes considerable time. In GPU environments, it can take tens of seconds including the compilation step. Creating a new instance per request makes production service virtually impossible, so you must maintain **a single instance for the process lifetime** and share it across all requests.

In practice, the code builds a singleton with `TTS(...).load(...).prepare(warmup=True)` in FastAPI's lifespan hook, stores it in `app.state.tts`, and all handlers reuse this one instance.

Concurrency is handled automatically. `Speaker` has an internal `threading.Lock`, so concurrent requests for the same speaker are automatically serialized while different speakers run in parallel — no separate pool or queue implementation needed.

::: details GPU backend prepares torch.compile as well
`TTS.prepare()` not only loads models for the CUDA backend but also applies `torch.compile` to all speakers and BERT at once.

Setting `warmup=True` runs a dummy inference pass to shift compilation costs into the prepare phase. This itself can take tens of seconds, so it must be done exactly once at app boot time. **Creating a new TTS per request triggers recompilation every time**, effectively paralyzing the server.

The CPU backend uses ONNX Runtime, so there is no separate compilation step and prepare is much faster.
:::

-> Implementation: [FastAPI Integration](/en/deploy/fastapi)

### 2. Bake Weights into the Image at Build Time

HayaKoe's recommended operational pattern is to **pack model weights entirely into the Docker image** so it starts immediately at runtime with no external network access.

For this, it provides `TTS.pre_download(device=...)` — a method that "fills the cache without initializing." Calling it during the Docker build stage embeds all necessary speaker files into the image, so runtime containers never need to access HF or S3.

This is an especially clean pattern for offline environments, behind firewalls, or when you do not want to expose HF/S3 credentials to runtime containers.

-> Implementation: [Docker Image](/en/deploy/docker)

## Section Layout

| Page | Content |
|---|---|
| [FastAPI Integration](/en/deploy/fastapi) | Singleton load in lifespan, `agenerate` / `astream`, concurrency |
| [Docker Image](/en/deploy/docker) | Build-time `pre_download`, BuildKit secret, multi-stage |
| [Backend Selection](/en/deploy/backend) | CPU (ONNX) vs GPU (PyTorch) trade-offs |
