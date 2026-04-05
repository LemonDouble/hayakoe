"""GPU ONNX vs PyTorch 벤치마크.

비교 대상:
  1. PyTorch FP32 GPU
  2. PyTorch FP16(BERT) + FP32(Synth) GPU
  3. ONNX FP32 CPU (기존 실험 4 baseline)
  4. ONNX FP32 GPU (CUDAExecutionProvider)
  5. ONNX Q8 CPU (BERT Q8 + Synth FP32)

사용법:
    cd hayakoe/
    python dev-tools/benchmark_gpu_onnx.py
"""

import gc
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

# ── 설정 ──
HF_REPO = "lemondouble/hayakoe"
SPEAKER = "jvnv-F1-jp"
WARMUP = 2
RUNS = 5

TEST_TEXTS = {
    "short": "こんにちは。",
    "med": "私はイレイナ。旅の魔女です。あちこちを旅しています。",
    "long": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
}

STYLE_VEC = None
HPS = None


def download_models():
    """HF에서 모델 다운로드 후 경로 반환."""
    from huggingface_hub import snapshot_download

    print("Downloading models from HF...")
    local = snapshot_download(HF_REPO)
    local = Path(local)

    paths = {
        "bert_pt_fp32": local / "pytorch" / "bert" / "fp32",
        "bert_onnx_fp32": local / "onnx" / "bert" / "fp32" / "bert.onnx",
        "bert_onnx_q8": local / "onnx" / "bert" / "q8" / "bert_q8.onnx",
        "speaker_pt": local / "pytorch" / "speakers" / SPEAKER,
        "synth_onnx_fp32": local / "onnx" / "speakers" / SPEAKER / "synthesizer.onnx",
        "synth_onnx_q8": local / "onnx" / "speakers" / SPEAKER / "synthesizer_q8.onnx",
    }

    # bert.onnx 는 .data 파일과 같이 있을 수 있음
    bert_onnx_data = local / "onnx" / "bert" / "fp32" / "bert.onnx.data"
    if bert_onnx_data.exists():
        paths["bert_onnx_fp32_data"] = bert_onnx_data

    return paths


def load_common(speaker_dir: Path):
    """스타일 벡터 + HPS 로드."""
    global STYLE_VEC, HPS
    from hayakoe.models.hyper_parameters import HyperParameters

    HPS = HyperParameters.load_from_json(speaker_dir / "config.json")
    style_vectors = np.load(speaker_dir / "style_vectors.npy")
    STYLE_VEC = style_vectors[0]  # Neutral


def benchmark_pytorch(device: str, bert_fp16: bool, paths: dict):
    """PyTorch 추론 벤치마크."""
    from hayakoe.constants import BERT_JP_REPO
    from hayakoe.models.infer import get_net_g, infer
    from hayakoe.nlp import bert_models

    label = f"PyTorch {'FP16' if bert_fp16 else 'FP32'} GPU"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # BERT 로드
    bert_models.unload_model()
    bert_models.unload_tokenizer()

    dtype = "float16" if bert_fp16 else "float32"
    bert_models.load_model(
        pretrained_model_name_or_path=str(paths["bert_pt_fp32"]),
        device_map=device,
        cache_dir=None,
    )
    if bert_fp16:
        bert_models._loaded_model.half()
    bert_models.load_tokenizer(pretrained_model_name_or_path=BERT_JP_REPO)

    # Synth 로드
    sf = sorted(paths["speaker_pt"].glob("*.safetensors"))[-1]
    net_g = get_net_g(str(sf), HPS.version, device, HPS)

    results = {}
    for name, text in TEST_TEXTS.items():
        times = []
        for i in range(WARMUP + RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                audio = infer(
                    text=text, sdp_ratio=0.2, noise_scale=0.6,
                    noise_scale_w=0.8, length_scale=1.0, sid=0,
                    language="JP", hps=HPS, net_g=net_g,
                    device=device, style_vec=STYLE_VEC,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i >= WARMUP:
                times.append(elapsed)

        avg = np.mean(times)
        audio_dur = len(audio) / HPS.data.sampling_rate
        results[name] = {"time": avg, "audio_dur": audio_dur, "rtf": audio_dur / avg}
        print(f"  {name:6s}: {avg:.3f}s (audio {audio_dur:.1f}s, RTF {audio_dur/avg:.1f}x)")

    # 정리
    del net_g
    bert_models.unload_model()
    bert_models.unload_tokenizer()
    torch.cuda.empty_cache()
    gc.collect()

    return label, results


def benchmark_onnx(provider: str, bert_path: Path, synth_path: Path, label: str):
    """ONNX Runtime 벤치마크."""
    import onnxruntime as ort
    from hayakoe.models.infer_onnx import infer_onnx

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    providers = [provider] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    bert_sess = ort.InferenceSession(str(bert_path), sess_opts, providers=providers)
    synth_sess = ort.InferenceSession(str(synth_path), sess_opts, providers=providers)

    actual_provider = bert_sess.get_providers()[0]
    print(f"  Provider: {actual_provider}")

    results = {}
    for name, text in TEST_TEXTS.items():
        times = []
        for i in range(WARMUP + RUNS):
            if "CUDA" in provider:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = infer_onnx(
                text=text, style_vec=STYLE_VEC, sdp_ratio=0.2,
                noise_scale=0.6, noise_scale_w=0.8, length_scale=1.0,
                sid=0, language="JP", hps=HPS,
                bert_session=bert_sess, synth_session=synth_sess,
            )
            if "CUDA" in provider:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i >= WARMUP:
                times.append(elapsed)

        avg = np.mean(times)
        audio_dur = len(audio) / HPS.data.sampling_rate
        results[name] = {"time": avg, "audio_dur": audio_dur, "rtf": audio_dur / avg}
        print(f"  {name:6s}: {avg:.3f}s (audio {audio_dur:.1f}s, RTF {audio_dur/avg:.1f}x)")

    del bert_sess, synth_sess
    gc.collect()

    return label, results


def print_summary(all_results: list[tuple[str, dict]]):
    """결과 요약 테이블."""
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")

    header = f"{'Config':40s} | {'short':>8s} | {'med':>8s} | {'long':>8s} | {'RTF(long)':>9s}"
    print(header)
    print("-" * len(header))

    for label, results in all_results:
        short = results.get("short", {}).get("time", 0)
        med = results.get("med", {}).get("time", 0)
        long_ = results.get("long", {}).get("time", 0)
        rtf = results.get("long", {}).get("rtf", 0)
        print(f"{label:40s} | {short:7.3f}s | {med:7.3f}s | {long_:7.3f}s | {rtf:8.1f}x")


def main():
    paths = download_models()
    load_common(paths["speaker_pt"])

    all_results = []

    # 1. PyTorch FP32 GPU
    all_results.append(benchmark_pytorch("cuda", bert_fp16=False, paths=paths))

    # 2. PyTorch FP16(BERT) GPU
    all_results.append(benchmark_pytorch("cuda", bert_fp16=True, paths=paths))

    # 3. ONNX FP32 CPU
    all_results.append(benchmark_onnx(
        "CPUExecutionProvider",
        paths["bert_onnx_fp32"], paths["synth_onnx_fp32"],
        "ONNX FP32 CPU",
    ))

    # 4. ONNX Q8 CPU (BERT Q8 + Synth FP32)
    all_results.append(benchmark_onnx(
        "CPUExecutionProvider",
        paths["bert_onnx_q8"], paths["synth_onnx_fp32"],
        "ONNX Q8(BERT)+FP32(Synth) CPU",
    ))

    # 5. ONNX FP32 GPU
    all_results.append(benchmark_onnx(
        "CUDAExecutionProvider",
        paths["bert_onnx_fp32"], paths["synth_onnx_fp32"],
        "ONNX FP32 GPU",
    ))

    # 6. ONNX Q8 GPU (BERT Q8 + Synth FP32)
    all_results.append(benchmark_onnx(
        "CUDAExecutionProvider",
        paths["bert_onnx_q8"], paths["synth_onnx_fp32"],
        "ONNX Q8(BERT)+FP32(Synth) GPU",
    ))

    print_summary(all_results)


if __name__ == "__main__":
    main()
