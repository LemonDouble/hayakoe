"""벤치마크 실행기 — TTS API로 추론 성능 측정."""

import contextlib
import gc
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from cli.ui.console import console


@contextlib.contextmanager
def _quiet_loading():
    """모델 로딩 시 라이브러리 노이즈를 억제한다.

    억제 대상:
      - hayakoe loguru 로거 (INFO)
      - HuggingFace Hub 다운로드 progress bar + 경고
      - tqdm progress bar (safetensors loading 등)
    """
    from loguru import logger as loguru_logger

    # loguru: 핸들러를 임시로 제거
    loguru_logger.disable("hayakoe")

    # HF Hub: progress bar 끄기
    old_hf_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # tqdm / HF warnings 억제
    old_tqdm_disable = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unauthenticated")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        try:
            yield
        finally:
            loguru_logger.enable("hayakoe")
            if old_hf_progress is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_hf_progress
            if old_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = old_tqdm_disable
            logging.getLogger("huggingface_hub").setLevel(logging.NOTSET)


WARMUP = 2
RUNS = 5
SPEAKER_NAME = "jvnv-F1-jp"

TEST_TEXTS = {
    "짧음": "こんにちは。",
    "중간": "私はイレイナ。旅の魔女です。あちこちを旅しています。",
    "김": (
        "吾輩は猫である。名前はまだ無い。"
        "どこで生れたかとんと見当がつかぬ。"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
    ),
}


@dataclass
class BenchmarkResult:
    device: str
    backend: str
    text_label: str
    text: str
    avg_time: float
    audio_duration: float

    @property
    def speed(self) -> float:
        """배속 (audio_duration / inference_time). 높을수록 빠름."""
        if self.avg_time == 0:
            return 0
        return self.audio_duration / self.avg_time


def _benchmark_device(device: str, progress, task, *, compile: bool = False) -> list[BenchmarkResult]:
    """단일 디바이스에서 벤치마크를 실행한다."""
    from hayakoe import TTS

    if compile:
        backend = "torch.compile"
    elif device == "cpu":
        backend = "ONNX"
    else:
        backend = "PyTorch"
    progress.update(task, description=f"[dim]{backend} ({device.upper()})[/dim] 모델 로딩...")

    with _quiet_loading():
        tts = TTS(device=device)
        speaker = tts.load(SPEAKER_NAME)
        if compile:
            speaker.optimize()
    sr = speaker._hps.data.sampling_rate

    results = []
    for label, text in TEST_TEXTS.items():
        progress.update(task, description=f"[dim]{backend} ({device.upper()})[/dim] {label}...")
        times = []

        for i in range(WARMUP + RUNS):
            if device != "cpu":
                import torch
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            audio_result = speaker.generate(text)
            audio_len = len(audio_result.data)

            if device != "cpu":
                import torch
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            if i >= WARMUP:
                times.append(elapsed)

        audio_duration = audio_len / sr
        results.append(BenchmarkResult(
            device=device,
            backend=backend,
            text_label=label,
            text=text,
            avg_time=float(np.mean(times)),
            audio_duration=audio_duration,
        ))
        progress.advance(task)

    # 정리
    del tts, speaker
    gc.collect()
    if device != "cpu":
        import torch
        torch.cuda.empty_cache()

    return results


def run_benchmark(devices: list[str], *, compile: bool = False) -> Path:
    """벤치마크를 실행하고 HTML 리포트를 생성한다.

    Returns:
        생성된 HTML 파일 경로.
    """
    total_steps = len(devices) * len(TEST_TEXTS)
    if compile:
        total_steps += len(TEST_TEXTS)  # torch.compile 벤치마크 추가
    all_results: list[BenchmarkResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("벤치마크 중", total=total_steps)

        for device in devices:
            results = _benchmark_device(device, progress, task)
            all_results.extend(results)

        if compile:
            results = _benchmark_device("cuda", progress, task, compile=True)
            all_results.extend(results)

    # 결과 테이블 출력
    console.print()
    console.print(
        "  [dim]배속 = 오디오 길이 ÷ 추론 시간 (높을수록 빠름)\n"
        "  예: 10.0x → 1초 분량의 음성을 0.1초 만에 생성[/dim]"
    )
    _print_summary(all_results)

    # HTML 리포트 생성
    from cli.benchmark.report import build_benchmark_html

    html = build_benchmark_html(all_results)

    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_{ts}.html"
    output_path.write_text(html, encoding="utf-8")

    return output_path


def _print_summary(results: list[BenchmarkResult]):
    """터미널에 요약 테이블 출력."""
    from rich.table import Table

    table = Table(
        title="벤치마크 결과",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        padding=(0, 1),
    )
    table.add_column("백엔드", style="bold bright_white")
    table.add_column("텍스트", style="bright_white")
    table.add_column("추론 시간", justify="right")
    table.add_column("오디오 길이", justify="right")
    table.add_column("배속", justify="right")

    for r in results:
        speed_style = "bold green" if r.speed >= 10 else "bold yellow" if r.speed >= 1 else "bold red"
        table.add_row(
            f"{r.backend} ({r.device.upper()})",
            r.text_label,
            f"{r.avg_time:.3f}s",
            f"{r.audio_duration:.1f}s",
            f"[{speed_style}]{r.speed:.1f}x[/{speed_style}]",
        )

    console.print()
    console.print(table)
