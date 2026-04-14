"""Measure HayaKoe memory footprint for a single scenario.

Run as subprocess so each scenario starts from a clean Python heap.

Usage:
    python run_one.py --device {cpu,cuda} --scenario {idle1,idle4,gen1,seq4,conc4}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any

import psutil


SPEAKERS = ["jvnv-F1-jp", "jvnv-F2-jp", "jvnv-M1-jp", "jvnv-M2-jp"]
TEXT_MEDIUM = "旅の途中で不思議な街に辿り着きました。石畳の道を歩いていると、小さなカフェが見えてきました。"


class MemoryPoller:
    """Background thread that samples process RSS at a fixed interval."""

    def __init__(self, interval: float = 0.05) -> None:
        self._proc = psutil.Process()
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0

    def start(self) -> None:
        self._running = True
        self.peak_rss_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def measure_base_rss() -> int:
    return psutil.Process().memory_info().rss


def load_tts(device: str, speaker_names: list[str]):
    from hayakoe import TTS

    tts = TTS(device=device)
    for name in speaker_names:
        tts.load(name)
    tts.prepare()
    return tts


def gpu_peak_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_idle(device: str, n_speakers: int) -> dict[str, Any]:
    speakers = SPEAKERS[:n_speakers]

    poller = MemoryPoller()
    poller.start()

    base_rss = measure_base_rss()
    tts = load_tts(device, speakers)

    # Let memory settle briefly
    time.sleep(0.5)

    rss_after_load = psutil.Process().memory_info().rss
    poller.stop()

    return {
        "base_rss_mb": base_rss / (1024 * 1024),
        "after_load_rss_mb": rss_after_load / (1024 * 1024),
        "peak_rss_mb": poller.peak_rss_bytes / (1024 * 1024),
        "gpu_peak_mb": gpu_peak_mb(),
    }


def run_gen(device: str, n_speakers: int, concurrent: bool) -> dict[str, Any]:
    speakers = SPEAKERS[:n_speakers]

    poller = MemoryPoller()
    poller.start()

    base_rss = measure_base_rss()
    tts = load_tts(device, speakers)

    # Warmup each speaker once (drops torch.compile cold-start from peak)
    for name in speakers:
        tts.speakers[name].generate(TEXT_MEDIUM)

    reset_gpu_peak()
    time.sleep(0.3)

    rss_after_load = psutil.Process().memory_info().rss

    # Reset peak tracker after warmup/load to isolate the generation workload
    poller.peak_rss_bytes = rss_after_load

    if concurrent and len(speakers) > 1:
        threads = []
        for name in speakers:
            t = threading.Thread(
                target=lambda n=name: tts.speakers[n].generate(TEXT_MEDIUM)
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for name in speakers:
            tts.speakers[name].generate(TEXT_MEDIUM)

    time.sleep(0.3)
    poller.stop()

    return {
        "base_rss_mb": base_rss / (1024 * 1024),
        "after_load_rss_mb": rss_after_load / (1024 * 1024),
        "peak_rss_mb": poller.peak_rss_bytes / (1024 * 1024),
        "gpu_peak_mb": gpu_peak_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["idle1", "idle4", "gen1", "seq4", "conc4"],
    )
    args = parser.parse_args()

    if args.scenario == "idle1":
        result = run_idle(args.device, 1)
    elif args.scenario == "idle4":
        result = run_idle(args.device, 4)
    elif args.scenario == "gen1":
        result = run_gen(args.device, 1, concurrent=False)
    elif args.scenario == "seq4":
        result = run_gen(args.device, 4, concurrent=False)
    elif args.scenario == "conc4":
        result = run_gen(args.device, 4, concurrent=True)
    else:
        raise ValueError(args.scenario)

    result["device"] = args.device
    result["scenario"] = args.scenario
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
