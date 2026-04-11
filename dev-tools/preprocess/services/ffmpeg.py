"""ffmpeg 오디오 추출."""

import asyncio
from pathlib import Path

from loguru import logger


async def extract_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 44100,
    progress_callback=None,
) -> Path:
    """미디어 파일에서 44.1kHz mono WAV 추출."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_duration = await get_duration(input_path) if progress_callback else 0

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        "-progress", "pipe:1",
        str(output_path),
    ]

    logger.info(f"ffmpeg: {input_path.name} → {output_path.name}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    async for line in proc.stdout:
        if progress_callback and total_duration > 0 and line.startswith(b"out_time_us="):
            try:
                us = int(line.split(b"=")[1])
                progress = min(us / 1_000_000 / total_duration, 0.99)
                await progress_callback(progress, "오디오 추출 중...")
            except (ValueError, ZeroDivisionError):
                pass

    await proc.wait()

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        raise RuntimeError(f"ffmpeg 실패: {stderr.decode()[-500:]}")

    return output_path


async def get_duration(path: Path) -> float:
    """ffprobe로 오디오 길이(초) 반환."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0
