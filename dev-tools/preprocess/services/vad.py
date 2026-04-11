"""Silero VAD 음성 세그먼팅."""

import asyncio
import json
from pathlib import Path

import soundfile as sf
from loguru import logger

# 모델 lazy load
_model = None
_utils = None


def _get_model():
    global _model, _utils
    if _model is None:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _model = model
        _utils = utils
    return _model, _utils


async def segment_audio(
    audio_path: Path,
    segments_dir: Path,
    min_segment_sec: float = 1.0,
    max_segment_sec: float = 15.0,
    threshold: float = 0.5,
    min_silence_ms: int = 50,
    progress_callback=None,
) -> dict:
    """Silero VAD로 음성 구간 감지 → 세그먼트 WAV 저장.

    Returns:
        {"source": "...", "segments": [...], "total_segments": N}
    """
    unclassified_dir = segments_dir / "unclassified"
    unclassified_dir.mkdir(parents=True, exist_ok=True)

    def _run_vad():
        import torch
        import librosa

        model, utils = _get_model()
        get_speech_timestamps = utils[0]

        audio_np, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        wav = torch.from_numpy(audio_np)

        timestamps = get_speech_timestamps(
            wav, model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=int(min_segment_sec * 1000),
            min_silence_duration_ms=min_silence_ms,
            return_seconds=True,
        )
        return timestamps

    logger.info(f"VAD 분석: {audio_path.name}")
    if progress_callback:
        await progress_callback(0.05, "VAD 모델 분석 중...")
    timestamps = await asyncio.to_thread(_run_vad)

    if progress_callback:
        await progress_callback(0.3, "오디오 로딩 중...")
    # 원본 SR로 오디오 로드 (세그먼트 저장용)
    audio_data, sr = await asyncio.to_thread(sf.read, str(audio_path))

    # 청크 목록 미리 계산
    all_chunks = []
    for ts in timestamps:
        start_sec = ts["start"]
        end_sec = ts["end"]
        duration = end_sec - start_sec

        if duration < min_segment_sec:
            continue

        if duration > max_segment_sec:
            t = start_sec
            while t < end_sec:
                chunk_end = min(t + max_segment_sec, end_sec)
                if chunk_end - t >= min_segment_sec:
                    all_chunks.append((t, chunk_end))
                t = chunk_end
        else:
            all_chunks.append((start_sec, end_sec))

    total_chunks = len(all_chunks)
    segments = []

    for seg_idx, (cs, ce) in enumerate(all_chunks):
        start_sample = int(cs * sr)
        end_sample = int(ce * sr)
        segment_audio = audio_data[start_sample:end_sample]

        seg_name = f"seg_{seg_idx:04d}.wav"
        sf.write(str(unclassified_dir / seg_name), segment_audio, sr)

        segments.append({
            "file": seg_name,
            "start": round(cs, 3),
            "end": round(ce, 3),
            "duration": round(ce - cs, 3),
        })

        if progress_callback and total_chunks > 0:
            p = 0.35 + (seg_idx + 1) / total_chunks * 0.6
            await progress_callback(p, f"세그먼트 저장 중... ({seg_idx + 1}/{total_chunks})")

    result = {
        "source": audio_path.name,
        "segments": segments,
        "total_segments": len(segments),
    }

    logger.info(f"VAD 완료: {len(segments)}개 세그먼트")
    return result
