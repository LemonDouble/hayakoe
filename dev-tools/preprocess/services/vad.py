"""Silero VAD 음성 세그먼팅."""

import asyncio
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


def _split_into_chunks(
    timestamps: list[dict],
    min_segment_sec: float,
    max_segment_sec: float,
) -> list[tuple[float, float]]:
    """VAD 타임스탬프를 min/max 길이 제약에 맞는 (start, end) 청크 목록으로 변환."""
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
    return all_chunks


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

        # 전체 디코드 2회를 피하기 위해 원본 SR로 한 번만 읽고,
        # VAD 입력(16kHz mono)은 메모리에서 변환.
        # sf.read는 (frames, channels) 배열이라 to_mono 전에 전치가 필요하고,
        # dtype="float32"는 Silero float32 모델과의 dtype 일치 조건.
        audio_data, sr = sf.read(str(audio_path), dtype="float32")
        y = audio_data.T if audio_data.ndim > 1 else audio_data
        y = librosa.to_mono(y)
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        wav = torch.from_numpy(y)

        timestamps = get_speech_timestamps(
            wav, model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=int(min_segment_sec * 1000),
            min_silence_duration_ms=min_silence_ms,
            return_seconds=True,
        )
        return timestamps, audio_data, sr

    logger.info(f"VAD 분석: {audio_path.name}")
    if progress_callback:
        progress_callback(0.05, "VAD 모델 분석 중...")
    timestamps, audio_data, sr = await asyncio.to_thread(_run_vad)

    if progress_callback:
        progress_callback(0.3, "세그먼트 저장 준비 중...")

    def _save_segments():
        # 수백 개 sf.write가 이벤트 루프를 블록하지 않도록 저장 루프 전체를 워커 스레드에서 실행
        all_chunks = _split_into_chunks(timestamps, min_segment_sec, max_segment_sec)
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
                progress_callback(p, f"세그먼트 저장 중... ({seg_idx + 1}/{total_chunks})")
        return segments

    segments = await asyncio.to_thread(_save_segments)

    result = {
        "source": audio_path.name,
        "segments": segments,
        "total_segments": len(segments),
    }

    logger.info(f"VAD 완료: {len(segments)}개 세그먼트")
    return result
