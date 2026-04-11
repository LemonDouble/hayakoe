"""Whisper 전사 (화자 배정된 세그먼트만)."""

import asyncio
import json
from pathlib import Path

from loguru import logger

import config

# 모델 lazy load
_model = None


def _get_model():
    global _model
    if _model is None:
        import whisper
        settings = config.get()
        _model = whisper.load_model(settings.whisper_model, device=settings.device)
    return _model


async def transcribe_video(
    video_dir: Path,
    data_dir: Path,
    language: str = "ja",
    progress_callback=None,
) -> list[dict]:
    """영상의 화자 배정 세그먼트를 Whisper로 전사.

    Returns:
        [{"file": "seg_0001.wav", "speaker": "나나미", "text": "...", "language": "ja"}, ...]
    """
    from services import speakers as speakers_svc

    speaker_list = speakers_svc.load(data_dir)
    segments_dir = video_dir / "segments"

    # 전사 대상: 화자에 배정된 세그먼트만
    wav_files = []
    for name in speaker_list:
        speaker_seg = segments_dir / name
        if not speaker_seg.exists():
            continue
        for wav in sorted(speaker_seg.glob("*.wav")):
            wav_files.append((wav, name))

    if not wav_files:
        return []

    total = len(wav_files)
    results = []

    def _transcribe_one(path: Path) -> str:
        model = _get_model()
        result = model.transcribe(str(path), language=language, task="transcribe")
        return result["text"].strip()

    for i, (wav_path, speaker) in enumerate(wav_files):
        if progress_callback:
            await progress_callback(i / total, f"전사 중: {wav_path.name} ({i + 1}/{total})")

        text = await asyncio.to_thread(_transcribe_one, wav_path)
        results.append({
            "file": wav_path.name,
            "speaker": speaker,
            "text": text,
            "language": language,
        })

    # atomic write
    output_path = video_dir / "transcription.json"
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    tmp.rename(output_path)

    logger.info(f"전사 완료: {len(results)}개 세그먼트")
    return results
