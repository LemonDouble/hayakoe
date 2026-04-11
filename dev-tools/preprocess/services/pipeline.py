"""개별 파이프라인 단계 실행 + 멱등성 보장."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from services import ffmpeg, separator, vad, video_manager


def _write_processing(video_dir: Path, stage: str, progress: float, message: str = ""):
    """processing.json에 진행률 기록 (atomic write)."""
    data = {
        "stage": stage,
        "progress": progress,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = video_dir / "processing.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.rename(video_dir / "processing.json")


def _remove_processing(video_dir: Path):
    p = video_dir / "processing.json"
    if p.exists():
        p.unlink()


def _write_error(video_dir: Path, stage: str, error: str):
    """error.json에 실패 정보 기록."""
    data = {
        "stage": stage,
        "message": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = video_dir / "error.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.rename(video_dir / "error.json")


def _clear_error(video_dir: Path):
    """error.json 삭제 (재실행 시)."""
    p = video_dir / "error.json"
    if p.is_file():
        p.unlink()


def _cleanup_tmp(video_dir: Path, name: str):
    """이전 크래시에서 남은 .tmp 파일/디렉토리 삭제."""
    tmp = video_dir / name
    if tmp.is_file():
        tmp.unlink()
    elif tmp.is_dir():
        shutil.rmtree(tmp)


async def run_extract(video_dir: Path):
    """오디오 추출 (source → extracted.wav)."""
    try:
        _clear_error(video_dir)
        source = video_manager.find_source(video_dir)
        if source is None:
            raise FileNotFoundError("소스 파일이 없습니다.")

        extracted = video_dir / "extracted.wav"
        if extracted.exists():
            logger.info(f"이미 추출됨: {video_dir.name}")
            return

        _cleanup_tmp(video_dir, "extracted.wav.tmp")
        tmp = video_dir / "extracted.wav.tmp"
        _write_processing(video_dir, "extract", 0, "오디오 추출 중...")

        async def _extract_progress(p, msg):
            _write_processing(video_dir, "extract", p, msg)

        await ffmpeg.extract_audio(source, tmp, progress_callback=_extract_progress)
        tmp.rename(extracted)
        logger.info(f"추출 완료: {video_dir.name}")
    except Exception as e:
        logger.exception(f"추출 실패: {video_dir.name}")
        _write_error(video_dir, "extract", str(e))
    finally:
        _remove_processing(video_dir)


async def run_separate(video_dir: Path):
    """배경음 제거 (extracted.wav → vocals.wav)."""
    try:
        _clear_error(video_dir)
        extracted = video_dir / "extracted.wav"
        if not extracted.exists():
            raise FileNotFoundError("extracted.wav가 없습니다. 먼저 추출을 실행하세요.")

        vocals = video_dir / "vocals.wav"
        if vocals.exists():
            logger.info(f"이미 분리됨: {video_dir.name}")
            return

        _cleanup_tmp(video_dir, "vocals.wav.tmp")
        tmp = video_dir / "vocals.wav.tmp"
        _write_processing(video_dir, "separate", 0, "배경음 제거 중...")
        await separator.separate_vocals(extracted, tmp)
        tmp.rename(vocals)
        logger.info(f"배경음 제거 완료: {video_dir.name}")
    except Exception as e:
        logger.exception(f"배경음 제거 실패: {video_dir.name}")
        _write_error(video_dir, "separate", str(e))
    finally:
        _remove_processing(video_dir)


async def run_vad(video_dir: Path, **vad_params):
    """VAD 세그먼팅 (vocals.wav → vad.json + segments/)."""
    try:
        _clear_error(video_dir)
        vocals = video_dir / "vocals.wav"
        if not vocals.exists():
            raise FileNotFoundError("vocals.wav가 없습니다. 먼저 배경음 제거를 실행하세요.")

        vad_json = video_dir / "vad.json"
        if vad_json.exists():
            logger.info(f"이미 세그먼팅됨: {video_dir.name}")
            return

        _cleanup_tmp(video_dir, "segments.tmp")
        _cleanup_tmp(video_dir, "vad.json.tmp")
        segments_tmp = video_dir / "segments.tmp"
        _write_processing(video_dir, "vad", 0, "VAD 세그먼팅 중...")

        async def _vad_progress(p, msg):
            _write_processing(video_dir, "vad", p, msg)

        result = await vad.segment_audio(vocals, segments_tmp, progress_callback=_vad_progress, **vad_params)

        # vad.json atomic write
        vad_tmp = video_dir / "vad.json.tmp"
        vad_tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        # segments.tmp → segments
        segments_dir = video_dir / "segments"
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
        segments_tmp.rename(segments_dir)
        vad_tmp.rename(vad_json)

        logger.info(f"VAD 완료: {video_dir.name}, {result['total_segments']}개 세그먼트")
    except Exception as e:
        logger.exception(f"VAD 실패: {video_dir.name}")
        _write_error(video_dir, "vad", str(e))
    finally:
        _remove_processing(video_dir)


async def run_transcription(video_dir: Path, data_dir: Path):
    """전사 실행."""
    from services import transcription

    try:
        _clear_error(video_dir)

        async def _progress(p, msg):
            _write_processing(video_dir, "transcribe", p, msg)

        _write_processing(video_dir, "transcribe", 0, "전사 시작...")
        await transcription.transcribe_video(video_dir, data_dir, progress_callback=_progress)
        logger.info(f"전사 완료: {video_dir.name}")
    except Exception as e:
        logger.exception(f"전사 실패: {video_dir.name}")
        _write_error(video_dir, "transcribe", str(e))
    finally:
        _remove_processing(video_dir)
