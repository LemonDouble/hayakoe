"""영상 업로드/목록/상태/삭제/rollback + 개별 단계 실행 API."""

import asyncio

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

import config
from services import classification, pipeline, separator, video_manager

router = APIRouter(prefix="/videos", tags=["videos"])


class RollbackRequest(BaseModel):
    stage: str


class VadRequest(BaseModel):
    min_segment_sec: float
    max_segment_sec: float
    threshold: float
    min_silence_ms: int


@router.get("")
def list_videos():
    return {"videos": video_manager.list_all(config.get().data_dir)}


@router.post("/upload")
async def upload_video(file: UploadFile):
    """영상 업로드 (저장만, 파이프라인은 수동 실행)."""
    data = await file.read()
    result = video_manager.create(config.get().data_dir, file.filename, data)
    return result


@router.get("/{video_id}/status")
def get_status(video_id: str):
    """영상 현재 상태 (폴링용)."""
    video_dir = _get_dir(video_id)
    stage = video_manager.get_stage(video_dir)
    processing = video_manager.get_processing_info(video_dir)
    meta = video_manager.get_meta(video_dir)
    error = video_manager.get_error(video_dir)
    source = video_manager.find_source(video_dir)
    result = {
        "stage": stage,
        "processing": processing,
        "filename": meta.get("original_filename", video_id),
        "error": error,
        "source_file": source.name if source else None,
    }

    # done/review 단계에서 화자별 요약 포함
    if stage in ("done", "review"):
        result["summary"] = classification.get_video_summary(video_dir, config.get().data_dir)

    return result


@router.post("/{video_id}/extract")
async def start_extract(video_id: str):
    """오디오 추출 실행."""
    video_dir = _get_dir(video_id)
    asyncio.create_task(pipeline.run_extract(video_dir))
    return {"status": "started"}


@router.post("/{video_id}/separate")
async def start_separate(video_id: str):
    """배경음 제거 실행."""
    if separator.is_busy():
        raise HTTPException(
            status_code=409,
            detail="다른 영상의 배경음 제거가 진행 중입니다.",
        )
    video_dir = _get_dir(video_id)
    asyncio.create_task(pipeline.run_separate(video_dir))
    return {"status": "started"}


@router.post("/{video_id}/vad")
async def start_vad(video_id: str, body: VadRequest):
    """VAD 세그먼팅 실행."""
    video_dir = _get_dir(video_id)
    asyncio.create_task(pipeline.run_vad(video_dir, **body.model_dump()))
    return {"status": "started"}


@router.post("/{video_id}/transcribe")
async def start_transcription(video_id: str):
    """전사 실행."""
    video_dir = _get_dir(video_id)
    asyncio.create_task(pipeline.run_transcription(video_dir, config.get().data_dir))
    return {"status": "started"}


@router.post("/{video_id}/rollback")
def rollback_video(video_id: str, body: RollbackRequest):
    """특정 단계부터 재처리 (해당 단계 + 이후 데이터 삭제)."""
    video_dir = _get_dir(video_id)
    try:
        video_manager.rollback(video_dir, body.stage)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"stage": video_manager.get_stage(video_dir)}


@router.delete("/{video_id}")
def delete_video(video_id: str):
    try:
        video_manager.delete(config.get().data_dir, video_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": video_id}


def _get_dir(video_id: str):
    try:
        return video_manager.get_dir(config.get().data_dir, video_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
