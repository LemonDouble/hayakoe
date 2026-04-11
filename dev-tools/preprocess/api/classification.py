"""세그먼트 분류 API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from services import classification as svc, video_manager

router = APIRouter(prefix="/videos/{video_id}/classification", tags=["classification"])


class ClassifyRequest(BaseModel):
    segment_file: str
    speaker: str  # 화자명 또는 "discarded"


def _video_dir(video_id: str):
    try:
        return video_manager.get_dir(config.get().data_dir, video_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("")
def get_classification(video_id: str):
    """분류 상태 조회."""
    vdir = _video_dir(video_id)
    state = svc.get_state(vdir)
    summary = svc.get_summary(vdir, config.get().data_dir)
    return {
        "done": state.get("done", False),
        "history_count": len(state.get("history", [])),
        "speakers": summary,
    }


@router.get("/segments")
def get_segments(video_id: str, offset: int = 0, limit: int = 20):
    """미분류 세그먼트 목록."""
    return svc.get_unclassified(_video_dir(video_id), offset, limit)


@router.post("/classify")
def classify_segment(video_id: str, body: ClassifyRequest):
    """세그먼트를 화자에 배정."""
    try:
        svc.classify(_video_dir(video_id), body.segment_file, body.speaker)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"classified": body.segment_file, "speaker": body.speaker}


@router.post("/undo")
def undo_classification(video_id: str):
    """마지막 분류 되돌리기."""
    entry = svc.undo(_video_dir(video_id))
    if entry is None:
        raise HTTPException(status_code=400, detail="되돌릴 분류가 없습니다.")
    return {"undone": entry}


@router.post("/done")
def mark_done(video_id: str):
    """분류 완료 표시."""
    svc.mark_done(_video_dir(video_id))
    return {"done": True}
