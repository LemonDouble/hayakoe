"""전사 결과 확인/수정 API."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from services import video_manager

router = APIRouter(prefix="/videos/{video_id}/review", tags=["review"])


class EditRequest(BaseModel):
    file: str
    text: str


class DeleteRequest(BaseModel):
    file: str


def _video_dir(video_id: str) -> Path:
    try:
        return video_manager.get_dir(config.get().data_dir, video_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _read_transcription(video_dir: Path) -> list[dict]:
    p = video_dir / "transcription.json"
    if not p.exists():
        raise HTTPException(status_code=400, detail="전사 데이터가 없습니다.")
    return json.loads(p.read_text())


def _write_transcription(video_dir: Path, data: list[dict]):
    tmp = video_dir / "transcription.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.rename(video_dir / "transcription.json")


@router.get("")
def get_transcriptions(video_id: str):
    """전사 결과 목록 조회."""
    vdir = _video_dir(video_id)
    entries = _read_transcription(vdir)
    return {"entries": entries, "total": len(entries)}


@router.post("/edit")
def edit_transcription(video_id: str, body: EditRequest):
    """세그먼트 전사 텍스트 수정."""
    vdir = _video_dir(video_id)
    entries = _read_transcription(vdir)

    found = False
    for entry in entries:
        if entry["file"] == body.file:
            entry["text"] = body.text
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"세그먼트 '{body.file}'를 찾을 수 없습니다.")

    _write_transcription(vdir, entries)
    return {"edited": body.file}


@router.post("/delete")
def delete_transcription(video_id: str, body: DeleteRequest):
    """세그먼트 전사 삭제."""
    vdir = _video_dir(video_id)
    entries = _read_transcription(vdir)

    new_entries = [e for e in entries if e["file"] != body.file]
    if len(new_entries) == len(entries):
        raise HTTPException(status_code=404, detail=f"세그먼트 '{body.file}'를 찾을 수 없습니다.")

    _write_transcription(vdir, new_entries)
    return {"deleted": body.file, "remaining": len(new_entries)}


@router.post("/done")
def mark_review_done(video_id: str):
    """검토 완료 표시."""
    vdir = _video_dir(video_id)
    if not (vdir / "transcription.json").exists():
        raise HTTPException(status_code=400, detail="전사 데이터가 없습니다.")

    done_file = vdir / "review_done.json"
    done_file.write_text(json.dumps({"done": True}))
    return {"done": True}
