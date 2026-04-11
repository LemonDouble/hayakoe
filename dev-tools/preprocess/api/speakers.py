"""화자 관리 API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from services import classification, speakers as svc

router = APIRouter(prefix="/speakers", tags=["speakers"])


class SpeakerCreate(BaseModel):
    name: str


class SpeakerRename(BaseModel):
    old_name: str
    new_name: str


@router.get("")
def list_speakers():
    return {"speakers": svc.load(config.get().data_dir)}


@router.post("")
def create_speaker(body: SpeakerCreate):
    try:
        speakers = svc.add(config.get().data_dir, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"speakers": speakers}


@router.put("")
def rename_speaker(body: SpeakerRename):
    try:
        speakers = svc.rename(config.get().data_dir, body.old_name, body.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"speakers": speakers}


@router.get("/summary")
def speakers_summary():
    """화자별 전체 영상 합산 세그먼트 수 + 총 길이."""
    summary = classification.get_total_summary(config.get().data_dir)
    return {"summary": summary}


@router.delete("/{name}")
def delete_speaker(name: str):
    try:
        speakers = svc.delete(config.get().data_dir, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"speakers": speakers}
