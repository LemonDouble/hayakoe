"""데이터셋 생성 API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from services import dataset

router = APIRouter(prefix="/dataset", tags=["dataset"])


class BuildRequest(BaseModel):
    val_ratio: float


@router.post("/build")
def build_dataset(body: BuildRequest):
    """전체 영상의 분류+전사 데이터를 합쳐 SBV2 데이터셋 생성."""
    try:
        result = dataset.build(config.get().data_dir, val_ratio=body.val_ratio)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
