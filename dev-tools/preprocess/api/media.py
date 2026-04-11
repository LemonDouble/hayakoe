"""오디오/비디오 파일 서빙 (Range 요청 지원)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

import config

router = APIRouter(prefix="/media", tags=["media"])

_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


@router.get("/{path:path}")
def serve_media(path: str, request: Request):
    """data_dir 내 미디어 파일 서빙."""
    file_path = config.get().data_dir / path

    # 경로 탈출 방지
    try:
        file_path.resolve().relative_to(config.get().data_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="접근 불가")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    suffix = file_path.suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")

    range_header = request.headers.get("range")
    if range_header:
        return _range_response(file_path, media_type, range_header)

    return FileResponse(file_path, media_type=media_type)


def _range_response(file_path: Path, media_type: str, range_header: str) -> StreamingResponse:
    file_size = file_path.stat().st_size
    range_spec = range_header.replace("bytes=", "")
    parts = range_spec.split("-")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else file_size - 1
    end = min(end, file_size - 1)
    content_length = end - start + 1

    def iter_file():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(8192, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )
