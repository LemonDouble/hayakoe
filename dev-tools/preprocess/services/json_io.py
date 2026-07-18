"""JSON 파일 atomic write 유틸."""

import json
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any, indent: int | None = 2):
    """tmp 파일에 쓴 뒤 rename — 부분 쓰기로 인한 파일 손상 방지."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent))
    tmp.rename(path)
