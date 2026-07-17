"""요청 경로가 허용된 base 디렉토리를 벗어나지 못하게 하는 헬퍼.

video_id / 세그먼트 파일명 / 화자명처럼 외부 입력이 파일 경로 조각으로
쓰이는 지점에서, '..' 나 절대경로로 base 밖을 건드리는 path traversal 을 막는다.
"""

from pathlib import Path


class UnsafePathError(Exception):
    """이어붙인 경로가 base 디렉토리 밖으로 벗어남 (path traversal)."""


def safe_join(base: Path, *parts: str) -> Path:
    """base 아래에 parts 를 이어붙인 절대경로를 반환한다.

    resolve() 로 정규화한 뒤 base 안에 있지 않으면 UnsafePathError 를 던진다.
    parts 에 절대경로가 섞여 있어도 base 를 벗어나므로 거부된다.
    """
    candidate = base.joinpath(*parts).resolve()
    base_resolved = base.resolve()
    if not candidate.is_relative_to(base_resolved):
        raise UnsafePathError(
            "허용 범위 밖 경로입니다: " + "/".join(str(p) for p in parts)
        )
    return candidate
