"""영상 디렉토리 관리 + 디스크 기반 상태 판정."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


# 파이프라인 단계 순서
STAGES = ["extract", "separate", "vad", "classify", "transcribe", "review"]


def _videos_dir(data_dir: Path) -> Path:
    return data_dir / "videos"


def _next_id(data_dir: Path) -> str:
    vdir = _videos_dir(data_dir)
    existing = [d.name for d in vdir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not existing:
        return "001"
    return f"{max(int(n) for n in existing) + 1:03d}"


def create(data_dir: Path, original_filename: str, source_data: bytes) -> dict:
    """영상 업로드 → 디렉토리 생성 + source 파일 저장."""
    video_id = _next_id(data_dir)
    video_dir = _videos_dir(data_dir) / video_id
    video_dir.mkdir(parents=True)

    suffix = Path(original_filename).suffix
    source_path = video_dir / f"source{suffix}"
    source_path.write_bytes(source_data)

    meta = {
        "original_filename": original_filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (video_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return {"id": video_id, "filename": original_filename}


def list_all(data_dir: Path) -> list[dict]:
    vdir = _videos_dir(data_dir)
    if not vdir.exists():
        return []
    result = []
    for d in sorted(vdir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        result.append({
            "id": d.name,
            "filename": meta["original_filename"],
            "uploaded_at": meta["uploaded_at"],
            "stage": get_stage(d),
        })
    return result


def get_error(video_dir: Path) -> dict | None:
    """error.json에서 에러 정보 읽기."""
    p = video_dir / "error.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def clear_error(video_dir: Path):
    """error.json 삭제."""
    p = video_dir / "error.json"
    if p.is_file():
        p.unlink()


def get_stage(video_dir: Path) -> str:
    """디스크 파일 존재 여부로 현재 단계 판정."""
    # 처리 중이면 processing 상태
    processing = video_dir / "processing.json"
    if processing.exists():
        try:
            data = json.loads(processing.read_text())
            return f"processing:{data['stage']}"
        except (json.JSONDecodeError, KeyError):
            pass

    if (video_dir / "review_done.json").exists():
        return "done"

    if (video_dir / "transcription.json").exists():
        return "review"

    cls_path = video_dir / "classification.json"
    if cls_path.exists():
        try:
            cls = json.loads(cls_path.read_text())
            if cls.get("done"):
                return "transcribe"
        except json.JSONDecodeError:
            pass
        return "classifying"

    if (video_dir / "vad.json").exists():
        return "classify"

    if (video_dir / "vocals.wav").exists():
        return "vad"

    if (video_dir / "extracted.wav").exists():
        return "separate"

    if find_source(video_dir):
        return "extract"

    return "empty"


def get_meta(video_dir: Path) -> dict:
    """meta.json 읽기."""
    meta_path = video_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def get_processing_info(video_dir: Path) -> dict | None:
    """processing.json에서 진행률 정보 읽기."""
    p = video_dir / "processing.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def find_source(video_dir: Path) -> Path | None:
    for f in video_dir.iterdir():
        if f.name.startswith("source.") and f.is_file():
            return f
    return None


def get_dir(data_dir: Path, video_id: str) -> Path:
    vdir = _videos_dir(data_dir) / video_id
    if not vdir.exists():
        raise FileNotFoundError(f"영상 '{video_id}'를 찾을 수 없습니다.")
    return vdir


def delete(data_dir: Path, video_id: str):
    shutil.rmtree(get_dir(data_dir, video_id))


def rollback(video_dir: Path, to_stage: str):
    """지정 단계부터 이후 모든 데이터 삭제."""
    if to_stage not in STAGES:
        raise ValueError(f"잘못된 단계: {to_stage}")

    idx = STAGES.index(to_stage)
    # 뒤에서부터 삭제 (downstream first)
    for stage in reversed(STAGES[idx:]):
        _clear_stage(video_dir, stage)

    # 에러 상태 초기화
    clear_error(video_dir)

    # tmp 파일 정리
    for tmp in video_dir.glob("*.tmp"):
        tmp.unlink() if tmp.is_file() else shutil.rmtree(tmp)
    tmp_seg = video_dir / "segments.tmp"
    if tmp_seg.exists():
        shutil.rmtree(tmp_seg)


def _clear_stage(video_dir: Path, stage: str):
    if stage == "review":
        _rm_file(video_dir / "review_done.json")
    elif stage == "transcribe":
        _rm_file(video_dir / "transcription.json")
        _rm_file(video_dir / "review_done.json")
    elif stage == "classify":
        _rollback_classification(video_dir)
        _rm_file(video_dir / "classification.json")
    elif stage == "vad":
        _rm_file(video_dir / "vad.json")
        _rm_dir(video_dir / "segments")
    elif stage == "separate":
        _rm_file(video_dir / "vocals.wav")
    elif stage == "extract":
        _rm_file(video_dir / "extracted.wav")


def _rollback_classification(video_dir: Path):
    """분류된 세그먼트를 전부 unclassified로 복원."""
    segments_dir = video_dir / "segments"
    if not segments_dir.exists():
        return
    unclassified = segments_dir / "unclassified"
    unclassified.mkdir(exist_ok=True)

    for d in list(segments_dir.iterdir()):
        if not d.is_dir() or d.name == "unclassified":
            continue
        for f in d.iterdir():
            if f.is_file():
                shutil.move(str(f), str(unclassified / f.name))
        shutil.rmtree(d)


def _rm_file(p: Path):
    if p.is_file():
        p.unlink()


def _rm_dir(p: Path):
    if p.is_dir():
        shutil.rmtree(p)
