"""화자 목록 관리 (speakers.json)."""

import json
import shutil
from pathlib import Path

from services.safe_path import safe_join


def _path(data_dir: Path) -> Path:
    return data_dir / "speakers.json"


def _validate_name(name: str) -> str:
    """화자 이름을 검증한다. 폴더명으로 쓰이므로 경로 구분자를 막는다."""
    name = name.strip()
    if not name:
        raise ValueError("화자 이름이 비어 있습니다.")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("화자 이름에 경로 구분자를 쓸 수 없습니다.")
    return name


def load(data_dir: Path) -> list[str]:
    p = _path(data_dir)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _save(data_dir: Path, speakers: list[str]):
    p = _path(data_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(speakers, ensure_ascii=False, indent=2))
    tmp.rename(p)


def add(data_dir: Path, name: str) -> list[str]:
    name = _validate_name(name)
    speakers = load(data_dir)
    if name in speakers:
        raise ValueError(f"화자 '{name}'이(가) 이미 존재합니다.")
    speakers.append(name)
    _save(data_dir, speakers)
    return speakers


def rename(data_dir: Path, old_name: str, new_name: str) -> list[str]:
    new_name = _validate_name(new_name)
    speakers = load(data_dir)
    if old_name not in speakers:
        raise ValueError(f"화자 '{old_name}'을(를) 찾을 수 없습니다.")
    if new_name in speakers:
        raise ValueError(f"화자 '{new_name}'이(가) 이미 존재합니다.")

    speakers[speakers.index(old_name)] = new_name
    _save(data_dir, speakers)

    # 모든 영상의 segments/{old_name} → {new_name} 폴더 rename
    videos_dir = data_dir / "videos"
    if not videos_dir.exists():
        return speakers
    for vdir in videos_dir.iterdir():
        if not vdir.is_dir():
            continue
        old_seg = safe_join(vdir / "segments", old_name)
        if old_seg.exists():
            old_seg.rename(safe_join(vdir / "segments", new_name))
        # classification.json 내 화자명 갱신
        _update_classification_speaker(vdir, old_name, new_name)

    return speakers


def delete(data_dir: Path, name: str) -> list[str]:
    speakers = load(data_dir)
    if name not in speakers:
        raise ValueError(f"화자 '{name}'을(를) 찾을 수 없습니다.")

    speakers.remove(name)
    _save(data_dir, speakers)

    # 모든 영상에서 해당 화자 세그먼트 → unclassified로 복원
    videos_dir = data_dir / "videos"
    if not videos_dir.exists():
        return speakers
    for vdir in videos_dir.iterdir():
        if not vdir.is_dir():
            continue
        speaker_seg = safe_join(vdir / "segments", name)
        unclassified = vdir / "segments" / "unclassified"
        if speaker_seg.exists() and unclassified.exists():
            for f in speaker_seg.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(unclassified / f.name))
            shutil.rmtree(speaker_seg)
        # classification.json에서 해당 화자 기록 제거
        _remove_classification_speaker(vdir, name)

    return speakers


def _update_classification_speaker(video_dir: Path, old_name: str, new_name: str):
    cls_path = video_dir / "classification.json"
    if not cls_path.exists():
        return
    cls = json.loads(cls_path.read_text())
    for entry in cls.get("history", []):
        if entry.get("speaker") == old_name:
            entry["speaker"] = new_name
    tmp = cls_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cls, ensure_ascii=False, indent=2))
    tmp.rename(cls_path)


def _remove_classification_speaker(video_dir: Path, name: str):
    cls_path = video_dir / "classification.json"
    if not cls_path.exists():
        return
    cls = json.loads(cls_path.read_text())
    cls["history"] = [e for e in cls.get("history", []) if e.get("speaker") != name]
    tmp = cls_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cls, ensure_ascii=False, indent=2))
    tmp.rename(cls_path)
