"""영상별 세그먼트 분류 (history 기반 undo 지원)."""

import json
import shutil
from pathlib import Path

import soundfile as sf


def _cls_path(video_dir: Path) -> Path:
    return video_dir / "classification.json"


def _load(video_dir: Path) -> dict:
    p = _cls_path(video_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"done": False, "history": []}


def _save(video_dir: Path, state: dict):
    p = _cls_path(video_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.rename(p)


def get_state(video_dir: Path) -> dict:
    return _load(video_dir)


def classify(video_dir: Path, segment_file: str, speaker: str):
    """세그먼트를 화자 폴더(또는 discarded)로 이동."""
    segments_dir = video_dir / "segments"
    unclassified = segments_dir / "unclassified"
    src = unclassified / segment_file

    if not src.exists():
        raise FileNotFoundError(f"세그먼트를 찾을 수 없습니다: {segment_file}")

    dst_dir = segments_dir / speaker
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst_dir / segment_file))

    state = _load(video_dir)
    state["history"].append({"segment": segment_file, "speaker": speaker})
    _save(video_dir, state)


def undo(video_dir: Path) -> dict | None:
    """마지막 분류 되돌리기. 되돌린 항목 반환."""
    state = _load(video_dir)
    if not state["history"]:
        return None

    entry = state["history"].pop()
    segments_dir = video_dir / "segments"
    src = segments_dir / entry["speaker"] / entry["segment"]
    dst = segments_dir / "unclassified" / entry["segment"]

    if src.exists():
        shutil.move(str(src), str(dst))

    _save(video_dir, state)
    return entry


def mark_done(video_dir: Path):
    """분류 완료 표시."""
    state = _load(video_dir)
    state["done"] = True
    _save(video_dir, state)


def get_unclassified(video_dir: Path, offset: int = 0, limit: int = 20) -> dict:
    """미분류 세그먼트 목록 + VAD 메타데이터."""
    segments_dir = video_dir / "segments" / "unclassified"
    if not segments_dir.exists():
        return {"segments": [], "total": 0, "classified": 0, "total_all": 0}

    # VAD 메타데이터 로드
    seg_meta = {}
    vad_path = video_dir / "vad.json"
    if vad_path.exists():
        vad_data = json.loads(vad_path.read_text())
        for seg in vad_data.get("segments", []):
            seg_meta[seg["file"]] = seg

    all_files = sorted(f.name for f in segments_dir.glob("seg_*.wav"))
    page = all_files[offset:offset + limit]

    segments = []
    for fname in page:
        meta = seg_meta.get(fname, {})
        segments.append({
            "file": fname,
            "start": meta.get("start", 0),
            "end": meta.get("end", 0),
            "duration": meta.get("duration", 0),
        })

    # 전체 세그먼트 수 계산
    total_all = 0
    classified = 0
    parent = video_dir / "segments"
    if parent.exists():
        for d in parent.iterdir():
            if d.is_dir():
                count = len(list(d.glob("seg_*.wav")))
                total_all += count
                if d.name != "unclassified":
                    classified += count

    return {
        "segments": segments,
        "total": len(all_files),
        "classified": classified,
        "total_all": total_all,
    }


def get_video_summary(video_dir: Path, data_dir: Path) -> list[dict]:
    """특정 영상의 화자별 세그먼트 수 + 총 재생시간."""
    from services import speakers as speakers_svc

    speaker_list = speakers_svc.load(data_dir)
    segments_dir = video_dir / "segments"
    result = []

    for name in speaker_list:
        speaker_seg = segments_dir / name
        if not speaker_seg.exists():
            continue
        count = 0
        total_duration = 0.0
        for wav in speaker_seg.glob("*.wav"):
            count += 1
            total_duration += sf.info(str(wav)).duration
        if count > 0:
            result.append({
                "name": name,
                "count": count,
                "total_duration": round(total_duration, 1),
            })

    discarded_dir = segments_dir / "discarded"
    if discarded_dir.exists():
        d_count = len(list(discarded_dir.glob("*.wav")))
        if d_count > 0:
            d_dur = sum(sf.info(str(w)).duration for w in discarded_dir.glob("*.wav"))
            result.append({
                "name": "discarded",
                "count": d_count,
                "total_duration": round(d_dur, 1),
            })

    return result


def get_total_summary(data_dir: Path) -> list[dict]:
    """화자별 전체 영상 합산 세그먼트 수 + 총 재생시간."""
    from services import speakers as speakers_svc

    speaker_list = speakers_svc.load(data_dir)
    videos_dir = data_dir / "videos"
    result = []

    for name in speaker_list:
        count = 0
        total_duration = 0.0
        if videos_dir.exists():
            for vdir in videos_dir.iterdir():
                if not vdir.is_dir():
                    continue
                speaker_seg = vdir / "segments" / name
                if speaker_seg.exists():
                    for wav in speaker_seg.glob("*.wav"):
                        count += 1
                        total_duration += sf.info(str(wav)).duration
        result.append({
            "name": name,
            "count": count,
            "total_duration": round(total_duration, 1),
        })

    return result


def get_summary(video_dir: Path, data_dir: Path) -> list[dict]:
    """화자별 세그먼트 수 + 총 재생시간 (전체 영상 합산)."""
    from services import speakers as speakers_svc

    speaker_list = speakers_svc.load(data_dir)
    result = []

    for name in speaker_list:
        count = 0
        total_duration = 0.0
        # 모든 영상에서 해당 화자의 세그먼트 합산
        videos_dir = data_dir / "videos"
        if videos_dir.exists():
            for vdir in videos_dir.iterdir():
                if not vdir.is_dir():
                    continue
                speaker_seg = vdir / "segments" / name
                if speaker_seg.exists():
                    for wav in speaker_seg.glob("*.wav"):
                        count += 1
                        total_duration += sf.info(str(wav)).duration

        result.append({
            "name": name,
            "count": count,
            "total_duration": round(total_duration, 1),
        })

    # discarded 통계 (현재 영상만)
    discarded_dir = video_dir / "segments" / "discarded"
    if discarded_dir and discarded_dir.exists():
        discard_count = len(list(discarded_dir.glob("*.wav")))
        if discard_count > 0:
            result.append({
                "name": "discarded",
                "count": discard_count,
                "total_duration": 0,
            })

    return result
