"""전체 영상의 분류+전사 데이터를 합쳐서 SBV2 학습 데이터셋 생성."""

import json
import random
import shutil
from pathlib import Path

import soundfile as sf
from loguru import logger

from services import speakers as speakers_svc

DEFAULT_STYLES = {
    "Neutral": 0,
    "Happy": 1,
    "Sad": 2,
    "Angry": 3,
    "Fear": 4,
    "Surprise": 5,
    "Disgust": 6,
}


def build(data_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> dict:
    """모든 영상에서 분류+전사 완료된 세그먼트를 합쳐 dataset/ 생성.

    Returns:
        {"speakers": {"나나미": {"count": N}}, "total": M}
    """
    speaker_list = speakers_svc.load(data_dir)
    if not speaker_list:
        raise ValueError("등록된 화자가 없습니다.")

    dataset_dir = data_dir / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    videos_dir = data_dir / "videos"
    # 영상별 전사 데이터 수집
    transcript_map: dict[str, dict[str, str]] = {}  # speaker → {filename: text}
    for vdir in sorted(videos_dir.iterdir()):
        if not vdir.is_dir():
            continue
        t_path = vdir / "transcription.json"
        if not t_path.exists():
            continue
        for entry in json.loads(t_path.read_text()):
            speaker = entry["speaker"]
            if speaker not in transcript_map:
                transcript_map[speaker] = {}
            # 파일명 충돌 방지: video_id prefix
            unique_name = f"{vdir.name}_{entry['file']}"
            transcript_map[speaker][unique_name] = entry["text"]

    # 화자별 오디오 복사 + esd 엔트리 생성
    speaker_entries: dict[str, list[str]] = {}
    speaker_durations: dict[str, list[float]] = {}  # entry별 duration

    for speaker in speaker_list:
        speaker_dir = dataset_dir / speaker
        audio_dir = speaker_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        durations = []
        for vdir in sorted(videos_dir.iterdir()):
            if not vdir.is_dir():
                continue
            seg_dir = vdir / "segments" / speaker
            if not seg_dir.exists():
                continue
            for wav in sorted(seg_dir.glob("*.wav")):
                unique_name = f"{vdir.name}_{wav.name}"
                text = transcript_map.get(speaker, {}).get(unique_name, "")
                if not text:
                    continue
                dst = audio_dir / unique_name
                shutil.copy2(wav, dst)
                entries.append(f"{dst.resolve()}|{speaker}|JP|{text}")
                info = sf.info(str(dst))
                durations.append(info.duration)

        speaker_entries[speaker] = entries
        speaker_durations[speaker] = durations

    total = sum(len(e) for e in speaker_entries.values())
    if total == 0:
        raise ValueError("유효한 데이터가 없습니다 (전사 결과 없음).")

    # 화자별 esd.list, train.list, val.list, config.json 생성
    speaker_stats = {}

    for speaker in speaker_list:
        entries = speaker_entries[speaker]
        durations = speaker_durations[speaker]
        if not entries:
            continue

        speaker_dir = dataset_dir / speaker

        # esd.list
        (speaker_dir / "esd.list").write_text("\n".join(entries) + "\n")

        # train/val 분할 (duration도 같은 순서로 셔플)
        random.seed(seed)
        paired = list(zip(entries, durations))
        random.shuffle(paired)
        val_count = max(1, int(len(paired) * val_ratio))
        val_pairs = paired[:val_count]
        train_pairs = paired[val_count:]

        (speaker_dir / "val.list").write_text("\n".join(e for e, _ in val_pairs) + "\n")
        (speaker_dir / "train.list").write_text("\n".join(e for e, _ in train_pairs) + "\n")

        total_dur = sum(durations)
        train_dur = sum(d for _, d in train_pairs)
        val_dur = sum(d for _, d in val_pairs)

        speaker_stats[speaker] = {
            "count": len(entries),
            "duration": round(total_dur, 1),
            "train_count": len(train_pairs),
            "train_duration": round(train_dur, 1),
            "val_count": len(val_pairs),
            "val_duration": round(val_dur, 1),
            "path": str(speaker_dir.resolve()),
        }

        # config.json
        spk2id = {speaker: 0}
        cfg = {
            "model_name": f"hayakoe_{speaker}",
            "version": "2.7.0-JP-Extra",
            "train": {
                "epochs": 500,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "lr_decay": 0.99996,
                "seed": seed,
                "log_interval": 200,
                "eval_interval": 1000,
                "segment_size": 16384,
            },
            "data": {
                "use_jp_extra": True,
                "training_files": str((speaker_dir / "train.list").resolve()),
                "validation_files": str((speaker_dir / "val.list").resolve()),
                "sampling_rate": 44100,
                "filter_length": 2048,
                "hop_length": 512,
                "win_length": 2048,
                "n_mel_channels": 128,
                "mel_fmin": 0.0,
                "mel_fmax": None,
                "add_blank": True,
                "n_speakers": 1,
                "cleaned_text": False,
                "spk2id": spk2id,
                "num_styles": len(DEFAULT_STYLES),
                "style2id": DEFAULT_STYLES,
            },
            "model": {
                "use_spk_conditioned_encoder": True,
                "use_noise_scaled_mas": True,
                "use_mel_posterior_encoder": False,
                "use_duration_discriminator": False,
                "use_wavlm_discriminator": True,
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 8, 2, 2],
                "n_layers_q": 3,
                "use_spectral_norm": False,
                "gin_channels": 512,
                "slm": {
                    "model": "microsoft/wavlm-base-plus",
                    "sr": 16000,
                    "hidden": 768,
                    "nlayers": 13,
                    "initial_channel": 64,
                },
            },
        }
        (speaker_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    logger.info(f"데이터셋 생성 완료: {total}개 엔트리, 화자 {len(speaker_stats)}명")
    return {
        "speakers": speaker_stats,
        "total": total,
        "dataset_dir": str(dataset_dir.resolve()),
    }
