"""데이터셋 탐색, 상태 확인, 활성화."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cli.i18n import t


DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "dataset"


@dataclass
class DatasetInfo:
    name: str
    path: Path          # 프로젝트 루트 (training/, exports/ 위치)
    data_dir: Path      # 데이터 파일 위치 (esd.list, config.json, train.list 등)
    utterance_count: int
    train_count: int
    val_count: int
    # 전처리 상태
    text_preprocessed: bool
    bert_done: int
    bert_total: int
    style_done: int
    style_total: int
    default_style_done: bool
    has_checkpoints: bool

    @property
    def all_preprocessed(self) -> bool:
        return (
            self.text_preprocessed
            and self.bert_done == self.bert_total
            and self.style_done == self.style_total
            and self.default_style_done
        )

    @property
    def status_label(self) -> str:
        if self.all_preprocessed:
            if self.has_checkpoints:
                return t("training.dataset.status_resume")
            return t("training.dataset.status_ready")
        return t("training.dataset.status_need_preprocess")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_text_preprocessed(train_list: Path) -> bool:
    """train.list 첫 줄의 필드 수로 G2P 완료 여부 판정."""
    if not train_list.exists():
        return False
    with open(train_list, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return False
    return len(first_line.split("|")) >= 7


def _count_feature_files(train_list: Path, val_list: Path, suffix: str, append: bool = False) -> tuple[int, int]:
    """train/val 리스트의 wav 경로에 대해 feature 파일 존재 여부 카운트.

    append=False: .wav → .{suffix} 치환 (bert.pt 등)
    append=True:  .wav.{suffix} 추가 (npy 등)
    """
    total = 0
    done = 0
    for list_path in [train_list, val_list]:
        if not list_path.exists():
            continue
        with open(list_path, encoding="utf-8") as f:
            for line in f:
                wav_path = line.strip().split("|")[0]
                if not wav_path:
                    continue
                total += 1
                if append:
                    feature_path = f"{wav_path}.{suffix}"
                else:
                    feature_path = wav_path.replace(".WAV", ".wav").replace(".wav", f".{suffix}")
                if os.path.exists(feature_path):
                    done += 1
    return done, total


def _get_model_name(config_path: Path) -> str:
    if not config_path.exists():
        return ""
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("model_name", "")


def _detect_data_dir(project_dir: Path) -> Path | None:
    """프로젝트 디렉토리에서 esd.list 위치를 감지. 없으면 None."""
    if (project_dir / "esd.list").exists():
        return project_dir
    sbv2_dir = project_dir / "sbv2_data"
    if (sbv2_dir / "esd.list").exists():
        return sbv2_dir
    return None


def _build_dataset_info(project_dir: Path, data_dir: Path) -> DatasetInfo:
    """프로젝트/데이터 디렉토리에서 DatasetInfo 생성."""
    esd_list = data_dir / "esd.list"
    train_list = data_dir / "train.list"
    val_list = data_dir / "val.list"
    config_path = data_dir / "config.json"

    text_done = _is_text_preprocessed(train_list)

    if text_done:
        bert_done, bert_total = _count_feature_files(train_list, val_list, "bert.pt")
        style_done, style_total = _count_feature_files(train_list, val_list, "npy", append=True)
    else:
        utterance_count = _count_lines(esd_list)
        bert_done, bert_total = 0, utterance_count
        style_done, style_total = 0, utterance_count

    model_name = _get_model_name(config_path)
    exports_dir = project_dir / "exports" / model_name if model_name else None
    default_style_done = (
        exports_dir is not None
        and (exports_dir / "style_vectors.npy").exists()
    )

    training_dir = project_dir / "training"
    has_checkpoints = (
        training_dir.exists()
        and any(training_dir.glob("G_*.pth"))
    )

    return DatasetInfo(
        name=project_dir.name,
        path=project_dir,
        data_dir=data_dir,
        utterance_count=_count_lines(esd_list),
        train_count=_count_lines(train_list),
        val_count=_count_lines(val_list),
        text_preprocessed=text_done,
        bert_done=bert_done,
        bert_total=bert_total,
        style_done=style_done,
        style_total=style_total,
        default_style_done=default_style_done,
        has_checkpoints=has_checkpoints,
    )


def discover_datasets() -> list[DatasetInfo]:
    """data/dataset/ 아래 모든 데이터셋을 탐색하고 상태를 확인."""
    if not DATA_DIR.exists():
        return []

    datasets = []
    for speaker_dir in sorted(DATA_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue

        data_dir = _detect_data_dir(speaker_dir)
        if data_dir is None:
            continue

        datasets.append(_build_dataset_info(speaker_dir, data_dir))

    return datasets


def scan_dataset(input_path: Path) -> DatasetInfo | None:
    """단일 경로에서 데이터셋을 로드. esd.list를 찾을 수 없으면 None."""
    input_path = input_path.resolve()

    if not input_path.is_dir():
        return None

    # Case 1: 입력 경로에 esd.list 직접 존재 (sbv2_data를 직접 지정한 경우)
    if (input_path / "esd.list").exists():
        if input_path.name == "sbv2_data":
            return _build_dataset_info(input_path.parent, input_path)
        return _build_dataset_info(input_path, input_path)

    # Case 2: sbv2_data/ 하위에 esd.list 존재
    sbv2_dir = input_path / "sbv2_data"
    if (sbv2_dir / "esd.list").exists():
        return _build_dataset_info(input_path, sbv2_dir)

    return None


def activate_dataset(dataset_path: Path):
    """training 모듈 import 전에 호출. HAYAKOE_PROJECT_DIR 설정 + config 캐시 리셋."""
    os.environ["HAYAKOE_PROJECT_DIR"] = str(dataset_path)

    # 이미 import된 경우 캐시 리셋
    import sys
    if "config" in sys.modules:
        import importlib
        import config as training_config
        training_config._config = None
        importlib.reload(training_config)
