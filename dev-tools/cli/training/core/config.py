"""
HayaKoe 프로젝트 디렉토리 기반 설정 생성.

SBV2 원본의 YAML 기반 config 시스템을 대체.
환경변수 HAYAKOE_PROJECT_DIR에서 프로젝트 경로를 읽고,
SBV2 스크립트가 기대하는 Config 객체를 생성.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch


cuda_available = torch.cuda.is_available()


class Resample_config:
    def __init__(self, in_dir: Path, out_dir: Path, sampling_rate: int = 44100):
        self.sampling_rate = sampling_rate
        self.in_dir = in_dir
        self.out_dir = out_dir


class Preprocess_text_config:
    def __init__(
        self,
        transcription_path: Path,
        cleaned_path: Path,
        train_path: Path,
        val_path: Path,
        config_path: Path,
        val_per_lang: int = 5,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        self.transcription_path = transcription_path
        self.cleaned_path = cleaned_path
        self.train_path = train_path
        self.val_path = val_path
        self.config_path = config_path
        self.val_per_lang = val_per_lang
        self.max_val_total = max_val_total
        self.clean = clean


class Bert_gen_config:
    def __init__(
        self,
        config_path: Path,
        num_processes: int = 1,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device if cuda_available else "cpu"
        self.use_multi_device = use_multi_device


class Style_gen_config:
    def __init__(
        self,
        config_path: Path,
        num_processes: int = 4,
        device: str = "cuda",
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device if cuda_available else "cpu"


class Train_ms_config:
    def __init__(
        self,
        config_path: Path,
        model_dir: str = "training",
        env: Optional[dict] = None,
        num_workers: int = 4,
        spec_cache: bool = True,
        keep_ckpts: int = 5,
    ):
        self.config_path = config_path
        self.model_dir = Path(model_dir)
        self.env = env or {}
        self.num_workers = num_workers
        self.spec_cache = spec_cache
        self.keep_ckpts = keep_ckpts


class Config:
    """프로젝트 디렉토리에서 SBV2 호환 설정 생성."""

    def __init__(self, project_dir: Path):
        self.dataset_path = project_dir
        self.dataset_root = project_dir.parent
        self.assets_root = project_dir / "exports"

        # flat layout (웹 전처리 출력) vs sbv2_data/ layout 자동 감지
        flat_config = project_dir / "config.json"
        sbv2_dir = project_dir / "sbv2_data"

        if flat_config.exists():
            # flat layout: 파일이 프로젝트 루트에 직접 존재
            data_dir = project_dir
            config_json = flat_config
        else:
            # sbv2_data/ layout (레거시)
            data_dir = sbv2_dir
            config_json = sbv2_dir / "config.json"

        # config.json에서 모델명 읽기
        if config_json.exists():
            with open(config_json, encoding="utf-8") as f:
                data = json.load(f)
            self.model_name = data.get("model_name", project_dir.name)
        else:
            self.model_name = project_dir.name

        self.out_dir = self.assets_root / self.model_name

        self.resample_config = Resample_config(
            in_dir=project_dir / "selected_speaker",
            out_dir=data_dir / "raw",
            sampling_rate=44100,
        )

        self.preprocess_text_config = Preprocess_text_config(
            transcription_path=data_dir / "esd.list",
            cleaned_path=data_dir / "esd.list.cleaned",
            train_path=data_dir / "train.list",
            val_path=data_dir / "val.list",
            config_path=config_json,
        )

        self.bert_gen_config = Bert_gen_config(config_path=config_json)
        self.style_gen_config = Style_gen_config(config_path=config_json)

        self.train_ms_config = Train_ms_config(
            config_path=config_json,
            model_dir="training",
        )


_config: Optional[Config] = None


def get_config() -> Config:
    """HAYAKOE_PROJECT_DIR 환경변수에서 프로젝트 경로를 읽어 Config 생성."""
    global _config
    if _config is not None:
        return _config

    project_dir = Path(os.environ.get("HAYAKOE_PROJECT_DIR", "."))
    _config = Config(project_dir)
    return _config
