import glob
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

from hayakoe.models.utils import (
    checkpoints,  # type: ignore # noqa: F401
    safetensors,  # type: ignore # noqa: F401
)


if TYPE_CHECKING:
    # tensorboard는 라이브러리로 설치된 경우 의존성에 포함되지 않으므로 타입 체크 시에만 임포트
    from torch.utils.tensorboard import SummaryWriter




def summarize(
    writer: "SummaryWriter",
    global_step: int,
    scalars: dict[str, float] = {},
    histograms: dict[str, Any] = {},
    images: dict[str, Any] = {},
    audios: dict[str, Any] = {},
    audio_sampling_rate: int = 22050,
) -> None:
    """
    지정된 데이터를 TensorBoard에 일괄 추가한다

    Args:
        writer (SummaryWriter): TensorBoard에 기록을 수행하는 객체
        global_step (int): 글로벌 스텝 수
        scalars (dict[str, float]): 스칼라 값 딕셔너리
        histograms (dict[str, Any]): 히스토그램 딕셔너리
        images (dict[str, Any]): 이미지 데이터 딕셔너리
        audios (dict[str, Any]): 오디오 데이터 딕셔너리
        audio_sampling_rate (int): 오디오 데이터의 샘플링 레이트
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def is_resuming(dir_path: Union[str, Path]) -> bool:
    """
    지정된 디렉토리 경로에 재개 가능한 모델이 존재하는지 여부를 반환한다

    Args:
        dir_path: 확인할 디렉토리 경로

    Returns:
        bool: 재개 가능한 모델이 존재하는지 여부
    """
    # JP-Extra 버전에서는 DUR이 없고 WD가 있는 등 변동이 있으므로 G만으로 판단
    g_list = glob.glob(os.path.join(dir_path, "G_*.pth"))
    # d_list = glob.glob(os.path.join(dir_path, "D_*.pth"))
    # dur_list = glob.glob(os.path.join(dir_path, "DUR_*.pth"))
    return len(g_list) > 0


def load_wav_to_torch(full_path: Union[str, Path]) -> tuple[torch.FloatTensor, int]:
    """
    지정된 오디오 파일을 읽어 PyTorch 텐서로 변환하여 반환한다

    Args:
        full_path (Union[str, Path]): 오디오 파일 경로

    Returns:
        tuple[torch.FloatTensor, int]: 오디오 데이터 텐서와 샘플링 레이트
    """

    # 이 함수는 학습 시 이외에는 사용되지 않으므로, 라이브러리로서의 hayakoe가
    # 무거운 scipy에 의존하지 않도록 지연 import 수행
    try:
        from scipy.io.wavfile import read
    except ImportError:
        raise ImportError("scipy is required to load wav file")

    sampling_rate, data = read(full_path)
    if data.ndim == 2:
        data = data.mean(axis=1)  # stereo → mono
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(
    filename: Union[str, Path], split: str = "|"
) -> list[list[str]]:
    """
    지정된 파일에서 파일 경로와 텍스트를 읽어온다

    Args:
        filename (Union[str, Path]): 파일 경로
        split (str): 파일 구분자 (기본값: "|")

    Returns:
        list[list[str]]: 파일 경로와 텍스트의 리스트
    """

    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_steps(model_path: Union[str, Path]) -> Optional[int]:
    """
    모델 경로에서 이터레이션 횟수를 가져온다

    Args:
        model_path (Union[str, Path]): 모델 경로

    Returns:
        Optional[int]: 이터레이션 횟수
    """

    matches = re.findall(r"\d+", model_path)  # type: ignore
    return matches[-1] if matches else None


