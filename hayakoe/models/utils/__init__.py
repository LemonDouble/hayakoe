import glob
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray

from hayakoe.logging import logger
from hayakoe.models.utils import checkpoints  # type: ignore # noqa: F401
from hayakoe.models.utils import safetensors  # type: ignore # noqa: F401


if TYPE_CHECKING:
    # tensorboard는 라이브러리로 설치된 경우 의존성에 포함되지 않으므로 타입 체크 시에만 임포트
    from torch.utils.tensorboard import SummaryWriter


__is_matplotlib_imported = False


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


def plot_spectrogram_to_numpy(spectrogram: NDArray[Any]) -> NDArray[Any]:
    """
    지정된 스펙트로그램을 이미지 데이터로 변환한다

    Args:
        spectrogram (NDArray[Any]): 스펙트로그램

    Returns:
        NDArray[Any]: 이미지 데이터
    """

    global __is_matplotlib_imported
    if not __is_matplotlib_imported:
        import matplotlib

        matplotlib.use("Agg")
        __is_matplotlib_imported = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(
    alignment: NDArray[Any], info: Optional[str] = None
) -> NDArray[Any]:
    """
    지정된 얼라인먼트를 이미지 데이터로 변환한다

    Args:
        alignment (NDArray[Any]): 얼라인먼트
        info (Optional[str]): 이미지에 추가할 정보

    Returns:
        NDArray[Any]: 이미지 데이터
    """

    global __is_matplotlib_imported
    if not __is_matplotlib_imported:
        import matplotlib

        matplotlib.use("Agg")
        __is_matplotlib_imported = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


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


def get_logger(
    model_dir_path: Union[str, Path], filename: str = "train.log"
) -> logging.Logger:
    """
    로거를 가져온다

    Args:
        model_dir_path (Union[str, Path]): 로그를 저장할 디렉토리 경로
        filename (str): 로그 파일 이름 (기본값: "train.log")

    Returns:
        logging.Logger: 로거
    """

    global logger
    logger = logging.getLogger(os.path.basename(model_dir_path))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    h = logging.FileHandler(os.path.join(model_dir_path, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


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


def check_git_hash(model_dir_path: Union[str, Path]) -> None:
    """
    모델 디렉토리에 .git 디렉토리가 존재할 경우 해시 값을 비교한다

    Args:
        model_dir_path (Union[str, Path]): 모델 디렉토리 경로
    """

    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            f"{source_dir} is not a git repository, therefore hash value comparison will be ignored."
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir_path, "githash")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved_hash = f.read()
        if saved_hash != cur_hash:
            logger.warning(
                f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)"
            )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cur_hash)
