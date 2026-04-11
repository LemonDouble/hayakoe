"""스타일 벡터 추출: 각 발화에 대해 화자 임베딩 생성 후 캐시."""

import argparse
import os
import warnings
from typing import Any

# pyannote 내부 경고 억제 (import 전에 설정해야 함)
# - torchcodec 미설치: waveform dict를 직접 전달하므로 불필요
# - TF32 비활성화 안내: ReproducibilityWarning(UserWarning), 스타일 벡터 품질에 무관
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*", category=UserWarning)

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from pyannote.audio import Inference, Model
from tqdm import tqdm

from config import get_config
from hayakoe.logging import logger
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
inference.to(torch.device(config.style_gen_config.device))


def get_style_vector(wav_path: str) -> NDArray[Any]:
    # soundfile로 읽어서 waveform dict를 전달 (torchcodec 불필요)
    data, sample_rate = sf.read(wav_path)
    if data.ndim == 2:
        data = data.mean(axis=1)  # stereo → mono
    waveform = torch.from_numpy(data).float().unsqueeze(0)  # (1, samples)
    return inference({"waveform": waveform, "sample_rate": sample_rate})  # type: ignore


def process_line(line: str) -> tuple[str, str | None]:
    wav_path = line.split("|")[0]
    npy_path = f"{wav_path}.npy"

    # 이미 생성된 파일은 스킵 (멱등성)
    if os.path.exists(npy_path):
        return line, None

    try:
        style_vec = get_style_vector(wav_path)
        if np.isnan(style_vec).any():
            logger.warning(f"NaN 스타일 벡터: {wav_path}")
            return line, "nan_error"
        np.save(npy_path, style_vec)
        return line, None
    except Exception as e:
        logger.error(f"스타일 벡터 오류: {wav_path}\n{e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=config.style_gen_config.config_path)
    args, _ = parser.parse_known_args()

    hps = HyperParameters.load_from_json(args.config)

    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    ok_lines: list[str] = []
    nan_lines: list[str] = []

    for line in tqdm(lines, file=SAFE_STDOUT, dynamic_ncols=True):
        result_line, error = process_line(line)
        if error is None:
            ok_lines.append(result_line)
        else:
            nan_lines.append(result_line)

    # NaN 발생한 파일은 학습 데이터에서 제거
    if nan_lines:
        nan_files = [l.split("|")[0] for l in nan_lines]
        logger.warning(f"NaN {len(nan_lines)}개 파일 제거: {nan_files}")

        # train/val에서 NaN 라인 제거 후 재작성
        with open(hps.data.training_files, encoding="utf-8") as f:
            train = f.readlines()
        with open(hps.data.validation_files, encoding="utf-8") as f:
            val = f.readlines()

        nan_set = set(nan_lines)
        with open(hps.data.training_files, "w", encoding="utf-8") as f:
            f.writelines(l for l in train if l not in nan_set)
        with open(hps.data.validation_files, "w", encoding="utf-8") as f:
            f.writelines(l for l in val if l not in nan_set)

    logger.info(f"스타일 벡터 생성 완료! 총 {len(ok_lines)}개 파일.")
