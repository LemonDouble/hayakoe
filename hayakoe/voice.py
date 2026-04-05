from typing import Any

import numpy as np
import pyworld
from numpy.typing import NDArray


def adjust_voice(
    fs: int,
    wave: NDArray[Any],
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> tuple[int, NDArray[Any]]:
    """
    음성의 피치와 억양을 조정한다.
    변경하면 약간 음질이 열화되므로, 둘 다 초기값 그대로라면 그대로 반환한다.

    Args:
        fs (int): 음성의 샘플링 주파수
        wave (NDArray[Any]): 음성 데이터
        pitch_scale (float, optional): 피치 높이. Defaults to 1.0.
        intonation_scale (float, optional): 억양의 평균으로부터의 변경 비율. Defaults to 1.0.

    Returns:
        tuple[int, NDArray[Any]]: 조정 후 음성 데이터의 샘플링 주파수와 음성 데이터
    """

    if pitch_scale == 1.0 and intonation_scale == 1.0:
        # 초기값인 경우, 음질 열화를 피하기 위해 그대로 반환
        return fs, wave

    # pyworld로 f0를 가공하여 합성
    # pyworld보다 더 좋은 것이 있을 수도 있지만……

    wave = wave.astype(np.double)

    # 품질이 좋아 보이니 일단 harvest를 사용
    f0, t = pyworld.harvest(wave, fs)

    sp = pyworld.cheaptrick(wave, f0, t, fs)
    ap = pyworld.d4c(wave, f0, t, fs)

    non_zero_f0 = [f for f in f0 if f != 0]
    f0_mean = sum(non_zero_f0) / len(non_zero_f0)

    for i, f in enumerate(f0):
        if f == 0:
            continue
        f0[i] = pitch_scale * f0_mean + intonation_scale * (f - f0_mean)

    wave = pyworld.synthesize(f0, sp, ap, fs)
    return fs, wave
