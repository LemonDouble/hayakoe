"""문장 경계 pause 계산 헬퍼 (torch-free).

CPU/ONNX 추론 경로와 PyTorch 추론 경로가 공유한다. duration predictor 가
내놓은 phoneme 별 frame 수를 문장 경계 무음 길이(초)로 변환하는 순수
numpy/python 로직만 담으며, torch / numba 에 의존하지 않는다. (이 함수들을
torch 를 top-level import 하는 ``infer.py`` 에 두면, torch 없는 CPU 환경에서
다중 문장 합성 시 ``ModuleNotFoundError: torch`` 로 크래시한다.)
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.nlp.symbols import SYMBOLS


_BOUNDARY_PUNCT_IDS = frozenset(
    SYMBOLS.index(p) for p in (".", "!", "?") if p in SYMBOLS
)


def find_boundary_punct_positions(phone_list: list[int]) -> list[int]:
    """phoneme 시퀀스에서 문장 경계 구두점(. ! ?)의 위치를 반환한다."""
    return [i for i, p in enumerate(phone_list) if p in _BOUNDARY_PUNCT_IDS]


def durations_to_boundary_pauses(
    durations: NDArray[Any],
    phone_list: list[int],
    punct_positions: list[int],
    num_sentences: int,
    hps: HyperParameters,
) -> list[float]:
    """예측된 phoneme별 frame 수를 문장 경계 pause 길이(초)로 변환한다."""
    num_boundaries = num_sentences - 1
    if not punct_positions or num_boundaries <= 0:
        return []

    hop_length = hps.data.hop_length
    sr = hps.data.sampling_rate

    pauses: list[float] = []
    for pos in punct_positions[:num_boundaries]:
        frames = float(durations[pos])
        if pos > 0 and phone_list[pos - 1] == 0:
            frames += float(durations[pos - 1])
        if pos + 1 < len(phone_list) and phone_list[pos + 1] == 0:
            frames += float(durations[pos + 1])
        pauses.append(frames * hop_length / sr)

    return pauses
