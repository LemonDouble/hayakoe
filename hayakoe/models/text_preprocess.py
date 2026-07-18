"""텍스트 → 음소 시퀀스 공통 전처리 (torch-free).

PyTorch 경로(``infer.get_text``), ONNX 경로(``infer_onnx.get_text_onnx``),
BERT 배치 경로(``api.speaker._preprocess_nlp``)가 공유한다. add_blank
intersperse 와 word2ph 보정처럼 어긋나면 BERT feature 정렬이 깨지는 로직을
한 곳에 둔다. torch 를 import 하지 않으므로 CPU/ONNX 경로에서도 안전하다.
"""

from __future__ import annotations

from typing import Optional

from hayakoe.constants import Languages
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
)


def intersperse(lst: list, item) -> list:
    """리스트 요소 사이에 아이템을 삽입한다."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def prepare_phone_sequences(
    text: str,
    hps: HyperParameters,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[str, list[int], list[int], list[int], list[int]]:
    """클린 텍스트 → 음소/톤/언어 시퀀스 변환 (add_blank·word2ph 보정 포함).

    Returns:
        ``(norm_text, phone, tone, language, word2ph)`` — BERT 특징 추출 직전
        까지의 전처리 결과.
    """
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        Languages.JP,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=hps.version.endswith("JP-Extra"),
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, Languages.JP)

    if hps.data.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    return norm_text, phone, tone, language, word2ph
