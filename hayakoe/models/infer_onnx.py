"""HayaKoe용 ONNX Runtime 추론.

CPU 속도 최적화를 위해 PyTorch 추론을 ONNX Runtime으로 대체한다.
두 개의 ONNX 모델을 필요로 한다: bert.onnx와 synthesizer.onnx.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from hayakoe.constants import Languages
from hayakoe.logging import logger
from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
)
from hayakoe.nlp.japanese.g2p import text_to_sep_kata
from hayakoe.nlp.symbols import SYMBOLS


def _intersperse(lst: list, item) -> list:
    """리스트 요소 사이에 아이템을 삽입한다."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _get_tokenizer():
    """JP BERT 토크나이저를 가져온다."""
    from hayakoe.nlp import bert_models

    if not bert_models.is_tokenizer_loaded():
        bert_models.load_tokenizer()
    return bert_models.load_tokenizer()


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_session,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray:
    """ONNX Runtime 세션을 사용하여 BERT 특징을 추출한다.

    nlp/japanese/bert_feature.py의 로직을 미러링하되 ONNX를 사용한다.
    """
    # PyTorch 버전과 동일한 전처리
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    tokenizer = _get_tokenizer()
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    res = onnx_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0][0]  # [seq_len, hidden_dim]

    style_res_mean = None
    if assist_text:
        style_inputs = tokenizer(assist_text, return_tensors="np")
        style_res = onnx_session.run(None, {
            "input_ids": style_inputs["input_ids"],
            "attention_mask": style_inputs["attention_mask"],
        })[0][0]
        style_res_mean = style_res.mean(axis=0)

    assert len(word2ph) == len(text) + 2, text
    phone_level_feature = []
    for i in range(len(word2ph)):
        if assist_text and style_res_mean is not None:
            repeat_feature = (
                np.tile(res[i], (word2ph[i], 1)) * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2ph[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)
    return phone_level_feature.T  # [hidden_dim, phone_len]


def get_text_onnx(
    text: str,
    hps: HyperParameters,
    bert_session,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """텍스트 전처리 + ONNX를 통한 BERT 특징 추출."""
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        Languages.JP,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, Languages.JP)

    if hps.data.add_blank:
        phone = _intersperse(phone, 0)
        tone = _intersperse(tone, 0)
        language = _intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    ja_bert = extract_bert_feature_onnx(
        norm_text,
        word2ph,
        bert_session,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert ja_bert.shape[-1] == len(phone), phone

    phone = np.array(phone, dtype=np.int64)
    tone = np.array(tone, dtype=np.int64)
    language = np.array(language, dtype=np.int64)
    return ja_bert, phone, tone, language


def infer_onnx(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,
    language: Languages,
    hps: HyperParameters,
    bert_session,
    synth_session,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> NDArray[Any]:
    """전체 ONNX 추론 파이프라인: 텍스트 → BERT → Synthesizer → 오디오."""
    ja_bert, phones, tones, lang_ids = get_text_onnx(
        text,
        hps,
        bert_session,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )

    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]

    # 배치 차원 추가
    x = phones[np.newaxis, :]                       # [1, phone_len]
    x_lengths = np.array([phones.shape[0]], dtype=np.int64)  # [1]
    t = tones[np.newaxis, :]                         # [1, phone_len]
    l = lang_ids[np.newaxis, :]                      # [1, phone_len]
    b = ja_bert[np.newaxis, :, :]                    # [1, 1024, phone_len]
    s = style_vec[np.newaxis, :].astype(np.float32)  # [1, 256]
    sid_arr = np.array([sid], dtype=np.int64)        # [1]

    output = synth_session.run(None, {
        "x": x,
        "x_lengths": x_lengths,
        "sid": sid_arr,
        "tone": t,
        "language": l,
        "bert": b.astype(np.float32),
        "style_vec": s,
        "noise_scale": np.array([noise_scale], dtype=np.float32),
        "length_scale": np.array([length_scale], dtype=np.float32),
        "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
        "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
    })

    audio = output[0][0, 0]  # [audio_len]
    return audio
