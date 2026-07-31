"""HayaKoe용 ONNX Runtime 추론.

CPU 속도 최적화를 위해 PyTorch 추론을 ONNX Runtime으로 대체한다.
두 개의 ONNX 모델을 필요로 한다: bert.onnx와 synthesizer.onnx.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.models.text_preprocess import prepare_phone_sequences
from hayakoe.nlp.japanese.g2p import text_to_sep_kata


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_session,
) -> NDArray:
    """ONNX Runtime 세션을 사용하여 BERT 특징을 추출한다.

    nlp/japanese/bert_feature.py의 로직을 미러링하되 ONNX를 사용한다.
    """
    from hayakoe.nlp import bert_models

    # PyTorch 버전과 동일한 전처리
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])

    tokenizer = bert_models.load_tokenizer()
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    res = onnx_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0][0]  # [seq_len, hidden_dim]

    assert len(word2ph) == len(text) + 2, text
    phone_level_feature = np.repeat(res[: len(word2ph)], word2ph, axis=0)
    return phone_level_feature.T  # [hidden_dim, phone_len]


def get_text_onnx(
    text: str,
    hps: HyperParameters,
    bert_session,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """텍스트 전처리 + ONNX를 통한 BERT 특징 추출."""
    norm_text, phone, tone, language, word2ph = prepare_phone_sequences(
        text,
        hps,
        given_phone=given_phone,
        given_tone=given_tone,
    )

    ja_bert = extract_bert_feature_onnx(
        norm_text,
        word2ph,
        bert_session,
    )
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
    hps: HyperParameters,
    bert_session,
    synth_session,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[NDArray[Any], NDArray[Any], Optional[NDArray[Any]]]:
    """전체 ONNX 추론 파이프라인: 텍스트 → BERT → Synthesizer → 오디오.

    Returns:
        ``(audio, phone_ids, durations)``. ``durations`` 는 phoneme 별 frame 수로,
        ``durations`` 출력을 포함해 내보낸 synthesizer 에서만 얻어지며 그렇지
        않으면 ``None`` 이다.
    """
    ja_bert, phones, tones, lang_ids = get_text_onnx(
        text,
        hps,
        bert_session,
        given_phone=given_phone,
        given_tone=given_tone,
    )

    # 배치 차원 추가
    x = phones[np.newaxis, :]                       # [1, phone_len]
    x_lengths = np.array([phones.shape[0]], dtype=np.int64)  # [1]
    t = tones[np.newaxis, :]                         # [1, phone_len]
    lang = lang_ids[np.newaxis, :]                   # [1, phone_len]
    b = ja_bert[np.newaxis, :, :]                    # [1, 1024, phone_len]
    s = style_vec[np.newaxis, :].astype(np.float32)  # [1, 256]
    sid_arr = np.array([sid], dtype=np.int64)        # [1]

    output = synth_session.run(None, {
        "x": x,
        "x_lengths": x_lengths,
        "sid": sid_arr,
        "tone": t,
        "language": lang,
        "bert": b.astype(np.float32),
        "style_vec": s,
        "noise_scale": np.array([noise_scale], dtype=np.float32),
        "length_scale": np.array([length_scale], dtype=np.float32),
        "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
        "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
    })

    audio = output[0][0, 0]  # [audio_len]
    durations = output[1][0] if len(output) > 1 else None  # [phone_len]
    return audio, phones, durations
