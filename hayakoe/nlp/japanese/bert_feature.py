from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from hayakoe.nlp import bert_models
from hayakoe.nlp.japanese.g2p import text_to_sep_kata


if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    일본어 텍스트에서 BERT 특징량을 추출한다 (PyTorch 추론)

    Args:
        text (str): 일본어 텍스트
        word2ph (list[int]): 원본 텍스트의 각 문자에 음소가 몇 개 할당되는지를 나타내는 리스트
        device (str): 추론에 사용할 디바이스
        assist_text (Optional[str], optional): 보조 텍스트 (기본값: None)
        assist_text_weight (float, optional): 보조 텍스트의 가중치 (기본값: 0.7)

    Returns:
        torch.Tensor: BERT 특징량
    """

    import torch

    # 각 단어가 몇 글자인지를 만드는 `word2ph`를 사용해야 하므로, 읽을 수 없는 문자는 반드시 무시한다
    # 그렇지 않으면 `word2ph`의 결과와 텍스트의 글자 수 결과의 정합성이 맞지 않는다
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(device=device)
    bert_models.transfer_model(device)

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer()
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu().float()
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
