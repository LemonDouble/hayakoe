from __future__ import annotations

from typing import TYPE_CHECKING

from hayakoe.nlp import bert_models
from hayakoe.nlp.japanese.g2p import text_to_sep_kata


if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
) -> torch.Tensor:
    """
    일본어 텍스트에서 BERT 특징량을 추출한다 (PyTorch 추론)

    Args:
        text (str): 일본어 텍스트
        word2ph (list[int]): 원본 텍스트의 각 문자에 음소가 몇 개 할당되는지를 나타내는 리스트
        device (str): 추론에 사용할 디바이스

    Returns:
        torch.Tensor: BERT 특징량
    """

    import torch

    # 각 단어가 몇 글자인지를 만드는 `word2ph`를 사용해야 하므로, 읽을 수 없는 문자는 반드시 무시한다
    # 그렇지 않으면 `word2ph`의 결과와 텍스트의 글자 수 결과의 정합성이 맞지 않는다
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(device=device)
    bert_models.transfer_model(device)

    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer()
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = res["hidden_states"][-3][0].float()

    assert len(word2ph) == len(text) + 2, text
    phone_level_feature = torch.repeat_interleave(
        res, torch.tensor(word2ph, device=res.device), dim=0
    )

    return phone_level_feature.T
