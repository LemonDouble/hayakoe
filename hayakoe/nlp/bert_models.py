"""
JP BERT 모델 (ku-nlp/deberta-v2-large-japanese-char-wwm) 로드/관리 모듈.
글로벌 싱글턴으로 모델과 토크나이저를 보유하며, 한 번 로드하면 어디서든 가져올 수 있음.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel

from hayakoe.constants import HF_REPO
from hayakoe.logging import logger


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    import torch


_loaded_model: Optional[PreTrainedModel] = None
_loaded_tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None


def load_model(
    pretrained_model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PreTrainedModel:
    global _loaded_model

    if _loaded_model is not None:
        return _loaded_model

    start_time = time.time()
    source = pretrained_model_name_or_path or HF_REPO
    kwargs: dict = {"cache_dir": cache_dir, "dtype": "float32"}
    if pretrained_model_name_or_path is None:
        kwargs["subfolder"] = "pytorch/bert/fp32"

    _loaded_model = AutoModelForMaskedLM.from_pretrained(source, **kwargs)

    if device:
        _loaded_model.to(device)

    logger.info(
        f"Loaded JP BERT model from {source} ({time.time() - start_time:.2f}s)"
    )
    return _loaded_model


def load_tokenizer(
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    global _loaded_tokenizer

    if _loaded_tokenizer is not None:
        return _loaded_tokenizer

    source = pretrained_model_name_or_path or HF_REPO
    kwargs: dict = {"cache_dir": cache_dir, "use_fast": True}
    if pretrained_model_name_or_path is None:
        kwargs["subfolder"] = "bert/tokenizer"

    _loaded_tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    logger.info(f"Loaded JP BERT tokenizer from {source}")
    return _loaded_tokenizer


def transfer_model(device: str) -> None:
    if _loaded_model is None:
        raise ValueError("JP BERT model is not loaded.")

    current_device = str(_loaded_model.device)
    if current_device.startswith(device):
        return

    _loaded_model.to(device)  # type: ignore
    logger.info(f"Transferred JP BERT model from {current_device} to {device}")


def compile_model() -> None:
    """글로벌 BERT 모델에 torch.compile을 적용한다.

    이미 compile 된 모델에 다시 호출해도 중첩 래핑이 쌓이지 않는다.
    """
    global _loaded_model

    if _loaded_model is None:
        raise ValueError("JP BERT model is not loaded.")

    import torch

    # OptimizedModule 이면 이미 compile 된 상태 — 다시 감싸면 forward 경로가
    # torch.compile(torch.compile(model)) 이 되어 추론이 깨질 수 있다.
    if isinstance(_loaded_model, torch._dynamo.OptimizedModule):
        return

    _loaded_model = torch.compile(_loaded_model, mode="default")
    logger.info("Applied torch.compile to JP BERT model")


def is_model_loaded() -> bool:
    return _loaded_model is not None


def is_tokenizer_loaded() -> bool:
    return _loaded_tokenizer is not None


def unload_model() -> None:
    global _loaded_model

    import torch

    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded JP BERT model")


def unload_tokenizer() -> None:
    global _loaded_tokenizer

    if _loaded_tokenizer is not None:
        del _loaded_tokenizer
        _loaded_tokenizer = None
        gc.collect()
        logger.info("Unloaded JP BERT tokenizer")
