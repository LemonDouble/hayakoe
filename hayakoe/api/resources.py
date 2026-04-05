from __future__ import annotations

from typing import Optional

from hayakoe.logging import logger
from hayakoe.nlp import bert_models


def prepare_bert(cache_dir: Optional[str] = None) -> None:
    """JP BERT 모델과 토크나이저를 로컬 캐시에 다운로드한다. GPU 불필요."""
    bert_models.load_model(device_map="cpu", cache_dir=cache_dir)
    bert_models.unload_model()
    bert_models.load_tokenizer(cache_dir=cache_dir)
    bert_models.unload_tokenizer()
    logger.info("JP BERT download complete.")


def load_bert(device: str, cache_dir: Optional[str] = None) -> None:
    """JP BERT 모델과 토크나이저를 다운로드(필요 시)하고 메모리에 로드한다."""
    if bert_models.is_model_loaded() and bert_models.is_tokenizer_loaded():
        bert_models.transfer_model(device)
        return

    bert_models.load_model(device_map=device, cache_dir=cache_dir)
    bert_models.load_tokenizer(cache_dir=cache_dir)
