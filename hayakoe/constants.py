import os
from pathlib import Path

from hayakoe.utils.strenum import StrEnum


BASE_DIR = Path(__file__).parent.parent


class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


# HuggingFace repository (공식 배포용 — 화자/BERT 기본 소스)
HF_REPO = "lemondouble/hayakoe"

# 기본 소스 URI — TTS.__init__ / TTS.load 에서 source 미지정 시 사용
DEFAULT_SPEAKER_SOURCE = f"hf://{HF_REPO}"
DEFAULT_BERT_SOURCE = f"hf://{HF_REPO}"


def default_cache_dir() -> Path:
    """기본 캐시 디렉토리.

    우선순위: ``HAYAKOE_CACHE`` env → ``$CWD/hayakoe_cache``.
    HuggingFace / S3 / 로컬 여부와 무관하게 모든 소스는 이 루트 아래에 캐싱된다.
    """
    env = os.environ.get("HAYAKOE_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / "hayakoe_cache"

# JP BERT model
BERT_JP_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"

# Default inference parameters
DEFAULT_STYLE = "Neutral"
