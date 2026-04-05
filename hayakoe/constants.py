from pathlib import Path

from hayakoe.utils.strenum import StrEnum


VERSION = "0.1.0"

BASE_DIR = Path(__file__).parent.parent


class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


# HuggingFace repository
HF_REPO = "lemondouble/hayakoe"

# JP BERT model
BERT_JP_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"

DEFAULT_USER_DICT_DIR = Path.home() / ".cache" / "hayakoe" / "user_dict"

# Default inference parameters
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0
