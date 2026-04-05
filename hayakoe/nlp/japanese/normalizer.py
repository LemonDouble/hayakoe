import json
import re
import unicodedata
from pathlib import Path

from num2words import num2words

from hayakoe.nlp.symbols import PUNCTUATIONS

# 영어→카타카나 외래어 사전 (지연 로드)
_EN_TO_KATA_DICT: dict[str, str] | None = None
_EN_WORD_PATTERN = re.compile(r"[A-Za-z]+")


def _get_en_to_kata_dict() -> dict[str, str]:
    global _EN_TO_KATA_DICT
    if _EN_TO_KATA_DICT is None:
        dict_path = Path(__file__).parent / "english_to_katakana.json"
        with open(dict_path, "r", encoding="utf-8") as f:
            _EN_TO_KATA_DICT = json.load(f)
    return _EN_TO_KATA_DICT


def _replace_english_with_katakana(text: str) -> str:
    """영단어를 외래어 사전으로 카타카나로 치환한다. 사전에 없는 단어는 그대로 남긴다."""
    d = _get_en_to_kata_dict()

    def _replace(m: re.Match) -> str:
        word = m.group()
        return d.get(word.lower(), word)

    return _EN_WORD_PATTERN.sub(_replace, text)


# 기호류 정규화 맵
__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    # NFKC 정규화 후의 하이픈/대시 변종을 모두 통상 반각 하이픈 - \u002d로 변환
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen,
    # "\u2011": "\u002d",  # ‑, Non-Breaking Hyphen, NFKC에 의해 \u2010으로 변환됨
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    # "～": "-",  # 이것은 장음 기호 「ー」로 취급하도록 변경
    # "~": "-",  # 이것도 장음 기호 「ー」로 취급하도록 변경
    "「": "'",
    "」": "'",
}
# 기호류 정규화 패턴
__REPLACE_PATTERN = re.compile("|".join(re.escape(p) for p in __REPLACE_MAP))
# 구두점 등의 정규화 패턴
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    # ↓ 히라가나, 카타카나, 한자
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    # ↓ 반각 알파벳 (대문자와 소문자)
    + r"\u0041-\u005A\u0061-\u007A"
    # ↓ 전각 알파벳 (대문자와 소문자)
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    # ↓ 그리스 문자
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    # ↓ "!", "?", "…", ",", ".", "'", "-", 단 `…`는 이미 `...`로 변환되어 있음
    + "".join(PUNCTUATIONS) + r"]+",  # fmt: skip
)
# 숫자/통화 기호 정규화 패턴
__CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
__CURRENCY_PATTERN = re.compile(r"([$¥£€])([0-9.]*[0-9])")
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")


def normalize_text(text: str) -> str:
    """
    일본어 텍스트를 정규화한다.
    결과는 정확히 다음 문자들로만 구성된다:
    - 히라가나
    - 카타카나 (전각 장음 기호 「ー」가 포함됨!)
    - 한자
    - 반각 알파벳 (대문자와 소문자)
    - 그리스 문자
    - `.` (마침표 `。`이나 `…`의 일부나 개행 등)
    - `,` (쉼표 `、`이나 `:` 등)
    - `?` (물음표 `？`)
    - `!` (느낌표 `！`)
    - `'` (`「`이나 `」` 등)
    - `-` (`―` (대시, 장음 기호가 아님)이나 `-` 등)

    주의사항:
    - 말줄임표 `…`는 `...`로 변환된다 (`なるほど…。` → `なるほど....`)
    - 숫자는 한자로 변환된다 (`1,100円` → `千百円`, `52.34` → `五十二点三四`)
    - 쉼표나 물음표 등의 위치/개수는 유지된다 (`??あ、、！！！` → `??あ,,!!!`)

    Args:
        text (str): 정규화할 텍스트

    Returns:
        str: 정규화된 텍스트
    """

    res = unicodedata.normalize("NFKC", text)  # 여기서 알파벳은 반각으로 변환됨
    res = _replace_english_with_katakana(res)  # 영단어→카타카나 외래어 변환
    res = __convert_numbers_to_words(res)  # 「100円」→「百円」 등
    # 「～」와 「〜」와 「~」도 장음 기호로 취급
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")
    res = res.replace("〜", "ー")

    res = replace_punctuation(res)  # 구두점 등 정규화, 읽을 수 없는 문자 삭제

    # 결합 문자의 탁점/반탁점 삭제
    # 통상의 「ば」 등은 그대로 유지되고, 「あ゛」은 위에서 「あ゙」가 되어 여기서 「あ」가 됨
    res = res.replace("\u3099", "")  # 결합 문자의 탁점 삭제, る゙ → る
    res = res.replace("\u309a", "")  # 결합 문자의 반탁점 삭제, な゚ → な
    return res


def replace_punctuation(text: str) -> str:
    """
    구두점 등을 「.」「,」「!」「?」「'」「-」로 정규화하고, OpenJTalk에서 읽기를 가져올 수 있는 것만 남긴다:
    한자, 히라가나, 카타카나, 알파벳, 그리스 문자

    Args:
        text (str): 정규화할 텍스트

    Returns:
        str: 정규화된 텍스트
    """

    # 구두점을 사전으로 치환
    replaced_text = __REPLACE_PATTERN.sub(lambda x: __REPLACE_MAP[x.group()], text)

    # 상기 이외의 문자를 삭제
    replaced_text = __PUNCTUATION_CLEANUP_PATTERN.sub("", replaced_text)

    return replaced_text


def __convert_numbers_to_words(text: str) -> str:
    """
    기호나 숫자를 일본어 문자 표현으로 변환한다.

    Args:
        text (str): 변환할 텍스트

    Returns:
        str: 변환된 텍스트
    """

    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), text)
    res = __CURRENCY_PATTERN.sub(lambda m: m[2] + __CURRENCY_MAP.get(m[1], m[1]), res)
    res = __NUMBER_PATTERN.sub(lambda m: num2words(m[0], lang="ja"), res)

    return res
