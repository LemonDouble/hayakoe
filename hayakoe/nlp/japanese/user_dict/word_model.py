"""
이 파일은 VOICEVOX 프로젝트의 VOICEVOX ENGINE에서 차용한 것입니다.
출처: https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/model.py#L207
라이선스: LGPL-3.0
자세한 내용은 이 파일과 같은 폴더에 있는 README.md를 참조하세요.
"""

from enum import Enum
from re import findall, fullmatch
from typing import List, Optional

from pydantic import BaseModel, Field, validator


USER_DICT_MIN_PRIORITY = 0
USER_DICT_MAX_PRIORITY = 10


class UserDictWord(BaseModel):
    """
    사전 컴파일에 사용되는 정보
    """

    surface: str = Field(title="표층형")
    priority: int = Field(
        title="우선도", ge=USER_DICT_MIN_PRIORITY, le=USER_DICT_MAX_PRIORITY
    )
    context_id: int = Field(title="문맥 ID", default=1348)
    part_of_speech: str = Field(title="품사")
    part_of_speech_detail_1: str = Field(title="품사 세분류1")
    part_of_speech_detail_2: str = Field(title="품사 세분류2")
    part_of_speech_detail_3: str = Field(title="품사 세분류3")
    inflectional_type: str = Field(title="활용형 타입")
    inflectional_form: str = Field(title="활용형")
    stem: str = Field(title="원형")
    yomi: str = Field(title="읽기")
    pronunciation: str = Field(title="발음")
    accent_type: int = Field(title="악센트형")
    mora_count: Optional[int] = Field(title="모라 수", default=None)
    accent_associative_rule: str = Field(title="악센트 결합 규칙")

    class Config:
        validate_assignment = True

    @validator("surface")
    def convert_to_zenkaku(cls, surface):
        return surface.translate(
            str.maketrans(
                "".join(chr(0x21 + i) for i in range(94)),
                "".join(chr(0xFF01 + i) for i in range(94)),
            )
        )

    @validator("pronunciation", pre=True)
    def check_is_katakana(cls, pronunciation):
        if not fullmatch(r"[ァ-ヴー]+", pronunciation):
            raise ValueError("발음은 유효한 카타카나여야 합니다.")
        sutegana = ["ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ヮ", "ッ"]
        for i in range(len(pronunciation)):
            if pronunciation[i] in sutegana:
                # 「キャット」처럼 작은 가나가 연속할 가능성이 있으므로,
                # 「ッ」에 관해서는 「ッ」 자체가 연속하는 경우와 「ッ」 뒤에 다른 작은 가나가 연속하는 경우만 무효로 한다
                if i < len(pronunciation) - 1 and (
                    pronunciation[i + 1] in sutegana[:-1]
                    or (
                        pronunciation[i] == sutegana[-1]
                        and pronunciation[i + 1] == sutegana[-1]
                    )
                ):
                    raise ValueError("유효하지 않은 발음입니다. (작은 가나의 연속)")
            if pronunciation[i] == "ヮ":
                if i != 0 and pronunciation[i - 1] not in ["ク", "グ"]:
                    raise ValueError(
                        "유효하지 않은 발음입니다. (「くゎ」「ぐゎ」 이외의 「ゎ」 사용)"
                    )
        return pronunciation

    @validator("mora_count", pre=True, always=True)
    def check_mora_count_and_accent_type(cls, mora_count, values):
        if "pronunciation" not in values or "accent_type" not in values:
            # 적절한 위치에서 오류가 발생하도록 함
            return mora_count

        if mora_count is None:
            rule_others = (
                "[イ][ェ]|[ヴ][ャュョ]|[トド][ゥ]|[テデ][ィャュョ]|[デ][ェ]|[クグ][ヮ]"
            )
            rule_line_i = "[キシチニヒミリギジビピ][ェャュョ]"
            rule_line_u = "[ツフヴ][ァ]|[ウスツフヴズ][ィ]|[ウツフヴ][ェォ]"
            rule_one_mora = "[ァ-ヴー]"
            mora_count = len(
                findall(
                    f"(?:{rule_others}|{rule_line_i}|{rule_line_u}|{rule_one_mora})",
                    values["pronunciation"],
                )
            )

        if not 0 <= values["accent_type"] <= mora_count:
            raise ValueError(
                "잘못된 악센트형입니다 ({})。 expect: 0 <= accent_type <= {}".format(
                    values["accent_type"], mora_count
                )
            )
        return mora_count


class PartOfSpeechDetail(BaseModel):
    """
    품사별 정보
    """

    part_of_speech: str = Field(title="품사")
    part_of_speech_detail_1: str = Field(title="품사 세분류1")
    part_of_speech_detail_2: str = Field(title="품사 세분류2")
    part_of_speech_detail_3: str = Field(title="품사 세분류3")
    # context_id는 사전의 좌/우 문맥 ID를 의미함
    # https://github.com/VOICEVOX/open_jtalk/blob/427cfd761b78efb6094bea3c5bb8c968f0d711ab/src/mecab-naist-jdic/_left-id.def
    context_id: int = Field(title="문맥 ID")
    cost_candidates: List[int] = Field(title="비용 퍼센타일")
    accent_associative_rules: List[str] = Field(title="악센트 결합 규칙 목록")


class WordTypes(str, Enum):
    """
    FastAPI에서 word_type 인수를 검증할 때 사용하는 클래스
    """

    PROPER_NOUN = "PROPER_NOUN"
    COMMON_NOUN = "COMMON_NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    SUFFIX = "SUFFIX"
