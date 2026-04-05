"""
이 파일은 VOICEVOX 프로젝트의 VOICEVOX ENGINE에서 차용한 것입니다.
출처: https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/user_dict/user_dict.py
라이선스: LGPL-3.0
자세한 내용은 이 파일과 같은 폴더에 있는 README.md를 참조하세요.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np

from hayakoe.constants import DEFAULT_USER_DICT_DIR
from hayakoe.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from hayakoe.nlp.japanese.user_dict.part_of_speech_data import (
    MAX_PRIORITY,
    MIN_PRIORITY,
    part_of_speech_data,
)
from hayakoe.nlp.japanese.user_dict.word_model import UserDictWord, WordTypes


# root_dir = engine_root()
# save_dir = get_save_dir()

# if not save_dir.is_dir():
#     save_dir.mkdir(parents=True)

user_dict_path = DEFAULT_USER_DICT_DIR / "user_dict.json"
compiled_dict_path = DEFAULT_USER_DICT_DIR / "user.dic"


# # 동시 쓰기 제어
# mutex_user_dict = threading.Lock()
# mutex_openjtalk_dict = threading.Lock()


# @mutex_wrapper(mutex_user_dict)
def _write_to_json(user_dict: Dict[str, UserDictWord], user_dict_path: Path) -> None:
    """
    사용자 사전 파일에 사용자 사전 데이터 쓰기
    Parameters
    ----------
    user_dict : Dict[str, UserDictWord]
        사용자 사전 데이터
    user_dict_path : Path
        사용자 사전 파일의 경로
    """
    user_dict_path.parent.mkdir(parents=True, exist_ok=True)
    converted_user_dict = {}
    for word_uuid, word in user_dict.items():
        word_dict = word.model_dump()
        word_dict["cost"] = _priority2cost(
            word_dict["context_id"], word_dict["priority"]
        )
        del word_dict["priority"]
        converted_user_dict[word_uuid] = word_dict
    # 사전에 JSON으로 변환 가능한지 확인
    user_dict_json = json.dumps(converted_user_dict, ensure_ascii=False)

    # 사용자 사전 파일에 쓰기
    user_dict_path.write_text(user_dict_json, encoding="utf-8")


# @mutex_wrapper(mutex_openjtalk_dict)
def update_dict(
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> None:
    random_string = uuid4()
    tmp_csv_path = compiled_dict_path.with_suffix(
        f".dict_csv-{random_string}.tmp"
    )
    tmp_compiled_path = compiled_dict_path.with_suffix(
        f".dict_compiled-{random_string}.tmp"
    )

    try:
        csv_text = ""
        user_dict = read_dict(user_dict_path=user_dict_path)
        for word_uuid in user_dict:
            word = user_dict[word_uuid]
            csv_text += (
                "{surface},{context_id},{context_id},{cost},{part_of_speech},"
                + "{part_of_speech_detail_1},{part_of_speech_detail_2},"
                + "{part_of_speech_detail_3},{inflectional_type},"
                + "{inflectional_form},{stem},{yomi},{pronunciation},"
                + "{accent_type}/{mora_count},{accent_associative_rule}\n"
            ).format(
                surface=word.surface,
                context_id=word.context_id,
                cost=_priority2cost(word.context_id, word.priority),
                part_of_speech=word.part_of_speech,
                part_of_speech_detail_1=word.part_of_speech_detail_1,
                part_of_speech_detail_2=word.part_of_speech_detail_2,
                part_of_speech_detail_3=word.part_of_speech_detail_3,
                inflectional_type=word.inflectional_type,
                inflectional_form=word.inflectional_form,
                stem=word.stem,
                yomi=word.yomi,
                pronunciation=word.pronunciation,
                accent_type=word.accent_type,
                mora_count=word.mora_count,
                accent_associative_rule=word.accent_associative_rule,
            )
        # 사전 데이터를 사전.csv에 임시 저장
        tmp_csv_path.write_text(csv_text, encoding="utf-8")

        # 사전.csv를 OpenJTalk용으로 컴파일
        # pyopenjtalk.create_user_dict(str(tmp_csv_path), str(tmp_compiled_path))
        pyopenjtalk.mecab_dict_index(str(tmp_csv_path), str(tmp_compiled_path))
        if not tmp_compiled_path.is_file():
            raise RuntimeError("사전 컴파일 중 오류가 발생했습니다.")

        # 컴파일된 사전 교체 및 로드
        pyopenjtalk.unset_user_dict()
        tmp_compiled_path.replace(compiled_dict_path)
        if compiled_dict_path.is_file():
            # pyopenjtalk.set_user_dict(str(compiled_dict_path.resolve(strict=True)))
            pyopenjtalk.update_global_jtalk_with_user_dict(str(compiled_dict_path))

    except Exception as e:
        print("Error: Failed to update dictionary.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise e

    finally:
        # 후처리
        if tmp_csv_path.exists():
            tmp_csv_path.unlink()
        if tmp_compiled_path.exists():
            tmp_compiled_path.unlink()


# @mutex_wrapper(mutex_user_dict)
def read_dict(user_dict_path: Path = user_dict_path) -> Dict[str, UserDictWord]:
    """
    사용자 사전 읽기
    Parameters
    ----------
    user_dict_path : Path
        사용자 사전 파일의 경로
    Returns
    -------
    result : Dict[str, UserDictWord]
        사용자 사전
    """
    # 지정된 사용자 사전이 존재하지 않는 경우, 빈 사전을 반환
    if not user_dict_path.is_file():
        return {}

    with user_dict_path.open(encoding="utf-8") as f:
        result: Dict[str, UserDictWord] = {}
        for word_uuid, word in json.load(f).items():
            # cost2priority로 변환할 때 context_id가 필요하지만,
            # 0.12 이전의 사전은 context_id가 하드코딩되어 있어 사용자 사전 내에 저장되어 있지 않음
            # 하드코딩되어 있던 context_id는 고유명사를 의미하므로, 고유명사의 context_id로 보완
            if word.get("context_id") is None:
                word["context_id"] = part_of_speech_data[
                    WordTypes.PROPER_NOUN
                ].context_id
            word["priority"] = _cost2priority(word["context_id"], word["cost"])
            del word["cost"]
            result[str(UUID(word_uuid))] = UserDictWord(**word)

    return result


def _create_word(
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: Optional[WordTypes] = None,
    priority: Optional[int] = None,
) -> UserDictWord:
    """
    단어 객체 생성
    Parameters
    ----------
    surface : str
        단어 정보
    pronunciation : str
        단어 정보
    accent_type : int
        단어 정보
    word_type : Optional[WordTypes]
        품사
    priority : Optional[int]
        우선도
    Returns
    -------
    : UserDictWord
        단어 객체
    """
    if word_type is None:
        word_type = WordTypes.PROPER_NOUN
    if word_type not in part_of_speech_data.keys():
        raise ValueError("알 수 없는 품사입니다")
    if priority is None:
        priority = 5
    if not MIN_PRIORITY <= priority <= MAX_PRIORITY:
        raise ValueError("우선도 값이 유효하지 않습니다")
    pos_detail = part_of_speech_data[word_type]
    return UserDictWord(
        surface=surface,
        context_id=pos_detail.context_id,
        priority=priority,
        part_of_speech=pos_detail.part_of_speech,
        part_of_speech_detail_1=pos_detail.part_of_speech_detail_1,
        part_of_speech_detail_2=pos_detail.part_of_speech_detail_2,
        part_of_speech_detail_3=pos_detail.part_of_speech_detail_3,
        inflectional_type="*",
        inflectional_form="*",
        stem="*",
        yomi=pronunciation,
        pronunciation=pronunciation,
        accent_type=accent_type,
        accent_associative_rule="*",
    )


def apply_word(
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: Optional[WordTypes] = None,
    priority: Optional[int] = None,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> str:
    """
    신규 단어 추가
    Parameters
    ----------
    surface : str
        단어 정보
    pronunciation : str
        단어 정보
    accent_type : int
        단어 정보
    word_type : Optional[WordTypes]
        품사
    priority : Optional[int]
        우선도
    user_dict_path : Path
        사용자 사전 파일의 경로
    compiled_dict_path : Path
        컴파일된 사전 파일의 경로
    Returns
    -------
    word_uuid : UserDictWord
        추가된 단어에 발행된 UUID
    """
    # 신규 단어 추가에 의한 사전 데이터 업데이트
    word = _create_word(
        surface=surface,
        pronunciation=pronunciation,
        accent_type=accent_type,
        word_type=word_type,
        priority=priority,
    )
    user_dict = read_dict(user_dict_path=user_dict_path)
    word_uuid = str(uuid4())
    user_dict[word_uuid] = word

    # 업데이트된 사전 데이터 저장 및 적용
    _write_to_json(user_dict, user_dict_path)
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_dict_path)

    return word_uuid


def rewrite_word(
    word_uuid: str,
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: Optional[WordTypes] = None,
    priority: Optional[int] = None,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> None:
    """
    기존 단어 덮어쓰기 업데이트
    Parameters
    ----------
    word_uuid : str
        단어 UUID
    surface : str
        단어 정보
    pronunciation : str
        단어 정보
    accent_type : int
        단어 정보
    word_type : Optional[WordTypes]
        품사
    priority : Optional[int]
        우선도
    user_dict_path : Path
        사용자 사전 파일의 경로
    compiled_dict_path : Path
        컴파일된 사전 파일의 경로
    """
    word = _create_word(
        surface=surface,
        pronunciation=pronunciation,
        accent_type=accent_type,
        word_type=word_type,
        priority=priority,
    )

    # 기존 단어 덮어쓰기에 의한 사전 데이터 업데이트
    user_dict = read_dict(user_dict_path=user_dict_path)
    if word_uuid not in user_dict:
        raise KeyError("UUID에 해당하는 단어를 찾을 수 없습니다")
    user_dict[word_uuid] = word

    # 업데이트된 사전 데이터 저장 및 적용
    _write_to_json(user_dict, user_dict_path)
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_dict_path)


def delete_word(
    word_uuid: str,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> None:
    """
    단어 삭제
    Parameters
    ----------
    word_uuid : str
        단어 UUID
    user_dict_path : Path
        사용자 사전 파일의 경로
    compiled_dict_path : Path
        컴파일된 사전 파일의 경로
    """
    # 기존 단어 삭제에 의한 사전 데이터 업데이트
    user_dict = read_dict(user_dict_path=user_dict_path)
    if word_uuid not in user_dict:
        raise KeyError("ID에 해당하는 단어를 찾을 수 없습니다")
    del user_dict[word_uuid]

    # 업데이트된 사전 데이터 저장 및 적용
    _write_to_json(user_dict, user_dict_path)
    update_dict(user_dict_path=user_dict_path, compiled_dict_path=compiled_dict_path)


def import_user_dict(
    dict_data: Dict[str, UserDictWord],
    override: bool = False,
    user_dict_path: Path = user_dict_path,
    compiled_dict_path: Path = compiled_dict_path,
) -> None:
    # 임포트할 사전 데이터의 유효성 검증
    for word_uuid, word in dict_data.items():
        UUID(word_uuid)
        assert isinstance(word, UserDictWord)
        for pos_detail in part_of_speech_data.values():
            if word.context_id == pos_detail.context_id:
                assert word.part_of_speech == pos_detail.part_of_speech
                assert (
                    word.part_of_speech_detail_1 == pos_detail.part_of_speech_detail_1
                )
                assert (
                    word.part_of_speech_detail_2 == pos_detail.part_of_speech_detail_2
                )
                assert (
                    word.part_of_speech_detail_3 == pos_detail.part_of_speech_detail_3
                )
                assert (
                    word.accent_associative_rule in pos_detail.accent_associative_rules
                )
                break
        else:
            raise ValueError("지원하지 않는 품사입니다")

    # 기존 사전 읽기
    old_dict = read_dict(user_dict_path=user_dict_path)

    # 사전 데이터 업데이트
    # 중복 항목 덮어쓰기
    if override:
        new_dict = {**old_dict, **dict_data}
    # 중복 항목 유지
    else:
        new_dict = {**dict_data, **old_dict}

    # 업데이트된 사전 데이터 저장 및 적용
    _write_to_json(user_dict=new_dict, user_dict_path=user_dict_path)
    update_dict(
        user_dict_path=user_dict_path,
        compiled_dict_path=compiled_dict_path,
    )


def _search_cost_candidates(context_id: int) -> List[int]:
    for value in part_of_speech_data.values():
        if value.context_id == context_id:
            return value.cost_candidates
    raise ValueError("품사 ID가 올바르지 않습니다")


def _cost2priority(context_id: int, cost: int) -> int:
    assert -32768 <= cost <= 32767
    cost_candidates = _search_cost_candidates(context_id)
    # cost_candidates 중에서 가장 가까운 값을 기반으로 priority를 반환
    # 참고: https://qiita.com/Krypf/items/2eada91c37161d17621d
    # 이 함수와 priority2cost 함수에 의해, 사전 파일의 cost를 조작해도 가장 가까운 priority의 cost로 덮어쓰여짐
    return MAX_PRIORITY - np.argmin(np.abs(np.array(cost_candidates) - cost)).item()


def _priority2cost(context_id: int, priority: int) -> int:
    assert MIN_PRIORITY <= priority <= MAX_PRIORITY
    cost_candidates = _search_cost_candidates(context_id)
    return cost_candidates[MAX_PRIORITY - priority]
