"""
이 파일은 VOICEVOX 프로젝트의 VOICEVOX ENGINE에서 차용한 것입니다.
출처: https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/user_dict/user_dict.py
라이선스: LGPL-3.0
자세한 내용은 이 파일과 같은 폴더에 있는 README.md를 참조하세요.

HayaKoe 용으로 축소·수정되었습니다.
원본은 사용자 사전을 JSON 파일에 영구 저장했지만, 이 버전은 프로세스 메모리에만
유지합니다. 프로세스가 종료되면 등록한 단어도 함께 사라집니다.
"""

import atexit
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pyopenjtalk

from hayakoe.nlp.japanese.user_dict.part_of_speech_data import (
    MAX_PRIORITY,
    MIN_PRIORITY,
    part_of_speech_data,
)
from hayakoe.nlp.japanese.user_dict.word_model import UserDictWord, WordTypes


# 프로세스 메모리 상에 유지되는 사용자 사전. 키는 단어 UUID.
_user_dict: dict[str, UserDictWord] = {}

# 현재 pyopenjtalk 에 로드되어 있는 컴파일된 사전이 위치한 임시 디렉터리.
# 재빌드 성공 시에만 새 디렉터리로 교체되고, 이전 디렉터리는 지워진다.
_current_dict_dir: Optional[Path] = None


def _cleanup_current_dict_dir() -> None:
    if _current_dict_dir is not None:
        shutil.rmtree(_current_dict_dir, ignore_errors=True)


atexit.register(_cleanup_current_dict_dir)


def _rebuild_compiled_dict(user_dict: dict[str, UserDictWord]) -> None:
    """주어진 사전을 임시 .dic 로 컴파일하고 pyopenjtalk 에 적용한다.

    실패 시 새로 만든 임시 디렉터리만 정리하고 예외를 전파하므로, 이전 사전
    상태 (``_current_dict_dir``) 는 그대로 유지된다. 성공 시에만 기존 디렉터리를
    새 디렉터리로 교체하고 이전 디렉터리를 삭제한다.
    """
    global _current_dict_dir

    new_dir = Path(tempfile.mkdtemp(prefix="hayakoe-user-dict-"))
    csv_path = new_dir / "user.csv"
    dic_path = new_dir / "user.dic"

    try:
        csv_text = ""
        for word in user_dict.values():
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
        csv_path.write_text(csv_text, encoding="utf-8")

        pyopenjtalk.mecab_dict_index(str(csv_path), str(dic_path))
        if not dic_path.is_file():
            raise RuntimeError("사전 컴파일 중 오류가 발생했습니다.")

        # update_global_jtalk_with_user_dict 가 내부적으로 새 OpenJTalk 인스턴스를
        # 만들어 교체하므로 별도의 unset 호출은 필요하지 않다.
        pyopenjtalk.update_global_jtalk_with_user_dict(str(dic_path))

    except Exception:
        shutil.rmtree(new_dir, ignore_errors=True)
        print("Error: Failed to update dictionary.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

    old_dir = _current_dict_dir
    _current_dict_dir = new_dir
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)


def _create_word(
    surface: str,
    pronunciation: str,
    accent_type: int,
    word_type: Optional[WordTypes] = None,
    priority: Optional[int] = None,
) -> UserDictWord:
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
) -> str:
    """신규 단어를 프로세스 메모리 상 사용자 사전에 추가하고 pyopenjtalk 에 반영한다.

    등록된 단어는 디스크에 영구 저장되지 않으며, 프로세스가 종료되면 함께 사라진다.
    재빌드가 실패하면 메모리 사전은 이전 상태 그대로 유지된다.

    Returns:
        추가된 단어에 발행된 UUID.
    """
    word = _create_word(
        surface=surface,
        pronunciation=pronunciation,
        accent_type=accent_type,
        word_type=word_type,
        priority=priority,
    )
    word_uuid = str(uuid4())
    candidate = {**_user_dict, word_uuid: word}
    _rebuild_compiled_dict(candidate)
    _user_dict[word_uuid] = word
    return word_uuid


def _search_cost_candidates(context_id: int) -> list[int]:
    for value in part_of_speech_data.values():
        if value.context_id == context_id:
            return value.cost_candidates
    raise ValueError("품사 ID가 올바르지 않습니다")


def _priority2cost(context_id: int, priority: int) -> int:
    assert MIN_PRIORITY <= priority <= MAX_PRIORITY
    cost_candidates = _search_cost_candidates(context_id)
    return cost_candidates[MAX_PRIORITY - priority]
