from hayakoe.nlp.japanese.g2p import g2p
from hayakoe.nlp.japanese.mora_list import (
    CONSONANTS,
    MORA_KATA_TO_MORA_PHONEMES,
    MORA_PHONEMES_TO_MORA_KATA,
)
from hayakoe.nlp.symbols import PUNCTUATIONS


def g2kata_tone(norm_text: str) -> list[tuple[str, int]]:
    """
    텍스트에서 카타카나와 악센트 쌍의 리스트를 반환한다.
    추론 시에만 사용되는 함수이므로, 항상 `raise_yomi_error=False`를 지정하여 g2p()를 호출하는 사양으로 되어 있다.

    Args:
        norm_text: 정규화된 텍스트.

    Returns:
        카타카나와 음높이의 리스트.
    """

    phones, tones, _ = g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    return phone_tone2kata_tone(list(zip(phones, tones)))


def phone_tone2kata_tone(phone_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    phone_tone의 phone 부분을 카타카나로 변환한다. 단, 처음과 마지막의 ("_", 0)은 무시한다.

    Args:
        phone_tone: 음소와 음높이의 리스트.

    Returns:
        카타카나와 음높이의 리스트.
    """

    phone_tone = phone_tone[1:]  # 처음의 ("_", 0)을 무시
    phones = [phone for phone, _ in phone_tone]
    tones = [tone for _, tone in phone_tone]
    result: list[tuple[str, int]] = []
    current_mora = ""
    for phone, next_phone, tone, next_tone in zip(phones, phones[1:], tones, tones[1:]):
        # zip 특성상 마지막의 ("_", 0)은 무시됨
        if phone in PUNCTUATIONS:
            result.append((phone, tone))
            continue
        if phone in CONSONANTS:  # n 이외의 자음인 경우
            assert current_mora == "", f"Unexpected {phone} after {current_mora}"
            assert tone == next_tone, f"Unexpected {phone} tone {tone} != {next_tone}"
            current_mora = phone
        else:
            # phone이 모음 또는 "N"인 경우
            current_mora += phone
            result.append((MORA_PHONEMES_TO_MORA_KATA[current_mora], tone))
            current_mora = ""

    return result


def kata_tone2phone_tone(kata_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone2kata_tone()`의 역변환을 수행한다.

    Args:
        kata_tone: 카타카나와 음높이의 리스트.

    Returns:
        음소와 음높이의 리스트.
    """

    result: list[tuple[str, int]] = [("_", 0)]
    for mora, tone in kata_tone:
        if mora in PUNCTUATIONS:
            result.append((mora, tone))
        else:
            consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
            if consonant is None:
                result.append((vowel, tone))
            else:
                result.append((consonant, tone))
                result.append((vowel, tone))
    result.append(("_", 0))

    return result
