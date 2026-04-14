import re
from typing import TypedDict


import pyopenjtalk

from hayakoe.logging import logger
from hayakoe.nlp import bert_models
from hayakoe.nlp.japanese.mora_list import MORA_KATA_TO_MORA_PHONEMES, VOWELS
from hayakoe.nlp.japanese.normalizer import replace_punctuation
from hayakoe.nlp.symbols import PUNCTUATIONS


def g2p(
    norm_text: str, use_jp_extra: bool = True, raise_yomi_error: bool = False
) -> tuple[list[str], list[int], list[int]]:
    """
    다른 곳에서 사용되는 메인 함수. `normalize_text()`로 정규화된 `norm_text`를 받아서,
    - phones: 음소의 리스트 (단, `!`이나 `,`이나 `.` 등 punctuation이 포함될 수 있음)
    - tones: 악센트의 리스트, 0(저)과 1(고)로 구성되며 phones와 같은 길이
    - word2ph: 원본 텍스트의 각 문자에 음소가 몇 개 할당되는지를 나타내는 리스트
    의 튜플을 반환한다.
    단, `phones`와 `tones`의 처음과 끝에 `_`가 들어가고, 이에 따라 `word2ph`의 처음과 마지막에 1이 추가된다.

    Args:
        norm_text (str): 정규화된 텍스트
        use_jp_extra (bool, optional): False인 경우, 「ん」의 음소를 "N"이 아닌 "n"으로 한다. Defaults to True.
        raise_yomi_error (bool, optional): False인 경우, 읽을 수 없는 문자가 "'"로 발음된다. Defaults to False.

    Returns:
        tuple[list[str], list[int], list[int]]: 음소의 리스트, 악센트의 리스트, word2ph의 리스트
    """

    # pyopenjtalk의 풀 컨텍스트 라벨을 사용하여 악센트를 추출하면, punctuation의 위치가 사라져 정보가 손실된다:
    # 「こんにちは、世界。」와 「こんにちは！世界。」와 「こんにちは！！！？？？世界……。」는 모두 같아진다.
    # 따라서, 먼저 punctuation 없는 음소와 악센트의 리스트를 만들고,
    # 그와 별도로 pyopenjtalk.run_frontend()에서 얻은 음소 리스트 (이쪽은 punctuation이 유지됨)를 사용하여,
    # 악센트 할당을 다시 수행함으로써 punctuation을 포함한 음소와 악센트의 리스트를 만든다.

    # punctuation이 모두 제거된, 음소와 악센트의 튜플 리스트 (「ん」은 「N」)
    phone_tone_list_wo_punct = __g2phone_tone_wo_punct(norm_text)

    # sep_text: 단어 단위의 단어 리스트
    # sep_kata: 단어 단위의 단어의 카타카나 읽기 리스트, 읽을 수 없는 문자는 raise_yomi_error=True이면 예외, False이면 읽을 수 없는 문자를 "'"로 반환
    sep_text, sep_kata = text_to_sep_kata(norm_text, raise_yomi_error=raise_yomi_error)

    # sep_phonemes: 각 단어별 음소 리스트의 리스트
    sep_phonemes = __handle_long([__kata_to_phoneme_list(i) for i in sep_kata])

    # phone_w_punct: sep_phonemes를 결합한, punctuation을 원래대로 유지한 음소열
    phone_w_punct: list[str] = []
    for i in sep_phonemes:
        phone_w_punct += i

    # punctuation 없는 악센트 정보를 사용하여, punctuation을 포함한 악센트 정보를 만든다
    phone_tone_list = __align_tones(phone_w_punct, phone_tone_list_wo_punct)
    # logger.debug(f"phone_tone_list:\n{phone_tone_list}")

    # word2ph는 엄밀한 해답이 불가능하므로 (「今日」「眼鏡」 등의 숙자훈이 존재),
    # Bert-VITS2에서는, 단어 단위의 분할을 사용하여 단어의 문자마다 대략 균등하게 음소를 분배한다

    # sep_text에서, 각 단어를 한 글자씩 분할하여 문자의 리스트 (의 리스트)를 만든다
    sep_tokenized: list[list[str]] = []
    for i in sep_text:
        if i not in PUNCTUATIONS:
            sep_tokenized.append(
                bert_models.load_tokenizer().tokenize(i)
            )  # 여기서 아마도 `i`가 문자 단위로 분할된다
        else:
            sep_tokenized.append([i])

    # 각 단어에 대해, 음소의 수와 문자의 수를 비교하여 균등하게 분배한다
    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        word2ph += __distribute_phone(phone_len, word_len)

    # 처음과 마지막에 `_` 기호를 추가, 악센트는 0(저), word2ph도 이에 맞춰 추가
    phone_tone_list = [("_", 0)] + phone_tone_list + [("_", 0)]
    word2ph = [1] + word2ph + [1]

    phones = [phone for phone, _ in phone_tone_list]
    tones = [tone for _, tone in phone_tone_list]

    assert len(phones) == sum(word2ph), f"{len(phones)} != {sum(word2ph)}"

    # use_jp_extra가 아닌 경우 "N"을 "n"으로 변환
    if not use_jp_extra:
        phones = [phone if phone != "N" else "n" for phone in phones]

    return phones, tones, word2ph


def text_to_sep_kata(
    norm_text: str, raise_yomi_error: bool = False
) -> tuple[list[str], list[str]]:
    """
    `normalize_text`로 정규화된 `norm_text`를 받아서 단어 분할하고,
    분할된 단어 리스트와 그 읽기 (카타카나 or 기호 1문자)의 리스트 튜플을 반환한다.
    단어 분할 결과는, `g2p()`의 `word2ph`에서 1문자당 할당할 음소 기호의 수를 결정하는 데 사용한다.
    예:
    `私はそう思う!って感じ?` →
    ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]

    Args:
        norm_text (str): 정규화된 텍스트
        raise_yomi_error (bool, optional): False인 경우, 읽을 수 없는 문자가 "'"로 발음된다. Defaults to False.

    Returns:
        tuple[list[str], list[str]]: 분할된 단어 리스트와 그 읽기 (카타카나 or 기호 1문자)의 리스트
    """

    # parsed: OpenJTalk의 분석 결과
    parsed = pyopenjtalk.run_frontend(norm_text)
    sep_text: list[str] = []
    sep_kata: list[str] = []

    for parts in parsed:
        # word: 실제 단어의 문자열
        # yomi: 그 읽기, 단 무성화 기호 `’`는 제거
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        """
        여기서 `yomi`가 취할 수 있는 값은 다음과 같을 것이다.
        - `word`가 통상 단어 → 통상의 읽기 (카타카나)
            (카타카나로 구성되며, 장음 기호도 포함될 수 있음, `アー` 등)
        - `word`가 `ー`로 시작 → `ーラー`나 `ーーー` 등
        - `word`가 구두점이나 공백 등 → `、`
        - `word`가 punctuation의 반복 → 전각으로 변환된 것
        기본적으로 punctuation은 1문자씩 분리되지만, 어느 정도 연속되면 하나로 합쳐진다.
        또한 `word`가 읽을 수 없는 키릴 문자나 아랍 문자 등이면 `、`가 되지만, 정규화에 의해 이 경우는 발생하지 않을 것이다.
        또한 원래 코드에서는 `yomi`가 공백인 경우의 처리가 있었지만, 이것은 발생하지 않을 것이다.
        처리해야 할 것은 `yomi`가 `、`인 경우뿐일 것이다.
        """
        assert yomi != "", f"Empty yomi: {word}"
        if yomi == "、":
            # word는 정규화되어 있으므로, `.`, `,`, `!`, `'`, `-`, `--` 중 하나
            if not set(word).issubset(set(PUNCTUATIONS)):  # 기호 반복인지 판정
                # 여기는 pyopenjtalk가 읽을 수 없는 문자 등일 때 발생한다
                ## 예외를 발생시키는 경우
                if raise_yomi_error:
                    raise YomiError(f"Cannot read: {word} in:\n{norm_text}")
                ## 예외를 발생시키지 않는 경우
                ## 읽을 수 없는 문자는 "'"로 취급한다
                logger.warning(
                    f'Cannot read: {word} in:\n{norm_text}, replaced with "\'"'
                )
                # word의 글자 수만큼 "'"을 추가
                yomi = "'" * len(word)
            else:
                # yomi를 원래 기호 그대로 변경
                yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)

    return sep_text, sep_kata


def adjust_word2ph(
    word2ph: list[int],
    generated_phone: list[str],
    given_phone: list[str],
) -> list[int]:
    """
    `g2p()`에서 얻은 `word2ph`를, generated_phone과 given_phone의 차분 정보를 사용하여 적절히 조정한다.
    generated_phone은 정규화된 읽기 텍스트에서 생성된 읽기 정보이지만,
    given_phone으로 같은 읽기 텍스트에 다른 읽기가 주어진 경우, 정규화된 읽기 텍스트의 각 문자에
    음소가 몇 개 할당되는지를 나타내는 word2ph의 합계값이 given_phone의 길이 (음소 수)와 일치하지 않을 수 있다.
    따라서 generated_phone과 given_phone의 차분을 취해 변경 지점에 대응하는 word2ph 요소의 값만 증감시켜,
    악센트에 대한 영향을 최소한으로 억제하면서 word2ph의 합계값을 given_phone의 길이 (음소 수)에 일치시킨다.

    Args:
        word2ph (list[int]): 단어별 음소 수의 리스트
        generated_phone (list[str]): 생성된 음소의 리스트
        given_phone (list[str]): 주어진 음소의 리스트

    Returns:
        list[int]: 수정된 word2ph의 리스트
    """

    # word2ph, generated_phone, given_phone 모두 선두와 말미에 더미 요소가 들어있으므로, 처리 편의상 이를 삭제
    # word2ph는 선두와 말미에 1이 들어있다 (반환 시 다시 추가함)
    word2ph = word2ph[1:-1]
    generated_phone = generated_phone[1:-1]
    given_phone = given_phone[1:-1]

    class DiffDetail(TypedDict):
        begin_index: int
        end_index: int
        value: list[str]

    class Diff(TypedDict):
        generated: DiffDetail
        given: DiffDetail

    def extract_differences(
        generated_phone: list[str], given_phone: list[str]
    ) -> list[Diff]:
        """
        최장 공통 부분열을 기반으로, 두 리스트의 다른 부분을 추출한다.
        """

        def longest_common_subsequence(
            X: list[str], Y: list[str]
        ) -> list[tuple[int, int]]:
            """
            두 리스트의 최장 공통 부분열의 인덱스 쌍을 반환한다.
            """
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            # LCS의 길이 구축
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            # LCS를 역방향으로 트레이스하여 인덱스 쌍을 취득
            index_pairs = []
            i, j = m, n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    index_pairs.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif L[i - 1][j] >= L[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            index_pairs.reverse()
            return index_pairs

        differences = []
        common_indices = longest_common_subsequence(generated_phone, given_phone)
        prev_x, prev_y = -1, -1

        # 공통 부분의 인덱스를 기반으로 차분을 추출
        for x, y in common_indices:
            diff_X = {
                "begin_index": prev_x + 1,
                "end_index": x,
                "value": generated_phone[prev_x + 1 : x],
            }
            diff_Y = {
                "begin_index": prev_y + 1,
                "end_index": y,
                "value": given_phone[prev_y + 1 : y],
            }
            if diff_X or diff_Y:
                differences.append({"generated": diff_X, "given": diff_Y})
            prev_x, prev_y = x, y
        # 마지막 비공통 부분을 추가
        if prev_x < len(generated_phone) - 1 or prev_y < len(given_phone) - 1:
            differences.append(
                {
                    "generated": {
                        "begin_index": prev_x + 1,
                        "end_index": len(generated_phone) - 1,
                        "value": generated_phone[prev_x + 1 : len(generated_phone) - 1],
                    },
                    "given": {
                        "begin_index": prev_y + 1,
                        "end_index": len(given_phone) - 1,
                        "value": given_phone[prev_y + 1 : len(given_phone) - 1],
                    },
                }
            )
        # generated.value와 given.value 모두 빈 요소를 differences에서 삭제
        for diff in differences[:]:
            if (
                len(diff["generated"]["value"]) == 0
                and len(diff["given"]["value"]) == 0
            ):
                differences.remove(diff)

        return differences

    # 두 리스트의 차분을 추출
    differences = extract_differences(generated_phone, given_phone)

    # word2ph를 기반으로 새로 만들 word2ph의 리스트
    ## 길이는 word2ph와 같지만, 내용은 0으로 초기화되어 있다
    adjusted_word2ph: list[int] = [0] * len(word2ph)
    # 현재 처리 중인 generated_phone의 인덱스
    current_generated_index = 0

    # word2ph의 요소 수 (=정규화된 읽기 텍스트의 문자 수)를 유지하면서, 차분 정보를 사용하여 word2ph를 수정
    ## 음소 수가 generated_phone과 given_phone에서 다른 경우에 이 adjust_word2ph()가 호출된다
    ## word2ph는 정규화된 읽기 텍스트의 문자 수에 대응하므로, 요소 수는 그대로 given_phone에서 증감한 음소 수에 맞춰 각 요소의 값을 증감한다
    for word2ph_element_index, word2ph_element in enumerate(word2ph):
        # 여기의 word2ph_element는, 정규화된 읽기 텍스트의 각 문자에 할당되는 음소 수를 나타낸다
        # 예를 들어 word2ph_element가 2이면, 해당 문자에는 2개의 음소 (예: "k", "a")가 할당된다
        # 음소 수만큼 루프를 돈다
        for _ in range(word2ph_element):
            # difference 중에 처리 중인 generated_phone에서 시작하는 차분이 있는지 확인
            current_diff: Diff | None = None
            for diff in differences:
                if diff["generated"]["begin_index"] == current_generated_index:
                    current_diff = diff
                    break
            # current_diff가 None이 아닌 경우, generated_phone에서 시작하는 차분이 있다
            if current_diff is not None:
                # generated에서 given으로 변경된 음소 수의 차분을 취득 (2 증가한 경우 +2, 2 감소한 경우 -2)
                diff_in_phonemes = \
                    len(current_diff["given"]["value"]) - len(current_diff["generated"]["value"])  # fmt: skip
                # adjusted_word2ph[(읽기 텍스트의 각 문자 인덱스)]에 위 차분을 반영
                adjusted_word2ph[word2ph_element_index] += diff_in_phonemes
            # adjusted_word2ph[(읽기 텍스트의 각 문자 인덱스)]에 처리 완료된 분의 음소로서 1을 더한다
            adjusted_word2ph[word2ph_element_index] += 1
            # 처리 중인 generated_phone의 인덱스를 진행시킨다
            current_generated_index += 1

    # 이 시점에서 given_phone의 길이와 adjusted_word2ph에 기록된 음소 수의 합계가 일치해야 한다
    assert len(given_phone) == sum(adjusted_word2ph), f"{len(given_phone)} != {sum(adjusted_word2ph)}"  # fmt: skip

    # generated_phone에서 given_phone 사이에 음소가 줄어든 경우 (예: a, sh, i, t, a -> a, s, u),
    # adjusted_word2ph 요소의 값이 1 미만이 될 수 있으므로, 1이 되도록 값을 올린다
    ## 이때, adjusted_word2ph에 기록된 음소 수의 합계를 바꾸지 않기 위해,
    ## 값을 1로 만든 만큼 오른쪽 이웃 요소에서 증가분의 차분을 차감한다
    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):  # fmt: skip
        # 만약 현재 요소가 1 미만이면
        if adjusted_word2ph_element < 1:
            # 값을 1로 만들기 위해 얼마나 더해야 하는지 계산
            diff = 1 - adjusted_word2ph_element
            # adjusted_word2ph[(읽기 텍스트의 각 문자 인덱스)]를 1로 만든다
            # 이로써, 해당 문자에 최소한으로 1개의 음소가 할당된다
            adjusted_word2ph[adjusted_word2ph_element_index] = 1
            # 다음 요소 중, 가장 가깝고 1 이상인 요소에서 diff를 뺀다
            # 이때, diff를 뺀 결과 해당 요소가 1 미만이 되는 경우, 그 요소의 다음 요소 중 가장 가깝고 1 이상인 요소에서 뺀다
            # 위를 반복하여 diff가 0이 될 때까지 계속한다
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break  # adjusted_word2ph의 끝에 도달한 경우 포기
                if adjusted_word2ph[adjusted_word2ph_element_index + i] - diff >= 1:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] -= diff
                    break
                else:
                    diff -= adjusted_word2ph[adjusted_word2ph_element_index + i] - 1
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 1
                    if diff == 0:
                        break

    # 반대로, generated_phone에서 given_phone 사이에 음소가 증가한 경우 (예: a, s, u -> a, sh, i, t, a),
    # 1문자당 7음소 이상 할당되는 경우가 있으므로, 최대 6음소로 제한한 뒤 삭감분의 차분을 다음 요소에 더한다
    # 다음 요소에 차분을 더한 결과 7음소 이상이 되는 경우, 그 차분을 더 다음 요소에 더한다
    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):  # fmt: skip
        if adjusted_word2ph_element > 6:
            diff = adjusted_word2ph_element - 6
            adjusted_word2ph[adjusted_word2ph_element_index] = 6
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break  # adjusted_word2ph의 끝에 도달한 경우 포기
                if adjusted_word2ph[adjusted_word2ph_element_index + i] + diff <= 6:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] += diff
                    break
                else:
                    diff -= 6 - adjusted_word2ph[adjusted_word2ph_element_index + i]
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 6
                    if diff == 0:
                        break

    # 이 시점에서 given_phone의 길이와 adjusted_word2ph에 기록된 음소 수의 합계가 일치하지 않는 경우,
    # 정규화된 읽기 텍스트와 given_phone이 현저히 괴리되어 있음을 나타낸다
    # 이때, 이 함수의 호출원인 get_text()에서 InvalidPhoneError가 발생한다

    # 처음에 삭제한 앞뒤의 더미 요소를 추가하여 반환한다
    return [1] + adjusted_word2ph + [1]


def __g2phone_tone_wo_punct(text: str) -> list[tuple[str, int]]:
    """
    텍스트에 대해, 음소와 악센트 (0 또는 1)의 쌍 리스트를 반환한다.
    단, "!" "." "?" 등의 비음소 기호 (punctuation)는 모두 사라진다 (포즈 기호도 남기지 않음).
    비음소 기호를 포함하는 처리는 `align_tones()`에서 수행된다.
    또한 「っ」은 "q"로, 「ん」은 "N"으로 변환된다.
    예: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]

    Args:
        text (str): 텍스트

    Returns:
        list[tuple[str, int]]: 음소와 악센트의 쌍 리스트
    """

    prosodies = __pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
    # logger.debug(f"prosodies: {prosodies}")
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0

    for i, letter in enumerate(prosodies):
        # 특수 기호 처리

        # 문두 기호, 무시한다
        if letter == "^":
            assert i == 0, "Unexpected ^"
        # 악센트구의 끝에 오는 기호
        elif letter in ("$", "?", "_", "#"):
            # 보유하고 있는 프레이즈를, 악센트 수치를 0-1로 수정하여 결과에 추가
            result.extend(__fix_phone_tone(current_phrase))
            # 말미에 오는 종료 기호, 무시 (문중의 의문문은 `_`가 된다)
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            # 나머지는 "_" (포즈)와 "#" (악센트구의 경계)뿐
            # 이들은 남기지 않고, 다음 악센트구에 대비한다.
            current_phrase = []
            # 0을 기준점으로 하여 거기서 상승/하강한다 (음수인 경우 위의 `fix_phone_tone`에서 수정됨)
            current_tone = 0
        # 악센트 상승 기호
        elif letter == "[":
            current_tone = current_tone + 1
        # 악센트 하강 기호
        elif letter == "]":
            current_tone = current_tone - 1
        # 그 외에는 통상의 음소
        else:
            if letter == "cl":  # 「っ」 처리
                letter = "q"
            # elif letter == "N":  # 「ん」 처리
            #     letter = "n"
            current_phrase.append((letter, current_tone))

    return result


__PYOPENJTALK_G2P_PROSODY_A1_PATTERN = re.compile(r"/A:([0-9\-]+)\+")
__PYOPENJTALK_G2P_PROSODY_A2_PATTERN = re.compile(r"\+(\d+)\+")
__PYOPENJTALK_G2P_PROSODY_A3_PATTERN = re.compile(r"\+(\d+)/")
__PYOPENJTALK_G2P_PROSODY_E3_PATTERN = re.compile(r"!(\d+)_")
__PYOPENJTALK_G2P_PROSODY_F1_PATTERN = re.compile(r"/F:(\d+)_")
__PYOPENJTALK_G2P_PROSODY_P3_PATTERN = re.compile(r"\-(.*?)\+")


def __pyopenjtalk_g2p_prosody(
    text: str, drop_unvoiced_vowels: bool = True
) -> list[str]:
    """
    ESPnet의 구현에서 인용, 대체로 변경 없음. 「ん」은 "N"인 것에 주의.
    ref: https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    ------------------------------------------------------------------------------------------

    입력 풀 컨텍스트 라벨에서 음소 + 운율 기호 시퀀스를 추출한다.

    이 알고리즘은 `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_를 기반으로 하며, r9y9의 조정이 포함되어 있다.

    Args:
        text (str): 입력 텍스트.
        drop_unvoiced_vowels (bool): 무성 모음을 제거할지 여부.

    Returns:
        List[str]: 음소 + 운율 기호의 리스트.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104
    """

    def _numeric_feature_by_regex(pattern: re.Pattern[str], s: str) -> int:
        match = pattern.search(s)
        if match is None:
            return -50
        return int(match.group(1))

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # 현재 음소
        p3 = __PYOPENJTALK_G2P_PROSODY_P3_PATTERN.search(lab_curr).group(1)  # type: ignore
        # 무성 모음을 통상 모음으로 취급
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 텍스트 처음과 끝의 sil 처리
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # 의문형인지 확인
                e3 = _numeric_feature_by_regex(
                    __PYOPENJTALK_G2P_PROSODY_E3_PATTERN, lab_curr
                )
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # 악센트 타입 및 위치 정보 (순방향 또는 역방향)
        a1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A1_PATTERN, lab_curr)
        a2 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A2_PATTERN, lab_curr)
        a3 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A3_PATTERN, lab_curr)

        # 악센트구 내의 모라 수
        f1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_F1_PATTERN, lab_curr)

        a2_next = _numeric_feature_by_regex(
            __PYOPENJTALK_G2P_PROSODY_A2_PATTERN, labels[n + 1]
        )
        # 악센트구 경계
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # 피치 하강
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # 피치 상승
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def __fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list`의 tone (악센트 값)을 0 또는 1의 범위로 수정한다.
    예: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]

    Args:
        phone_tone_list (list[tuple[str, int]]): 음소와 악센트의 쌍 리스트

    Returns:
        list[tuple[str, int]]: 수정된 음소와 악센트의 쌍 리스트
    """

    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def __handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    """
    프레이즈별로 나뉜 음소 (장음 기호가 그대로인) 리스트의 리스트 `sep_phonemes`를 받아서,
    그 장음 기호를 처리하여 음소의 리스트의 리스트를 반환한다.
    기본적으로는 직전의 음소를 늘리지만, 직전의 음소가 모음이 아닌 경우 또는 문두인 경우,
    아마도 장음 기호와 대시를 혼동한 것으로 추정되므로, 대시에 대응하는 음소 `-`로 변환한다.

    Args:
        sep_phonemes (list[list[str]]): 프레이즈별로 나뉜 음소의 리스트의 리스트

    Returns:
        list[list[str]]: 장음 기호를 처리한 음소의 리스트의 리스트
    """

    for i in range(len(sep_phonemes)):
        if len(sep_phonemes[i]) == 0:
            # 공백 문자 등으로 리스트가 빈 경우
            continue
        if sep_phonemes[i][0] == "ー":
            if i != 0:
                prev_phoneme = sep_phonemes[i - 1][-1]
                if prev_phoneme in VOWELS:
                    # 모음과 「ん」 뒤의 장음이므로, 해당 모음으로 변환
                    sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
                else:
                    # 「。ーー」 등 아마도 예상치 못한 장음 기호
                    # 대시를 혼동한 것으로 추정됨
                    sep_phonemes[i][0] = "-"
            else:
                # 문두에 장음 기호가 왔으며, 이는 대시를 혼동한 것으로 추정됨
                sep_phonemes[i][0] = "-"
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]

    return sep_phonemes


__KATAKANA_PATTERN = re.compile(r"[\u30A0-\u30FF]+")
__MORA_PATTERN = re.compile(
    "|".join(
        map(re.escape, sorted(MORA_KATA_TO_MORA_PHONEMES.keys(), key=len, reverse=True))
    )
)
__LONG_PATTERN = re.compile(r"(\w)(ー*)")


def __kata_to_phoneme_list(text: str) -> list[str]:
    """
    원칙적으로 카타카나인 `text`를 받아서, 그대로 음소 기호의 리스트로 변환한다.
    주의사항:
    - punctuation이나 그 반복이 들어온 경우, punctuation들을 그대로 리스트로 반환한다.
    - 문두에 오는 「ー」는 그대로 「ー」로 남긴다 (`handle_long()`에서 처리됨)
    - 문중의 「ー」는 앞 음소 기호의 마지막 음소 기호로 변환된다.
    예:
    `ーーソーナノカーー` → ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    `?` → ["?"]
    `!?!?!?!?!` → ["!", "?", "!", "?", "!", "?", "!", "?", "!"]

    Args:
        text (str): 카타카나 텍스트

    Returns:
        list[str]: 음소 기호의 리스트
    """

    if set(text).issubset(set(PUNCTUATIONS)):
        return list(text)
    # `text`가 카타카나 (`ー` 포함)로만 구성되어 있는지 확인
    if __KATAKANA_PATTERN.fullmatch(text) is None:
        raise ValueError(f"Input must be katakana only: {text}")

    def mora2phonemes(mora: str) -> str:
        consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
        if consonant is None:
            return f" {vowel}"
        return f" {consonant} {vowel}"

    spaced_phonemes = __MORA_PATTERN.sub(lambda m: mora2phonemes(m.group()), text)

    # 장음 기호 「ー」 처리
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))  # type: ignore
    spaced_phonemes = __LONG_PATTERN.sub(long_replacement, spaced_phonemes)

    return spaced_phonemes.strip().split(" ")


def __align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """
    예: …私は、、そう思う。
    phones_with_punct:
        [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
        [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("_", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
        [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]

    Args:
        phones_with_punct (list[str]): punctuation을 포함하는 음소의 리스트
        phone_tone_list (list[tuple[str, int]]): punctuation을 포함하지 않는 음소와 악센트의 쌍 리스트

    Returns:
        list[tuple[str, int]]: punctuation을 포함하는 음소와 악센트의 쌍 리스트
    """

    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            # 남은 punctuation이 있는 경우 → (punctuation, 0)을 추가
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            # phone_tone_list의 현재 음소와 일치하는 경우 → tone을 거기서 취득, (phone, tone)을 추가
            result.append((phone, phone_tone_list[tone_index][1]))
            # 탐색 index를 1 진행
            tone_index += 1
        elif phone in PUNCTUATIONS:
            # phone이 punctuation인 경우 → (phone, 0)을 추가
            result.append((phone, 0))
        else:
            logger.debug(f"phones: {phones_with_punct}")
            logger.debug(f"phone_tone_list: {phone_tone_list}")
            logger.debug(f"result: {result}")
            logger.debug(f"tone_index: {tone_index}")
            logger.debug(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")

    return result


def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """
    왼쪽에서 오른쪽으로 1씩 분배하고, 다시 왼쪽에서 오른쪽으로 1씩 늘리는 방식으로,
    음소의 수 `n_phone`을 단어의 수 `n_word`에 분배한다.

    Args:
        n_phone (int): 음소의 수
        n_word (int): 단어의 수

    Returns:
        list[int]: 단어별 음소 수의 리스트
    """

    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1

    return phones_per_word


class YomiError(Exception):
    """
    OpenJTalk에서 읽기를 올바르게 가져올 수 없는 부분이 있을 때 발생하는 예외.
    기본적으로 「학습 전처리의 텍스트 처리 시」에는 발생시키고, 그 외의 경우에는
    raise_yomi_error=False로 설정하여 이 예외가 발생하지 않도록 한다.
    """
