# 커스텀 단어 등록

HayaKoe 는 일본어 형태소 분석을 위해 pyopenjtalk + OpenJTalk 사전을 사용합니다.

이 사전에는 일반적인 어휘는 거의 다 들어있지만, 희귀한 고유명사·외래어·신조어·상호명 같은 건 빠져 있을 수 있습니다.

그런 단어는 문장 안에서 이상하게 끊기거나 한 글자씩 읽혀 버리는데, 직접 등록해서 원하는 발음으로 고정할 수 있습니다.

::: info 영어 단어는 이미 약 22만 개가 내장돼 있습니다
HayaKoe 는 pyopenjtalk 에 텍스트를 넘기기 **전** 단계에서, 영어 단어를 가타카나로 치환하는 내부 정규화 사전을 가지고 있습니다.

이 사전에는 221,587 개의 영어 단어와 대응 가타카나가 미리 들어 있어서, `OpenAI`·`GitHub` 같은 일반적인 영어 고유명사는 따로 `add_word()` 를 호출하지 않아도 자연스럽게 읽힙니다.

`add_word()` 는 이 정규화 단계 뒤, **pyopenjtalk 의 일본어 형태소 분석 단계** 에 영향을 주는 별개 레이어입니다.

즉, 주로 잘못 읽히는 일본어 고유명사·희귀어·신조어를 고정하는 용도라고 생각하시면 됩니다.
:::

## 가장 짧은 예

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)

speaker.generate("担々麺が食べたい。").save("tantanmen.wav")
```

`prepare()` 가 끝난 뒤 언제든지 `add_word()` 를 호출할 수 있고, 이후의 `generate()` 부터 바로 반영됩니다.

등록 전/후로 같은 문장이 어떻게 달라지는지 들어보면 감이 빠릅니다.

`担々麺` 의 `々` 는 OpenJTalk 사전에서 독립 기호로 처리되는 탓에, 아무 설정 없이 합성하면 "タン / メン" 두 덩어리로 쪼개져 버립니다.

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="등록 전 — 担々麺이 タン / メン 으로 끊김" src="/hayakoe/samples/custom-words/tantanmen_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="등록 후 — タンタンメン 하나로 이어짐" src="/hayakoe/samples/custom-words/tantanmen_after.wav" />

## 영어 브랜드명·신조어도 같은 방식

앞서 22만 개 영어 단어는 자동으로 가타카나로 바뀐다고 말씀드렸지만, **사전에 없는 브랜드명·제품명·신조어** 는 그 자동 치환이 안 먹힙니다.

그런 경우엔 pyopenjtalk 가 글자 단위로 쪼개서 읽어 버리기 때문에, 역시 `add_word()` 로 고쳐줘야 합니다.

예로 이 라이브러리 이름인 `HayaKoe` 를 그대로 읽히면 `エイチ・エー・ワイ・エー・ケー・オー・イー` 처럼 알파벳을 하나씩 어색하게 불러 버립니다.

```python
text = "HayaKoeは速い日本語の音声合成ライブラリです。"

# 등록 전
speaker.generate(text).save("before.wav")

# 원하는 가타카나 발음으로 고정
tts.add_word(surface="HayaKoe", reading="ハヤコエ", accent=0)

# 등록 후
speaker.generate(text).save("after.wav")
```

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="등록 전 — HayaKoe 가 알파벳 하나씩 읽힘" src="/hayakoe/samples/custom-words/hayakoe_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="등록 후 — ハヤコエ 로 자연스럽게 이어짐" src="/hayakoe/samples/custom-words/hayakoe_after.wav" />

이처럼 22만 개 사전에 없는 단어는 눈에 띄는 대로 `add_word()` 로 잡아 주면 됩니다.

## 인자 세 개

### `surface` — 문장에 쓰는 글자 그대로

실제 입력 문장에 나타나는 **글자 그대로** 를 넘기면 됩니다.

한자·히라가나·가타카나·영문 어느 쪽이든 괜찮고, 섞여 있어도 됩니다.

### `reading` — 가타카나 발음

어떻게 소리 낼지를 **가타카나** 로 지정합니다.

히라가나·한자·로마자는 허용되지 않습니다.

::: tip 가타카나만 받습니다
내부 검증이 엄격해서 다른 문자가 섞이면 `ValidationError` 가 납니다.

히라가나밖에 모른다면 먼저 가타카나로 변환해서 넘겨주세요.
:::

### `accent` — 억양 (모르면 `0`)

일본어는 단어 안에서 소리가 한 번 "뚝" 떨어지는 지점이 있습니다.

그 지점을 숫자로 지정하는 값인데, **잘 모르겠으면 `0` 으로 두세요**.

대부분의 단어는 `0` 만으로도 충분히 자연스럽게 합성되고, 써 보고 억양이 어색하면 그때 `1`, `2`, ... 로 하나씩 올려가면서 가장 자연스러운 값을 찾으면 됩니다.

::: details 좀 더 정확히 알고 싶다면
일본어 악센트는 "어느 **모라** 다음에 피치가 한 번 떨어지는가" 로 표현됩니다.

- `accent=0` — 피치가 끝까지 떨어지지 않음 (평판형)
- `accent=1` — 첫 모라 다음에 떨어짐
- `accent=n` — n 번째 모라 다음에 떨어짐
- 최대값은 해당 단어의 모라 수까지

**모라** 는 일본어 발음의 기본 리듬 단위로, 글자 수와 꼭 같지는 않습니다.

작은 `ッ`·`ャュョ` 는 앞 모라에 붙어서 한 모라가 되고, `ン` 과 장음 `ー` 는 그 자체가 한 모라로 셉니다.

- `タンタンメン` — 6 모라 (タ/ン/タ/ン/メ/ン)
- `ハヤコエ` — 4 모라 (ハ/ヤ/コ/エ)
- `キャット` — 3 모라 (キャ/ッ/ト)
:::

## 여러 단어는 그냥 누적됩니다

```python
tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)
tts.add_word(surface="檸檬", reading="レモン", accent=0)
tts.add_word(surface="魚", reading="サカナ", accent=0)

speaker.generate("檸檬と担々麺と魚").save("mix.wav")
```

`add_word()` 는 기존 등록을 지우지 않고 추가합니다.

같은 `surface` 를 두 번 등록하지 않도록만 주의하면 됩니다.

## 프로세스가 끝나면 사라집니다

::: warning 영구 저장되지 않음
등록된 단어는 **프로세스 메모리에만** 유지됩니다.

파이썬 프로세스가 종료되면 전부 사라지고, 다음 실행에서는 다시 `add_word()` 를 호출해야 합니다.

매번 등록하는 게 번거로우면, 애플리케이션 시작 루틴 (서버라면 startup 훅) 에 한 곳에 모아 두는 걸 권장합니다.
:::

이건 의도된 동작입니다.

로컬 캐시에 단어가 쌓이면서 프로세스 간 상태가 꼬이거나 "예전에 뭘 등록했는지 기억 안 나는" 상황을 피하기 위해, HayaKoe 는 사용자 사전을 디스크에 절대 남기지 않도록 설계되어 있습니다.

## 동사·형용사는 활용형별로 따로 등록

등록되는 모든 단어는 내부적으로 **고유명사** 로 처리됩니다.

그래서 한 가지 형태로만 등장하는 이름·브랜드명·외래어는 이 API 만으로 거의 다 해결됩니다.

반대로 **동사나 형용사처럼 활용형이 바뀌는 단어** 는, `surface` 가 정확히 일치해야 교체가 일어나기 때문에 기본형 하나만 등록하면 나머지 형태는 잡히지 않습니다.

필요한 활용형을 각각 `add_word()` 로 한 번씩 더 등록해 주세요.

```python
# "ググる" (구글 검색하다) 같은 신조어 동사를 활용형별로 등록
tts.add_word(surface="ググる",     reading="ググル",     accent=0)  # 기본형
tts.add_word(surface="ググった",   reading="ググッタ",   accent=0)  # 과거
tts.add_word(surface="ググって",   reading="ググッテ",   accent=0)  # 연결형 (~하고)
tts.add_word(surface="ググります", reading="ググリマス", accent=0)  # 정중형
tts.add_word(surface="ググらない", reading="ググラナイ", accent=0)  # 부정형

# 형용사도 동일 — "エモい" 를 활용형별로
tts.add_word(surface="エモい",     reading="エモイ",     accent=0)
tts.add_word(surface="エモかった", reading="エモカッタ", accent=0)
tts.add_word(surface="エモく",     reading="エモク",     accent=0)
```

실제 문장에서 써 보다가 "어, 이 형태는 여전히 이상하게 읽힌다" 싶을 때마다 하나씩 추가해 나가는 식으로 쓰면 됩니다.
