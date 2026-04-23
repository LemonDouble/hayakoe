# Custom Word Registration

HayaKoe uses pyopenjtalk + the OpenJTalk dictionary for Japanese morphological analysis.

This dictionary covers most common vocabulary, but may be missing rare proper nouns, loanwords, neologisms, or brand names.

Such words may be awkwardly split or read character by character in sentences, but you can register them to fix the pronunciation.

::: info Approximately 220,000 English words are already built in
HayaKoe has an internal normalization dictionary that converts English words to katakana **before** passing text to pyopenjtalk.

This dictionary contains 221,587 English words with corresponding katakana, so common English proper nouns like `OpenAI` or `GitHub` are read naturally without calling `add_word()`.

`add_word()` affects a separate layer — the **pyopenjtalk Japanese morphological analysis stage** that comes after this normalization step.

In other words, think of it primarily as a tool for fixing mispronounced Japanese proper nouns, rare words, and neologisms.
:::

## Shortest Example

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)

speaker.generate("担々麺が食べたい。").save("tantanmen.wav")
```

You can call `add_word()` any time after `prepare()` finishes, and it takes effect from the next `generate()` call onward.

Listening to the same sentence before and after registration makes the difference immediately clear.

The `々` in `担々麺` is treated as an independent symbol in the OpenJTalk dictionary, so without any settings, synthesis splits it into two chunks: "tan / men".

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="Before — 担々麺 split as tan / men" src="/hayakoe/samples/custom-words/tantanmen_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="After — Reads as tantanmen in one breath" src="/hayakoe/samples/custom-words/tantanmen_after.wav" />

## English Brand Names and Neologisms Work the Same Way

As mentioned, 220,000 English words are automatically converted to katakana, but **brand names, product names, and neologisms not in the dictionary** will not be auto-converted.

In those cases, pyopenjtalk will read them character by character, so you need to fix them with `add_word()`.

For example, if the library name `HayaKoe` is read as-is, it comes out as something like "H-A-Y-A-K-O-E" with each letter spoken awkwardly.

```python
text = "HayaKoeは速い日本語の音声合成ライブラリです。"

# Before registration
speaker.generate(text).save("before.wav")

# Fix with desired katakana pronunciation
tts.add_word(surface="HayaKoe", reading="ハヤコエ", accent=0)

# After registration
speaker.generate(text).save("after.wav")
```

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="Before — HayaKoe read letter by letter" src="/hayakoe/samples/custom-words/hayakoe_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="After — Reads naturally as HayaKoe" src="/hayakoe/samples/custom-words/hayakoe_after.wav" />

Simply fix words not in the 220,000-word dictionary with `add_word()` as you encounter them.

## The Three Arguments

### `surface` — The Word as It Appears in the Sentence

Pass the **exact characters** as they appear in your input sentence.

Kanji, hiragana, katakana, or English are all fine, and mixing is acceptable too.

### `reading` — Katakana Pronunciation

Specify how it should be pronounced in **katakana**.

Hiragana, kanji, and romaji are not accepted.

::: tip Only katakana is accepted
Internal validation is strict — mixing in other character types will raise a `ValidationError`.

If you only know the hiragana, convert it to katakana first before passing it.
:::

### `accent` — Accent (Use `0` If Unsure)

Japanese has a point within a word where the pitch drops abruptly.

This value specifies that point as a number, but **if you are unsure, just set it to `0`**.

Most words synthesize naturally enough with `0` alone, and if the accent sounds off, you can try `1`, `2`, ... incrementally until you find the most natural value.

::: details For a more precise understanding
Japanese accent is expressed as "after which **mora** does the pitch drop once."

- `accent=0` — Pitch never drops until the end (flat pattern)
- `accent=1` — Drops after the first mora
- `accent=n` — Drops after the n-th mora
- Maximum value is the number of morae in the word

A **mora** is the basic rhythmic unit of Japanese pronunciation and does not always equal the number of characters.

Small `ッ` and `ャュョ` attach to the preceding mora to form one, while `ン` and the long vowel `ー` each count as one mora on their own.

- `タンタンメン` — 6 morae (ta/n/ta/n/me/n)
- `ハヤコエ` — 4 morae (ha/ya/ko/e)
- `キャット` — 3 morae (kya/tsu/to)
:::

## Multiple Words Simply Accumulate

```python
tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)
tts.add_word(surface="檸檬", reading="レモン", accent=0)
tts.add_word(surface="魚", reading="サカナ", accent=0)

speaker.generate("檸檬と担々麺と魚").save("mix.wav")
```

`add_word()` adds without clearing existing registrations.

Just be careful not to register the same `surface` twice.

## Registrations Are Lost When the Process Ends

::: warning Not persistently stored
Registered words are kept **only in process memory**.

When the Python process terminates, everything is lost and you must call `add_word()` again on the next run.

If re-registering every time is tedious, we recommend gathering all registrations in one place in your application startup routine (e.g., the server's startup hook).
:::

This is by design.

To avoid situations where words accumulate in a local cache, state gets tangled across processes, or you can't remember what was registered before, HayaKoe is designed to never save the user dictionary to disk.

## Verbs and Adjectives Need Separate Registration Per Conjugation

All registered words are internally treated as **proper nouns**.

This means the API alone handles most names, brand names, and loanwords that appear in only one form.

On the other hand, **verbs and adjectives that change form through conjugation** require an exact `surface` match for replacement, so registering only the base form will miss the other forms.

Register each needed conjugation form with a separate `add_word()` call.

```python
# Register the neologism verb "ググる" (to google) for each conjugation
tts.add_word(surface="ググる",     reading="ググル",     accent=0)  # base form
tts.add_word(surface="ググった",   reading="ググッタ",   accent=0)  # past
tts.add_word(surface="ググって",   reading="ググッテ",   accent=0)  # connective (~and)
tts.add_word(surface="ググります", reading="ググリマス", accent=0)  # polite form
tts.add_word(surface="ググらない", reading="ググラナイ", accent=0)  # negative form

# Same for adjectives — "エモい" by conjugation
tts.add_word(surface="エモい",     reading="エモイ",     accent=0)
tts.add_word(surface="エモかった", reading="エモカッタ", accent=0)
tts.add_word(surface="エモく",     reading="エモク",     accent=0)
```

Just add new forms whenever you encounter one that still reads incorrectly in actual sentences.
