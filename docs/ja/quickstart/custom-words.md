# カスタム単語の登録

HayaKoe は日本語の形態素解析に pyopenjtalk + OpenJTalk 辞書を使用しています。

この辞書には一般的な語彙はほぼすべて含まれていますが、まれな固有名詞・外来語・新語・商号などは漏れている場合があります。

そのような単語は文中で不自然に区切られたり一文字ずつ読まれてしまいますが、直接登録して望みの発音に固定できます。

::: info 英語の単語は既に約22万語が内蔵されています
HayaKoe は pyopenjtalk にテキストを渡す **前の** 段階で、英単語をカタカナに置換する内部正規化辞書を持っています。

この辞書には 221,587 個の英単語と対応カタカナが事前に含まれているため、`OpenAI`・`GitHub` のような一般的な英語固有名詞は別途 `add_word()` を呼ばなくても自然に読まれます。

`add_word()` はこの正規化段階の後、**pyopenjtalk の日本語形態素解析段階** に影響を与える別のレイヤーです。

つまり、主に誤読される日本語の固有名詞・希少語・新語を固定する用途だとお考えください。
:::

## 最も短い例

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)

speaker.generate("担々麺が食べたい。").save("tantanmen.wav")
```

`prepare()` の完了後いつでも `add_word()` を呼べ、以降の `generate()` からすぐに反映されます。

登録前/後で同じ文章がどう変わるか聴き比べると感覚がつかめます。

`担々麺` の `々` は OpenJTalk 辞書で独立記号として処理されるため、何も設定せずに合成すると「タン / メン」の2つに分かれてしまいます。

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="登録前 — 担々麺が タン / メン に分断" src="/hayakoe/samples/custom-words/tantanmen_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="登録後 — タンタンメン とひとつながりに" src="/hayakoe/samples/custom-words/tantanmen_after.wav" />

## 英語ブランド名・新語も同じ方法

先ほど22万語の英単語は自動でカタカナに変わると述べましたが、**辞書にないブランド名・製品名・新語** はその自動置換が効きません。

その場合は pyopenjtalk が文字単位で分割して読んでしまうため、やはり `add_word()` で修正する必要があります。

例としてこのライブラリ名 `HayaKoe` をそのまま読ませると `エイチ・エー・ワイ・エー・ケー・オー・イー` のようにアルファベットを一つずつ不自然に読み上げてしまいます。

```python
text = "HayaKoeは速い日本語の音声合成ライブラリです。"

# 登録前
speaker.generate(text).save("before.wav")

# 望みのカタカナ発音に固定
tts.add_word(surface="HayaKoe", reading="ハヤコエ", accent=0)

# 登録後
speaker.generate(text).save("after.wav")
```

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="登録前 — HayaKoe がアルファベット1文字ずつ読まれる" src="/hayakoe/samples/custom-words/hayakoe_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="登録後 — ハヤコエ と自然につながる" src="/hayakoe/samples/custom-words/hayakoe_after.wav" />

このように22万語辞書にない単語は見つけ次第 `add_word()` で対処すれば大丈夫です。

## 3つの引数

### `surface` — 文中で使う文字そのまま

実際の入力文に現れる **文字そのまま** を渡します。

漢字・ひらがな・カタカナ・英文どれでも構いませんし、混在していても問題ありません。

### `reading` — カタカナ発音

どう発声するかを **カタカナ** で指定します。

ひらがな・漢字・ローマ字は許可されません。

::: tip カタカナのみ受け付けます
内部バリデーションが厳格で、他の文字が混ざると `ValidationError` が発生します。

ひらがなしかわからない場合は、先にカタカナに変換してから渡してください。
:::

### `accent` — アクセント（わからなければ `0`）

日本語は単語の中で音が一度「ストン」と落ちるポイントがあります。

そのポイントを数値で指定する値ですが、**よくわからなければ `0` にしてください**。

ほとんどの単語は `0` だけでも十分自然に合成され、使ってみてアクセントが不自然なら、そのとき `1`、`2`、... と一つずつ上げていき最も自然な値を見つけてください。

::: details もう少し正確に知りたい場合
日本語のアクセントは「どの **モーラ** の後にピッチが一度落ちるか」で表現されます。

- `accent=0` — ピッチが最後まで落ちない（平板型）
- `accent=1` — 1モーラ目の後に落ちる
- `accent=n` — n モーラ目の後に落ちる
- 最大値はその単語のモーラ数まで

**モーラ** は日本語の発音の基本リズム単位で、文字数と必ず一致するわけではありません。

小さい `ッ`・`ャュョ` は前のモーラに付いて1モーラになり、`ン` と長音 `ー` はそれ自体が1モーラとして数えられます。

- `タンタンメン` — 6モーラ（タ/ン/タ/ン/メ/ン）
- `ハヤコエ` — 4モーラ（ハ/ヤ/コ/エ）
- `キャット` — 3モーラ（キャ/ッ/ト）
:::

## 複数の単語はそのまま蓄積されます

```python
tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)
tts.add_word(surface="檸檬", reading="レモン", accent=0)
tts.add_word(surface="魚", reading="サカナ", accent=0)

speaker.generate("檸檬と担々麺と魚").save("mix.wav")
```

`add_word()` は既存の登録を消さずに追加します。

同じ `surface` を2回登録しないようにだけ注意してください。

## プロセス終了で消えます

::: warning 永続保存されません
登録された単語は **プロセスメモリにのみ** 保持されます。

Python プロセスが終了するとすべて消え、次回実行時には再度 `add_word()` を呼ぶ必要があります。

毎回登録するのが面倒なら、アプリケーション起動ルーチン（サーバーなら startup フック）に一箇所にまとめておくことを推奨します。
:::

これは意図的な動作です。

ローカルキャッシュに単語が溜まってプロセス間で状態が絡まったり「以前何を登録したか覚えていない」状況を避けるため、HayaKoe はユーザー辞書をディスクに一切残さない設計になっています。

## 動詞・形容詞は活用形ごとに別途登録

登録されるすべての単語は内部的に **固有名詞** として扱われます。

そのため1つの形態でしか出現しない名前・ブランド名・外来語はこの API だけでほぼすべて解決できます。

一方、**動詞や形容詞のように活用形が変わる単語** は、`surface` が正確に一致しないと置換が行われないため、基本形1つだけ登録しても他の形態は対処されません。

必要な活用形をそれぞれ `add_word()` で1回ずつ追加登録してください。

```python
# "ググる"（Google検索する）のような新語動詞を活用形ごとに登録
tts.add_word(surface="ググる",     reading="ググル",     accent=0)  # 基本形
tts.add_word(surface="ググった",   reading="ググッタ",   accent=0)  # 過去
tts.add_word(surface="ググって",   reading="ググッテ",   accent=0)  # 連用形（〜して）
tts.add_word(surface="ググります", reading="ググリマス", accent=0)  # 丁寧形
tts.add_word(surface="ググらない", reading="ググラナイ", accent=0)  # 否定形

# 形容詞も同様 — "エモい" を活用形ごとに
tts.add_word(surface="エモい",     reading="エモイ",     accent=0)
tts.add_word(surface="エモかった", reading="エモカッタ", accent=0)
tts.add_word(surface="エモく",     reading="エモク",     accent=0)
```

実際の文中で使ってみて「あ、この形態はまだおかしく読まれる」と気づいたら1つずつ追加していく、という使い方で十分です。
