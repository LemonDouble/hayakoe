# 注册自定义词语

HayaKoe 使用 pyopenjtalk + OpenJTalk 词典进行日语形态素分析。

该词典包含了大多数常用词汇,但稀有专有名词、外来语、新造词、商标名等可能缺失。

这类词在句子中可能会被奇怪地断开或逐字读出,您可以手动注册以固定为想要的发音。

::: info 已内置约 22 万个英语单词
HayaKoe 在将文本传给 pyopenjtalk **之前**,有一个内部规范化字典将英语单词替换为片假名。

该字典预置了 221,587 个英语单词及其对应片假名,因此 `OpenAI`·`GitHub` 等常见英语专有名词无需调用 `add_word()` 就能自然地朗读。

`add_word()` 影响的是规范化步骤之后的 **pyopenjtalk 日语形态素分析阶段**,是一个独立的层。

也就是说,主要用于修正读错的日语专有名词、稀有词和新造词。
:::

## 最简示例

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)

speaker.generate("担々麺が食べたい。").save("tantanmen.wav")
```

在 `prepare()` 完成后随时可以调用 `add_word()`,之后的 `generate()` 会立即生效。

对比注册前后同一句话的差异,可以快速找到感觉。

`担々麺` 中的 `々` 在 OpenJTalk 词典中被作为独立符号处理,因此不做任何设置直接合成时会被拆成"タン / メン"两段。

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="注册前 — 担々麺 被断成 タン / メン" src="/hayakoe/samples/custom-words/tantanmen_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="注册后 — タンタンメン 连贯读出" src="/hayakoe/samples/custom-words/tantanmen_after.wav" />

## 英语品牌名·新造词也是同样方式

前面提到 22 万个英语单词会自动转换为片假名,但 **字典中没有的品牌名·产品名·新造词** 无法触发自动转换。

这种情况下 pyopenjtalk 会逐字符拆分朗读,同样需要用 `add_word()` 修正。

例如本库名称 `HayaKoe` 如果直接朗读,会变成 `エイチ・エー・ワイ・エー・ケー・オー・イー` 这样逐个字母生硬地读出。

```python
text = "HayaKoeは速い日本語の音声合成ライブラリです。"

# 注册前
speaker.generate(text).save("before.wav")

# 固定为想要的片假名发音
tts.add_word(surface="HayaKoe", reading="ハヤコエ", accent=0)

# 注册后
speaker.generate(text).save("after.wav")
```

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="注册前 — HayaKoe 逐字母读出" src="/hayakoe/samples/custom-words/hayakoe_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="注册后 — ハヤコエ 连贯自然" src="/hayakoe/samples/custom-words/hayakoe_after.wav" />

像这样,22 万词典中没有的单词看到一个就用 `add_word()` 修正一个即可。

## 三个参数

### `surface` — 在句子中原样出现的文字

传入实际输入句子中 **原样出现的文字** 即可。

汉字·平假名·片假名·英文均可,混合也可以。

### `reading` — 片假名发音

用 **片假名** 指定如何发音。

不允许使用平假名·汉字·罗马字。

::: tip 仅接受片假名
内部验证比较严格,混入其他文字会抛出 `ValidationError`。

如果只知道平假名,请先转换为片假名再传入。
:::

### `accent` — 声调(不确定时填 `0`)

日语中单词内有一个音高"突降"的位置。

这个值用数字指定该位置,**不确定的话填 `0`**。

大多数单词仅用 `0` 就能合成得足够自然,如果听起来声调不对,再逐一尝试 `1`、`2`、... 找到最自然的值即可。

::: details 想更准确了解的话
日语声调用"第几个 **莫拉** 之后音高下降一次"来表示。

- `accent=0` — 音高直到末尾都不下降(平板型)
- `accent=1` — 第一个莫拉之后下降
- `accent=n` — 第 n 个莫拉之后下降
- 最大值为该词的莫拉数

**莫拉** 是日语发音的基本节奏单位,与字符数不一定相同。

小写 `ッ`·`ャュョ` 附着在前一个莫拉上算作一个莫拉,`ン` 和长音 `ー` 本身各算一个莫拉。

- `タンタンメン` — 6 莫拉 (タ/ン/タ/ン/メ/ン)
- `ハヤコエ` — 4 莫拉 (ハ/ヤ/コ/エ)
- `キャット` — 3 莫拉 (キャ/ッ/ト)
:::

## 多个词语直接累加

```python
tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)
tts.add_word(surface="檸檬", reading="レモン", accent=0)
tts.add_word(surface="魚", reading="サカナ", accent=0)

speaker.generate("檸檬と担々麺と魚").save("mix.wav")
```

`add_word()` 不会清除已有注册,而是追加。

只需注意不要重复注册相同的 `surface`。

## 进程结束后会消失

::: warning 不会永久保存
注册的词语 **仅保留在进程内存中**。

Python 进程终止后全部消失,下次运行时需要重新调用 `add_word()`。

如果觉得每次都注册很麻烦,建议集中放在应用启动例程(服务器的话放在 startup 钩子)中。
:::

这是有意为之的行为。

为了避免词语在本地缓存中不断累积、导致进程间状态混乱或"记不清之前注册了什么"的情况,HayaKoe 设计为绝不将用户词典写入磁盘。

## 动词·形容词需要按活用形分别注册

注册的所有词语在内部都作为 **固有名词** 处理。

因此只以一种形式出现的名字·品牌名·外来语用这个 API 基本都能解决。

反之,**像动词和形容词这样活用形会变化的词**,由于需要 `surface` 完全匹配才会触发替换,只注册基本形的话其他形式不会被识别。

请将需要的活用形各自调用 `add_word()` 注册一次。

```python
# "ググる"(用 Google 搜索)这样的新造动词按活用形注册
tts.add_word(surface="ググる",     reading="ググル",     accent=0)  # 基本形
tts.add_word(surface="ググった",   reading="ググッタ",   accent=0)  # 过去时
tts.add_word(surface="ググって",   reading="ググッテ",   accent=0)  # 连接形
tts.add_word(surface="ググります", reading="ググリマス", accent=0)  # 敬体形
tts.add_word(surface="ググらない", reading="ググラナイ", accent=0)  # 否定形

# 形容词同理 — "エモい" 按活用形注册
tts.add_word(surface="エモい",     reading="エモイ",     accent=0)
tts.add_word(surface="エモかった", reading="エモカッタ", accent=0)
tts.add_word(surface="エモく",     reading="エモク",     accent=0)
```

在实际句子中使用时,每当遇到"这个形式还是读得不对"的情况,逐个添加即可。
