# 註冊自訂詞彙

HayaKoe 使用 pyopenjtalk + OpenJTalk 字典進行日語形態素分析。

該字典包含了大多數常用詞彙,但稀有專有名詞、外來語、新造詞、商標名等可能缺失。

這類詞在句子中可能會被奇怪地斷開或逐字讀出,您可以手動註冊以固定為想要的發音。

::: info 已內建約 22 萬個英語單字
HayaKoe 在將文本傳給 pyopenjtalk **之前**,有一個內部正規化字典將英語單字替換為片假名。

該字典預置了 221,587 個英語單字及其對應片假名,因此 `OpenAI`・`GitHub` 等常見英語專有名詞無需呼叫 `add_word()` 就能自然地朗讀。

`add_word()` 影響的是正規化步驟之後的 **pyopenjtalk 日語形態素分析階段**,是一個獨立的層。

也就是說,主要用於修正讀錯的日語專有名詞、稀有詞和新造詞。
:::

## 最簡範例

```python
from hayakoe import TTS

tts = TTS().load("jvnv-F1-jp").prepare()
speaker = tts.speakers["jvnv-F1-jp"]

tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)

speaker.generate("担々麺が食べたい。").save("tantanmen.wav")
```

在 `prepare()` 完成後隨時可以呼叫 `add_word()`,之後的 `generate()` 會立即生效。

對比註冊前後同一句話的差異,可以快速找到感覺。

`担々麺` 中的 `々` 在 OpenJTalk 字典中被作為獨立符號處理,因此不做任何設定直接合成時會被拆成「タン / メン」兩段。

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="註冊前 — 担々麺 被斷成 タン / メン" src="/hayakoe/samples/custom-words/tantanmen_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="註冊後 — タンタンメン 連貫讀出" src="/hayakoe/samples/custom-words/tantanmen_after.wav" />

## 英語品牌名・新造詞也是同樣方式

前面提到 22 萬個英語單字會自動轉換為片假名,但 **字典中沒有的品牌名・產品名・新造詞** 無法觸發自動轉換。

這種情況下 pyopenjtalk 會逐字元拆分朗讀,同樣需要用 `add_word()` 修正。

例如本函式庫名稱 `HayaKoe` 如果直接朗讀,會變成 `エイチ・エー・ワイ・エー・ケー・オー・イー` 這樣逐個字母生硬地讀出。

```python
text = "HayaKoeは速い日本語の音声合成ライブラリです。"

# 註冊前
speaker.generate(text).save("before.wav")

# 固定為想要的片假名發音
tts.add_word(surface="HayaKoe", reading="ハヤコエ", accent=0)

# 註冊後
speaker.generate(text).save("after.wav")
```

<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="註冊前 — HayaKoe 逐字母讀出" src="/hayakoe/samples/custom-words/hayakoe_before.wav" />
<SpeakerSample badge="つくよみちゃん" badgeIcon="/hayakoe/images/speakers/tsukuyomi.png" name="註冊後 — ハヤコエ 連貫自然" src="/hayakoe/samples/custom-words/hayakoe_after.wav" />

像這樣,22 萬字典中沒有的單字看到一個就用 `add_word()` 修正一個即可。

## 三個參數

### `surface` — 在句子中原樣出現的文字

傳入實際輸入句子中 **原樣出現的文字** 即可。

漢字・平假名・片假名・英文均可,混合也可以。

### `reading` — 片假名發音

用 **片假名** 指定如何發音。

不允許使用平假名・漢字・羅馬字。

::: tip 僅接受片假名
內部驗證比較嚴格,混入其他文字會拋出 `ValidationError`。

如果只知道平假名,請先轉換為片假名再傳入。
:::

### `accent` — 聲調(不確定時填 `0`)

日語中單字內有一個音高「突降」的位置。

這個值用數字指定該位置,**不確定的話填 `0`**。

大多數單字僅用 `0` 就能合成得足夠自然,如果聽起來聲調不對,再逐一嘗試 `1`、`2`、... 找到最自然的值即可。

::: details 想更準確了解的話
日語聲調用「第幾個 **莫拉** 之後音高下降一次」來表示。

- `accent=0` — 音高直到末尾都不下降(平板型)
- `accent=1` — 第一個莫拉之後下降
- `accent=n` — 第 n 個莫拉之後下降
- 最大值為該詞的莫拉數

**莫拉** 是日語發音的基本節奏單位,與字元數不一定相同。

小寫 `ッ`・`ャュョ` 附著在前一個莫拉上算作一個莫拉,`ン` 和長音 `ー` 本身各算一個莫拉。

- `タンタンメン` — 6 莫拉 (タ/ン/タ/ン/メ/ン)
- `ハヤコエ` — 4 莫拉 (ハ/ヤ/コ/エ)
- `キャット` — 3 莫拉 (キャ/ッ/ト)
:::

## 多個詞彙直接累加

```python
tts.add_word(surface="担々麺", reading="タンタンメン", accent=0)
tts.add_word(surface="檸檬", reading="レモン", accent=0)
tts.add_word(surface="魚", reading="サカナ", accent=0)

speaker.generate("檸檬と担々麺と魚").save("mix.wav")
```

`add_word()` 不會清除已有註冊,而是追加。

只需注意不要重複註冊相同的 `surface`。

## 行程結束後會消失

::: warning 不會永久儲存
註冊的詞彙 **僅保留在行程記憶體中**。

Python 行程終止後全部消失,下次執行時需要重新呼叫 `add_word()`。

如果覺得每次都註冊很麻煩,建議集中放在應用程式啟動例程(伺服器的話放在 startup 鉤子)中。
:::

這是有意為之的行為。

為了避免詞彙在本地快取中不斷累積、導致行程間狀態混亂或「記不清之前註冊了什麼」的情況,HayaKoe 設計為絕不將使用者字典寫入硬碟。

## 動詞・形容詞需要按活用形分別註冊

註冊的所有詞彙在內部都作為 **固有名詞** 處理。

因此只以一種形式出現的名字・品牌名・外來語用這個 API 基本都能解決。

反之,**像動詞和形容詞這樣活用形會變化的詞**,由於需要 `surface` 完全匹配才會觸發替換,只註冊基本形的話其他形式不會被識別。

請將需要的活用形各自呼叫 `add_word()` 註冊一次。

```python
# "ググる"(用 Google 搜索)這樣的新造動詞按活用形註冊
tts.add_word(surface="ググる",     reading="ググル",     accent=0)  # 基本形
tts.add_word(surface="ググった",   reading="ググッタ",   accent=0)  # 過去時
tts.add_word(surface="ググって",   reading="ググッテ",   accent=0)  # 連接形
tts.add_word(surface="ググります", reading="ググリマス", accent=0)  # 敬體形
tts.add_word(surface="ググらない", reading="ググラナイ", accent=0)  # 否定形

# 形容詞同理 — "エモい" 按活用形註冊
tts.add_word(surface="エモい",     reading="エモイ",     accent=0)
tts.add_word(surface="エモかった", reading="エモカッタ", accent=0)
tts.add_word(surface="エモく",     reading="エモク",     accent=0)
```

在實際句子中使用時,每當遇到「這個形式還是讀得不對」的情況,逐個新增即可。
