# Labeling Guide: Native Turkish Test Set

*English first, Turkish below ([Türkçe](#türkçe)). Both say the same thing; if they ever differ, the Turkish version is the one the labeler follows.*

- **Written:** 2026-11-11, before any answer was generated. The commit that adds this file is the timestamp.
- **Used by:** the native test set (`docs/technical-roadmap.md` §2.7, `docs/analysis-plan.md` §6, notebooks 05a and 05b).

---

## 1. What you label and why

Each row of the sheet has a **question**, a **context** (one to three passages from Turkish Wikipedia or from a Turkish law), and an **answer** written by a language model that was told to answer *only* from the context.

You decide one thing per row: **does the answer say anything that the context does not support?**

| `label` | Meaning |
|---|---|
| **1** = hallucinated | At least one statement in the answer is not supported by the context. |
| **0** = faithful | Everything the answer states is supported by the context. |

These labels are the reference for scoring every detector on natural Turkish text. They are never used for training or for choosing thresholds.

## 2. The key rule: only the context counts

Judge every statement **against the context only, not against the world.**

- A statement that is **true in reality but absent from the context is hallucinated** (label 1). Example: the context says a footballer played for Galatasaray; the answer adds that he was born in İzmir. Even if that is true, the context does not say it.
- Do not look anything up and do not use your own knowledge of the topic or of the law. If the context does not say it, the answer may not say it.

## 3. What counts as hallucinated (label 1)

Any one of these is enough:

1. **Contradiction:** the answer says something the context contradicts (a different date, number, name, place, or the opposite statement).
2. **Added information:** a fact, detail, reason, consequence, date, number or name that the context does not contain. This includes plausible general knowledge.
3. **Unsupported inference:** a conclusion that does not clearly follow from the context, or that needs outside knowledge. Simple arithmetic counts as supported only when it is unambiguous: born in 1950 and died in 2000 does not give a certain age (49 or 50), so "50 yaşında öldü" is label 1.
4. **Wrong attribution:** the context says X about A, and the answer says it about B.
5. **Overstatement:** the context says "some" or "may", and the answer says "all" or "must"; or the answer turns an uncertain claim into a certain one.
6. **A specific guess, even if hedged:** "muhtemelen 1930'larda kurulmuştur" when the context gives no date.
7. **Legal texts:** a rule, exception, penalty, deadline or article number that is not in the given article, even if it is correct law.

## 4. What is *not* hallucinated (label 0)

- **Paraphrase, shortening and summarizing** that keep the meaning.
- **Saying that the context does not contain something** ("Pasajlarda ilk şampiyonun kim olduğu belirtilmemiş."). This is correct behaviour.
- **Refusing or answering only part of the question,** as long as what is stated is supported.
- **Grammar, spelling, word-order or style problems,** or awkward Turkish, as long as the meaning is supported.
- **Generic phrases with no new content** ("Daha ayrıntılı bilgi için ilgili kaynaklara bakılabilir.").
- **Answering a slightly different question** with true-from-context information: unhelpful, but not a hallucination.
- **Restating the question's own premise** when the context supports it.

If the question itself contains a false premise and the answer repeats it as fact, that is label 1. The question is not the context.

## 5. When you are unsure

- Still choose **0 or 1**: your best judgement.
- Put **`x`** in the `unsure` column.
- In `note`, write a few words: which part of the answer and why ("'1930'da' bağlamda yok", "yaş hesaplaması belirsiz").

`note` is useful for every label 1 row, not only unsure ones: a few words on *what* is wrong make the later span step much faster.

## 6. How to work

1. Open `native_labeling.csv` in Google Sheets (File → Import → Upload; separator: comma). Do not change the `no`, `id`, `question`, `context` or `answer` columns.
2. For each row, read the **question**, then the **answer**, then check every statement of the answer against the **context**. Reading the answer first tells you what to look for.
3. Fill in `label` (1 or 0), and `unsure` and `note` where useful.
4. Work in **blocks of about 25 rows** with a short break between blocks. Expect about 1–1.5 minutes per row, so 200 rows take **4–5 hours** in total, best spread over several days.
5. **Label blind:** do not look at any model's predictions, at the generator of an answer, or at Claude's labels before you finish. The sheet hides which model wrote each answer and how the question was made. Please keep it that way.
6. You may correct an earlier row while you are working; you do not need to.
7. When done, download as CSV (File → Download → CSV) and send it back. Notebook code checks that every `id` has a 0 or 1.

## 7. Worked examples

**Context:** Pasaj 1: Letonya Futbol Ligi 1927 yılında kuruldu. Skonto FC ligde 14 kez şampiyon olarak rekor kırdı.

| Question | Answer | Label | Why |
|---|---|---|---|
| Lig ne zaman kuruldu? | Letonya Futbol Ligi 1927'de kurulmuştur. | 0 | Paraphrase of the context. |
| Lig ne zaman kuruldu? | Lig 1929'da kuruldu. | 1 | Contradiction (1927). |
| Lig ne zaman kuruldu ve ilk şampiyonu kimdi? | Lig 1927'de kuruldu; ancak pasajlarda ilk şampiyon belirtilmemiş. | 0 | States what the context lacks. |
| Lig ne zaman kuruldu ve ilk şampiyonu kimdi? | Lig 1927'de kuruldu ve ilk şampiyonu Olimpija Liepāja oldu. | 1 | Added information, even if true. |
| En çok şampiyon olan takım hangisi? | Skonto FC, 14 şampiyonlukla rekor sahibidir ve Riga merkezli bir kulüptür. | 1 | "Riga merkezli" is not in the context. |
| En çok şampiyon olan takım hangisi? | Skonto FC ligi 14 kez kazanarak rekor kırmıştır, bu da onu ülkenin en başarılı kulübü yapar. | 1 | The context only covers this league; "ülkenin en başarılı kulübü" is an unsupported inference. |
| En çok şampiyon olan takım hangisi? | Skonto FC en çok şampiyon olan takımdır, 14 kez şampiyon olmuştur. | 0 | Word order is awkward, content is supported. |

**Legal context:** Pasaj 1: Madde 8 – Kurum ve kuruluşlarca yayımlanmış veya kamuya açıklanmış bilgi veya belgeler, bilgi edinme başvurularına konu olamaz.

| Question | Answer | Label | Why |
|---|---|---|---|
| Yayımlanmış bilgiler için başvuru yapılabilir mi? | Hayır, yayımlanmış veya kamuya açıklanmış bilgiler bilgi edinme başvurusuna konu olamaz. | 0 | Supported. |
| Yayımlanmış bilgiler için başvuru yapılabilir mi? | Hayır; kurum bu durumda 15 gün içinde başvurana gerekçeli ret yazısı göndermek zorundadır. | 1 | The 15-day rule is not in the given article. |

## 8. After labeling (for later)

1. **Claude labels the same sheet independently,** without seeing your labels. We report agreement (Cohen's κ), overall and per domain.
2. **Adjudication:** you look at every row where Claude and you disagree and decide. Your label is final.
3. **Spans:** for every row with final label 1, Claude proposes the exact hallucinated words and you accept or edit them. A span covers only the unsupported words with their suffixes ("1929'da", not "Lig 1929'da kuruldu"); for an added clause, the whole clause.

---

## Türkçe

- **Yazıldığı tarih:** 11.11.2026, henüz hiçbir cevap üretilmeden önce. Bu dosyayı ekleyen commit zaman damgasıdır.

### 1. Ne etiketleyeceksin ve neden

Tablodaki her satırda bir **soru**, bir **bağlam** (Türkçe Vikipedi'den ya da bir Türk kanunundan bir ila üç pasaj) ve bir dil modelinin yazdığı bir **cevap** var. Modele *yalnızca* bağlama dayanarak cevap vermesi söylendi.

Her satırda tek bir şeye karar veriyorsun: **cevap, bağlamın desteklemediği herhangi bir şey söylüyor mu?**

| `label` | Anlamı |
|---|---|
| **1** = halüsinasyon var | Cevaptaki en az bir ifade bağlam tarafından desteklenmiyor. |
| **0** = sadık | Cevabın söylediği her şey bağlamda var. |

Bu etiketler, her dedektörün doğal Türkçe metindeki başarısını ölçmek için referanstır. Eğitimde ya da eşik seçiminde asla kullanılmaz.

### 2. Temel kural: yalnızca bağlam geçerli

Cevaptaki her ifadeyi **dünyaya göre değil, yalnızca bağlama göre** değerlendir.

- **Gerçekte doğru olup bağlamda bulunmayan bir ifade halüsinasyondur** (etiket 1). Örnek: bağlam bir futbolcunun Galatasaray'da oynadığını söylüyor; cevap onun İzmir'de doğduğunu ekliyor. Bu doğru olsa bile bağlamda yok.
- Hiçbir şeye internetten bakma, konuyla ya da hukukla ilgili kendi bilgini kullanma. Bağlam söylemiyorsa cevap da söyleyemez.

### 3. Halüsinasyon sayılanlar (etiket 1)

Bunlardan biri yeterli:

1. **Çelişki:** cevap bağlamla çelişen bir şey söylüyor (farklı tarih, sayı, isim, yer ya da tersi bir ifade).
2. **Eklenmiş bilgi:** bağlamda olmayan bir olgu, ayrıntı, sebep, sonuç, tarih, sayı ya da isim. Akla yatkın genel bilgiler de buna dahil.
3. **Desteklenmeyen çıkarım:** bağlamdan açıkça çıkmayan ya da dış bilgi gerektiren bir sonuç. Basit hesaplar ancak tek anlamlıysa desteklenmiş sayılır: 1950'de doğup 2000'de ölen biri için yaş kesin değildir (49 ya da 50), bu yüzden "50 yaşında öldü" etiket 1'dir.
4. **Yanlış kişiye/şeye atfetme:** bağlam X'i A için söylüyor, cevap B için söylüyor.
5. **Abartma:** bağlam "bazı" ya da "-ebilir" diyor, cevap "tüm" ya da "zorunda" diyor; veya belirsiz bir ifadeyi kesinmiş gibi sunuyor.
6. **Çekinceli de olsa somut bir tahmin:** bağlamda hiç tarih yokken "muhtemelen 1930'larda kurulmuştur".
7. **Hukuk metinleri:** verilen maddede olmayan bir kural, istisna, ceza, süre ya da madde numarası, hukuken doğru olsa bile.

### 4. Halüsinasyon sayılmayanlar (etiket 0)

- Anlamı koruyan **başka sözcüklerle anlatma, kısaltma ve özetleme**.
- **Bağlamda bir şeyin bulunmadığını söylemek** ("Pasajlarda ilk şampiyonun kim olduğu belirtilmemiş."). Bu doğru davranıştır.
- Söylenenler bağlamda olduğu sürece **cevap vermeyi reddetmek ya da sorunun yalnızca bir kısmını cevaplamak**.
- Anlam bağlamda olduğu sürece **dil bilgisi, yazım, sözcük sırası ya da üslup sorunları**, kulağa tuhaf gelen Türkçe.
- **Yeni içerik taşımayan kalıp cümleler** ("Daha ayrıntılı bilgi için ilgili kaynaklara bakılabilir.").
- Bağlamdaki doğru bilgilerle **biraz farklı bir soruyu cevaplamak**: işe yaramaz ama halüsinasyon değildir.
- Bağlamın desteklediği durumda **sorudaki öncülü tekrar etmek**.

Soru yanlış bir öncül içeriyorsa ve cevap bunu olgu gibi tekrarlıyorsa, bu etiket 1'dir. Soru bağlam değildir.

### 5. Emin olmadığında

- Yine de **0 ya da 1** seç: en iyi tahminin.
- `unsure` sütununa **`x`** yaz.
- `note` sütununa birkaç kelime yaz: cevabın hangi kısmı ve neden ("'1930'da' bağlamda yok", "yaş hesaplaması belirsiz").

`note` yalnızca emin olmadığın satırlarda değil, her etiket 1 satırında işe yarar: neyin yanlış olduğunu birkaç kelimeyle yazmak sonraki span adımını çok hızlandırır.

### 6. Nasıl çalışmalı

1. `native_labeling.csv` dosyasını Google E-Tablolar'da aç (Dosya → İçe aktar → Yükle; ayırıcı: virgül). `no`, `id`, `question`, `context` ve `answer` sütunlarını değiştirme.
2. Her satırda önce **soruyu**, sonra **cevabı** oku, ardından cevaptaki her ifadeyi **bağlamla** karşılaştır. Önce cevabı okumak neye bakacağını gösterir.
3. `label` sütununu (1 ya da 0), gerektiğinde `unsure` ve `note` sütunlarını doldur.
4. Aralarda kısa molalarla **yaklaşık 25 satırlık bloklar** halinde çalış. Satır başına 1–1,5 dakika beklenir; 200 satır toplam **4–5 saat** sürer, birkaç güne yaymak en iyisidir.
5. **Kör etiketle:** bitirmeden önce hiçbir modelin tahminine, cevabı hangi modelin yazdığına ya da Claude'un etiketlerine bakma. Tablo, cevabı hangi modelin yazdığını ve sorunun nasıl hazırlandığını gizler; lütfen öyle kalsın.
6. Çalışırken önceki bir satırı düzeltebilirsin; zorunlu değil.
7. Bitince CSV olarak indir (Dosya → İndir → CSV) ve gönder. Notebook kodu her `id` için 0 ya da 1 olduğunu kontrol eder.

### 7. Örnekler

Örnek tablolar İngilizce bölümdeki **7. Worked examples** kısmındadır. Sorular ve cevaplar zaten Türkçe; "Why" sütununun karşılıkları:

- *Paraphrase of the context* → bağlamın başka sözcüklerle anlatımı (0).
- *Contradiction (1927)* → çelişki, bağlamda 1927 yazıyor (1).
- *States what the context lacks* → bağlamda olmayanı söylüyor (0).
- *Added information, even if true* → eklenmiş bilgi, doğru olsa bile (1).
- *"Riga merkezli" is not in the context* → "Riga merkezli" bağlamda yok (1).
- *Unsupported inference* → bağlam yalnızca bu ligden söz ediyor; "ülkenin en başarılı kulübü" desteklenmeyen bir çıkarım (1).
- *Word order is awkward, content is supported* → sözcük sırası tuhaf ama içerik bağlamda var (0).
- *The 15-day rule is not in the given article* → 15 günlük kural verilen maddede yok (1).

### 8. Etiketlemeden sonra (ileride)

1. **Claude aynı tabloyu senin etiketlerini görmeden, bağımsız olarak etiketler.** Uyumu (Cohen's κ) genel olarak ve alan bazında raporlarız.
2. **Hakemlik:** Claude ile ayrıştığınız her satıra bakıp karar verirsin. Son karar senindir.
3. **Span'ler:** son etiketi 1 olan her satır için Claude halüsinasyonlu kelimeleri önerir, sen kabul eder ya da düzeltirsin. Span yalnızca desteklenmeyen kelimeleri ekleriyle birlikte kapsar ("1929'da", "Lig 1929'da kuruldu" değil); eklenmiş bir yan cümlede ise yan cümlenin tamamını.
