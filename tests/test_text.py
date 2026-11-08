from hdtr.text import digits, ends_like_sentence, words


def test_apostrophe_suffix_stays_with_stem():
    assert words("Başvurular 25 Ocak'a kadar sürer.") == ["Başvurular", "25", "Ocak'a", "kadar", "sürer"]


def test_curly_apostrophe_and_numbers():
    assert words("Ücret 15.000 TL’dir, oran 3,5'tir.") == ["Ücret", "15.000", "TL’dir", "oran", "3,5'tir"]


def test_turkish_capitals_are_word_characters():
    assert words("İSTANBUL ve IŞIK") == ["İSTANBUL", "ve", "IŞIK"]


def test_digits_ignore_separators():
    assert digits("1.000 ve 1,000") == "10001000"


def test_sentence_end_detection():
    assert ends_like_sentence("Bu bir cümledir.")
    assert not ends_like_sentence("Bu cümle yarıda kes")
