from vocab.dictionary import Dictionary, EJDict, WordNetDict, default_senses, suggest_senses


def test_wordnet_groups_by_pos_sorted_by_freq_and_deduped(wn_path):
    wn = WordNetDict(wn_path)
    assert wn.lookup("network") == {"名詞": ["ネットワーク", "網", "放送網"], "動詞": ["人脈を作る"]}


def test_wordnet_pos_order(wn_path):
    assert list(WordNetDict(wn_path).lookup("fast")) == ["形容詞", "副詞"]


def test_wordnet_compound_uses_underscore(wn_path):
    assert WordNetDict(wn_path).lookup("neural network") == {"名詞": ["ニューラルネットワーク"]}
    assert WordNetDict(wn_path).has("neural network")


def test_ejdict_split_and_variant_headwords(ej_path):
    ej = EJDict(ej_path)
    assert ej.lookup("network") == ["網状組織", "(コンピューターの)ネットワーク"]
    assert ej.lookup("colour") == ["色"]
    assert ej.lookup("state-of-the-art") == ["最新式の"]


def test_dictionary_lookup_lemmatizes(wn_path, ej_path):
    r = Dictionary(WordNetDict(wn_path), EJDict(ej_path)).lookup("Networks,")
    assert r["input"] == "Networks," and r["lemma"] == "network" and r["found"] is True
    assert r["wordnet"]["名詞"][0] == "ネットワーク"
    assert r["ejdict"][0] == "網状組織"


def test_dictionary_not_found(wn_path, ej_path):
    r = Dictionary(WordNetDict(wn_path), EJDict(ej_path)).lookup("transformerz")
    assert r == {"input": "transformerz", "lemma": "transformerz", "found": False,
                 "wordnet": {}, "english_pos": [], "ejdict": []}


def test_dictionary_without_files(tmp_path):
    d = Dictionary.from_dir(tmp_path)
    assert d.available is False
    assert d.lookup("network")["found"] is False


def test_default_senses():
    assert default_senses({"名詞": ["a", "b", "c", "d"], "動詞": ["e"]}) == [
        {"pos": "名詞", "meanings": ["a", "b", "c"]}, {"pos": "動詞", "meanings": ["e"]}]


def test_english_pos_without_japanese(wn_path):
    wn = WordNetDict(wn_path)
    assert wn.lookup("recurrent") == {}
    assert wn.english_pos("recurrent") == ["形容詞"]
    assert wn.english_pos("go") == ["名詞", "動詞"]


def test_suggest_prefers_wordnet():
    r = {"wordnet": {"名詞": ["a", "b", "c", "d"]}, "english_pos": ["名詞"], "ejdict": ["x"]}
    assert suggest_senses(r) == [{"pos": "名詞", "meanings": ["a", "b", "c"]}]


def test_suggest_falls_back_to_ejdict_with_english_pos():
    r = {"wordnet": {}, "english_pos": ["形容詞"], "ejdict": ["再発する", "x", "y", "z"]}
    assert suggest_senses(r) == [{"pos": "形容詞", "meanings": ["再発する", "x", "y"]}]


def test_suggest_falls_back_to_other_when_pos_unknown():
    assert suggest_senses({"wordnet": {}, "english_pos": [], "ejdict": ["…経由で"]}) == [
        {"pos": "その他", "meanings": ["…経由で"]}]
    assert suggest_senses({"wordnet": {}, "english_pos": ["名詞", "動詞"], "ejdict": ["行く"]}) == [
        {"pos": "その他", "meanings": ["行く"]}]
    assert suggest_senses({"wordnet": {}, "english_pos": [], "ejdict": []}) == []


def _d(wn_path, ej_path):
    return Dictionary(WordNetDict(wn_path), EJDict(ej_path))


def test_inflection_entry_resolves_to_base(wn_path, ej_path):
    d = _d(wn_path, ej_path)
    assert d.lookup("went")["lemma"] == "go"
    assert d.lookup("ran")["lemma"] == "run"
    # 変化形以外の意味もある語はそのまま
    assert d.lookup("found")["lemma"] == "found"


def test_wordnet_lemma_without_japanese_is_not_preferred(wn_path, ej_path):
    assert _d(wn_path, ej_path).lookup("rates")["lemma"] == "rate"


def test_acronym_headword_has_lower_priority(wn_path, ej_path):
    d = _d(wn_path, ej_path)
    assert d.lookup("uses")["lemma"] == "use"
    r = d.lookup("LSTM")
    assert r["lemma"] == "lstm" and r["found"] is True and r["ejdict"] == ["長短期記憶"]
