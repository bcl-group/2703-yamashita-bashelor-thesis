from vocab.lemmatize import find_lemma, normalize

WORDS = {"network", "study", "process", "use", "compute", "stop", "run",
         "analysis", "bus", "datum", "data", "neural network", "state-of-the-art"}
exists = WORDS.__contains__


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Networks, ") == "networks"
    assert normalize("“model”") == "model"
    assert normalize("State-of-the-art.") == "state-of-the-art"
    assert normalize("Neural  Network") == "neural network"
    assert normalize("123 !!") == ""


def test_rules():
    assert find_lemma("networks", exists) == ("network", True)
    assert find_lemma("studies", exists) == ("study", True)
    assert find_lemma("processes", exists) == ("process", True)
    assert find_lemma("used", exists) == ("use", True)
    assert find_lemma("computing", exists) == ("compute", True)
    assert find_lemma("stopped", exists) == ("stop", True)


def test_irregular():
    assert find_lemma("ran", exists) == ("run", True)


def test_existing_word_is_not_changed():
    assert find_lemma("analysis", exists) == ("analysis", True)
    assert find_lemma("bus", exists) == ("bus", True)
    assert find_lemma("data", exists) == ("data", True)


def test_not_found_returns_input():
    assert find_lemma("Transformerz", exists) == ("transformerz", False)


REAL_LIKE = {"not", "note", "nod", "node", "cod", "code", "plan", "plane", "li", "lie",
             "hop", "hope", "car", "care", "rat", "rate", "tun", "tune", "us", "use",
             "process", "study", "stop", "compute", "model", "plan"}


def test_rules_prefer_real_base_over_short_stem():
    ex = REAL_LIKE.__contains__
    cases = {"notes": "note", "nodes": "node", "codes": "code", "planes": "plane", "lies": "lie",
             "hoped": "hope", "cared": "care", "rated": "rate", "tuned": "tune", "used": "use",
             "processes": "process", "studies": "study", "stopped": "stop", "computing": "compute",
             "planned": "plan", "modeling": "model"}
    assert {w: find_lemma(w, ex)[0] for w in cases} == cases


def test_normalize_strips_possessive():
    assert normalize("model’s") == "model"
    assert normalize("model's,") == "model"
    assert normalize("models'") == "models"
    assert normalize("don't") == "don't"
