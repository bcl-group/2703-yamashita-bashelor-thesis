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
