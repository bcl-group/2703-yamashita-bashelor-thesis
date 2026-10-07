import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3

import pytest

# (英語 lemma, synset, 品詞, freq, 日本語訳)
_WN_ROWS = [
    ("network", "A-n", "n", 23, ["ネットワーク", "網"]),
    ("network", "B-n", "n", 4, ["ネットワーク", "放送網"]),
    ("network", "C-v", "v", None, ["人脈を作る"]),
    ("neural_network", "D-n", "n", None, ["ニューラルネットワーク"]),
    ("fast", "E-r", "r", None, ["速く"]),
    ("fast", "F-a", "a", None, ["速い"]),
    ("recurrent", "G-a", "a", None, []),
    ("go", "H-v", "v", 5, []),
    ("go", "I-n", "n", 1, []),
]

_EJ_TEXT = (
    "network\t網状組織 / (コンピューターの)ネットワーク\n"
    "neural network\t神経回路網\n"
    "state-of-the-art\t最新式の\n"
    "color,colour\t色\n"
)


@pytest.fixture
def wn_path(tmp_path):
    """本番の wnjpn.db と同じ列を持つ小さな DB。"""
    path = tmp_path / "wnjpn.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE word (wordid integer primary key, lang text, lemma text, pron text, pos text);
        CREATE TABLE sense (synset text, wordid integer, lang text, rank text, lexid integer, freq integer, src text);
        CREATE TABLE synset (synset text, pos text, name text, src text);
    """)
    eng_ids: dict[str, int] = {}
    for lemma, synset, pos, freq, jpns in _WN_ROWS:
        if lemma not in eng_ids:
            eng_ids[lemma] = con.execute(
                "INSERT INTO word (lang, lemma, pos) VALUES ('eng', ?, ?)", (lemma, pos)).lastrowid
        con.execute("INSERT INTO synset VALUES (?, ?, ?, 'test')", (synset, pos, lemma))
        con.execute("INSERT INTO sense (synset, wordid, lang, freq) VALUES (?, ?, 'eng', ?)",
                    (synset, eng_ids[lemma], freq))
        for jpn in jpns:
            jid = con.execute(
                "INSERT INTO word (lang, lemma, pos) VALUES ('jpn', ?, ?)", (jpn, pos)).lastrowid
            con.execute("INSERT INTO sense (synset, wordid, lang) VALUES (?, ?, 'jpn')", (synset, jid))
    con.commit()
    con.close()
    return path


@pytest.fixture
def ej_path(tmp_path):
    path = tmp_path / "ejdict.tsv"
    path.write_text(_EJ_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def client(tmp_path, wn_path, ej_path):
    from app import create_app
    from vocab.dictionary import Dictionary, EJDict, WordNetDict
    from vocab.store import VocabStore

    store = VocabStore(tmp_path / "v.json")
    app = create_app(store, Dictionary(WordNetDict(wn_path), EJDict(ej_path)))
    return app.test_client()
