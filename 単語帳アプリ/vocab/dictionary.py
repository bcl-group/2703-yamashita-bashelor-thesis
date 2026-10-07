"""日本語WordNet と EJDict の検索。"""

import sqlite3
from pathlib import Path

from .lemmatize import find_lemma

POS_ORDER: list[str] = ["名詞", "動詞", "形容詞", "副詞", "前置詞", "接続詞", "代名詞", "助動詞", "その他"]
WN_POS: dict[str, str] = {"n": "名詞", "v": "動詞", "a": "形容詞", "s": "形容詞", "r": "副詞"}

_WN_SQL = """
SELECT sy.pos, jw.lemma
FROM word w
JOIN sense s   ON s.wordid = w.wordid AND s.lang = 'eng'
JOIN synset sy ON sy.synset = s.synset
JOIN sense js  ON js.synset = s.synset AND js.lang = 'jpn'
JOIN word jw   ON jw.wordid = js.wordid
WHERE w.lang = 'eng' AND w.lemma = ?
ORDER BY COALESCE(s.freq, 0) DESC, s.synset, js.rowid
"""

_WN_POS_SQL = """
SELECT DISTINCT sy.pos
FROM word w
JOIN sense s   ON s.wordid = w.wordid AND s.lang = 'eng'
JOIN synset sy ON sy.synset = s.synset
WHERE w.lang = 'eng' AND w.lemma = ?
"""


def _wn_key(lemma: str) -> str:
    # WordNet の複合語は neural_network のように _ でつなぐ
    return lemma.replace(" ", "_")


class WordNetDict:
    def __init__(self, path: Path):
        self._con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self._lemmas: set[str] | None = None

    def has(self, lemma: str) -> bool:
        if self._lemmas is None:
            rows = self._con.execute("SELECT lemma FROM word WHERE lang = 'eng'")
            self._lemmas = {r[0] for r in rows}
        return _wn_key(lemma) in self._lemmas

    def lookup(self, lemma: str) -> dict[str, list[str]]:
        """品詞名 → 訳（頻度の高い synset 順、重複なし）。キーは POS_ORDER 順。"""
        groups: dict[str, list[str]] = {}
        for pos, jpn in self._con.execute(_WN_SQL, (_wn_key(lemma),)):
            meanings = groups.setdefault(WN_POS[pos], [])
            if jpn not in meanings:
                meanings.append(jpn)
        return {p: groups[p] for p in POS_ORDER if p in groups}

    def english_pos(self, lemma: str) -> list[str]:
        """英語WordNet上の品詞（日本語訳の有無によらない）。POS_ORDER 順。"""
        found = {WN_POS[r[0]] for r in self._con.execute(_WN_POS_SQL, (_wn_key(lemma),))}
        return [p for p in POS_ORDER if p in found]


class EJDict:
    def __init__(self, path: Path):
        self._entries: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            head, sep, body = line.partition("\t")
            if not sep:
                continue
            for h in head.split(","):
                key = h.strip().lower()
                if key:
                    old = self._entries.get(key)
                    self._entries[key] = f"{old} / {body}" if old else body

    def has(self, lemma: str) -> bool:
        return lemma in self._entries

    def lookup(self, lemma: str) -> list[str]:
        out: list[str] = []
        for m in self._entries.get(lemma, "").split(" / "):
            # 『』は EJDict の強調記号なので外す
            m = m.replace("『", "").replace("』", "").strip()
            if m and m not in out:
                out.append(m)
        return out


class Dictionary:
    def __init__(self, wordnet: WordNetDict | None, ejdict: EJDict | None):
        self.wordnet = wordnet
        self.ejdict = ejdict

    @property
    def available(self) -> bool:
        return self.wordnet is not None or self.ejdict is not None

    @classmethod
    def from_dir(cls, data_dir: Path) -> "Dictionary":
        wn = data_dir / "wnjpn.db"
        ej = data_dir / "ejdict.tsv"
        return cls(WordNetDict(wn) if wn.exists() else None,
                   EJDict(ej) if ej.exists() else None)

    def _exists(self, lemma: str) -> bool:
        return any(d is not None and d.has(lemma) for d in (self.wordnet, self.ejdict))

    def lookup(self, text: str) -> dict:
        lemma, found = find_lemma(text, self._exists)
        return {
            "input": text,
            "lemma": lemma,
            "found": found,
            "wordnet": self.wordnet.lookup(lemma) if self.wordnet else {},
            "english_pos": self.wordnet.english_pos(lemma) if self.wordnet else [],
            "ejdict": self.ejdict.lookup(lemma) if self.ejdict else [],
        }


def default_senses(wordnet: dict[str, list[str]], top: int = 3) -> list[dict]:
    return [{"pos": pos, "meanings": ms[:top]} for pos, ms in wordnet.items()]


def suggest_senses(result: dict, top: int = 3) -> list[dict]:
    """保存内容の初期値。日本語WordNetに訳がなければ EJDict の訳で補う。"""
    if result["wordnet"]:
        return default_senses(result["wordnet"], top)
    if not result["ejdict"]:
        return []
    pos = result["english_pos"]
    # 品詞が1つに決まらない場合は「その他」に入れ、画面で付け替えてもらう
    return [{"pos": pos[0] if len(pos) == 1 else "その他", "meanings": result["ejdict"][:top]}]
