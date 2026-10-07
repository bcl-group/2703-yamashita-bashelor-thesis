"""日本語WordNet と EJDict の検索。"""

import re
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

# EJDict の「goの過去」「childの複数形」のような変化形だけの訳
_INFLECTION = re.compile(r"^([a-z]+)\s?の[^/;()]*?(過去|分詞|複数|単数|現在形|比較級|最上級)[^/;()]*$")

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
        self._jpn_lemmas: set[str] | None = None

    def has(self, lemma: str) -> bool:
        """日本語訳がある語か。"""
        if self._jpn_lemmas is None:
            # 相関サブクエリは遅い（約15秒）ので、synset の集合を作って Python 側で絞る
            jpn = {r[0] for r in self._con.execute("SELECT synset FROM sense WHERE lang = 'jpn'")}
            rows = self._con.execute(
                "SELECT w.lemma, s.synset FROM word w JOIN sense s ON s.wordid = w.wordid "
                "WHERE w.lang = 'eng' AND s.lang = 'eng'")
            self._jpn_lemmas = {lemma for lemma, synset in rows if synset in jpn}
        return _wn_key(lemma) in self._jpn_lemmas

    def has_english(self, lemma: str) -> bool:
        """英語WordNetに載っている語か（日本語訳の有無によらない）。"""
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
        # USES（略語）と uses が小文字で衝突しないよう、全部大文字の見出しは分けておく
        self._acronyms: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            head, sep, body = line.partition("\t")
            if not sep:
                continue
            for h in head.split(","):
                h = h.strip()
                if not h:
                    continue
                table = self._acronyms if len(h) > 1 and h.isupper() else self._entries
                key = h.lower()
                old = table.get(key)
                table[key] = f"{old} / {body}" if old else body

    def has(self, lemma: str) -> bool:
        return lemma in self._entries

    def has_acronym(self, lemma: str) -> bool:
        return lemma in self._acronyms

    def lookup(self, lemma: str) -> list[str]:
        body = self._entries.get(lemma) or self._acronyms.get(lemma, "")
        out: list[str] = []
        for m in body.split(" / "):
            # 『』は EJDict の強調記号なので外す
            m = m.replace("『", "").replace("』", "").strip()
            if m and m not in out:
                out.append(m)
        return out

    def base_of(self, lemma: str) -> str | None:
        """訳が変化形の説明だけなら原形を返す（went → go）。"""
        meanings = self.lookup(lemma)
        matches = [_INFLECTION.match(m) for m in meanings]
        if meanings and all(matches):
            return matches[0].group(1)
        return None


class Dictionary:
    def __init__(self, wordnet: WordNetDict | None, ejdict: EJDict | None):
        self.wordnet = wordnet
        self.ejdict = ejdict

    def warm_up(self) -> None:
        """見出し語の一覧を先に読み込み、最初の検索を待たせない。"""
        if self.wordnet is not None:
            self.wordnet.has("")
            self.wordnet.has_english("")

    @property
    def available(self) -> bool:
        return self.wordnet is not None or self.ejdict is not None

    @classmethod
    def from_dir(cls, data_dir: Path) -> "Dictionary":
        wn = data_dir / "wnjpn.db"
        ej = data_dir / "ejdict.tsv"
        return cls(WordNetDict(wn) if wn.exists() else None,
                   EJDict(ej) if ej.exists() else None)

    def _has_translation(self, lemma: str) -> bool:
        return any(d is not None and d.has(lemma) for d in (self.wordnet, self.ejdict))

    def _has_any(self, lemma: str) -> bool:
        return ((self.wordnet is not None and self.wordnet.has_english(lemma))
                or (self.ejdict is not None and self.ejdict.has_acronym(lemma)))

    def lookup(self, text: str) -> dict:
        # 訳のある語を優先し、なければ略語や訳のない WordNet 語も候補にする
        lemma, found = find_lemma(text, self._has_translation)
        if not found:
            lemma, found = find_lemma(text, self._has_any)
        if self.ejdict is not None and not (self.wordnet and self.wordnet.has(lemma)):
            base = self.ejdict.base_of(lemma)
            if base and self._has_translation(base):
                lemma = base
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
