"""入力の正規化と原形の決定。"""

import re
from typing import Callable

from .irregular import IRREGULAR

_EDGE = re.compile(r"^[^a-z]+|[^a-z]+$")
_SPACES = re.compile(r"\s+")
_POSSESSIVE = re.compile(r"'s$")
_VOWELS = set("aeiou")


def normalize(text: str) -> str:
    """前後の英字以外と所有格 's を除き、連続空白を1つにして小文字化する。"""
    s = text.strip().lower().replace("’", "'").replace("‘", "'")
    s = _EDGE.sub("", _POSSESSIVE.sub("", _EDGE.sub("", s)))
    return _SPACES.sub(" ", s)


def _undouble(stem: str) -> list[str]:
    # stopp → stop のように、末尾の子音重複を1つにする
    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
        return [stem[:-1]]
    return []


def _rule_candidates(word: str) -> list[str]:
    if len(word) < 4:
        return []
    out: list[str] = []
    # notes → note を not より、hoped → hope を hop より先に試す
    if word.endswith("s") and not word.endswith("ss"):
        out.append(word[:-1])
    if word.endswith("ies"):
        out.append(word[:-3] + "y")
    if word.endswith("es"):
        out.append(word[:-2])
    if word.endswith("ied"):
        out.append(word[:-3] + "y")
    if word.endswith("ed"):
        stem = word[:-2]
        out += [stem + "e", stem, *_undouble(stem)]
    if word.endswith("ing"):
        stem = word[:-3]
        out += [stem + "e", stem, *_undouble(stem)]
    return out


def find_lemma(word: str, exists: Callable[[str], bool]) -> tuple[str, bool]:
    """辞書にある原形を探す。見つからなければ (正規化した入力, False)。"""
    w = normalize(word)
    candidates = [w]
    if w in IRREGULAR:
        candidates.append(IRREGULAR[w])
    candidates += _rule_candidates(w)
    for c in candidates:
        if c and exists(c):
            return c, True
    return w, False
