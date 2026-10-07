"""vocab.json の読み書きとクイズ出題。"""

from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

from .dictionary import POS_ORDER


class CorruptVocabError(Exception):
    pass


def _now_str(now: datetime | None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def quiz_weight(entry: dict, now: datetime) -> float:
    """w = (1 + wrong) * 1/(1 + correct) * (1 + d/7)。d は最終出題からの日数（未出題なら 7）。"""
    r = entry["review"]
    if r["last"] is None:
        d = 7.0
    else:
        d = max(0.0, (now - datetime.fromisoformat(r["last"])).total_seconds() / 86400)
    return (1 + r["wrong"]) * (1 / (1 + r["correct"])) * (1 + d / 7)


def _clean_senses(senses: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in senses:
        pos = s.get("pos")
        if pos not in POS_ORDER:
            raise ValueError(f"品詞が不正です: {pos}")
        meanings: list[str] = []
        for m in s.get("meanings", []):
            m = str(m).strip()
            if m and m not in meanings:
                meanings.append(m)
        if meanings:
            out.append({"pos": pos, "meanings": meanings})
    if not out:
        raise ValueError("意味を1つ以上入力してください")
    return out


class VocabStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise CorruptVocabError(f"{self.path} を読み込めません: {e}") from e
            if not isinstance(data, list):
                raise CorruptVocabError(f"{self.path} の形式が不正です（配列ではありません）")
            self._entries = {e["word"]: e for e in data}

    def _save(self) -> None:
        # 一時ファイルに書いてから置き換え、途中で落ちても元のファイルを壊さない
        text = json.dumps(list(self._entries.values()), ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise

    def list(self, q: str = "", pos: str = "", sort: str = "new") -> list[dict]:
        q = q.strip().lower()
        out = []
        for e in self._entries.values():
            if pos and all(s["pos"] != pos for s in e["senses"]):
                continue
            if q:
                texts = [e["word"]] + [m.lower() for s in e["senses"] for m in s["meanings"]]
                if not any(q in t for t in texts):
                    continue
            out.append(e)
        if sort == "alpha":
            return sorted(out, key=lambda e: e["word"])
        return sorted(out, key=lambda e: e["created_at"], reverse=True)

    def get(self, word: str) -> dict | None:
        return self._entries.get(word.strip().lower())

    def put(self, word: str, senses: list[dict], example: str = "",
            now: datetime | None = None) -> dict:
        word = word.strip().lower()
        if not word:
            raise ValueError("単語が空です")
        cleaned = _clean_senses(senses)
        ts = _now_str(now)
        old = self._entries.get(word)
        entry = {
            "word": word,
            "senses": cleaned,
            "example": example.strip(),
            "created_at": old["created_at"] if old else ts,
            "updated_at": ts,
            "review": old["review"] if old else {"correct": 0, "wrong": 0, "last": None},
        }
        self._entries[word] = entry
        self._save()
        return entry

    def delete(self, word: str) -> bool:
        if self._entries.pop(word.strip().lower(), None) is None:
            return False
        self._save()
        return True

    def record(self, word: str, result: str, now: datetime | None = None) -> dict:
        if result not in ("correct", "wrong"):
            raise ValueError(f"result が不正です: {result}")
        entry = self._entries[word.strip().lower()]
        entry["review"][result] += 1
        entry["review"]["last"] = _now_str(now)
        self._save()
        return entry

    def next_quiz(self, exclude: str | None = None, rng: random.Random | None = None,
                  now: datetime | None = None) -> dict | None:
        if not self._entries:
            return None
        rng = rng or random.Random()
        now = now or datetime.now()
        pool = [e for e in self._entries.values() if e["word"] != exclude] or list(self._entries.values())
        return rng.choices(pool, weights=[quiz_weight(e, now) for e in pool])[0]
