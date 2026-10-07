import random
from datetime import datetime, timedelta

import pytest

from vocab.store import CorruptVocabError, VocabStore, quiz_weight

NOW = datetime(2026, 10, 7, 21, 0, 0)
S = [{"pos": "名詞", "meanings": ["ネットワーク"]}]


def test_put_get_roundtrip_keeps_japanese(tmp_path):
    p = tmp_path / "vocab.json"
    VocabStore(p).put("Network", S, "a network.", now=NOW)
    assert "ネットワーク" in p.read_text(encoding="utf-8")
    e = VocabStore(p).get("network")
    assert e["word"] == "network" and e["example"] == "a network."
    assert e["created_at"] == "2026-10-07T21:00:00"
    assert e["review"] == {"correct": 0, "wrong": 0, "last": None}


def test_put_cleans_meanings(tmp_path):
    e = VocabStore(tmp_path / "v.json").put("net", [
        {"pos": "名詞", "meanings": [" 網 ", "", "網", "ネット"]},
        {"pos": "動詞", "meanings": [" "]}], now=NOW)
    assert e["senses"] == [{"pos": "名詞", "meanings": ["網", "ネット"]}]


def test_put_rejects_empty_and_bad_pos(tmp_path):
    st = VocabStore(tmp_path / "v.json")
    with pytest.raises(ValueError):
        st.put("net", [{"pos": "名詞", "meanings": []}])
    with pytest.raises(ValueError):
        st.put("net", [{"pos": "形容動詞", "meanings": ["x"]}])


def test_update_keeps_created_and_review(tmp_path):
    st = VocabStore(tmp_path / "v.json")
    st.put("net", S, now=NOW)
    st.record("net", "wrong", now=NOW)
    e = st.put("net", S, "new", now=NOW + timedelta(days=1))
    assert e["created_at"] == "2026-10-07T21:00:00"
    assert e["updated_at"] == "2026-10-08T21:00:00"
    assert e["review"]["wrong"] == 1


def test_list_search_filter_sort(tmp_path):
    st = VocabStore(tmp_path / "v.json")
    st.put("beta", [{"pos": "動詞", "meanings": ["走る"]}], now=NOW)
    st.put("alpha", S, now=NOW + timedelta(hours=1))
    st.put("gamma", S, now=NOW - timedelta(hours=1))
    assert [e["word"] for e in st.list()] == ["alpha", "beta", "gamma"]
    assert [e["word"] for e in st.list(sort="alpha")] == ["alpha", "beta", "gamma"]
    assert [e["word"] for e in st.list(q="走")] == ["beta"]
    assert [e["word"] for e in st.list(q="ALP")] == ["alpha"]
    assert [e["word"] for e in st.list(pos="名詞")] == ["alpha", "gamma"]


def test_delete(tmp_path):
    st = VocabStore(tmp_path / "v.json")
    st.put("net", S)
    assert st.delete("net") is True and st.delete("net") is False
    assert VocabStore(tmp_path / "v.json").get("net") is None


def test_corrupt_file_is_detected_and_not_overwritten(tmp_path):
    p = tmp_path / "v.json"
    p.write_text("{broken", encoding="utf-8")
    with pytest.raises(CorruptVocabError):
        VocabStore(p)
    assert p.read_text(encoding="utf-8") == "{broken"


def test_quiz_weight():
    e = {"review": {"correct": 1, "wrong": 3, "last": "2026-10-07T21:00:00"}}
    assert quiz_weight(e, NOW + timedelta(days=7)) == pytest.approx(4 * 0.5 * 2)
    assert quiz_weight({"review": {"correct": 0, "wrong": 0, "last": None}}, NOW) == pytest.approx(2.0)


def test_next_quiz_excludes_previous_but_not_only_word(tmp_path):
    st = VocabStore(tmp_path / "v.json")
    assert st.next_quiz() is None
    st.put("only", S)
    assert st.next_quiz(exclude="only")["word"] == "only"
    st.put("other", S)
    rng = random.Random(0)
    assert all(st.next_quiz(exclude="only", rng=rng)["word"] == "other" for _ in range(20))


def test_record(tmp_path):
    st = VocabStore(tmp_path / "v.json")
    st.put("net", S)
    e = st.record("net", "correct", now=NOW)
    assert e["review"] == {"correct": 1, "wrong": 0, "last": "2026-10-07T21:00:00"}
    with pytest.raises(ValueError):
        st.record("net", "maybe")
    with pytest.raises(KeyError):
        st.record("nothing", "correct")
