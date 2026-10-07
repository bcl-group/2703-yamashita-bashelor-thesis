NET = {"senses": [{"pos": "名詞", "meanings": ["網"]}], "example": ""}


def test_status(client):
    assert client.get("/api/status").get_json()["dictionary"] is True


def test_lookup(client):
    r = client.get("/api/lookup?word=Networks").get_json()
    assert r["lemma"] == "network" and r["saved"] is None
    assert r["default"][0] == {"pos": "名詞", "meanings": ["ネットワーク", "網", "放送網"]}


def test_lookup_rejects_empty(client):
    assert client.get("/api/lookup?word=123").status_code == 400


def test_put_then_lookup_shows_saved(client):
    assert client.put("/api/words/network", json=NET).status_code == 200
    assert client.get("/api/lookup?word=network").get_json()["saved"]["senses"] == NET["senses"]


def test_put_word_with_space(client):
    body = {"senses": [{"pos": "名詞", "meanings": ["神経回路網"]}], "example": ""}
    assert client.put("/api/words/neural%20network", json=body).get_json()["word"] == "neural network"


def test_put_invalid(client):
    assert client.put("/api/words/x", json={"senses": [], "example": ""}).status_code == 400


def test_list_delete(client):
    client.put("/api/words/net", json=NET)
    assert [e["word"] for e in client.get("/api/words").get_json()] == ["net"]
    assert client.delete("/api/words/net").status_code == 204
    assert client.delete("/api/words/net").status_code == 404


def test_quiz(client):
    assert client.get("/api/quiz/next").get_json() is None
    client.put("/api/words/net", json=NET)
    assert client.get("/api/quiz/next").get_json()["word"] == "net"
    assert client.post("/api/quiz/net", json={"result": "correct"}).get_json()["review"]["correct"] == 1
    assert client.post("/api/quiz/net", json={"result": "x"}).status_code == 400
    assert client.post("/api/quiz/none", json={"result": "correct"}).status_code == 404


def test_is_running_detects_listening_port():
    import socket

    from app import is_running

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert is_running("127.0.0.1", port) is True
    assert is_running("127.0.0.1", port) is False
