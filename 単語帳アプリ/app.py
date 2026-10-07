"""英語論文単語帳アプリ（Flask）。

起動: uv run python 単語帳アプリ/app.py → http://127.0.0.1:5000 がブラウザで開く
（--no-browser でブラウザを開かない）
"""

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request

from vocab.dictionary import POS_ORDER, Dictionary, suggest_senses
from vocab.lemmatize import normalize
from vocab.store import CorruptVocabError, VocabStore

APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def is_running(host: str, port: int) -> bool:
    """そのポートで既にサーバが待ち受けているか。"""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def create_app(store: VocabStore, dictionary: Dictionary) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({"dictionary": dictionary.available, "pos_list": POS_ORDER})

    @app.get("/api/lookup")
    def lookup():
        word = request.args.get("word", "")
        if not normalize(word):
            return jsonify({"error": "英単語を入力してください"}), 400
        r = dictionary.lookup(word)
        r["default"] = suggest_senses(r)
        r["saved"] = store.get(r["lemma"])
        return jsonify(r)

    @app.get("/api/words")
    def list_words():
        a = request.args
        return jsonify(store.list(q=a.get("q", ""), pos=a.get("pos", ""), sort=a.get("sort", "new")))

    @app.put("/api/words/<path:word>")
    def put_word(word):
        body = request.get_json(silent=True) or {}
        try:
            entry = store.put(word, body.get("senses", []), body.get("example", ""))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(entry)

    @app.delete("/api/words/<path:word>")
    def delete_word(word):
        if not store.delete(word):
            return jsonify({"error": "見つかりません"}), 404
        return "", 204

    @app.get("/api/quiz/next")
    def quiz_next():
        return jsonify(store.next_quiz(exclude=request.args.get("exclude") or None))

    @app.post("/api/quiz/<path:word>")
    def quiz_record(word):
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(store.record(word, body.get("result", "")))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except KeyError:
            return jsonify({"error": "見つかりません"}), 404

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="論文単語帳")
    parser.add_argument("--no-browser", action="store_true", help="起動時にブラウザを開かない")
    args = parser.parse_args()

    # 二重起動した場合は、既存のサーバをブラウザで開くだけにする
    if is_running(HOST, PORT):
        print(f"既に起動しています: {URL}")
        if not args.no_browser:
            webbrowser.open(URL)
        return

    try:
        store = VocabStore(APP_DIR / "vocab.json")
    except CorruptVocabError as e:
        sys.exit(f"エラー: {e}\nvocab.json を修正してから再起動してください。")
    dictionary = Dictionary.from_dir(APP_DIR / "data")
    if not dictionary.available:
        print("辞書がありません: uv run python 単語帳アプリ/setup_dict.py を実行してください")
    dictionary.warm_up()
    if not args.no_browser:
        # サーバが待ち受けを始めてから開く
        threading.Timer(1.0, webbrowser.open, args=(URL,)).start()
    print(f"{URL} で起動しました。このウィンドウを閉じると終了します。")
    create_app(store, dictionary).run(host=HOST, port=PORT)


if __name__ == "__main__":
    main()
