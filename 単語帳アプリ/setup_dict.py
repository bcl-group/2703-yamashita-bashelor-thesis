"""辞書（日本語WordNet / EJDict）をダウンロードして data/ に置く。

実行: uv run python 単語帳アプリ/setup_dict.py
"""

import gzip
import shutil
import string
import urllib.request
from pathlib import Path
from typing import Iterable

APP_DIR = Path(__file__).resolve().parent
WNJPN_URL = "https://github.com/bond-lab/wnja/releases/download/v1.1/wnjpn.db.gz"
EJDICT_URL = "https://raw.githubusercontent.com/kujirahand/EJDict/master/src/{letter}.txt"


def concat_ejdict(texts: Iterable[str]) -> str:
    """各ファイルを連結し、空行とタブを含まない行を捨てる。"""
    lines = [line for t in texts for line in t.splitlines() if "\t" in line]
    return "\n".join(lines) + "\n"


def _download_wnjpn(dest: Path) -> None:
    part = dest.with_suffix(".part")
    print(f"日本語WordNet をダウンロード中（約60MB）: {WNJPN_URL}")
    with urllib.request.urlopen(WNJPN_URL) as res, gzip.GzipFile(fileobj=res) as gz, open(part, "wb") as f:
        shutil.copyfileobj(gz, f, length=1 << 20)
    part.replace(dest)


def _download_ejdict(dest: Path) -> None:
    texts = []
    for letter in string.ascii_lowercase:
        print(f"EJDict をダウンロード中: {letter}.txt", end="\r")
        with urllib.request.urlopen(EJDICT_URL.format(letter=letter)) as res:
            texts.append(res.read().decode("utf-8"))
    print()
    dest.write_text(concat_ejdict(texts), encoding="utf-8")


def main(data_dir: Path = APP_DIR / "data") -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, download in (("wnjpn.db", _download_wnjpn), ("ejdict.tsv", _download_ejdict)):
        dest = data_dir / name
        if dest.exists():
            print(f"{name} は既にあるので省略します")
            continue
        download(dest)
        print(f"{name} を作成しました")


if __name__ == "__main__":
    main()
