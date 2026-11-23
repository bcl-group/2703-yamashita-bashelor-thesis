"""
4 分野（A: Markov MLP / B: RNN 1 個（W_hh）/ C: MLP 3 個 / D: RNN 3 個）の最良を、検証 17 本だけで決める。
テスト 26 本・最終テスト 54 本の値は見ない。

決め方:
    1. 設定 = タグから seed（末尾の _s0 / _s1 ...）を除いたもの
    2. 設定ごとに、seed の平均で「検証の 100% 一致の本数」→「検証の平均 Γ」の順に比べる（大きいほうがよい）
    3. 同点なら seed の本数が多いほう（ばらつきを測れるので）
結果: results/final/selection.json と、標準出力の表
使い方: uv run python -X utf8 results/select_best.py
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'results'))
import make_ranking as mr   # noqa: E402

FIELD = {'Markov MLP': 'A', 'RNN 1 個': 'B', 'MLP 3 個': 'C', 'RNN 3 個': 'D'}


def config_of(tag):
    return re.sub(r'_s\d+$', '', tag)


def main():
    rows = [r for r in mr.collect() if r['val_n100'] is not None and r['arch']]
    groups = defaultdict(list)
    for r in rows:
        cat = mr.CATEGORY.get(r['arch'])
        if cat in FIELD:
            groups[(FIELD[cat], r['src'], config_of(r['tag']))].append(r)
    best = {}
    table = []
    for (f, src, cfg), rs in groups.items():
        n = len(rs)
        key = (sum(r['val_n100'] for r in rs) / n, sum(r['val_mean'] for r in rs) / n, n)
        item = (f, cfg, src, key, sorted(r['tag'] for r in rs), rs[0]['arch'])
        table.append(item)
        if f not in best or key > best[f][3]:
            best[f] = item
    for f in 'ABCD':
        print(f'== {f} 上位 6 ==')
        for t in sorted([t for t in table if t[0] == f], key=lambda t: t[3], reverse=True)[:6]:
            print(f'  {t[1]:24s} {t[5]:36s} 検証 100% {t[3][0]:5.2f}/17  平均 Γ {t[3][1]:.4f}  seed {t[3][2]}')
    out = {f: {'config': b[1], 'src': b[2], 'tags': b[4], 'arch': b[5],
               'val_n100_mean': b[3][0], 'val_mean': b[3][1], 'n_seed': b[3][2]} for f, b in sorted(best.items())}
    os.makedirs(os.path.join(ROOT, 'results', 'final'), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'results', 'final', 'selection.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
