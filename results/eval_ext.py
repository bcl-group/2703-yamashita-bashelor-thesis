"""
上位の HH 代理モデルを、追加のテスト（experiments/hh_surrogate/hh_bench_ext.py の fi / seeds / long / ood）で採点する。
モデルの読み込みは results/rescore_gamma.py の MODELS をそのまま使う。結果は results/ext/<タグ>.json。

使い方（リポジトリのルートで）:
    uv run python -X utf8 results/eval_ext.py mm_rel                 # 1 モデル、4 セットすべて
    uv run python -X utf8 results/eval_ext.py mm_rel --suites fi ood
    並列: bash results/ext/run_all.sh
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'results'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
import hh_bench_ext as hx                 # noqa: E402
from rescore_gamma import get_model       # noqa: E402

OUT = os.path.join(ROOT, 'results', 'ext')
# 追加テストにかけるモデル（各分類の上位と基準。seed 違いも 1 本）
TOP = ['hh_modular_ou', 'k_3_w3', 'mm_rel', 'mm_rel_s1', 'mm_sv05', 'mm_w256', 'mm_rel_relu', 'cl_p_mlp']


def main():
    if sys.argv[1:] == ['--list']:
        print('\n'.join(TOP))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='*', default=TOP)
    ap.add_argument('--suites', nargs='+', default=hx.SUITES)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    for tag in a.tags:
        path = os.path.join(OUT, tag.replace(":", "_").replace("+", "_") + ".json")
        res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
        run = get_model(tag)
        for s in a.suites:
            res[s] = hx.score_suite(run, s, label=tag)
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)


if __name__ == '__main__':
    main()
