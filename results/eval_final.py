"""
最終テスト（experiments/hh_surrogate/hh_bench_ext.py の final、54 本）で HH 代理モデルを採点する。
結果は results/final/<タグ>.json。

約束（結果の見せ方を正直にするため）:
    - final は最終候補を決めた後に 1 回だけ使う。結果を見てからモデルや学習を直したら、その結果は「開発用」になる
    - そのため、すでに結果があるタグは上書きしない（やり直すときは --force を付け、理由を PROGRESS.md に書く）
    - 採点の仕組みの確認には、候補モデルではなく次の 2 つを使う
        _hh   : HH そのもの（正解と同じ。Γ = 1 になるはず）
        _rest : ずっと静止電位を出すモデル（発火しない。Γ = 0 になるはず）

使い方（リポジトリのルートで）:
    uv run python -X utf8 results/eval_final.py _hh _rest                       # 仕組みの確認
    uv run python -X utf8 results/eval_final.py mm_sv05 r9_mlp_v2_s0 ...        # 最終候補（1 回だけ）
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'results'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import hh_bench_ext as hx                 # noqa: E402
from neuron.hh import simulate, V_REST     # noqa: E402

OUT = os.path.join(ROOT, 'results', 'final')
SANITY = os.path.join(OUT, 'sanity')


def run_hh(I):
    # HH の正解（キャッシュ）をそのまま返す。I (B, N) -> V (B, N-1)。採点の仕組みの確認用（解き直すと 11 分かかる）
    _, I_ref, Vt, _ = hx.reference('final')
    assert np.array_equal(I, I_ref)
    V = np.full((I.shape[0], I.shape[1] - 1), V_REST)
    for b, v in enumerate(Vt):
        V[b, :len(v)] = v
    return V


def run_rest(I):
    return np.full((I.shape[0], I.shape[1] - 1), V_REST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--force', action='store_true', help='既存の結果を上書きする（理由を PROGRESS.md に残すこと）')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    for tag in a.tags:
        sanity = tag.startswith('_')
        out_dir = SANITY if sanity else OUT
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, tag.replace(':', '_').replace('+', '_') + '.json')
        if os.path.exists(path) and not (a.force or sanity):
            print('スキップ（採点済み。final は 1 回だけ使う）:', os.path.relpath(path, ROOT))
            continue
        if sanity:
            run = {'_hh': run_hh, '_rest': run_rest}[tag]
        else:
            import torch
            from rescore_gamma import get_model
            torch.set_num_threads(a.threads)
            run = get_model(tag)
        res = hx.score_suite(run, 'final', label=tag)
        json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)


if __name__ == '__main__':
    main()
