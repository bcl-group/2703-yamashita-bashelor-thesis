"""
学習済みの MLP 3 個を評価し、応答（数値と図）を保存する。train.py の最後にも同じ処理が走る。

使い方（リポジトリのルートで）:
    # 新しい形式の実行フォルダ（results/mlp3/<tag>/model_b*.pt）を評価し直す
    uv run python -X utf8 experiments/mlp3/evaluate.py results/mlp3/m3_hdf_s0
    # 旧形式の重み（_b1.pt などの前の部分）を指定して、新しいフォルダに書き出す
    uv run python -X utf8 experiments/mlp3/evaluate.py results/hh_improve/m3_hdf_s0 --out results/mlp3/m3_hdf_s0
    # 最終テスト 54 本（最終候補を決めた後に 1 回だけ。結果があれば上書きしない）
    uv run python -X utf8 experiments/mlp3/evaluate.py results/mlp3/m3_hdf_s0 --final

保存するもの（<out>/）: responses_<set>.npz, score_<set>.json, figs/<set>/*.png（outputs.py を参照）
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_sets as es       # noqa: E402
import outputs               # noqa: E402
from model import MLP3, V_LO_TRAIN, V_LO_FINAL   # noqa: E402


def evaluate(model, out_dir, sets=('val', 'test'), label=None):
    """sets の各セットで応答を保存して採点する。Returns: {set: score}"""
    os.makedirs(out_dir, exist_ok=True)
    return {s: outputs.save_responses(out_dir, model, es.reference(s),
                                      V_LO_FINAL if s == 'final' else V_LO_TRAIN, label=label)
            for s in sets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='実行フォルダ（model_b1.pt を含む）か、旧形式の重みの前の部分')
    ap.add_argument('--out', default=None, help='保存先（省略時は path が実行フォルダならそこ）')
    ap.add_argument('--final', action='store_true', help='最終テスト 54 本だけを採点する')
    ap.add_argument('--force', action='store_true', help='最終テストの結果があっても上書きする（理由を PROGRESS.md に書く）')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.device != 'cpu':
        torch.set_default_device(a.device)
    is_run = os.path.isdir(a.path)
    prefix = os.path.join(a.path, 'model') if is_run else a.path
    out = a.out or (a.path if is_run else None)
    if out is None:
        ap.error('旧形式の重みを評価するときは --out を指定する')
    model = MLP3.load(prefix).to(a.device)
    if a.final:
        if os.path.exists(os.path.join(out, 'score_final.json')) and not a.force:
            print('スキップ: 最終テストは採点済み（final は 1 回だけ使う）')
            return
        evaluate(model, out, ('final',), label=os.path.basename(out))
    else:
        evaluate(model, out, label=os.path.basename(out))


if __name__ == '__main__':
    main()
