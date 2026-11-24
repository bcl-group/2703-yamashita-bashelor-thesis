"""
発表スライド（11/25）用：開発用テスト 26 本のうち、4 モデルのどれかが外した電流で閉ループの波形を保存する。
    .venv/Scripts/python.exe -X utf8 results/slides_1125/run_test_traces.py
出力: results/slides_1125/test_traces.npz（電流ごとに t [ms], I, V_hh, V_A..V_D）
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'results'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
import rescore_gamma as rg   # noqa: E402
import hh_bench as hb        # noqa: E402

NAMES = ['ramp 0-20', 'ramp 0-30', 'sin 10Hz 7.5±7.5', 'OU 5±3']
TAGS = {'A': 'r8_mlp_hd_w256', 'B': 'sp_r_init', 'C': 'm3_hdf_s0', 'D': 'rnn3:p_h64_cl1'}

names, I, Vt, lens = hb.reference(val=False)
idx = [names.index(n) for n in NAMES]
N = max(lens[i] for i in idx)
Isub = I[idx, :N]
out = {'names': np.array(NAMES)}
for j, i in enumerate(idx):
    out[f'I{j}'] = I[i, :lens[i] - 1]
    out[f'hh{j}'] = Vt[i]
for k, tag in TAGS.items():
    Vp = rg.get_model(tag)(Isub)
    for j, i in enumerate(idx):
        out[f'{k}{j}'] = Vp[j, :lens[i] - 1]
    print(k, 'done', flush=True)
np.savez_compressed(os.path.join(HERE, 'test_traces.npz'), **out)
