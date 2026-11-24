"""
発表スライド（11/25）用：検証用の電流 1 本で、HH と 4 分野の最良モデルの閉ループ波形を保存する。
    uv run python -X utf8 results/slides_1125/run_traces.py
出力: results/slides_1125/traces.npz（t [ms], I, V_hh, V_A..V_D）
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'results'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
import rescore_gamma as rg   # noqa: E402
import hh_bench as hb        # noqa: E402

NAME = 'v OU 7±5 (tau 5ms)'
TAGS = {'A': 'r8_mlp_hd_w256', 'B': 'sp_r_init', 'C': 'm3_hdf_s0', 'D': 'rnn3:p_h64_cl1'}

names, I, Vt, lens = hb.reference(val=True)
i = names.index(NAME)
n = lens[i]
Ii = I[i:i + 1, :n]
out = {'t': np.arange(n - 1) * hb.DT, 'I': Ii[0, :n - 1], 'V_hh': Vt[i]}
for k, tag in TAGS.items():
    out['V_' + k] = rg.get_model(tag)(Ii)[0]
    print(k, tag, 'done', flush=True)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traces.npz'), **out)
