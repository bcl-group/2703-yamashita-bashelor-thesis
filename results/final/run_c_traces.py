"""
最終テストのうち代表 8 本で C（m3_hdf_s0）を回し、I と V の波形を保存して図にする。
採点（results/final/m3_hdf_s0.json）はやり直さない。モデルも同じなので、結果は採点時と同じ波形になる。
長い電流（20 s、4.3 s）は時間がかかるので除いた。
    .venv/Scripts/python.exe -X utf8 results/final/run_c_traces.py
出力: results/final/c_traces.npz, results/final/fig_c_iv_final.png
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'results'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import hh_bench_ext as hx            # noqa: E402
from rescore_gamma import get_model  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'final')
DT = 0.01  # [ms]
PICK = [
    'threshold: const 7.69',
    'transient: step 8.0 for 147ms',
    'tracking: sin 2.6Hz 9.2±6.4',
    'tracking: ramp 0-21.3 in 577ms',   # 外した（Γ -0.17）
    'tracking: ramp 0-26.4 in 882ms',   # 外した（Γ -0.12）
    'fluct: OU 5.4±4.9 tau 3.5ms',      # 外した（Γ 0.88）
    'fluct: OU 11.1±3.3 tau 3.5ms',     # 外した（Γ 0.96）
    'ood: sin 100Hz 11.4±14.1',
]

names, I, Vt, lens = hx.reference('final')
idx = [names.index(n) for n in PICK]
N = max(lens[i] for i in idx)
torch.set_num_threads(8)
Vp = get_model('m3_hdf_s0')(np.ascontiguousarray(I[idx, :N]))
out = {'names': np.array(PICK)}
for j, i in enumerate(idx):
    out[f'I{j}'] = I[i, :lens[i] - 1]
    out[f'hh{j}'] = np.asarray(Vt[i])
    out[f'C{j}'] = Vp[j, :lens[i] - 1]
np.savez_compressed(os.path.join(OUT, 'c_traces.npz'), **out)

import matplotlib                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt       # noqa: E402
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})
fig, axs = plt.subplots(4, 4, figsize=(17, 10), sharex='col',
                        gridspec_kw={'height_ratios': [1, 2, 1, 2], 'hspace': 0.12, 'wspace': 0.18})
for j, name in enumerate(PICK):
    r, c = 2 * (j // 4), j % 4
    vh, vc, Ij = out[f'hh{j}'], out[f'C{j}'], out[f'I{j}']
    t = np.arange(len(vh)) * DT
    axs[r, c].plot(t, Ij, color='#444444', lw=0.9)
    axs[r, c].set_title(name, fontsize=11)
    axs[r + 1, c].plot(t, vh, color='#bdbdbd', lw=2.6, label='HH（正解）')
    axs[r + 1, c].plot(t, vc, color='#1baf7a', lw=0.9, label='C MLP 3 個')
    axs[r + 1, c].set_ylim(-85, 55)
    axs[r + 1, c].set_xlabel('時刻 $t$ [ms]')
for r in (0, 2):
    axs[r, 0].set_ylabel('$I$ [µA/cm²]')
    axs[r + 1, 0].set_ylabel('$V$ [mV]')
axs[1, 0].legend(loc='upper left', fontsize=9, frameon=False)
fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.06)
fig.savefig(os.path.join(OUT, 'fig_c_iv_final.png'), dpi=130)
print('saved')
