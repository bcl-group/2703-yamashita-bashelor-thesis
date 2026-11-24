"""
C（MLP 3 個、m3_hdf_s0）の入力電流 I と膜電位 V を、HH（正解）と重ねて描く。
    .venv/Scripts/python.exe -X utf8 results/slides_1125/plot_c_iv.py                      # 開発用テスト 4 本
    .venv/Scripts/python.exe -X utf8 results/slides_1125/plot_c_iv.py results/final/c_traces.npz results/final/fig_c_iv_final.png
npz の中身: names, I{j}, hh{j}, C{j}（j = 0, 1, ...）
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})
DT = 0.01  # [ms]
HH_COL, C_COL = '#222222', '#1baf7a'


def spikes(v, th=0.0):
    # 0 mV を下から上に越えた時刻 [ms]
    i = np.where((v[:-1] < th) & (v[1:] >= th))[0]
    return i * DT


def gamma(tr, tp, delta=2.0):
    # coincidence factor Γ（±2 ms 以内を一致とみなす。採点と同じ定義の簡易版）
    if len(tr) == 0 and len(tp) == 0:
        return 1.0
    T = 0.0
    used = np.zeros(len(tp), bool)
    nc = 0
    for s in tr:
        k = np.where((~used) & (np.abs(tp - s) <= delta))[0]
        if len(k):
            used[k[0]] = True
            nc += 1
    return nc, len(tr), len(tp)


src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'test_traces.npz')
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'fig_c_iv.png')
d = np.load(src)
names = list(d['names'])
n = len(names)
ncol = min(n, 4)
nrow = (n + ncol - 1) // ncol
fig = plt.figure(figsize=(4.3 * ncol, 5.0 * nrow + 0.4))
# 段ごとに [I, V, 空き] の 3 行。空き行で、下の段のタイトルと上の段の軸ラベルが重ならないようにする
hr = ([1, 2.2, 0.55] * nrow)[:-1]
gs = fig.add_gridspec(len(hr), ncol, height_ratios=hr, hspace=0.12, wspace=0.18)
axs = np.empty((2 * nrow, ncol), dtype=object)
for g in range(nrow):
    for c in range(ncol):
        axs[2 * g, c] = fig.add_subplot(gs[3 * g, c])
        axs[2 * g + 1, c] = fig.add_subplot(gs[3 * g + 1, c], sharex=axs[2 * g, c])
        axs[2 * g, c].tick_params(labelbottom=False)
for j, name in enumerate(names):
    r, c = 2 * (j // ncol), j % ncol
    I, vh, vc = d[f'I{j}'], d[f'hh{j}'], d[f'C{j}']
    t = np.arange(len(vh)) * DT
    nc, nh, np_ = gamma(spikes(vh), spikes(vc))
    axs[r, c].plot(t, I, color='#666666', lw=0.9)
    axs[r, c].set_title(name, fontsize=11)
    axs[r + 1, c].plot(t, vh, color=HH_COL, lw=1.4, label='HH（正解）')
    axs[r + 1, c].plot(t, vc, color=C_COL, lw=1.4, ls=(0, (3, 2)), label='C MLP 3 個')
    axs[r + 1, c].set_ylim(-85, 62)
    axs[r + 1, c].text(0.99, 0.99, f'スパイク HH {nh} / C {np_}（一致 {nc}）', transform=axs[r + 1, c].transAxes,
                       ha='right', va='top', fontsize=9.5, color='#333333')
    axs[r + 1, c].set_xlabel('時刻 $t$ [ms]')
for r in range(0, 2 * nrow, 2):
    axs[r, 0].set_ylabel('$I$ [µA/cm²]')
    axs[r + 1, 0].set_ylabel('$V$ [mV]')
h, l = axs[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper right', ncol=2, fontsize=11, frameon=False)
top_in = 0.75  # 凡例の分の余白 [inch]
H = fig.get_figheight()
fig.subplots_adjust(left=0.06, right=0.99, top=1 - top_in / H, bottom=0.55 / H)
fig.savefig(dst, dpi=140)
print('saved', dst)
