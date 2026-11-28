"""
C（MLP 3 個）の SiLU と ReLU を seed 0〜2 で並べた学習曲線（6 本を 1 枚に）。
    .venv/Scripts/python.exe -X utf8 results/hh_improve/plot_m3_loss_compare.py
入力: results/hh_improve/logs/<タグ>.log, <タグ>.json（plot_m3_loss.py と同じ）
出力: results/hh_improve/fig_m3_act_loss_compare.png
    上 2 行: 損失（行 = 活性化関数、列 = seed。線は 5 点の移動平均）
    下 1 行: 検証 Γ（SiLU と ReLU を同じパネルに重ねる）
"""
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
ACTS = [('SiLU', 'm3_hdf_s{}', '#2a78d6'), ('ReLU', 'm3_hdf_relu_s{}', '#d0453a')]
MLP_COLS = ['#2a78d6', '#e8673a', '#1baf7a']


def load(tag):
    it, L, dag = [], [], []
    for line in open(os.path.join(HERE, 'logs', tag + '.log'), encoding='utf-8', errors='replace'):
        m = re.match(r'pre\s+(\d+) loss RNN1 (\S+) RNN2 (\S+) RNN3 (\S+)', line)
        if m:
            it.append(int(m.group(1)))
            L.append([float(m.group(k)) for k in (2, 3, 4)])
        elif line.startswith('DAgger'):
            dag.append(it[-1] if it else 0)
    d = json.load(open(os.path.join(HERE, tag + '.json'), encoding='utf-8'))
    return np.array(it), np.array(L), dag, np.array(d['eval']), d


fig, axs = plt.subplots(3, 3, figsize=(15, 9.5), sharex=True,
                        gridspec_kw={'height_ratios': [1.3, 1.3, 1], 'hspace': 0.35, 'wspace': 0.12})
w = 5
for s in range(3):
    for r, (act, fmt, col) in enumerate(ACTS):
        tag = fmt.format(s)
        it, L, dag, ev, d = load(tag)
        ax = axs[r, s]
        for k in range(3):
            sm = np.convolve(L[:, k], np.ones(w) / w, mode='valid')
            ax.plot(it[w - 1:], sm, color=MLP_COLS[k], lw=1.5, label=f'MLP {k + 1}')
        ax.set_yscale('log')
        ax.set_ylim(1e-9, 1)
        for x in dag:
            ax.axvline(x, color='#999999', lw=0.8, ls='--')
            axs[2, s].axvline(x, color='#999999', lw=0.8, ls='--')
        b = d['best']
        ax.set_title(f'{act}  seed {s}（検証で選んだ iter {b[2] // 1000}k）', loc='left', fontsize=11)
        if s == 0:
            ax.set_ylabel(f'{act}\n損失（学習データ）')
        else:
            ax.set_yticklabels([])
        a2 = axs[2, s]
        a2.plot(ev[:, 0], ev[:, 3], 'o-', color=col, ms=4, lw=1.3, label=act)
        a2.plot(b[2], [e[3] for e in ev if e[0] == b[2]][-1], 's', color=col, ms=9, mfc='none', mew=1.6)
    a2 = axs[2, s]
    a2.set_ylim(0.6, 1.02)
    a2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    a2.set_xlim(0, 112000)
    a2.set_xticks([0, 25000, 50000, 75000, 100000])
    a2.set_xticklabels(['0', '25k', '50k', '75k', '100k'])
    a2.set_xlabel('iteration')
    a2.set_title(f'検証 Γ  seed {s}（□ = 選んだ iter）', loc='left', fontsize=11)
    if s == 0:
        a2.set_ylabel('検証 Γ')
    else:
        a2.set_yticklabels([])
h1, l1 = axs[0, 0].get_legend_handles_labels()
h2, l2 = axs[2, 0].get_legend_handles_labels()
h1.append(plt.Line2D([], [], color='#999999', lw=0.8, ls='--'))
l1.append('DAgger')
fig.legend(h1 + h2, l1 + l2, loc='upper right', ncol=6, frameon=False, fontsize=11)
fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.07)
out = os.path.join(HERE, 'fig_m3_act_loss_compare.png')
fig.savefig(out, dpi=130)
print('saved', out)
