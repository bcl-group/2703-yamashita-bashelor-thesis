"""
C（MLP 3 個、m3_hdf_*）の学習曲線。損失（3 個の MLP それぞれ）と、検証の Γ の推移を描く。
    .venv/Scripts/python.exe -X utf8 results/hh_improve/plot_m3_loss.py [タグ]
入力:
    results/hh_improve/logs/<タグ>.log   学習の標準出力（学内サーバから持ってくる）
        "pre  N loss RNN1 a RNN2 b RNN3 c" の行（500 iter ごと。名前は RNN だが中身は MLP 1〜3）
        "DAgger k: ..." の行（DAgger の追加の位置）
    results/hh_improve/<タグ>.json        eval: [iter, 検証の 100% 一致の本数, 平均一致率, Γ]
出力: results/hh_improve/fig_<タグ>_loss.png
"""
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else 'm3_hdf_s0'
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})

it, L = [], []
dag = []          # DAgger で状態を足した時点（直前の pre の iter）
ev_log = []       # ログの検証の行: [iter, 100% 一致の本数, 平均一致率, Γ]
for line in open(os.path.join(HERE, 'logs', tag + '.log'), encoding='utf-8', errors='replace'):
    m = re.match(r'pre\s+(\d+) loss RNN1 (\S+) RNN2 (\S+) RNN3 (\S+)', line)
    v = re.match(r'\[\S+ it (\d+) val\] 100%: (\d+)/\d+\s+平均 (\S+)\s+Γ (\S+)', line)
    if v:
        ev_log.append([float(x) for x in v.groups()])
    elif m:
        it.append(int(m.group(1)))
        L.append([float(m.group(k)) for k in (2, 3, 4)])
    elif line.startswith('DAgger'):
        dag.append(it[-1] if it else 0)
it, L = np.array(it), np.array(L)
jp = os.path.join(HERE, tag + '.json')
# 学習中は JSON がまだないので、ログの検証の行を使う
ev = np.array(json.load(open(jp, encoding='utf-8'))['eval']) if os.path.exists(jp) else np.array(ev_log).reshape(-1, 4)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})
cols = ['#2a78d6', '#e8673a', '#1baf7a']
labs = ['MLP 1', 'MLP 2', 'MLP 3']
for k in range(3):
    # 1 点ずつはミニバッチ 1 回の値でばらつくので、移動平均（5 点 = 2500 iter）も重ねる
    a1.plot(it, L[:, k], color=cols[k], lw=0.6, alpha=0.35)
    w = 5
    if len(it) >= w:
        sm = np.convolve(L[:, k], np.ones(w) / w, mode='valid')
        a1.plot(it[w - 1:], sm, color=cols[k], lw=1.8, label=labs[k])
a1.set_yscale('log')
# 6 本（SiLU・ReLU × seed 0〜2）で比べるので、軸の範囲は固定する
a1.set_ylim(1e-9, 1)
a1.set_xlim(0, 112000)
a1.set_title(tag, loc='left', fontsize=12)
a1.set_ylabel('損失（学習データ）')
a1.legend(frameon=False, fontsize=10)
for x in dag:
    for ax in (a1, a2):
        ax.axvline(x, color='#999999', lw=0.8, ls='--')
if dag:
    a1.text(dag[0], a1.get_ylim()[1], ' DAgger', va='top', fontsize=10, color='#777777')
a2.plot(ev[:, 0], ev[:, 3], 'o-', color='#222222', label='検証 Γ')
for x, n, g in zip(ev[:, 0], ev[:, 1], ev[:, 3]):
    a2.annotate(f'{int(n)}/17', (x, g), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=10)
a2.set_ylim(0.6, 1.05)
a2.set_ylabel('検証 Γ')
a2.set_xlabel('iteration')
fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.09)
out = os.path.join(HERE, f'fig_{tag}_loss.png')
fig.savefig(out, dpi=140)
print('saved', out)
