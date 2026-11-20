# 追加テストとテスト 26 本の「電流ごと」の結果から傾向を見る図（results/ext/trends.png）
# 使い方（リポジトリのルートで）: uv run python -X utf8 results/ext/plot_trends.py
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
import hh_bench_ext as hx   # noqa: E402

plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
EXT = os.path.join(ROOT, 'results', 'ext')
GAM = os.path.join(ROOT, 'results', 'gamma')
MODELS = [('hh_modular_ou', 'RNN 3 個 基準', '#888888'), ('k_3_w3', 'RNN 3 個 微調整', '#4f81bd'),
          ('mm_rel', 'Markov MLP', '#4e8a2e'), ('mm_rel_s1', 'Markov MLP（seed 1）', '#9bbb59')]
ext = {t: json.load(open(os.path.join(EXT, t + '.json'), encoding='utf-8')) for t, _, _ in MODELS
       if os.path.exists(os.path.join(EXT, t + '.json'))}

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
amps = hx.FI_AMPS

# (a) f–I 曲線（後半 500 ms の発火率）
a = ax[0, 0]
first = next(iter(ext.values()))
a.plot(amps, [r['rate'][0] for r in first['fi']['rows']], 'k-', lw=3, alpha=0.3, label='正解の HH')
for t, lab, c in MODELS:
    if t in ext:
        a.plot(amps, [r['rate'][1] for r in ext[t]['fi']['rows']], '-', color=c, lw=1.2, label=lab)
a.set_xlabel('定常電流 [µA/cm²]'); a.set_ylabel('発火率 [Hz]（後半 500 ms）')
a.set_title('(a) f–I 曲線: 発火率はどのモデルも正解とほぼ重なる'); a.legend(fontsize=8)

# (b) 定常電流ごとの Γ（1000 ms）
a = ax[0, 1]
for t, lab, c in MODELS:
    if t in ext:
        a.plot(amps, [r['gamma'] for r in ext[t]['fi']['rows']], 'o-', color=c, ms=3, lw=1, label=lab)
a.axvline(6.5, color='k', lw=0.8, ls=':')
a.text(6.7, 0.05, '反復発火の開始 6.5', fontsize=8)
a.set_xlabel('定常電流 [µA/cm²]'); a.set_ylabel('Γ（1000 ms）')
a.set_ylim(-0.1, 1.05)
a.set_title('(b) 定常電流ごとの Γ: 発火率は合っていても時刻がずれていく'); a.legend(fontsize=8)

# (c) seed 10 本ずつ: 系統ごとの Γ（平均 ± 標準偏差）
a = ax[1, 0]
fams = list(first['seeds']['summary']['gamma_by_family'])
w = 0.8 / len(ext)
for k, (t, lab, c) in enumerate([m for m in MODELS if m[0] in ext]):
    g = ext[t]['seeds']['summary']['gamma_by_family']
    a.bar(np.arange(len(fams)) + k * w, [g[f][0] for f in fams], w, yerr=[g[f][1] for f in fams],
          color=c, capsize=2, label=lab)
a.set_xticks(np.arange(len(fams)) + w * (len(ext) - 1) / 2)
a.set_xticklabels([f.replace(' (tau 5ms)', '\n(τ 5 ms)').replace('random ', 'random\n') for f in fams], fontsize=8)
a.set_ylim(0.8, 1.01); a.set_ylabel('Γ（各系統 10 本の平均 ± 標準偏差）')
a.set_title('(c) seed 10 本ずつ: ゆらぐ電流では Markov MLP が上'); a.legend(fontsize=8, loc='lower left')

# (d) テスト 26 本の電流ごとの Γ（モデル × 電流）
a = ax[1, 1]
rows = []
for t, lab, c in MODELS:
    d = json.load(open(os.path.join(GAM, t + '.json'), encoding='utf-8'))
    rows.append([r['gamma'] for r in d['rows']])
    names = [r['name'] for r in d['rows']]
im = a.imshow(np.array(rows), aspect='auto', cmap='viridis', vmin=0, vmax=1)
a.set_yticks(range(len(MODELS))); a.set_yticklabels([m[1] for m in MODELS], fontsize=8)
a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=90, fontsize=7)
plt.colorbar(im, ax=a, fraction=0.03, label='Γ')
a.set_title('(d) テスト 26 本の電流ごとの Γ: 外れるのはランプと定常の閾値付近')

fig.tight_layout()
fig.savefig(os.path.join(EXT, 'trends.png'), dpi=120)
print('saved', os.path.join(EXT, 'trends.png'))
