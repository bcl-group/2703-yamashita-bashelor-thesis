# 第1・2ラウンドの結果の図（26 電流の 100% 一致数）
import json, os, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
D = os.path.dirname(os.path.abspath(__file__))
L = lambda t: json.load(open(os.path.join(D, t + '.json'), encoding='utf-8'))
r1 = [('基準 (H50)', 19), ('H64×2層', 'a3_h64x2'), ('H128', 'a3_h128'), ('積の特徴量', 'a3_h64_feat'),
      ('H64', 'a3_h64'), ('雑音0.01', 'a3_h64_noise001'), ('雑音0.1', 'a3_h64_noise01'),
      ('閾値下×10', 'a3_h64_wsub10'), ('記憶抑制', 'a3_h64_nomem'),
      ('RNN1個 H64', 's_h64'), ('RNN1個 +aux', 's_h64_aux'), ('RNN1個 +leak', 's_h64_aux_leak')]
names, vals, cols = [], [], []
for n, t in r1:
    v = t if isinstance(t, int) else L(t)['closed_loop']['n100']
    names.append(n); vals.append(v)
    cols.append('#888888' if n.startswith('基準') else ('#c0504d' if n.startswith('RNN1個') else '#4f81bd'))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].barh(names[::-1], vals[::-1], color=cols[::-1], height=0.6)
ax[0].axvline(19, color='k', lw=1, ls='--')
for i, v in enumerate(vals[::-1]):
    ax[0].text(v + 0.3, i, str(v), va='center', fontsize=9)
ax[0].set_xlim(0, 26); ax[0].set_xlabel('100% 一致の電流数（26 本中）')
ax[0].set_title('第1ラウンド: 教師強制で 1 ステップ学習（青 RNN3個のRNN₃、赤 RNN1個）')
for tag, lab, c in [('c123_lr1e-6', '3ブロック lr1e-6', '#4f81bd'), ('c3_lr1e-6', 'RNN₃のみ lr1e-6', '#9bbb59'),
                    ('c3_lr1e-5', 'RNN₃のみ lr1e-5', '#c0504d')]:
    e = L(tag)['eval']
    ax[1].plot([x[0] for x in e], [x[1] for x in e], 'o-', lw=2, ms=6, color=c, label=lab)
ax[1].axhline(19, color='k', lw=1, ls='--', label='基準 19')
ax[1].set_xlabel('微調整の更新回数'); ax[1].set_ylabel('100% 一致の電流数')
ax[1].set_ylim(10, 23); ax[1].legend(fontsize=9)
ax[1].set_title('第2ラウンド: 3 ブロックの閉ループ微調整（K=100 = 1 ms）')
fig.tight_layout(); fig.savefig(os.path.join(D, 'round12.png'), dpi=130)
