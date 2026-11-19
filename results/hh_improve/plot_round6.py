# 第6ラウンドの図: Markov MLP（事前学習のみ）の変種の検証 17 本・テスト 26 本と、これまでの最良との比較
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
D = os.path.dirname(os.path.abspath(__file__))
rows = [  # (名前, 検証 17 本, テスト 26 本, 色)
    ('RNN 3 個 基準', 11, 19, '#888888'),
    ('RNN 3 個 閉ループ微調整', 13, 22, '#4f81bd'),
    ('RNN 1 個（再帰）最良', 1, None, '#c0504d'),
    ('式組込み MLP（ハイブリッド）', 17, 23, '#bbbbbb'),
    ('MLP 128×2, 5 万回（標準化誤差）', 12, 20, '#9bbb59'),
    ('MLP 128×2, 相対誤差', 15, 25, '#4e8a2e'),
    ('MLP 128×2, 相対誤差, 摂動 1/4', 12, 22, '#9bbb59'),
    ('MLP 128×2, 相対誤差, 摂動 ×2.5', 12, 20, '#9bbb59'),
    ('MLP 256×3, 相対誤差', 11, 23, '#9bbb59'),
    ('MLP 256×3, 相対誤差, 15 万回', 16, 23, '#9bbb59'),
]
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
y = list(range(len(rows)))[::-1]
for k, (col, tot, title) in enumerate([(1, 17, '検証 17 本'), (2, 26, 'テスト 26 本')]):
    for yi, r in zip(y, rows):
        v = r[col]
        if v is None:
            ax[k].text(0.3, yi, '未評価', va='center', fontsize=9, color='#666666')
            continue
        ax[k].barh(yi, v, color=r[3], height=0.6)
        ax[k].text(v + 0.2, yi, f'{v}/{tot}', va='center', fontsize=9)
    ax[k].set_xlim(0, tot + 3); ax[k].set_title(f'100% 一致（{title}）')
    ax[k].axvline(tot, color='k', lw=0.8, ls=':')
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows])
fig.suptitle('Markov MLP（入力 V,m,h,n,I → 出力 ΔV,Δm,Δh,Δn、再帰なし、事前学習のみ）とこれまでの最良', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(D, 'round6.png'), dpi=130)
