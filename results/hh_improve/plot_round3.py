# 第3ラウンドの図: 検証用 17 電流での推移と、検証で選んだ時点の 26 本テストの成績
import os, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
D = os.path.dirname(os.path.abspath(__file__))
def val_curve(tag):
    pts = []
    for line in open(os.path.join(D, 'logs', tag + '.log'), encoding='utf-8', errors='ignore'):
        m = re.match(r'\[\S+ it (\d+) val\] 100%: (\d+)/17', line)
        if m:
            pts.append((int(m.group(1)), int(m.group(2))))
    return pts
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.2, 1]})
for tag, lab, c in [('k_3_w3', 'RNN₃のみ, 閾値付近×3', '#9bbb59'), ('k_123_w3', '3ブロック, 閾値付近×3', '#4f81bd'),
                    ('k_123_w1', '3ブロック, 重みなし', '#c0504d')]:
    p = val_curve(tag)
    ax[0].plot([x for x, _ in p], [y for _, y in p], 'o-', lw=2, ms=6, color=c, label=lab)
for x0, x1, K in [(0, 200, '1 ms'), (200, 400, '5 ms'), (400, 600, '20 ms')]:
    ax[0].axvspan(x0, x1, color='#000000', alpha=0.04 if K == '5 ms' else 0)
    ax[0].text((x0 + x1) / 2, 14.6, 'K=' + K, ha='center', fontsize=9, color='#555555')
ax[0].axhline(11, color='k', lw=1, ls='--', label='基準 11')
ax[0].set_xlim(-10, 610); ax[0].set_ylim(8, 15)
ax[0].set_xlabel('微調整の更新回数（K=2000 の段階は 500 回目で中断）'); ax[0].set_ylabel('100% 一致（検証 17 本中）')
ax[0].set_title('検証用 17 電流での推移'); ax[0].legend(fontsize=9, loc='lower left')
names = ['基準', '第2R 最高\n(26本で選択)', '3ブロック\n(検証で選択)', 'RNN₃のみ\n(検証で選択)']
vals = [19, 21, 21, 22]
cols = ['#888888', '#cccccc', '#4f81bd', '#9bbb59']
b = ax[1].bar(names, vals, color=cols, width=0.6)
for r, v in zip(b, vals):
    ax[1].text(r.get_x() + r.get_width() / 2, v + 0.2, f'{v}/26', ha='center', fontsize=10)
ax[1].set_ylim(15, 24); ax[1].set_ylabel('100% 一致（テスト 26 本中）')
ax[1].set_title('26 本テストの成績（第2R 最高は参考値）')
fig.tight_layout(); fig.savefig(os.path.join(D, 'round3.png'), dpi=130)
