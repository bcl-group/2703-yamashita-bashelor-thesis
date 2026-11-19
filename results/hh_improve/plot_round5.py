# 第5ラウンドの図: 単一ネットの各変種（検証 17 本の最高）と、26 本テストの比較
import os, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
D = os.path.dirname(os.path.abspath(__file__))
def best_val(tag):
    b = (0, 0.0)
    for line in open(os.path.join(D, 'logs', tag + '.log'), encoding='utf-8', errors='ignore'):
        m = re.match(r'\[\S+ it \d+ val\] 100%: (\d+)/17\s+平均 ([\d.]+)', line)
        if m:
            b = max(b, (int(m.group(1)), float(m.group(2))))
    return b
rows = [('V のみ戻す, tanh RNN', 'cl_v_rnn'), ('V のみ戻す, GRU', 'cl_v_gru'),
        ('V,m,h,n を戻す, tanh RNN', 'cl_s_rnn'), ('V,m,h,n を戻す, RNN H128', 'cl_s_rnn_h128'),
        ('V,m,h,n を戻す, GRU', 'cl_s_gru'), ('V,m,h,n を戻す, 式の損失なし', 'cl_s_rnn_noonpol'),
        ('Markov MLP（再帰なし）', 'cl_m_mlp'), ('電流の式を組み込んだ MLP', 'cl_p_mlp')]
vals = [best_val(t) for _, t in rows]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
y = list(range(len(rows)))[::-1]
cols = ['#c0504d'] * 6 + ['#4f81bd', '#9bbb59']
ax[0].barh(y, [v[1] for v in vals], color=cols, height=0.6)
for yi, (n100, mean) in zip(y, vals):
    ax[0].text(mean + 0.01, yi, f'{n100}/17（平均 {mean:.2f}）', va='center', fontsize=9)
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows])
ax[0].set_xlim(0, 1.25); ax[0].set_xlabel('平均一致率（検証 17 本、最良の時点）')
ax[0].set_title('RNN 1 個型の変種（第5ラウンド、途中で中断）')
names = ['RNN 3 個\n基準', 'RNN 3 個\n閉ループ微調整', 'Markov MLP', '電流の式を\n組み込んだ MLP']
tv = [19, 22, 10, 23]
c2 = ['#888888', '#4f81bd', '#4f81bd', '#9bbb59']
b = ax[1].bar(names, tv, color=c2, width=0.6)
for r, v in zip(b, tv):
    ax[1].text(r.get_x() + r.get_width() / 2, v + 0.3, f'{v}/26', ha='center', fontsize=10)
ax[1].set_ylim(0, 27); ax[1].set_ylabel('100% 一致（テスト 26 本中）')
ax[1].set_title('26 本テスト（検証で選んだ時点）')
fig.tight_layout(); fig.savefig(os.path.join(D, 'round5.png'), dpi=130)
