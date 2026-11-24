"""
C（m3_hdf_s0）が外した ramp 2 本の拡大図。発火の始まりと最後を、HH（正解）と重ねて描く。
    .venv/Scripts/python.exe -X utf8 results/final/plot_c_ramp_zoom.py
入力: results/final/c_traces.npz（run_c_traces.py で保存）
出力: results/final/fig_c_ramp_zoom.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})
DT = 0.01  # [ms]
HH_COL, C_COL = '#222222', '#1baf7a'


def spikes(v):
    i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    return i * DT


d = np.load(os.path.join(HERE, 'c_traces.npz'))
names = list(d['names'])
picks = [names.index('tracking: ramp 0-21.3 in 577ms'), names.index('tracking: ramp 0-26.4 in 882ms')]

fig, axs = plt.subplots(2, 2, figsize=(13, 7.2), gridspec_kw={'hspace': 0.5, 'wspace': 0.15})
for r, j in enumerate(picks):
    vh, vc, I = d[f'hh{j}'], d[f'C{j}'], d[f'I{j}']
    t = np.arange(len(vh)) * DT
    sh, sc = spikes(vh), spikes(vc)
    n = min(len(sh), len(sc))
    diff = sc[:n] - sh[:n]
    wins = [(sh[0] - 25, sh[0] + 45, f'発火の始まり：C は {diff[0]:.1f} ms 遅れる'),
            (t[-1] - 70, t[-1], f'最後の 70 ms：遅れは {diff[-1]:.1f} ms のまま')]
    for c, (t0, t1, sub) in enumerate(wins):
        ax = axs[r, c]
        s = (t >= t0) & (t <= t1)
        ax.plot(t[s], vh[s], color=HH_COL, lw=1.6, label='HH（正解）')
        ax.plot(t[s], vc[s], color=C_COL, lw=1.6, ls=(0, (3, 2)), label='C MLP 3 個')
        # 対応するスパイクの組を矢印で結ぶ（ずれの向きと大きさ）
        for k in range(n):
            if t0 <= sh[k] <= t1 and t0 <= sc[k] <= t1:
                ax.annotate('', xy=(sc[k], 48), xytext=(sh[k], 48),
                            arrowprops=dict(arrowstyle='->', color='#d0453a', lw=1.3))
        ax.set_xlim(t0, t1)
        ax.set_ylim(-85, 58)
        ax.set_title(f'{names[j]}\n{sub}', fontsize=12, loc='left')
        ax.set_xlabel('時刻 $t$ [ms]')
        # 入力電流は右軸に薄く
        ax2 = ax.twinx()
        ax2.plot(t[s], I[s], color='#999999', lw=1.0, ls=':')
        ax2.set_ylim(0, 40)
        ax2.spines['top'].set_visible(False)
        if c == 1:
            ax2.set_ylabel('$I$ [µA/cm²]（点線）', color='#777777')
        else:
            ax2.set_yticklabels([])
        ax2.tick_params(colors='#777777')
    axs[r, 0].set_ylabel('$V$ [mV]')
h, l = axs[0, 0].get_legend_handles_labels()
h.append(plt.Line2D([], [], color='#d0453a', lw=1.3))
l.append('HH → C のスパイク時刻のずれ')
fig.legend(h, l, loc='upper right', ncol=3, frameon=False, fontsize=11)
fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.08)
fig.savefig(os.path.join(HERE, 'fig_c_ramp_zoom.png'), dpi=140)
print('saved')
