"""
OU 電流の波形を図にする（説明用）。

  (a) 振幅 σ の意味: OU 7±5（τ = 5 ms）に mean ± σ、mean ± 2σ の帯を重ねる
  (b) 相関時間 τ の意味: 同じ乱数列（同じ seed）で τ だけ変えた 4 本（σ = 3 で共通）
  (c) 学習データの電流: level_ou_current（基準 U[0,20] を 100〜1000 ms 保つ + OU σ~U[0,4]、τ = 1 ms）

実行:
    uv run python -X utf8 results/ou_examples/plot_ou_waveforms.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from neuron.current import ou_current, level_ou_current   # noqa: E402

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

DT = 0.01                       # [ms]
BLUE = '#2a78d6'
BAND1, BAND2 = '#b7d3f6', '#e3eefc'
INK2, GRID = '#5f5e5a', '#e6e5e0'
OUT = Path(__file__).with_name('ou_waveforms.png')


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#b4b2a9')


fig = plt.figure(figsize=(11, 10.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.0, 1.0], hspace=0.55, wspace=0.25)

# (a) 振幅 σ ---------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
t = np.arange(0, 500, DT)
m, s, tau = 7.0, 5.0, 5.0
I = ou_current(t, mean=m, sigma=s, tau=tau, seed=9)
ax.axhspan(m - 2 * s, m + 2 * s, color=BAND2, lw=0)
ax.axhspan(m - s, m + s, color=BAND1, lw=0)
ax.axhline(m, color=INK2, lw=1, ls='--')
ax.plot(t, I, color=BLUE, lw=1.0)
for y, lab in [(m, 'mean = 7'), (m + s, '+σ = 12'), (m - s, '−σ = 2'),
               (m + 2 * s, '+2σ = 17'), (m - 2 * s, '−2σ = −3')]:
    ax.text(503, y, lab, va='center', fontsize=9, color=INK2)
ax.set_xlim(0, 500)
ax.set_ylabel('I [µA/cm²]')
ax.set_xlabel('時刻 [ms]')
ax.set_title('(a) 振幅 ＝ σ：OU 7±5（τ = 5 ms）。濃い帯 ±σ に約 68%、薄い帯 ±2σ に約 95% の時間が入る',
             loc='left', fontsize=11)
style(ax)

# (b) 相関時間 τ ------------------------------------------------------------
t = np.arange(0, 200, DT)
axes = []
for i, tau in enumerate([0.1, 1.0, 5.0, 10.0]):
    ax = fig.add_subplot(gs[1, i], sharey=axes[0] if axes else None)
    I = ou_current(t, mean=7.0, sigma=3.0, tau=tau, seed=3)     # 同じ seed → 同じ乱数列
    ax.axhline(7.0, color=INK2, lw=1, ls='--')
    ax.plot(t, I, color=BLUE, lw=0.8 if tau < 1 else 1.0)
    fc = 1.0 / (2 * np.pi * tau * 1e-3)
    ax.set_title('τ = {:g} ms\n折れ点 f_c ≈ {:,.0f} Hz'.format(tau, fc), fontsize=10)
    ax.set_xlim(0, 200)
    ax.set_xlabel('時刻 [ms]')
    if i == 0:
        ax.set_ylabel('I [µA/cm²]')
    else:
        ax.tick_params(labelleft=False)
    style(ax)
    axes.append(ax)
axes[0].set_ylim(-4, 18)
fig.text(0.125, axes[0].get_position().y1 + 0.045,
         '(b) 波長の代わり ＝ τ：mean 7・σ 3 は共通、τ だけ変える。τ が大きいほど 1 回の揺れが長く続く',
         fontsize=11)

# (c) 学習データ -------------------------------------------------------------
ax = fig.add_subplot(gs[2, :])
t = np.arange(0, 3000, DT)
I = level_ou_current(t, seed=1)
ax.plot(t, I, color=BLUE, lw=0.6)
ax.set_xlim(0, 3000)
ax.set_ylabel('I [µA/cm²]')
ax.set_xlabel('時刻 [ms]')
ax.set_title('(c) 学習データ ou（level_ou_current）：基準 U[0, 20] を 100〜1000 ms 保ち、区間ごとの σ ~ U[0, 4]、τ = 1 ms',
             loc='left', fontsize=11)
style(ax)

fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(OUT)
