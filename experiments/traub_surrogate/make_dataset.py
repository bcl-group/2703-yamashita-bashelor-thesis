"""
Traub モデル（19 コンパートメント）の分割型代理モデル用の教師データを作る。

HH の分割型（experiments/hh_surrogate/hh_modular.py）と同じく、陽的オイラー法（dt = 0.01 ms）の
1 ステップを 1:1 で置き換えるので、全コンパートメントの全状態変数を毎ステップ保存する。

入力電流（細胞体だけ）:
    I(t) = clip( level(t) + sigma(t) * z(t), i_min, i_max )   [nA]
    neuron.current.level_ou_current と同じ作り方（基準レベルを hold ms 保ち、OU の揺らぎを足す）。
    Traub は 0.1〜0.3 nA でゆっくりしたバースト、0.8〜1.2 nA で反復発火する。

初期状態:
    initial_state() から、軌道ごとにランダムな長さ（0〜warmup_max ms）だけ同じ作り方の電流で
    計算を進め、その終わりの状態から記録を始める。K_AHP の q や Ca の χ のような遅い変数も
    軌道ごとにばらつかせるため。

保存するファイル（data/traub/teacher_modular.npz）:
    t       : 時刻 [ms]。shape (N+1,)
    I_nA    : 細胞体への電流 [nA]。shape (n_traj, N+1)。I_nA[:, k] は t[k] からの 1 ステップに使う
    V, xi, m, h, s, r, n, a, b, c, q : 状態。shape (n_traj, N+1, NC)、float64。
              [:, k] は t[k] の値（更新前）。k = N は最後の更新後の値
    warmup_ms : 助走の長さ [ms]。shape (n_traj,)
    meta    : 条件の JSON 文字列

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/traub_surrogate/make_dataset.py
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from neuron.traub import step, initial_state, compartment_conductance, NC, SOMA, area
from neuron.current import level_ou_current

STATE_NAMES = ['V', 'xi', 'm', 'h', 's', 'r', 'n', 'a', 'b', 'c', 'q']
# 1 nA = 1e-3 uA。traub.step() は電流密度 [uA/cm^2] を受け取るので細胞体の面積で割る
NA_TO_DENSITY = 1e-3 / area[SOMA]

CFG = dict(
    name='teacher_modular',
    n_traj=12, T=150.0, dt=0.01, warmup_max=1000.0, seed=2026,
    # 電流 [nA]
    level_min=-0.2, level_max=1.2, hold_min=20.0, hold_max=150.0,
    sigma_max=0.2, tau=1.0, i_min=-0.4, i_max=1.5,
)


def make_currents(rng, n_traj, n_steps, cfg):
    t = np.arange(n_steps) * cfg['dt']
    return np.stack([
        level_ou_current(t, cfg['level_min'], cfg['level_max'], cfg['hold_min'],
                         cfg['hold_max'], cfg['sigma_max'], cfg['tau'],
                         cfg['i_min'], cfg['i_max'], seed=int(rng.integers(2**31)))
        for _ in range(n_traj)])


def to_inj(I_nA):
    # (B,) [nA] -> (B, NC) [uA/cm^2]。細胞体以外は 0
    inj = np.zeros((len(I_nA), NC))
    inj[:, SOMA] = I_nA * NA_TO_DENSITY
    return inj


def main():
    cfg = CFG
    rng = np.random.default_rng(cfg['seed'])
    dt = cfg['dt']
    g_comp = compartment_conductance()
    B = cfg['n_traj']
    N = int(round(cfg['T'] / dt))
    start = time.time()

    # 1. 助走：長さを軌道ごとに変える。まだ助走中の軌道だけ状態を進める
    n_warm = (rng.uniform(0, cfg['warmup_max'], size=B) / dt).astype(int)
    I_warm = make_currents(rng, B, max(n_warm.max(), 1), cfg)
    state = initial_state((B,))
    for k in range(n_warm.max()):
        new = step(state, to_inj(I_warm[:, k]), dt, g_comp)
        active = k < n_warm
        state = {name: np.where(active[:, None], new[name], state[name]) for name in state}
    print('warmup done: {:.0f} s'.format(time.time() - start))

    # 2. 本番：全状態を毎ステップ保存する
    I_nA = make_currents(rng, B, N + 1, cfg)
    hist = {name: np.empty((B, N + 1, NC)) for name in STATE_NAMES}
    for k in range(N + 1):
        for name in STATE_NAMES:
            hist[name][:, k] = state[name]
        if k < N:
            state = step(state, to_inj(I_nA[:, k]), dt, g_comp)
    print('simulate done: {:.0f} s'.format(time.time() - start))

    t = np.arange(N + 1) * dt
    out = os.path.join(ROOT, 'data', 'traub', cfg['name'] + '.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, t=t, I_nA=I_nA, warmup_ms=n_warm * dt, meta=json.dumps(cfg), **hist)
    print('saved:', os.path.normpath(out),
          '({:.0f} MB)'.format(os.path.getsize(out) / 2**20))

    # 3. 確認用の図：電流と細胞体・樹状突起の膜電位
    fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    for k, ax in enumerate(axes.T.ravel()):
        if k >= B:
            break
        ax.plot(t, hist['V'][k, :, SOMA], color='#2a78d6', lw=0.8, label='細胞体 (8)')
        ax.plot(t, hist['V'][k, :, 13], color='#e87ba4', lw=0.8, label='尖端樹状突起 (13)')
        ax2 = ax.twinx()
        ax2.plot(t, I_nA[k], color='#888780', lw=0.6)
        ax2.set_ylim(-1, 6)
        ax2.set_yticks([])
        ax.set_title('軌道 {}（助走 {:.0f} ms）'.format(k, n_warm[k] * dt), fontsize=9)
        ax.set_ylabel('V [mV]')
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('時刻 [ms]')
    fig.suptitle('Traub の教師データ（灰色は細胞体への電流、右軸は表示しない）')
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    fig.tight_layout()
    fig.savefig(out.replace('.npz', '.png'), dpi=110)
    print('saved:', os.path.normpath(out.replace('.npz', '.png')))


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    main()
