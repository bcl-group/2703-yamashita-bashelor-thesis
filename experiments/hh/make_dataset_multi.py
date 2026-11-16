"""
軌道を複数本まとめて作る HH の教師データ。dt = 0.01 ms の陽的オイラー法（src/neuron/hh.py と同じ式）。

作れるデータ（引数で選ぶ）:
    multi : Learning flow functions of spiking systems（Aguiar ら, L4DC 2024）§4.1 の作り方を移したもの。
            800 本 x 100 ms、初期状態ランダム、10 ms ごとの階段電流（teacher_thesis と同じ分布）
    ou    : Traub の教師データ（隣の段階A）と同じ「ゆっくり変わる基準レベル + 揺らぎ」の電流で、
            揺らぎを OU 過程（相関時間 1 ms）にしたもの。100 本 x 1000 ms、初期状態ランダム
    near  : 発火の閾値付近（I≈2.3 の 1 発、I≈6〜10 の反復発火の始まり）を多く含む電流。基準レベルを
            折れ線にしてゆっくりしたランプも入れる。100 本 x 1000 ms、初期状態ランダム

    | 項目       | 論文 §4.1                          | multi                     | ou                                    |
    |------------|------------------------------------|---------------------------|---------------------------------------|
    | 軌道       | 800 本 x 500 ms                    | 800 本 x 100 ms           | 100 本 x 1000 ms                      |
    | 初期状態   | V ~ U[-100, 100] mV, m,n ~ U[0, 1] | V ~ U[-90, 50] mV, m,h,n ~ U[0, 1] | 同左                         |
    | 入力       | 100 ms ごとに U[0, 1] uA           | 10 ms ごとに U[0, 20]     | 基準 U[0, 20] を 100〜1000 ms 保つ + OU(σ ~ U[0, 4], τ = 1 ms)、[-5, 30] に収める |
    | 1ステップ  | Δ = 10 ms                          | dt = 0.01 ms              | dt = 0.01 ms                          |

どちらも、各時刻の I(t) を入れて x(t+dt) を出す、HH のオイラー法と同じ1:1の入出力になる。
初期状態をランダムにするのは、閉ループで予測がずれたときに入る「1本の軌道の上にない状態」も
学習データに入れるためである。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh/make_dataset_multi.py ou
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from neuron import hh
from neuron.hh import alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
from neuron.current import random_pulse_current, level_ou_current, knot_ou_current

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'hh')

DT = 0.01
V0_RANGE = (-90.0, 50.0)

PRESETS = {
    'multi': {
        'file': 'teacher_multi.npz', 'n_traj': 800, 'T': 100.0, 'seed': 2024,
        'note': 'Learning flow functions §4.1 の作り方（軌道800本・初期状態ランダム）を dt=0.01 ms の HH に移したもの',
        'current': (random_pulse_current, {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0}),
    },
    'ou': {
        'file': 'teacher_ou.npz', 'n_traj': 100, 'T': 1000.0, 'seed': 2025,
        'note': '基準レベル（U[0,20] を 100〜1000 ms 保つ）+ OU の揺らぎ（σ~U[0,4], τ=1 ms）。初期状態ランダム',
        'current': (level_ou_current, {'level_min': 0.0, 'level_max': 20.0,
                                       'hold_min': 100.0, 'hold_max': 1000.0,
                                       'sigma_max': 4.0, 'tau': 1.0,
                                       'i_min': -5.0, 'i_max': 30.0}),
    },
    # 発火の閾値付近を多く含める。HH（V_REST から一定電流）は I≈2.3 で 1 発、I≈6.0〜6.4 で反復発火に
    # 切り替わり、ゆっくり増やすと I≈9.8（ホップ分岐）付近まで静止状態にとどまる。
    # 基準レベルを折れ線（20〜500 ms おきの節点を直線でつなぐ）にして、ゆっくりしたランプも含める
    'near': {
        'file': 'teacher_near.npz', 'n_traj': 100, 'T': 1000.0, 'seed': 2026,
        'note': '閾値付近を多く含む折れ線の基準レベル（節点 20〜500 ms おき）+ OU の揺らぎ（σ~U[0,1.5]）。初期状態ランダム',
        'current': (knot_ou_current, {'ranges': ((1.5, 3.5, 0.3), (5.0, 8.0, 0.4),
                                                 (8.0, 11.0, 0.15), (0.0, 20.0, 0.15)),
                                      'knot_min': 20.0, 'knot_max': 500.0,
                                      'sigma_max': 1.5, 'tau': 1.0,
                                      'i_min': -5.0, 'i_max': 30.0}),
    },
}


def simulate_batch(I, V0, m0, h0, n0, dt=DT):
    """
    hh.simulate と同じ陽的オイラー法を、軌道の本数ぶんまとめて計算する。

    Args:
        I (np.ndarray): 入力電流。shape (B, N)。
        V0, m0, h0, n0 (np.ndarray): 初期状態。shape (B,)。

    Returns:
        tuple: V, m, h, n。どれも shape (B, N)。添字 [:, k] が時刻 k*dt の値。
    """
    B, N = I.shape
    V, m, h, n = (np.array(a, dtype=float) for a in (V0, m0, h0, n0))
    out = np.empty((4, B, N))
    for k in range(N):
        out[0, :, k], out[1, :, k], out[2, :, k], out[3, :, k] = V, m, h, n
        I_Na = hh.g_Na * m ** 3 * h * (V - hh.E_Na)
        I_K = hh.g_K * n ** 4 * (V - hh.E_K)
        I_L = hh.g_L * (V - hh.E_L)
        dV = (I[:, k] - I_Na - I_K - I_L) / hh.C_m
        dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n
        V = V + dV * dt
        m = m + dm * dt
        h = h + dh * dt
        n = n + dn * dt
    return out[0], out[1], out[2], out[3]


def plot_examples(t, I, V, path, n_show=4):
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.0 * n_show), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, V[i], color='#2a78d6', linewidth=0.6)
        ax.set_ylabel('V [mV]', fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, I[i], color='#eb6834', linewidth=0.5, alpha=0.7)
        ax2.set_ylabel('I [uA/cm2]', fontsize=8)
        ax.set_title('trajectory #{}'.format(i), fontsize=9)
    axes[-1].set_xlabel('Time [ms]')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(name):
    cfg = PRESETS[name]
    rng = np.random.default_rng(cfg['seed'])
    N = int(round(cfg['T'] / DT))
    t = np.arange(N) * DT
    n_traj = cfg['n_traj']

    # 入力は軌道ごとに別の seed で作る。dt = 0.01 ms の各時刻に I(t) が1つずつ対応する
    func, params = cfg['current']
    I = np.stack([func(t, seed=cfg['seed'] + 1 + i, **params) for i in range(n_traj)])

    V0 = rng.uniform(*V0_RANGE, size=n_traj)
    m0, h0, n0 = rng.uniform(0.0, 1.0, size=(3, n_traj))

    start = time.time()
    V, m, h, n = simulate_batch(I, V0, m0, h0, n0)
    elapsed = time.time() - start

    ok = np.all(np.isfinite(V), axis=1) & (np.abs(V).max(axis=1) < 1000)
    if not ok.all():
        print(f'発散した軌道を除く: {np.sum(~ok)} 本')
    I, V, m, h, n = I[ok], V[ok], m[ok], h[ok], n[ok]

    n_spikes = np.sum((V[:, :-1] < 0.0) & (V[:, 1:] >= 0.0), axis=1)
    meta = {
        'note': cfg['note'], 'dt_ms': DT, 'T_ms': cfg['T'], 'n_traj': int(ok.sum()),
        'seed': cfg['seed'], 'V0_range': V0_RANGE, 'gate0_range': (0.0, 1.0),
        'current': dict(params, kind=func.__name__),
        'spikes_per_traj_mean': float(n_spikes.mean()),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, cfg['file'])
    np.savez_compressed(path, t=t, I_ext=I, V=V, m=m, h=h, n=n,
                        meta=json.dumps(meta, ensure_ascii=False))
    png = path.replace('.npz', '.png')
    plot_examples(t, I, V, png)
    print(f'軌道 {ok.sum()} 本 x {N} ステップ、計算 {elapsed:.1f} 秒')
    print(f'スパイク数/本: 平均 {n_spikes.mean():.2f}, 最小 {n_spikes.min()}, 最大 {n_spikes.max()}')
    print(f'V の範囲: {V.min():.1f} 〜 {V.max():.1f} mV、I の範囲: {I.min():.1f} 〜 {I.max():.1f}')
    print('保存:', os.path.normpath(path), f'({os.path.getsize(path) / 1024 ** 2:.0f} MB)')
    print('図  :', os.path.normpath(png))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ou')
