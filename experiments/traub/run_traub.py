"""
Traub モデル（19 コンパートメント）に細胞体から電流を入れ、膜電位を描く。

snsbook の図 3.20 を再現する:
    a: 細胞体に 1e-4 uA を流し続ける（200 ms）→ 細胞体のバースト発火
    b: 1000 ms から 1 ms だけ 1e-2 uA を入れる（2000 ms）→ 樹状突起への伝わり方

実行方法（リポジトリのルートから）:
    uv run python experiments/traub/run_traub.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from neuron.traub import simulate, NC, SOMA, area

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'traub')


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)
    dt = 0.01

    # 図 3.20a: 一定電流
    t = np.arange(0, 200.0, dt)                       # shape (20000,)
    I_inj = np.zeros((len(t), NC))                     # shape (20000, 19)
    I_inj[:, SOMA] = 1.e-4 / area[SOMA]
    V_hist, _ = simulate(t, I_inj)

    plt.figure(figsize=(8, 3))
    plt.plot(t, V_hist[:, SOMA])
    plt.xlabel('t [ms]')
    plt.ylabel('V [mV]')
    plt.title('soma (compartment 8), I = 1e-4 uA')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'fig3_20a.png'), dpi=150)

    # 図 3.20b: 1000 ms から 1 ms だけパルス
    t = np.arange(0, 2000.0, dt)
    I_inj = np.zeros((len(t), NC))
    I_inj[(t >= 1000) & (t < 1001), SOMA] = 1.e-2 / area[SOMA]
    V_hist, _ = simulate(t, I_inj)

    plt.figure(figsize=(8, 3))
    mask = (t >= 1000) & (t <= 1050)
    for i in [7, 10, 13, 16]:  # gnuplot の列 9, 12, 15, 18（列 1 が t なので i = 列 - 2）
        plt.plot(t[mask], V_hist[mask, i], label=f'compartment {i}')
    plt.xlabel('t [ms]')
    plt.ylabel('V [mV]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'fig3_20b.png'), dpi=150)
    plt.show()
