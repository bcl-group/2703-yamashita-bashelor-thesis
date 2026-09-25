"""
Hodgkin-Huxley モデルのシミュレーション。

このファイルを直接実行すると、ステップ電流を与えたときの膜電位とゲート変数の
時間変化をグラフに表示する。

    python HH.py

他のファイルから simulate() を呼べば、任意の入力電流に対する
V, m, h, n の時系列を取得できる（教師データの作成などに使う）。

    from HH import simulate
    V, m, h, n = simulate(t, I_ext)

膜電位の表記
------------
静止膜電位を -65 mV とする表記を使っている。
Hodgkin と Huxley の原論文や棟近先輩の卒業論文では静止膜電位を 0 mV とする
表記が使われており、レート関数や反転電位の値の表れ方が異なる。
両者は V の原点の取り方が違うだけで等価なモデルである。
"""

import numpy as np
import matplotlib.pyplot as plt

# 入力電流を作る関数は current.py にまとめてある。
# 新しい電流の形を試したいときは current.py に関数を1つ足せばよい。
from current import (
    step_current,
    constant_current,
    ramp_current,
    sin_current,
    pulse_train_current,
    random_pulse_current,
    random_timing_pulse_current,
)

C_m  = 1.0      # 膜容量 (uF/cm^2)
g_Na = 120.0    # ナトリウムの最大コンダクタンス (mS/cm^2)
g_K  = 36.0     # カリウムの最大コンダクタンス (mS/cm^2)
g_L  = 0.3      # 漏れ電流（リーク）の最大コンダクタンス (mS/cm^2)

E_Na = 50.0     # ナトリウムの平衡電位 (mV)
E_K  = -77.0    # カリウムの平衡電位 (mV)
E_L  = -54.387  # 漏れ電流の平衡電位 (mV)

V_REST = -65.0  # 静止膜電位 (mV)。膜電位の初期値に使う


def alpha_n(V):
    if abs(V + 55.0) < 1e-6: return 0.1
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

def alpha_m(V):
    if abs(V + 40.0) < 1e-6: return 1.0
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def steady_state_gates(V):
    """
    ある膜電位における各ゲート変数の定常値を返す。

    ゲート変数の式 dx/dt = alpha(V)(1-x) - beta(V)x は、
    x = alpha / (alpha + beta) のとき dx/dt = 0 となる。
    シミュレーション開始時にゲート変数が動き出さないよう、
    この値を初期値として使う。

    Parameters
    ----------
    V : float
        膜電位 (mV)。

    Returns
    -------
    tuple of float
        (m, h, n) の定常値。
    """
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    return m, h, n


def simulate(t, I_ext, V0=V_REST):
    """
    陽的 Euler 法で Hodgkin-Huxley モデルを解く。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。等間隔であること。shape = (ステップ数,)
    I_ext : numpy.ndarray
        各時刻に与える外部電流 (uA/cm^2)。shape = t.shape
    V0 : float
        膜電位の初期値 (mV)。

    Returns
    -------
    V, m, h, n : numpy.ndarray
        各時刻の膜電位とゲート変数。shape はいずれも t.shape

    Notes
    -----
    ゲート変数の初期値は V0 における定常値とする（steady_state_gates を参照）。
    """
    dt = t[1] - t[0]

    # 初期値
    V = float(V0)
    m, h, n = steady_state_gates(V)

    # 結果を記録する配列
    V_hist = np.zeros(len(t))
    m_hist = np.zeros(len(t))
    h_hist = np.zeros(len(t))
    n_hist = np.zeros(len(t))

    for i in range(len(t)):
        # 現在の値を記録（更新前の値がその時刻の値）
        V_hist[i] = V
        m_hist[i] = m
        h_hist[i] = h
        n_hist[i] = n

        # 各イオン電流の計算
        I_Na = g_Na * (m ** 3) * h * (V - E_Na)
        I_K  = g_K * (n ** 4) * (V - E_K)
        I_L  = g_L * (V - E_L)

        # 微分方程式の右辺
        dV = (I_ext[i] - I_Na - I_K - I_L) / C_m
        dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n

        # Euler 法による更新
        V += dV * dt
        m += dm * dt
        h += dh * dt
        n += dn * dt

    return V_hist, m_hist, h_hist, n_hist


# ==================================================================
# このファイルを直接実行したときの処理
# ==================================================================

if __name__ == '__main__':
    dt = 0.01       # タイムステップ (ms)
    T  = 50.0       # 総シミュレーション時間 (ms)
    t  = np.arange(0, T, dt)

    # --------------------------------------------------------------
    # 入力電流の選択（ここを書き換えるだけで入力を切り替えられる）
    # --------------------------------------------------------------
    I_ext = step_current(t, amplitude=10.0, t_start=10.0, t_end=40.0)

    # --- 他の入力を試すときは、上をコメントアウトして下のどれかを使う ---
    # I_ext = constant_current(t, amplitude=10.0)                       # 定常電流
    # I_ext = ramp_current(t, i_start=0.0, i_end=20.0)                  # 発火閾値の確認
    # I_ext = sin_current(t, amplitude=10.0, freq=50.0, offset=10.0)    # sin 波入力
    # I_ext = pulse_train_current(t, amplitude=20.0, width=1.0, interval=10.0)  # パルス列

    # --- 棟近先輩の教師電流を再現する場合（上の T も 9000.0 に変えること）---
    # I_ext = random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=0)   # スライド p.7 の条件
    # I_ext = random_pulse_current(t, i_min=0.0, i_max=20.0, interval=10.0, seed=0)    # 卒業論文 Algorithm 2 の条件

    V_hist, m_hist, h_hist, n_hist = simulate(t, I_ext)

    # V(t) の局所極大値を探す
    peak_idx = np.where(
        (V_hist[1:-1] > V_hist[:-2]) & (V_hist[1:-1] > V_hist[2:])
    )[0] + 1

    # その中で膜電位が大きい順に2つ選び、時間順に並べる
    top2_idx = peak_idx[np.argsort(V_hist[peak_idx])[-2:]]
    top2_idx = top2_idx[np.argsort(t[top2_idx])]

    print("極大値2つの時刻 t [ms] =", t[top2_idx])
    print("そのときの膜電位 V [mV] =", V_hist[top2_idx])

    plt.figure(figsize=(10, 8))

    # 1段：膜電位と入力電流
    plt.subplot(2, 1, 1)
    plt.title("Hodgkin-Huxley Model")
    plt.plot(t, V_hist, 'b-', label="Membrane Potential (V)")
    plt.ylabel("Membrane Potential (mV)", color='b')
    plt.tick_params(axis='y', labelcolor='b')
    plt.grid(True)

    # 入力電流を右軸に重ねて表示
    ax2 = plt.gca().twinx()
    ax2.plot(t, I_ext, 'r-', label="Input Current (I_ext)")
    ax2.set_ylabel(r"Input Current ($\mu$A/cm$^2$)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 下段：ゲート変数 (m, n, h) の変化
    plt.subplot(2, 1, 2)
    plt.plot(t, m_hist, 'g-', label="m (Na activation)")
    plt.plot(t, h_hist, 'g--', label="h (Na inactivation)")
    plt.plot(t, n_hist, 'm-', label="n (K activation)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Gating Probability")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
