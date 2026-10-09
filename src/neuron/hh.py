
"""
Hodgkin-Huxley（HH）モデルを陽的オイラー法で数値計算する。
単位:
    時刻 t: ms
    膜電位 V、反転電位 E_ion: mV
    電流 I: uA/cm^2
    コンダクタンス g: mS/cm^2
    膜容量 C_m: uF/cm^2
"""
import numpy as np

# 膜容量 [uF/cm^2]
C_m  = 1.0

# 各イオンチャネルの最大コンダクタンス [mS/cm^2]（チャネルが全部開いたときの流れやすさ）
g_Na = 120.0
g_K  = 36.0
g_L  = 0.3

# 反転電位 [mV]（そのイオンの電流が 0 になる膜電位）
E_Na = 50.0
E_K  = -77.0
E_L  = 10.6 - 65.0

# 静止電位 [mV]
V_REST = -65.0


def alpha_n(V):
    """
    K チャネルの活性化ゲート n が「閉 → 開」に変わる速さ α_n を求める。

    α_n(V) = 0.01 (V + 55) / (1 - exp(-(V + 55) / 10))

    V = -55 では分子と分母がどちらも 0 になる（0/0）。そのときは極限値 0.1 を返す。

    Args:
        V (float | np.ndarray): 膜電位 [mV]。スカラーでも配列でもよい。

    Returns:
        np.ndarray: α_n [1/ms]。V と同じ shape（スカラーなら 0 次元配列）。
    """
    x = V + 55.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) < 1e-6, 0.1, 0.01 * x / (1.0 - np.exp(-x / 10.0)))

def beta_n(V):
    """
    K チャネルの活性化ゲート n が「開 → 閉」に変わる速さ β_n を求める。

    β_n(V) = 0.125 exp(-(V + 65) / 80)

    Args:
        V (float | np.ndarray): 膜電位 [mV]。

    Returns:
        float | np.ndarray: β_n [1/ms]。V と同じ shape。
    """
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

def alpha_m(V):
    """
    Na チャネルの活性化ゲート m が「閉 → 開」に変わる速さ α_m を求める。

    α_m(V) = 0.1 (V + 40) / (1 - exp(-(V + 40) / 10))

    V = -40 では 0/0 になるので、そのときは極限値 1.0 を返す。

    Args:
        V (float | np.ndarray): 膜電位 [mV]。スカラーでも配列でもよい。

    Returns:
        np.ndarray: α_m [1/ms]。V と同じ shape（スカラーなら 0 次元配列）。
    """
    x = V + 40.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) < 1e-6, 1.0, 0.1 * x / (1.0 - np.exp(-x / 10.0)))

def beta_m(V):
    """
    Na チャネルの活性化ゲート m が「開 → 閉」に変わる速さ β_m を求める。

    β_m(V) = 4 exp(-(V + 65) / 18)

    Args:
        V (float | np.ndarray): 膜電位 [mV]。

    Returns:
        float | np.ndarray: β_m [1/ms]。V と同じ shape。
    """
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    """
    Na チャネルの不活性化ゲート h が「閉 → 開」に変わる速さ α_h を求める。

    α_h(V) = 0.07 exp(-(V + 65) / 20)

    Args:
        V (float | np.ndarray): 膜電位 [mV]。

    Returns:
        float | np.ndarray: α_h [1/ms]。V と同じ shape。
    """
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    """
    Na チャネルの不活性化ゲート h が「開 → 閉」に変わる速さ β_h を求める。

    β_h(V) = 1 / (1 + exp(-(V + 35) / 10))

    Args:
        V (float | np.ndarray): 膜電位 [mV]。

    Returns:
        float | np.ndarray: β_h [1/ms]。V と同じ shape。
    """
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def steady_state_gates(V):
    """
    膜電位を V に固定したときの、ゲート変数の定常値 m_∞, h_∞, n_∞ を求める。

    dx/dt = α_x(1 - x) - β_x x 
    dx/dt = 0のとき
    x_∞ = α_x / (α_x + β_x)

    Args:
        V (float | np.ndarray): 膜電位 [mV]。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (m_∞, h_∞, n_∞)。値は 0〜1。
    """
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    return m, h, n


def simulate(t, I_ext, V0=V_REST):
    """
    入力電流 I_ext に対する HH モデルの応答を、陽的オイラー法で計算する。
    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
            刻み幅 dt は t[1] - t[0] から求める。
        I_ext (np.ndarray): 各時刻の入力電流 [uA/cm^2]。shape (N,)。
        V0 (float): 膜電位の初期値 [mV]。ゲート変数の初期値は V0 での定常値にする。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (V_hist, m_hist, h_hist, n_hist)。どれも shape (N,)、float64。
            V_hist は膜電位 [mV]、ほかの 3 つはゲート変数（0〜1)
    """
    dt = t[1] - t[0]

    V = float(V0)
    m, h, n = steady_state_gates(V)

    V_hist = np.zeros(len(t))
    m_hist = np.zeros(len(t))
    h_hist = np.zeros(len(t))
    n_hist = np.zeros(len(t))

    for i in range(len(t)):
        V_hist[i] = V
        m_hist[i] = m
        h_hist[i] = h
        n_hist[i] = n

        I_Na = g_Na * (m ** 3) * h * (V - E_Na)
        I_K  = g_K * (n ** 4) * (V - E_K)
        I_L  = g_L * (V - E_L)

        dV = (I_ext[i] - I_Na - I_K - I_L) / C_m
        dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n

        V += dV * dt
        m += dm * dt
        h += dh * dt
        n += dn * dt

    return V_hist, m_hist, h_hist, n_hist
