"""
Traub らの海馬 CA3 錐体細胞モデル（19 コンパートメント）を陽的オイラー法で数値計算する。

元論文:
    R. D. Traub, R. K. S. Wong, R. Miles, H. Michelson (1991).
    A model of a CA3 hippocampal pyramidal neuron incorporating voltage-clamp data
    on intrinsic conductances. J. Neurophysiol. 66(2), 635-650.

実装は「スパイキングニューラルネットワーク」本（snsbook）のリスト 3.8
code/part1/multi/traub.c を NumPy に移植したもの。C 版はコンパートメントごとに
for ループを回すが、ここでは 19 個のコンパートメントを shape (NC,) の配列でまとめて計算する。
さらに先頭にバッチ次元を付けた shape (B, NC) の状態も扱え、B 個のニューロン（それぞれ別の
注入電流）を同時に計算できる。状態の shape は一般に (..., NC) で、最後の軸がコンパートメント。

コンパートメントの並び（0 〜 18）:
    0 〜 7   : 基底樹状突起側（7 が細胞体寄り）
    8        : 細胞体（電流はここに入れる）
    9 〜 18  : 尖端樹状突起側
    ※ C 版ではインデックス 8 が細胞体。g_Na, g_K_A が最大なのも 8。

単位:
    時刻 t: ms
    膜電位 V、反転電位 V_ion: mV
    電流密度 I: uA/cm^2
    コンダクタンス密度 g: mS/cm^2
    膜容量 C_m: uF/cm^2
    長さ・半径: cm、面積: cm^2
    χ（細胞内 Ca 濃度に比例する量）: 任意単位
"""
import numpy as np

# コンパートメント数
NC = 19
# 電流を入れる細胞体のインデックス
SOMA = 8

# 膜容量 [uF/cm^2]
C_m = 3.0
# 細胞内の抵抗率 [kΩ·cm]
R_i = 0.1
# χ の減衰速度 [1/ms]
BETA = 0.075

# 反転電位 [mV]。論文は静止電位を 0 mV とした値で書かれているので V_LEAK だけずらす
V_LEAK = -60.0
V_Na = 115.0 + V_LEAK
V_Ca = 140.0 + V_LEAK
V_K  = -15.0 + V_LEAK

# 各コンパートメントの最大コンダクタンス [mS/cm^2]。shape (NC,)
g_Na    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 15.0, 30.0, 15.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
g_K_DR  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 5.0, 15.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
g_K_A   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
g_K_C   = np.array([0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 5.0, 20.0, 10.0, 20.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 0.0])
g_K_AHP = np.array([0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0])
g_Ca    = np.array([0.0, 5.0, 5.0, 12.0, 12.0, 12.0, 5.0, 8.0, 4.0, 8.0, 5.0, 17.0, 17.0, 17.0, 10.0, 10.0, 5.0, 5.0, 0.0])
g_leak  = np.full(NC, 0.1)

# Ca 電流が χ に変わる係数 φ。shape (NC,)
phi = np.array([7769.0] * 7 + [34530.0, 17402.0, 26404.0] + [5941.0] * 9)

# 形状。半径 rad [cm]、長さ len_ [cm]、膜面積 area [cm^2]。shape (NC,)
rad   = np.array([2.89e-4] * 8 + [4.23e-4] + [2.42e-4] * 10)
len_  = np.array([1.20e-2] * 8 + [1.25e-2] + [1.10e-2] * 10)
area  = np.array([2.188e-5] * 8 + [3.320e-5] + [1.673e-5] * 10)


def _x_over_expm1(x, k):
    """
    x / (exp(x / k) - 1) を求める。x = 0 では 0/0 になるので極限値 k を返す。

    Args:
        x (np.ndarray): shape (..., NC)。
        k (float): 正の定数。

    Returns:
        np.ndarray: x と同じ shape。
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) < 1e-9, k, x / np.expm1(x / k))


# 以下の α, β はすべて静止電位からのずれ u = V - V_LEAK [mV] の関数で、単位は [1/ms]。
# 要素ごとの計算なので、入力も出力も同じ shape (..., NC)。

def alpha_m(u): return 0.32 * _x_over_expm1(13.1 - u, 4.0)
def beta_m(u):  return 0.28 * _x_over_expm1(u - 40.1, 5.0)

def alpha_s(u): return 1.6 / (1.0 + np.exp(-0.072 * (u - 65.0)))
def beta_s(u):  return 0.02 * _x_over_expm1(u - 51.1, 5.0)

def alpha_n(u): return 0.016 * _x_over_expm1(35.1 - u, 5.0)
def beta_n(u):  return 0.25 * np.exp((20.0 - u) / 40.0)

def alpha_c(u):
    return np.where(u <= 50.0,
                    np.exp((u - 10.0) / 11.0 - (u - 6.5) / 27.0) / 18.975,
                    2.0 * np.exp(-(u - 6.5) / 27.0))

def beta_c(u):
    return np.where(u <= 50.0, 2.0 * np.exp(-(u - 6.5) / 27.0) - alpha_c(u), 0.0)

def alpha_a(u): return 0.02 * _x_over_expm1(13.1 - u, 10.0)
def beta_a(u):  return 0.0175 * _x_over_expm1(u - 40.1, 10.0)

def alpha_h(u): return 0.128 * np.exp((17.0 - u) / 18.0)
def beta_h(u):  return 4.0 / (1.0 + np.exp((40.0 - u) / 5.0))

def alpha_r(u): return np.where(u <= 0.0, 0.005, np.exp(-u / 20.0) / 200.0)
def beta_r(u):  return np.where(u <= 0.0, 0.0, 0.005 - alpha_r(u))

def alpha_b(u): return 0.0016 * np.exp((-13.0 - u) / 18.0)
def beta_b(u):  return 0.05 / (1.0 + np.exp((10.1 - u) / 5.0))

# q（K_AHP のゲート）だけは膜電位ではなく χ の関数
def alpha_q(xi): return np.minimum(0.2e-4 * xi, 0.01)
def beta_q(xi):  return np.full_like(xi, 0.001)


# 膜電位で決まるゲート変数と、その (α, β)
VOLTAGE_GATES = {
    'm': (alpha_m, beta_m),  # Na 活性化
    'h': (alpha_h, beta_h),  # Na 不活性化
    's': (alpha_s, beta_s),  # Ca 活性化
    'r': (alpha_r, beta_r),  # Ca 不活性化
    'n': (alpha_n, beta_n),  # K_DR 活性化
    'a': (alpha_a, beta_a),  # K_A 活性化
    'b': (alpha_b, beta_b),  # K_A 不活性化
    'c': (alpha_c, beta_c),  # K_C 活性化
}


def _tau_inf(alpha, beta):
    """
    α, β から時定数 τ = 1/(α+β) [ms] と定常値 x∞ = α/(α+β) を求める。
    """
    s = alpha + beta
    return 1.0 / s, alpha / s


def compartment_conductance():
    """
    隣り合うコンパートメント間のコンダクタンスを求める。

    コンパートメント i と i+1 の中心どうしを結ぶ抵抗は、それぞれの半分の長さの抵抗の和
        R = R_i·(len_i/2)/(π r_i²) + R_i·(len_{i+1}/2)/(π r_{i+1}²)
    なので、コンダクタンスは
        g = 1/R = 2 / (R_i·len_i/(π r_i²) + R_i·len_{i+1}/(π r_{i+1}²))  [mS]

    Returns:
        np.ndarray: shape (NC-1,)。要素 i が i と i+1 の間のコンダクタンス [mS]。
    """
    r_axial = R_i * len_ / (np.pi * rad ** 2)  # shape (NC,)
    return 2.0 / (r_axial[:-1] + r_axial[1:])


def initial_state(batch_shape=()):
    """
    初期状態を作る。全コンパートメントを V_LEAK にし、ゲート変数をその電位での定常値にする。
    χ は Ca 電流と減衰がつり合う値 -φ·I_Ca/β、q はその χ での定常値にする。

    Args:
        batch_shape (tuple[int, ...]): コンパートメントの前に付けるバッチの shape。
            () なら 1 個のニューロン、(B,) なら B 個のニューロン。

    Returns:
        dict[str, np.ndarray]: キーは 'V', 'xi', 'm', 'h', 's', 'r', 'n', 'a', 'b', 'c', 'q'。
            値はどれも shape (*batch_shape, NC)、float64。
    """
    V = np.full((*batch_shape, NC), V_LEAK)
    u = V - V_LEAK
    state = {'V': V}
    for name, (alpha, beta) in VOLTAGE_GATES.items():
        state[name] = _tau_inf(alpha(u), beta(u))[1]

    # area を掛けた Ca 電流 [uA]（密度ではない）
    I_Ca = g_Ca * area * state['s'] ** 2 * state['r'] * (V - V_Ca)
    state['xi'] = -I_Ca * phi / BETA
    state['q'] = _tau_inf(alpha_q(state['xi']), beta_q(state['xi']))[1]
    return state


def step(state, I_inj, dt, g_comp):
    """
    陽的オイラー法で 1 ステップ（dt）進めた状態を返す。

    膜電位（コンパートメント i）:
        C_m dV_i/dt = -I_ion,i + I_comp,i + I_inj,i
        I_comp,i = [g_{i-1,i}(V_{i-1} - V_i) + g_{i,i+1}(V_{i+1} - V_i)] / area_i
    ゲート変数 x:
        dx/dt = (x∞(V) - x) / τ(V)
    χ:
        dχ/dt = -φ·I_Ca·area - β·χ

    Args:
        state (dict[str, np.ndarray]): 現在の状態。値は shape (..., NC)。
        I_inj (np.ndarray): 注入電流密度 [uA/cm^2]。state の値と同じ shape (..., NC)。
        dt (float): 時間刻み [ms]。
        g_comp (np.ndarray): compartment_conductance() の返り値。shape (NC-1,)。

    Returns:
        dict[str, np.ndarray]: 次の時刻の状態。入力の state は書き換えない。
    """
    V, xi = state['V'], state['xi']
    m, h, s, r = state['m'], state['h'], state['s'], state['r']
    n, a, b, c, q = state['n'], state['a'], state['b'], state['c'], state['q']
    u = V - V_LEAK

    # イオン電流 [uA/cm^2]。shape (..., NC)。パラメータ g_* は shape (NC,) で最後の軸に揃う
    I_ion = (g_leak * (V - V_LEAK)
             + g_Na * m ** 2 * h * (V - V_Na)
             + g_Ca * s ** 2 * r * (V - V_Ca)
             + g_K_DR * n * (V - V_K)
             + g_K_A * a * b * (V - V_K)
             + g_K_AHP * q * (V - V_K)
             + g_K_C * c * np.minimum(1.0, xi / 250.0) * (V - V_K))

    # 隣のコンパートメントから流れ込む電流 [uA/cm^2]。両端は片側だけ
    I_flow = g_comp * (V[..., 1:] - V[..., :-1])  # i+1 から i へ流れる電流 [uA]。shape (..., NC-1)
    I_comp = np.zeros_like(V)
    I_comp[..., :-1] += I_flow
    I_comp[..., 1:]  -= I_flow
    I_comp /= area

    new = {'V': V + dt / C_m * (-I_ion + I_comp + I_inj)}
    for name, (alpha, beta) in VOLTAGE_GATES.items():
        tau, x_inf = _tau_inf(alpha(u), beta(u))
        new[name] = state[name] + dt / tau * (x_inf - state[name])

    tau_q, q_inf = _tau_inf(alpha_q(xi), beta_q(xi))
    new['q'] = q + dt / tau_q * (q_inf - q)

    I_Ca = g_Ca * area * s ** 2 * r * (V - V_Ca)  # [uA]
    new['xi'] = xi + dt * (-phi * I_Ca - BETA * xi)
    return new


def simulate(t, I_inj):
    """
    注入電流 I_inj に対する Traub モデルの応答を計算する。

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        I_inj (np.ndarray): 各時刻・各コンパートメントの注入電流密度 [uA/cm^2]。
            1 個なら shape (N, NC)、B 個まとめて計算するなら shape (N, B, NC)。
            細胞体だけに電流 I [uA] を入れるなら I_inj[..., SOMA] = I / area[SOMA]。

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]:
            V_hist: 膜電位 [mV]。I_inj と同じ shape。V_hist[i] は時刻 t[i] の値（更新前）。
            hist: 全状態変数の履歴。キーは initial_state() と同じ、値は I_inj と同じ shape。
    """
    dt = t[1] - t[0]
    g_comp = compartment_conductance()
    state = initial_state(I_inj.shape[1:-1])

    hist = {k: np.zeros((len(t), *v.shape)) for k, v in state.items()}
    for i in range(len(t)):
        for k, v in state.items():
            hist[k][i] = v
        state = step(state, I_inj[i], dt, g_comp)
    return hist['V'], hist
