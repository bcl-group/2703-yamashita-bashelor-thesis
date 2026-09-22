import numpy as np
import matplotlib.pyplot as plt

C_m  = 1.0      # 膜容量 (uF/cm^2)
g_Na = 120.0    # ナトリウムの最大コンダクタンス (mS/cm^2)
g_K  = 36.0     # カリウムの最大コンダクタンス (mS/cm^2)
g_L  = 0.3      # 漏れ電流（リーク）の最大コンダクタンス (mS/cm^2)

E_Na = 50.0     # ナトリウムの平衡電位 (mV)
E_K  = -77.0    # カリウムの平衡電位 (mV)
E_L  = -54.387  # 漏れ電流の平衡電位 (mV)

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

dt = 0.01       # タイムステップ (ms)
T  = 50.0       # 総シミュレーション時間 (ms)
t  = np.arange(0, T, dt)

# 外部からの注入電流 (10ms から 40ms の間だけ 10 uA/cm^2 流す)
I_ext = np.zeros(len(t))
I_ext[(t >= 10.0) & (t <= 40.0)] = 10.0

V = -65.0  # 初期膜電位 (mV)

# 各ゲート変数の無限大時間での定常状態の値 (x_inf = alpha / (alpha + beta))
n = alpha_n(V) / (alpha_n(V) + beta_n(V))
m = alpha_m(V) / (alpha_m(V) + beta_m(V))
h = alpha_h(V) / (alpha_h(V) + beta_h(V))

# 結果を記録する配列
V_hist = np.zeros(len(t))
n_hist = np.zeros(len(t))
m_hist = np.zeros(len(t))
h_hist = np.zeros(len(t))

for i in range(len(t)):
    # 現在の値を記録
    V_hist[i] = V
    n_hist[i] = n
    m_hist[i] = m
    h_hist[i] = h

    # 各イオン電流の計算
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K  = g_K * (n**4) * (V - E_K)
    I_L  = g_L * (V - E_L)

    # 微分方程式の更新 (次の時間ステップの値を計算)
    dV = (I_ext[i] - I_Na - I_K - I_L) / C_m
    dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h

    # 値の更新
    V += dV * dt
    n += dn * dt
    m += dm * dt
    h += dh * dt

# V(t) の局所極大値を探す
peak_idx = np.where(
    (V_hist[1:-1] > V_hist[:-2]) & (V_hist[1:-1] > V_hist[2:])
)[0] + 1

# その中で膜電位が大きい順に2つ選ぶ
top2_idx = peak_idx[np.argsort(V_hist[peak_idx])[-2:]]

# 時間順に並べる
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
ax2.set_ylabel("Input Current ($\mu$A/cm$^2$)", color='r')
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

