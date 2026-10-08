import numpy as np

C_m  = 1.0
g_Na = 120.0
g_K  = 36.0
g_L  = 0.3

E_Na = 50.0
E_K  = -77.0
E_L  = -54.387

V_REST = -65.0


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
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    return m, h, n


def simulate(t, I_ext, V0=V_REST):
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
