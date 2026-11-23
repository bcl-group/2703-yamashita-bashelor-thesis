"""
Traub モデル（19 区画、src/neuron/traub.py）の陽的 Euler の 1 ステップを PyTorch で書いたもの。
B 個の細胞をバッチの行に並べ、GPU でまとめて計算する（速度の比較と、学習中に教師をその場で作るため）。

式は src/neuron/traub.py の step() と同じ。違いは状態の持ち方だけ:
    V  : (B, NC)      膜電位 [mV]
    xi : (B, NC)      χ（Ca 濃度に比例する量）
    G  : (B, NC, 9)   ゲート変数。列は GATES = m, h, s, r, n, a, b, c, q
    I  : (B,)         細胞体への電流 [nA]（細胞体以外は 0）
check() で NumPy 版と fp64 で一致することを確かめる。
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from neuron import traub as T   # noqa: E402

NC, SOMA = T.NC, T.SOMA
GATES = ['m', 'h', 's', 'r', 'n', 'a', 'b', 'c', 'q']
NA_TO_DENSITY = 1e-3 / T.area[T.SOMA]


def _xe(x, k):
    # x / (exp(x/k) - 1)。x = 0 の近くでは極限値 k
    small = x.abs() < 1e-9
    xs = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, torch.full_like(x, k), xs / torch.expm1(xs / k))


def rates(u):
    """u = V - V_LEAK: (..., NC) -> α, β: (..., NC, 8)（ゲート m, h, s, r, n, a, b, c の順）"""
    am = 0.32 * _xe(13.1 - u, 4.0)
    bm = 0.28 * _xe(u - 40.1, 5.0)
    ah = 0.128 * torch.exp((17.0 - u) / 18.0)
    bh = 4.0 / (1.0 + torch.exp((40.0 - u) / 5.0))
    as_ = 1.6 / (1.0 + torch.exp(-0.072 * (u - 65.0)))
    bs = 0.02 * _xe(u - 51.1, 5.0)
    ar = torch.where(u <= 0.0, torch.full_like(u, 0.005), torch.exp(-u / 20.0) / 200.0)
    br = torch.where(u <= 0.0, torch.zeros_like(u), 0.005 - ar)
    an = 0.016 * _xe(35.1 - u, 5.0)
    bn = 0.25 * torch.exp((20.0 - u) / 40.0)
    aa = 0.02 * _xe(13.1 - u, 10.0)
    ba = 0.0175 * _xe(u - 40.1, 10.0)
    ab_ = 0.0016 * torch.exp((-13.0 - u) / 18.0)
    bb = 0.05 / (1.0 + torch.exp((10.1 - u) / 5.0))
    e = torch.exp(-(u - 6.5) / 27.0)
    ac = torch.where(u <= 50.0, torch.exp((u - 10.0) / 11.0 - (u - 6.5) / 27.0) / 18.975, 2.0 * e)
    bc = torch.where(u <= 50.0, 2.0 * e - ac, torch.zeros_like(u))
    a = torch.stack([am, ah, as_, ar, an, aa, ab_, ac], -1)
    b = torch.stack([bm, bh, bs, br, bn, ba, bb, bc], -1)
    return a, b


class Params:
    """区画ごとの定数を、指定したデバイス・型のテンソルにして持つ"""

    def __init__(self, dev, dtype):
        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), device=dev, dtype=dtype)
        self.gNa, self.gKDR, self.gKA = f(T.g_Na), f(T.g_K_DR), f(T.g_K_A)
        self.gKC, self.gKAHP, self.gCa, self.gL = f(T.g_K_C), f(T.g_K_AHP), f(T.g_Ca), f(T.g_leak)
        self.phi, self.area = f(T.phi), f(T.area)
        self.gcomp = f(T.compartment_conductance())
        inj = np.zeros(NC)
        inj[SOMA] = NA_TO_DENSITY
        self.inj = f(inj)        # (NC,)。電流 [nA] を掛けると注入電流密度 [uA/cm^2]
        self.dev, self.dtype = dev, dtype


def init_state(B, P):
    # NumPy 版の initial_state() と同じ値
    s = T.initial_state((B,))
    f = lambda x: torch.as_tensor(x, device=P.dev, dtype=P.dtype)
    G = torch.stack([f(s[g]) for g in GATES], -1)
    return f(s['V']), f(s['xi']), G


def ionic(V, xi, G, P):
    m, h, s, r, n, a, b, c, q = G.unbind(-1)
    VK = T.V_K
    return (P.gL * (V - T.V_LEAK) + P.gNa * m ** 2 * h * (V - T.V_Na) + P.gCa * s ** 2 * r * (V - T.V_Ca)
            + P.gKDR * n * (V - VK) + P.gKA * a * b * (V - VK) + P.gKAHP * q * (V - VK)
            + P.gKC * c * torch.clamp(xi / 250.0, max=1.0) * (V - VK))


def step(state, I_nA, P, dt=0.01):
    """陽的 Euler の 1 ステップ。state = (V, xi, G)、I_nA: (B,)"""
    V, xi, G = state
    u = V - T.V_LEAK
    flow = P.gcomp * (V[:, 1:] - V[:, :-1])                       # (B, NC-1)
    zero = torch.zeros_like(V[:, :1])
    I_comp = (torch.cat([flow, zero], 1) - torch.cat([zero, flow], 1)) / P.area
    V_new = V + dt / T.C_m * (-ionic(V, xi, G, P) + I_comp + I_nA[:, None] * P.inj)
    a, b = rates(u)
    x8 = G[..., :8]
    G8 = x8 + dt * (a * (1 - x8) - b * x8)                        # dt/τ (x∞ - x) = dt (α(1-x) - βx)
    aq = torch.clamp(0.2e-4 * xi, max=0.01)
    q = G[..., 8]
    q_new = q + dt * (aq * (1 - q) - 0.001 * q)
    I_Ca = P.gCa * P.area * G[..., 2] ** 2 * G[..., 3] * (V - T.V_Ca)
    xi_new = xi + dt * (-P.phi * I_Ca - T.BETA * xi)
    return V_new, xi_new, torch.cat([G8, q_new[..., None]], -1)


def check(n_steps=3000, B=4, seed=0):
    """NumPy 版（src/neuron/traub.py）と fp64 で比べる。最大の差を返す"""
    rng = np.random.default_rng(seed)
    I = rng.uniform(-0.2, 1.2, size=(n_steps, B))
    P = Params(torch.device('cpu'), torch.float64)
    g_comp = T.compartment_conductance()
    s_np = T.initial_state((B,))
    s_t = init_state(B, P)
    err = 0.0
    for k in range(n_steps):
        inj = np.zeros((B, NC))
        inj[:, SOMA] = I[k] * NA_TO_DENSITY
        s_np = T.step(s_np, inj, 0.01, g_comp)
        s_t = step(s_t, torch.as_tensor(I[k]), P)
        if k % 500 == 499 or k == n_steps - 1:
            err = max(err, float(np.abs(s_t[0].numpy() - s_np['V']).max()),
                      float(np.abs(s_t[1].numpy() - s_np['xi']).max() / max(1.0, np.abs(s_np['xi']).max())))
            for j, g in enumerate(GATES):
                err = max(err, float(np.abs(s_t[2][..., j].numpy() - s_np[g]).max()))
    return err


if __name__ == '__main__':
    print('NumPy 版との最大の差（V [mV]、χ は相対、ゲート）:', check())
