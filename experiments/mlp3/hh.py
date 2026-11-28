"""
HH モデルの陽的オイラー法（torch 版）。MLP 3 個の各ブロックの正解はここの式で厳密に計算する。

定数は src/neuron/hh.py と同じ（静止電位 -65 mV の表記）。dt = 0.01 ms。

    rates(V)          V (...)            -> αβ (..., 6) = α_m, β_m, α_h, β_h, α_n, β_n   [1/ms]
    d_gates(ab, g)    αβ (..., 6), g (..., 3) -> Δ(m, h, n)                               ブロック 2 の正解
    d_V(g, V, I)      g (..., 3), V (...), I (...) -> ΔV [mV]                             ブロック 3 の正解
    simulate(I)       I (B, N) -> V (B, N+1), G (B, N+1, 3)       静止状態から N ステップ
    states(I, stride) I (B, N) -> X (B * ceil(N / stride), 5)     stride ごとの状態 [V, m, h, n, I]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from neuron import hh as H   # noqa: E402

DT = 0.01
V_REST = H.V_REST
RATE_NAMES = ['α_m', 'β_m', 'α_h', 'β_h', 'α_n', 'β_n']


def rates(V):
    # α_m と α_n は V = -40, -55 mV で 0/0 になるので、その点だけ極限値（1, 0.1）にする
    xm, xn = V + 40.0, V + 55.0
    safe = lambda x: torch.where(x.abs() < 1e-6, torch.full_like(x, 1e-6), x)
    am = torch.where(xm.abs() < 1e-6, torch.ones_like(V), 0.1 * safe(xm) / (1 - torch.exp(-safe(xm) / 10)))
    an = torch.where(xn.abs() < 1e-6, torch.full_like(V, 0.1), 0.01 * safe(xn) / (1 - torch.exp(-safe(xn) / 10)))
    bm = 4.0 * torch.exp(-(V + 65) / 18)
    ah = 0.07 * torch.exp(-(V + 65) / 20)
    bh = 1.0 / (1 + torch.exp(-(V + 35) / 10))
    bn = 0.125 * torch.exp(-(V + 65) / 80)
    return torch.stack([am, bm, ah, bh, an, bn], -1)


def d_gates(ab, g):
    # Δx = dt (α_x (1 - x) - β_x x)、x = m, h, n
    a, b = ab[..., 0::2], ab[..., 1::2]
    return DT * (a * (1 - g) - b * g)


def d_V(g, V, I):
    # ΔV = dt (I - I_ion) / C_m、I_ion = g_Na m^3 h (V - E_Na) + g_K n^4 (V - E_K) + g_L (V - E_L)
    m, h, n = g[..., 0], g[..., 1], g[..., 2]
    I_ion = H.g_Na * m ** 3 * h * (V - H.E_Na) + H.g_K * n ** 4 * (V - H.E_K) + H.g_L * (V - H.E_L)
    return DT * (I - I_ion) / H.C_m


def rest_state(B, dtype=torch.float64):
    # 静止状態: V (B,) = -65 mV、g (B, 3) = その V での定常値
    V = torch.full((B,), V_REST, dtype=dtype)
    g = torch.tensor(H.steady_state_gates(V_REST), dtype=dtype).repeat(B, 1)
    return V, g


@torch.no_grad()
def simulate(I):
    B, N = I.shape
    V, g = rest_state(B)
    Vs, Gs = [V], [g]
    for k in range(N):
        V, g = V + d_V(g, V, I[:, k]), g + d_gates(rates(V), g)
        Vs.append(V)
        Gs.append(g)
    return torch.stack(Vs, 1), torch.stack(Gs, 1)


@torch.no_grad()
def states(I, stride=1):
    # 長い電流でもメモリを抑えるため、stride ステップごとの状態だけを残す
    B, N = I.shape
    V, g = rest_state(B)
    out = []
    for k in range(N):
        if k % stride == 0:
            out.append(torch.cat([V[:, None], g, I[:, k, None]], 1))
        V, g = V + d_V(g, V, I[:, k]), g + d_gates(rates(V), g)
    return torch.stack(out, 1).reshape(-1, 5)
