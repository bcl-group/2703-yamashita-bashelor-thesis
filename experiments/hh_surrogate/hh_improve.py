"""
HH 代理モデルの精度改善の実験（PyTorch）。2 つの型を同じ道具で学習・評価する。

  (A) RNN 3 個（分割型）: ブロックごとに教師強制で学習する。
      どのブロックの正解も「その時刻の入力の厳密な関数」（HH のオイラー法の式そのもの）なので、
      入力に雑音を足しても正解を式から計算し直せる（--noise）。閉ループでずれた状態でも正しい 1 ステップを学ぶ。
        RNN1: V -> α, β(6)          RNN2: α, β, m, h, n -> Δm, Δh, Δn          RNN3: m, h, n, V, I -> ΔV
  (B) RNN 1 個（丸ごと型）: [V(t), I(t)] -> ΔV。V は予測値を戻す。ゲート変数は隠れ状態が持つ。
      補助の出力（隠れ状態から m, h, n を当てる）を付けられる（--aux）。

改良の候補（コマンドの引数）:
    --H, --layers        幅と段数（段を重ねた RNN = nn.RNN(num_layers)）
    --iters, --lr        Adam + 余弦減衰（lr -> lr/100）
    --noise              入力の雑音（窓ごとの一定のずれ + 毎ステップの小さな揺らぎ）。正解は式から計算し直す
    --wsub               閾値下（V < -50 mV）のサンプルの損失の重み
    --rollout K          (B) 閉ループで K ステップ回した誤差を損失にする段階を追加

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_improve.py block --block 3 --H 128 --tag b3_h128
    uv run python -X utf8 experiments/hh_surrogate/hh_improve.py single --H 128 --aux 1 --tag s_h128
    uv run python -X utf8 experiments/hh_surrogate/hh_improve.py eval3 --b3 results/hh_improve/b3_h128.pt
結果は results/hh_improve/<tag>.pt / <tag>.json に保存する。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_bench as hb
import hh_modular as hm
import test_currents as tc
from neuron import hh as H_

OUT_DIR = os.path.join(hb.ROOT, 'results', 'hh_improve')
DT = 0.01
N_VALID_TRAJ = 10          # 最後の 10 軌道は 1 ステップ誤差の検証用（学習に使わない）


# ---------------- HH の式（torch、float64） ----------------

def rates(V):
    # V: (...) -> (..., 6) = α_m, β_m, α_h, β_h, α_n, β_n
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
    # ab: (..., 6), g: (..., 3) -> Δ(m, h, n)
    a, b = ab[..., 0::2], ab[..., 1::2]
    return DT * (a * (1 - g) - b * g)


def d_V(g, V, I):
    m, h, n = g[..., 0], g[..., 1], g[..., 2]
    I_ion = (H_.g_Na * m ** 3 * h * (V - H_.E_Na) + H_.g_K * n ** 4 * (V - H_.E_K)
             + H_.g_L * (V - H_.E_L))
    return DT * (I - I_ion) / H_.C_m


# ---------------- モデル ----------------

class Net(nn.Module):
    """標準化 + 段を重ねた tanh RNN + 線形の読み出し（+ 補助の読み出し）。"""

    def __init__(self, d_in, H, layers, d_out, d_aux=0, feat=0, leak=0.0, cell='rnn', phys=False):
        super().__init__()
        self.cfg = dict(d_in=d_in, H=H, layers=layers, d_out=d_out, d_aux=d_aux, feat=feat, leak=leak,
                        cell=cell, phys=phys)
        d_rnn = d_in + {0: 0, 2: 6, 3: 4}[feat]
        if cell == 'mlp':
            # 再帰のない MLP（Markov 型）。layers 段の Linear + SiLU
            mods, d = [], d_rnn
            for _ in range(layers):
                mods += [nn.Linear(d, H), nn.SiLU()]
                d = H
            self.mlp = nn.Sequential(*mods)
        elif cell == 'gru':
            self.rnn = nn.GRU(d_rnn, H, num_layers=layers, batch_first=True)
        else:
            self.rnn = nn.RNN(d_rnn, H, num_layers=layers, nonlinearity='tanh', batch_first=True)
        self.out = nn.Linear(H, d_out)
        self.aux = nn.Linear(H, d_aux) if d_aux else None
        for k, dim in [('im', d_rnn), ('is_', d_rnn), ('om', d_out), ('os_', d_out)]:
            self.register_buffer(k, torch.zeros(dim) if k.endswith('m') else torch.ones(dim))

    def expand(self, x):
        # 積の特徴量を足す（feat=3: RNN3 に m^3 h, n^4, m^3 h V, n^4 V。feat=2: RNN2 に α x, β x）
        f = self.cfg['feat']
        if f == 3:
            m, h, n, V = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            a, b = m ** 3 * h, n ** 4
            return torch.cat([x, torch.stack([a, b, a * V, b * V], -1)], -1)
        if f == 2:
            return torch.cat([x, x[..., 0:6:2] * x[..., 6:9], x[..., 1:6:2] * x[..., 6:9]], -1)
        return x

    def set_scalers(self, X, Y):
        # X: (n, d_in), Y: (n, d_out) の物理単位のサンプルから平均と標準偏差を決める
        X = self.expand(X)
        self.im.copy_(X.mean(0)); self.is_.copy_(X.std(0) + 1e-12)
        self.om.copy_(Y.mean(0)); self.os_.copy_(Y.std(0) + 1e-12)

    def forward(self, x, h=None):
        # x: (B, L, d_in) 物理単位 -> y: (B, L, d_out) 物理単位、標準化した y、隠れ状態の列、最後の隠れ状態
        u = (self.expand(x) - self.im) / self.is_
        if self.cfg['leak'] > 0:
            hs = self.zeros(x.shape[0], u.dtype) if h is None else h
            zs = []
            for k in range(u.shape[1]):
                hs = self._step(u[:, k], hs)
                zs.append(hs[-1])
            z, hN = torch.stack(zs, 1), hs
        else:
            z, hN = self.rnn(u, h)
        ys = self.out(z)
        return ys * self.os_ + self.om, ys, z, hN

    def cell(self, x, hs):
        # 1 ステップ（閉ループ用）。x: (B, d_in)、hs: list[(B, H)] -> y (B, d_out), hs
        hs = self._step((self.expand(x) - self.im) / self.is_, hs)
        return self.out(hs[-1]) * self.os_ + self.om, hs

    def _step(self, u, hs):
        # 標準化済みの入力 u で 1 ステップ。leak > 0 なら漏れ積分型 h <- h + leak (tanh(...) - h)
        if self.cfg.get('cell', 'rnn') == 'mlp':
            return [self.mlp(u)]
        new = []
        for l in range(self.cfg['layers']):
            Wi, Wh = getattr(self.rnn, f'weight_ih_l{l}'), getattr(self.rnn, f'weight_hh_l{l}')
            if self.cfg.get('cell', 'rnn') == 'gru':
                # GRU: r, z, n の 3 つのゲート（torch.nn.GRU と同じ式）
                gi = u @ Wi.T + getattr(self.rnn, f'bias_ih_l{l}')
                gh = hs[l] @ Wh.T + getattr(self.rnn, f'bias_hh_l{l}')
                ir, iz, i_n = gi.chunk(3, -1)
                hr, hz, h_n = gh.chunk(3, -1)
                r, zg = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
                z = (1 - zg) * torch.tanh(i_n + r * h_n) + zg * hs[l]
                new.append(z)
                u = z
                continue
            b = getattr(self.rnn, f'bias_ih_l{l}') + getattr(self.rnn, f'bias_hh_l{l}')
            z = torch.tanh(u @ Wi.T + hs[l] @ Wh.T + b)
            if self.cfg['leak'] > 0:
                z = hs[l] + self.cfg['leak'] * (z - hs[l])
            new.append(z)
            u = z
        return new

    def zeros(self, B, dtype):
        n = 1 if self.cfg.get('cell', 'rnn') == 'mlp' else self.cfg['layers']
        return [torch.zeros(B, self.cfg['H'], dtype=dtype) for _ in range(n)]


def from_simple(model, in_s, out_s):
    """hh_modular の SimpleRNN（numpy）を Net に写す。"""
    net = Net(model.input_dim, model.hidden_dim, 1, model.output_dim).double()
    with torch.no_grad():
        net.rnn.weight_ih_l0.copy_(torch.from_numpy(model.W_xh.T.copy()))
        net.rnn.weight_hh_l0.copy_(torch.from_numpy(model.W_hh.T.copy()))
        net.rnn.bias_ih_l0.copy_(torch.from_numpy(model.b_h.copy()))
        net.rnn.bias_hh_l0.zero_()
        net.out.weight.copy_(torch.from_numpy(model.W_hy.T.copy()))
        net.out.bias.copy_(torch.from_numpy(model.b_y.copy()))
        net.im.copy_(torch.from_numpy(in_s.mean)); net.is_.copy_(torch.from_numpy(in_s.std))
        net.om.copy_(torch.from_numpy(out_s.mean)); net.os_.copy_(torch.from_numpy(out_s.std))
    return net


def save_net(net, path, info):
    torch.save({'cfg': net.cfg, 'state': net.state_dict(), 'info': info}, path)


def load_net(path):
    if path.endswith('.npz'):
        return from_simple(*hm.load_block(path)[:3])
    d = torch.load(path, weights_only=False)
    net = Net(**d['cfg'])
    net.load_state_dict(d['state'])
    return net.double()


# ---------------- データ ----------------

class Windows:
    """
    教師データから長さ L の窓をランダムに切り出す。
    paths はカンマ区切りで複数指定できる（軌道の長さが違ってもよい）。ファイルは軌道の本数に比例して選ぶ。
    検証用は最初のファイルの最後の N_VALID_TRAJ 本だけ。
    """

    def __init__(self, paths, seed=0):
        if isinstance(paths, str):
            paths = paths.split(',')
        self.files = []
        for p in paths:
            d = hb.load_teacher(p, dtype=np.float32)
            self.files.append({k: torch.from_numpy(v) for k, v in d.items()})
        self.d = self.files[0]
        self.n_traj, self.N = self.d['V'].shape
        self.rng = np.random.default_rng(seed)

    def _sample_one(self, d, B, L, lo, hi, s_min):
        N = d['V'].shape[1]
        s_min = min(s_min, N - L - 2)
        ti = self.rng.integers(lo, hi, size=B)
        si = self.rng.integers(s_min, N - L - 1, size=B)
        idx = torch.from_numpy(si[:, None] + np.arange(L + 1)[None])
        ti = torch.from_numpy(ti)[:, None]
        g = lambda k: d[k][ti, idx].double()                         # (B, L+1)
        return g('I'), g('V'), torch.stack([g('m'), g('h'), g('n')], -1)

    def sample(self, B, L, valid=False, s_min=5000):
        nt = self.n_traj
        if valid or len(self.files) == 1:
            lo, hi = (nt - N_VALID_TRAJ, nt) if valid else (0, nt - N_VALID_TRAJ)
            return self._sample_one(self.d, B, L, lo, hi, s_min)
        sizes = np.array([nt - N_VALID_TRAJ] + [f['V'].shape[0] for f in self.files[1:]], float)
        fi = self.rng.choice(len(self.files), size=B, p=sizes / sizes.sum())
        out = []
        for j, f in enumerate(self.files):
            b = int((fi == j).sum())
            if b:
                hi = nt - N_VALID_TRAJ if j == 0 else f['V'].shape[0]
                out.append(self._sample_one(f, b, L, 0, hi, s_min))
        return tuple(torch.cat(x) for x in zip(*out))


def perturb(V, G, sigma, rng_t):
    """
    窓ごとの一定のずれ（標準偏差 V: 2σ mV、ゲート: 0.02σ）と、毎ステップの揺らぎ（その 1/5）を足す。
    窓の半分は雑音なし（きれいな軌道での精度も保つ）。
    """
    if sigma <= 0:
        return V, G
    B = V.shape[0]
    sigma = sigma * (torch.rand(B, 1, generator=rng_t, dtype=V.dtype) < 0.5)
    V = V + sigma * (2.0 * torch.randn(B, 1, generator=rng_t, dtype=V.dtype)
                     + 0.4 * torch.randn(V.shape, generator=rng_t, dtype=V.dtype))
    G = G + sigma[..., None] * (0.02 * torch.randn(B, 1, 3, generator=rng_t, dtype=G.dtype)
                     + 0.004 * torch.randn(G.shape, generator=rng_t, dtype=G.dtype))
    return V, G.clamp(0.0, 1.0)


def block_xy(block, I, V, G):
    # 窓 (B, L+1) からブロックの入力 X (B, L, d_in) と、式で計算した正解 Y (B, L, d_out)
    I, V, G = I[:, :-1], V[:, :-1], G[:, :-1]
    if block == 1:
        return V[..., None], rates(V)
    if block == 2:
        ab = rates(V)
        return torch.cat([ab, G], -1), d_gates(ab, G)
    return torch.cat([G, V[..., None], I[..., None]], -1), d_V(G, V, I)[..., None]


def lr_at(it, iters, lr):
    # 余弦減衰: lr -> lr/100。最初の 2% は線形に上げる
    w = max(1, iters // 50)
    if it < w:
        return lr * (it + 1) / w
    return lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - w) / max(1, iters - w))))


# ---------------- (A) ブロックの学習 ----------------

def train_block(a):
    torch.manual_seed(a.seed)
    rng_t = torch.Generator().manual_seed(a.seed)
    data = Windows(a.data, a.seed)
    d_in, d_out = {1: (1, 6), 2: (9, 3), 3: (5, 1)}[a.block]
    net = Net(d_in, a.H, a.layers, d_out, feat=a.block if (a.feat and a.block > 1) else 0)
    if a.whh0:
        with torch.no_grad():
            for l in range(a.layers):
                getattr(net.rnn, f'weight_hh_l{l}').zero_()
    I, V, G = data.sample(256, a.L)
    X, Y = block_xy(a.block, I, V, G)
    net.set_scalers(X.reshape(-1, d_in).float(), Y.reshape(-1, d_out).float())
    if a.init:
        net = load_net(a.init).float()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    log = {'args': vars(a), 'loss': [], 'valid': []}
    t0 = time.time()
    for it in range(a.iters):
        for gr in opt.param_groups:
            gr['lr'] = lr_at(it, a.iters, a.lr)
        I, V, G = data.sample(a.batch, a.L)
        V, G = perturb(V, G, a.noise, rng_t)
        X, Y = block_xy(a.block, I, V, G)
        _, ys, _, _ = net(X.float())
        e2 = ((ys - (Y.float() - net.om) / net.os_) ** 2).mean(-1)          # (B, L)
        Vb = V[:, :-1]
        w = 1.0 + (a.wsub - 1.0) * ((Vb > -70) & (Vb < -40)).float()
        loss = (w * e2).sum() / w.sum()
        if a.whh_pen > 0:
            loss = loss + a.whh_pen * sum(getattr(net.rnn, f'weight_hh_l{l}').pow(2).mean()
                                          for l in range(a.layers))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        log['loss'].append(float(loss))
        if (it + 1) % a.log_every == 0 or it + 1 == a.iters:
            v = block_valid(net, data, a.block, a.L)
            log['valid'].append((it + 1, v))
            print('it {:5d} loss {:.3e}  valid RMSE 全体 {:.3e} 閾値下 {:.3e}  ({:.0f} s)'.format(
                it + 1, np.mean(log['loss'][-a.log_every:]), v['all'], v['sub'], time.time() - t0),
                flush=True)
    path = os.path.join(OUT_DIR, a.tag + '.pt')
    save_net(net, path, {'block': a.block, 'valid': log['valid'][-1][1]})
    net = net.double()
    b = {1: a.b1, 2: a.b2, 3: a.b3}
    b[a.block] = path
    res = eval3(b[1], b[2], b[3], a.tag)
    log['closed_loop'] = res
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


@torch.no_grad()
def block_valid(net, data, block, L):
    # 検証用の軌道で、出力ごとの RMSE（物理単位）を出力全体と閾値下に分けて
    st = data.rng.bit_generator.state
    data.rng = np.random.default_rng(12345)
    I, V, G = data.sample(64, L, valid=True)
    data.rng.bit_generator.state = st
    X, Y = block_xy(block, I, V, G)
    y, _, _, _ = net(X.float())
    e = (y.double() - Y)
    sub = (V[:, :-1] < -50)
    return {'all': float(e.pow(2).mean().sqrt()), 'sub': float(e[sub].pow(2).mean().sqrt()),
            'per_out': e.pow(2).mean((0, 1)).sqrt().tolist()}


# ---------------- (A) 3 ブロックの閉ループ ----------------

def run3(nets, I_np):
    n1, n2, n3 = nets
    B, N = I_np.shape
    I = torch.from_numpy(I_np)
    V = torch.full((B,), H_.V_REST, dtype=torch.float64)
    g = torch.tensor(H_.steady_state_gates(H_.V_REST), dtype=torch.float64).repeat(B, 1)
    h1, h2, h3 = n1.zeros(B, torch.float64), n2.zeros(B, torch.float64), n3.zeros(B, torch.float64)
    out = torch.empty(B, N - 1, dtype=torch.float64)
    with torch.no_grad():
        for k in range(N - 1):
            ab, h1 = n1.cell(V[:, None], h1)
            ab = ab.clamp(min=0.0)
            dg, h2 = n2.cell(torch.cat([ab, g], 1), h2)
            dv, h3 = n3.cell(torch.cat([g, V[:, None], I[:, k, None]], 1), h3)
            V = V + dv[:, 0]
            g = (g + dg).clamp(0.0, 1.0)
            out[:, k] = V
    return out.numpy()


BASE = os.path.join(hb.ROOT, 'results', 'hh_modular_ou')


def eval3(p1, p2, p3, label):
    nets = [load_net(p) for p in (p1, p2, p3)]
    return hb.score(lambda I: run3(nets, I), label)


# ---------------- (A) 3 ブロックの閉ループ微調整 ----------------

def chain_loss(nets, I, V, G, a):
    """
    窓 (B, burn+K+1) で、最初の burn ステップは各ブロックに正しい入力を入れて隠れ状態を作り、
    残り K ステップを 3 ブロックの閉ループで回す。損失は 2 つの和:
      traj   : 予測した軌道と正解の差 ((V̂-V)/20)^2 + ((ĝ-g)/0.1)^2 の平均
      onpol  : 予測した状態 (V̂, ĝ) での各ブロックの出力と、その状態での HH の式の値との差（標準化した単位）
               どのブロックの正解も入力の厳密な関数なので、閉ループでずれた状態でも正解が計算できる（DAgger と同じ考え）
    """
    n1, n2, n3 = nets
    B = I.shape[0]
    h = [n.zeros(B, torch.float64) for n in nets]
    for k in range(a.burn):                              # 教師強制で隠れ状態を作る（勾配は流さない）
        with torch.no_grad():
            _, h[0] = n1.cell(V[:, k, None], h[0])
            _, h[1] = n2.cell(torch.cat([rates(V[:, k]), G[:, k]], 1), h[1])
            _, h[2] = n3.cell(torch.cat([G[:, k], V[:, k, None], I[:, k, None]], 1), h[2])
    v, g = V[:, a.burn], G[:, a.burn]
    traj = onpol = 0.0
    for k in range(a.burn, a.burn + a.K):
        ab, h[0] = n1.cell(v[:, None], h[0])
        dg, h[1] = n2.cell(torch.cat([ab.clamp(min=0.0), g], 1), h[1])
        dv, h[2] = n3.cell(torch.cat([g, v[:, None], I[:, k, None]], 1), h[2])
        if a.onpol > 0:
            vd, gd = v.detach(), g.detach()
            ab_t = rates(vd)
            onpol = onpol + (((ab - ab_t) / n1.os_) ** 2).mean() \
                + (((dg - d_gates(ab.detach().clamp(min=0.0) if a.local2 else ab_t, gd)) / n2.os_) ** 2).mean() \
                + (((dv[:, 0] - d_V(gd, vd, I[:, k])) / n3.os_) ** 2).mean()
        v = v + dv[:, 0]
        g = (g + dg).clamp(0.0, 1.0)
        Vt = V[:, k + 1]
        w = 1.0 + (a.wthr - 1.0) * ((Vt > -65) & (Vt < -45)).double()     # 閾値付近の重み
        traj = traj + (w * ((v - Vt) / 20.0) ** 2).mean() + (((g - G[:, k + 1]) / 0.1) ** 2).mean()
    return traj / a.K, (onpol / a.K if a.onpol > 0 else torch.zeros(()))


def train_chain(a):
    torch.manual_seed(a.seed)
    data = Windows(a.data, a.seed)
    paths = [a.b1, a.b2, a.b3]
    nets = [load_net(p).double() for p in paths]
    train = [int(c) for c in a.train]                   # 微調整するブロック（例 '3' や '123'）
    params = [p for i in train for p in nets[i - 1].parameters()]
    for i, n in enumerate(nets):
        n.requires_grad_((i + 1) in train)
    opt = torch.optim.Adam(params, lr=a.lr)
    log = {'args': vars(a), 'traj': [], 'onpol': [], 'eval': []}
    t0 = time.time()
    best = (-1, -1.0, 0)
    save = lambda tag: [save_net(nets[i - 1], os.path.join(OUT_DIR, f'{tag}_b{i}.pt'), {}) for i in train]
    def evaluate(it):
        # 最良の時点は検証用の電流（hb.VAL）で選ぶ。26 本のテストは最後に 1 回だけ使う
        nonlocal best
        r = hb.score(lambda I_: run3(nets, I_), '{} it {} val'.format(a.tag, it), val=True)
        log['eval'].append((it, r['n100'], r['mean']))
        if (r['n100'], r['mean']) > best:
            best = (r['n100'], r['mean'], it)
            save(a.tag)
        return r
    # K の段階（例 '100:300,500:300,2000:300' = K=100 を 300 回、K=500 を 300 回、...）
    sched = [tuple(int(x) for x in st.split(':')) for st in a.K_sched.split(',')] if a.K_sched         else [(a.K, a.iters)]
    Ks = [K for K, n in sched for _ in range(n)]
    a.iters = len(Ks)
    evaluate(0)
    for it in range(a.iters):
        for gr in opt.param_groups:
            gr['lr'] = lr_at(it, a.iters, a.lr)
        a.K = Ks[it]
        I, V, G = data.sample(a.batch, a.burn + a.K)
        traj, onpol = chain_loss(nets, I, V, G, a)
        loss = a.wtraj * traj + a.onpol * onpol
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log['traj'].append(float(traj)); log['onpol'].append(float(onpol))
        if (it + 1) % a.log_every == 0:
            print('it {:5d} K {} traj {:.3e} onpol {:.3e}  ({:.0f} s)'.format(
                it + 1, a.K, np.mean(log['traj'][-a.log_every:]), np.mean(log['onpol'][-a.log_every:]),
                time.time() - t0), flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            evaluate(it + 1)
    log['best'] = best
    for i in train:                                   # 検証で最良だった時点に戻して 26 本で評価
        nets[i - 1] = load_net(os.path.join(OUT_DIR, f'{a.tag}_b{i}.pt'))
    log['test'] = hb.score(lambda I_: run3(nets, I_), '{} (val best it {})'.format(a.tag, best[2]))
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


# ---------------- (B) 丸ごと型: 閉ループを主にした学習 ----------------

def gen_current(B, N, rng):
    """
    学習用の電流をその場で作る (B, N)。評価の電流とは独立に、切り替え・ランプ・揺らぎ・短いパルスを混ぜる。
      基準レベル: 2〜50 ms ごとに切り替える。0（30%）か U[-5, 25]。区間ごとに「一定」か「直線で次へ」（30%）
      揺らぎ: OU（τ = 1〜5 ms、σ = U[0, 3]）
      短いパルス: 平均 50 ms に 1 回、幅 0.5〜2 ms、高さ U[10, 40]
    """
    I = np.zeros((B, N))
    for b in range(B):
        t = 0
        lvl = 0.0 if rng.random() < 0.3 else rng.uniform(-5, 25)
        while t < N:
            L = max(1, int(rng.uniform(2, 50) / DT))
            nxt = 0.0 if rng.random() < 0.3 else rng.uniform(-5, 25)
            seg = np.linspace(lvl, nxt, L) if rng.random() < 0.3 else np.full(L, lvl)
            I[b, t:t + L] = seg[:N - t]
            t += L
            lvl = nxt
        tau, sig = rng.uniform(1, 5), rng.uniform(0, 3)
        a = math.exp(-DT / tau)
        e = rng.standard_normal(N) * sig * math.sqrt(1 - a * a)
        ou = np.zeros(N)
        for k in range(1, N):
            ou[k] = a * ou[k - 1] + e[k]
        I[b] += ou
        for k in rng.integers(0, N, size=rng.poisson(N * DT / 50)):
            I[b, k:k + int(rng.uniform(0.5, 2) / DT)] += rng.uniform(10, 40)
    return torch.from_numpy(I)


@torch.no_grad()
def hh_sim(I):
    # 静止状態から HH の陽的オイラー法。I: (B, N) -> V: (B, N+1), G: (B, N+1, 3)
    B, N = I.shape
    V = torch.full((B,), H_.V_REST, dtype=torch.float64)
    g = torch.tensor(H_.steady_state_gates(H_.V_REST), dtype=torch.float64).repeat(B, 1)
    Vs, Gs = [V], [g]
    for k in range(N):
        V, g = V + d_V(g, V, I[:, k]), g + d_gates(rates(V), g)
        Vs.append(V)
        Gs.append(g)
    return torch.stack(Vs, 1), torch.stack(Gs, 1)


def exact_step(x):
    # x: (..., 5) = V, m, h, n, I -> HH のオイラー法の 1 ステップの増分 (..., 4) = ΔV, Δm, Δh, Δn
    V, g, I = x[..., 0], x[..., 1:4], x[..., 4]
    return torch.cat([d_V(g, V, I)[..., None], d_gates(rates(V), g)], -1)


Q_MAX = 0.2


def phys_targets(V):
    # HH の式でのゲートの定常値 g∞ = α/(α+β) と、1 ステップで近づく割合 q = dt (α+β)。V: (B,) -> (B, 3), (B, 3)
    ab = rates(V)
    a_, b_ = ab[..., 0::2], ab[..., 1::2]
    return a_ / (a_ + b_), DT * (a_ + b_)


def state_step(net, x, hs):
    """
    状態を戻す型の 1 ステップ。x: (B, 5) = V, m, h, n, I -> 増分 (B, 4), hs, 補助の出力
      通常: ネットが [ΔV, Δm, Δh, Δn] を直接出す
      phys: ネットは V だけを見て g∞ と q を出す（sigmoid）。Δg = q (g∞ - g)、ΔV は HH の電流の式
            （ゲートの速さの関数だけを学習し、電流の式は既知として組み込む）
    """
    if net.cfg.get('phys'):
        z, hs = net.cell(x[:, :1], hs)
        ginf, q = torch.sigmoid(z[:, :3]), Q_MAX * torch.sigmoid(z[:, 3:])
        g = x[:, 1:4]
        dV = d_V(g, x[:, 0], x[:, 4])
        return torch.cat([dV[:, None], q * (ginf - g)], 1), hs, (ginf, q)
    y, hs = net.cell(x, hs)
    return y, hs, None


def run_state(net, I_np, rest_ms=20.0):
    # 状態を戻す型の閉ループ。入力 [V, m, h, n, I]、出力 [ΔV, Δm, Δh, Δn]
    B, N = I_np.shape
    I = torch.from_numpy(I_np)
    hs = net.zeros(B, torch.float64)
    V = torch.full((B,), H_.V_REST, dtype=torch.float64)
    g = torch.tensor(H_.steady_state_gates(H_.V_REST), dtype=torch.float64).repeat(B, 1)
    out = torch.empty(B, N - 1, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(int(rest_ms / DT)):
            _, hs, _ = state_step(net, torch.cat([V[:, None], g, torch.zeros(B, 1, dtype=torch.float64)], 1), hs)
        for k in range(N - 1):
            y, hs, _ = state_step(net, torch.cat([V[:, None], g, I[:, k, None]], 1), hs)
            V = (V + y[:, 0]).clamp(-100.0, 60.0)
            g = (g + y[:, 1:]).clamp(0.0, 1.0)
            out[:, k] = V
    return out.numpy()


def train_single_cl(a):
    """
    評価と同じ条件（静止状態から、隠れ状態 0、最初に rest_ms だけ静止状態と I=0 を入れる）で閉ループに回して学習する。
      --state なし: 入力 [V, I]、出力 ΔV。ゲート変数は隠れ状態が持つ
      --state あり: 入力 [V, m, h, n, I]、出力 [ΔV, Δm, Δh, Δn]。予測したゲート変数も次の入力に戻す
                    出力の正解は入力の厳密な関数なので、ずれた状態でも式から計算できる（--onpol の損失）
    戻す状態は「予測 + α (正解 - 予測)」。α は 1（教師強制）から 0（完全な閉ループ）へ下げる。
    長い系列は seg ステップごとに区切って更新する（truncated BPTT。隠れ状態と状態は切り離して引き継ぐ）。
    V は [-100, 60] mV、ゲートは [0, 1] に制限する（評価でも同じ）。
    """
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    st = a.state or a.phys
    d_in, d_out = (1, 6) if a.phys else ((5, 4) if st else (2, 1))
    net = Net(d_in, a.H, a.layers, d_out, d_aux=3 if (a.aux > 0 and not st) else 0, cell=a.cell, phys=a.phys)
    I0 = gen_current(32, 20000, rng)
    V0, G0 = hh_sim(I0)
    if a.phys:
        Vflat = V0.reshape(-1, 1).float()
        net.im.copy_(Vflat.mean(0)); net.is_.copy_(Vflat.std(0))
    elif st:
        X0 = torch.cat([V0[:, :-1, None], G0[:, :-1], I0[..., None]], -1)
        net.set_scalers(X0.reshape(-1, 5).float(), exact_step(X0).reshape(-1, 4).float())
    else:
        net.set_scalers(torch.stack([V0[:, :-1], I0], -1).reshape(-1, 2).float(),
                        (V0[:, 1:] - V0[:, :-1]).reshape(-1, 1).float())
    if a.init:
        d = torch.load(a.init, weights_only=False)
        net = Net(**d['cfg'])
        net.load_state_dict(d['state'])
    net = net.float()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    log = {'args': vars(a), 'loss': [], 'eval': []}
    path = os.path.join(OUT_DIR, a.tag + '.pt')
    best = (-1, -1.0, 0)
    t0 = time.time()
    n_rest = int(a.rest_ms / DT)
    N = int(a.ep_ms / DT)
    runner = run_state if st else run_single

    def evaluate(it):
        nonlocal best
        net.double()
        r = hb.score(lambda I_: runner(net, I_, a.rest_ms), '{} it {} val'.format(a.tag, it), val=True)
        net.float()
        log['eval'].append((it, r['n100'], r['mean']))
        if (r['n100'], r['mean']) > best[:2]:
            best = (r['n100'], r['mean'], it)
            save_net(net, path, {'it': it})

    evaluate(0)
    if a.pre_iters > 0:
        # 事前学習（Markov 型向け）: 教師の状態に摂動を足し、式で計算した 1 ステップの正解に合わせる
        bI = gen_current(a.bank, N, rng)
        bV, bG = hh_sim(bI)
        X = torch.cat([bV[:, :-1, None], bG[:, :-1], bI[..., None]], -1).reshape(-1, 5)
        popt = torch.optim.Adam(net.parameters(), lr=a.pre_lr)
        for pi in range(a.pre_iters):
            for gr in popt.param_groups:
                gr['lr'] = lr_at(pi, a.pre_iters, a.pre_lr)
            x = X[torch.from_numpy(rng.integers(0, X.shape[0], size=a.pre_batch))].clone()
            mask = (torch.rand(a.pre_batch, 1, dtype=x.dtype) < 0.5).to(x.dtype)
            x[:, :1] += mask * a.pre_sv * torch.randn(a.pre_batch, 1, dtype=x.dtype)
            x[:, 1:4] = (x[:, 1:4] + mask * a.pre_sg * torch.randn(a.pre_batch, 3, dtype=x.dtype)).clamp(0, 1)
            xf = x.float()
            if a.phys:
                _, _, (ginf, q) = state_step(net, xf, net.zeros(a.pre_batch, torch.float32))
                gt, qt = phys_targets(x[:, 0])
                ploss = (((ginf - gt.float()) / 0.1) ** 2).mean() + (((q - qt.float()) / 0.01) ** 2).mean()
            else:
                y, _, _ = state_step(net, xf, net.zeros(a.pre_batch, torch.float32))
                yt = exact_step(x).float()
                if a.pre_rel > 0:
                    # ΔV は増分の大きさで割った相対誤差（下限 pre_rel mV）の Huber。静止付近の小さな ΔV も同じ重みで合わせる
                    e0 = (y[:, 0] - yt[:, 0]) / yt[:, 0].abs().clamp(min=a.pre_rel)
                    ploss = nn.functional.huber_loss(e0, torch.zeros_like(e0), delta=1.0)                         + (((y[:, 1:] - yt[:, 1:]) / net.os_[1:]) ** 2).mean()
                else:
                    ploss = (((y - yt) / net.os_) ** 2).mean()
            popt.zero_grad()
            ploss.backward()
            popt.step()
            if (pi + 1) % 500 == 0:
                print('pre {:5d} loss {:.3e}  ({:.0f} s)'.format(pi + 1, ploss.item(), time.time() - t0), flush=True)
        evaluate(0)
    for it in range(a.iters):
        alpha = max(0.0, 1.0 - it / max(1, a.alpha_iters))
        for gr in opt.param_groups:
            gr['lr'] = lr_at(it, a.iters, a.lr)
        if it % a.bank_every == 0:            # 教師の軌道の在庫を作り直す（HH の計算はバッチが大きくても時間が同じ）
            bI = gen_current(a.bank, N, rng)
            bV, bG = hh_sim(bI)
        idx = torch.from_numpy(rng.integers(0, a.bank, size=a.batch))
        If, Vf, Gf = bI[idx].float(), bV[idx].float(), bG[idx].float()
        hs = net.zeros(a.batch, torch.float32)
        with torch.no_grad():
            r0 = torch.cat([Vf[:, :1], Gf[:, 0], torch.zeros(a.batch, 1)], 1) if st else \
                torch.stack([Vf[:, 0], torch.zeros(a.batch)], 1)
            for _ in range(n_rest):
                hs = state_step(net, r0, hs)[1] if st else net.cell(r0, hs)[1]
        v, g = Vf[:, 0], Gf[:, 0]
        tot = 0.0
        for s0 in range(0, N, a.seg):
            loss = 0.0
            for k in range(s0, min(N, s0 + a.seg)):
                vin = v + alpha * (Vf[:, k] - v)
                if st:
                    gin = g + alpha * (Gf[:, k] - g)
                    x = torch.cat([vin[:, None], gin, If[:, k, None]], 1)
                    y, hs, ph = state_step(net, x, hs)
                    if a.onpol > 0 and a.phys:
                        gt, qt = phys_targets(x[:, 0].detach().double())
                        loss = loss + a.onpol * ((((ph[0] - gt.float()) / 0.1) ** 2).mean(-1)
                                                 + (((ph[1] - qt.float()) / 0.01) ** 2).mean(-1))
                    elif a.onpol > 0:
                        yt = exact_step(x.detach().double()).float()
                        loss = loss + a.onpol * (((y - yt) / net.os_) ** 2).mean(-1)
                    v = (vin + y[:, 0]).clamp(-100.0, 60.0)
                    g = (gin + y[:, 1:]).clamp(0.0, 1.0)
                    loss = loss + (((g - Gf[:, k + 1]) / 0.1) ** 2).mean(-1)
                else:
                    y, hs = net.cell(torch.stack([vin, If[:, k]], 1), hs)
                    v = (vin + y[:, 0]).clamp(-100.0, 60.0)
                    if net.aux is not None:
                        loss = loss + a.aux * ((net.aux(hs[-1]) - (Gf[:, k + 1] - G_MEAN) / G_STD) ** 2).mean(-1)
                loss = loss + ((v - Vf[:, k + 1]) / 20.0) ** 2
            loss = loss.mean() / a.seg
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            hs = [h.detach() for h in hs]
            v, g = v.detach(), g.detach()
        log['loss'].append(tot / math.ceil(N / a.seg))
        if (it + 1) % a.log_every == 0:
            print('it {:5d} α {:.2f} loss {:.3e}  ({:.0f} s)'.format(
                it + 1, alpha, np.mean(log['loss'][-a.log_every:]), time.time() - t0), flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            evaluate(it + 1)
    d = torch.load(path, weights_only=False)
    net = Net(**d['cfg'])
    net.load_state_dict(d['state'])
    net = net.double()
    log['best'] = best
    log['test'] = hb.score(lambda I_: runner(net, I_, a.rest_ms),
                           '{} (val best it {})'.format(a.tag, best[2]))
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


# ---------------- (B) 丸ごと型 ----------------

def single_loss(net, I, V, G, a, rng_t):
    # 教師強制: 入力 [V(t), I(t)]、正解 ΔV。最初の burn ステップは隠れ状態を作る区間として損失に入れない
    Vin = V
    if a.noise > 0:
        Vin = V + a.noise * 0.4 * torch.randn(V.shape, generator=rng_t, dtype=V.dtype)
    X = torch.stack([Vin[:, :-1], I[:, :-1]], -1)
    Y = (V[:, 1:] - Vin[:, :-1])[..., None]          # 雑音を足した V から正しい次の V へ戻す量
    _, ys, z, _ = net(X.float())
    e2 = ((ys - (Y.float() - net.om) / net.os_) ** 2)[:, a.burn:, 0]
    Vb = V[:, a.burn:-1]
    w = 1.0 + (a.wsub - 1.0) * ((Vb > -70) & (Vb < -40)).float()
    loss = (w * e2).sum() / w.sum()
    if net.aux is not None:
        gz = (G[:, :-1].float() - G_MEAN) / G_STD
        loss = loss + a.aux * ((net.aux(z) - gz) ** 2)[:, a.burn:].mean()
    return loss


G_MEAN = torch.tensor([0.1, 0.5, 0.4])
G_STD = torch.tensor([0.2, 0.2, 0.15])


def rollout_loss(net, I, V, a):
    # 窓の前半 burn ステップは教師強制で隠れ状態を作り、残り K ステップを閉ループで回す
    hs = net.zeros(I.shape[0], torch.float32)
    If, Vf = I.float(), V.float()
    for k in range(a.burn):
        _, hs = net.cell(torch.stack([Vf[:, k], If[:, k]], 1), hs)
    v = Vf[:, a.burn]
    loss = 0.0
    for k in range(a.burn, a.burn + a.rollout):
        y, hs = net.cell(torch.stack([v, If[:, k]], 1), hs)
        v = v + y[:, 0]
        loss = loss + ((v - Vf[:, k + 1]) / 20.0) ** 2
    return loss.mean() / a.rollout


def run_single(net, I_np, rest_ms=20.0):
    # 静止状態の隠れ状態を作るため、最初に rest_ms だけ V = V_REST, I = 0 を入れてから閉ループ
    B, N = I_np.shape
    I = torch.from_numpy(I_np)
    hs = net.zeros(B, torch.float64)
    V = torch.full((B,), H_.V_REST, dtype=torch.float64)
    out = torch.empty(B, N - 1, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(int(rest_ms / DT)):
            _, hs = net.cell(torch.stack([V, torch.zeros_like(V)], 1), hs)
        for k in range(N - 1):
            y, hs = net.cell(torch.stack([V, I[:, k]], 1), hs)
            V = (V + y[:, 0]).clamp(-100.0, 60.0)
            out[:, k] = V
    return out.numpy()


def train_single(a):
    torch.manual_seed(a.seed)
    rng_t = torch.Generator().manual_seed(a.seed)
    data = Windows(a.data, a.seed)
    net = Net(2, a.H, a.layers, 1, d_aux=3 if a.aux > 0 else 0, leak=a.leak)
    I, V, G = data.sample(256, a.L)
    net.set_scalers(torch.stack([V[:, :-1], I[:, :-1]], -1).reshape(-1, 2).float(),
                    (V[:, 1:] - V[:, :-1]).reshape(-1, 1).float())
    if a.init:
        d = torch.load(a.init, weights_only=False)
        net.load_state_dict(d['state'])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    log = {'args': vars(a), 'loss': [], 'valid': []}
    t0 = time.time()
    path = os.path.join(OUT_DIR, a.tag + '.pt')
    for it in range(a.iters):
        for gr in opt.param_groups:
            gr['lr'] = lr_at(it, a.iters, a.lr)
        if a.rollout > 0:
            I, V, G = data.sample(a.batch, a.burn + a.rollout)
            loss = rollout_loss(net, I, V, a)
        else:
            I, V, G = data.sample(a.batch, a.L)
            loss = single_loss(net, I, V, G, a, rng_t)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        log['loss'].append(float(loss))
        if (it + 1) % a.log_every == 0 or it + 1 == a.iters:
            print('it {:5d} loss {:.3e}  ({:.0f} s)'.format(
                it + 1, np.mean(log['loss'][-a.log_every:]), time.time() - t0), flush=True)
            save_net(net, path, {})
        if a.eval_every and (it + 1) % a.eval_every == 0 and it + 1 < a.iters:
            r = hb.score(lambda I_: run_single(net.double(), I_), '{} it {}'.format(a.tag, it + 1))
            net.float()
            log['valid'].append((it + 1, r['n100'], r['mean']))
    save_net(net, path, {})
    res = hb.score(lambda I_: run_single(net.double(), I_), a.tag)
    log['closed_loop'] = res
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--tag', required=False, default='test')
    common.add_argument('--data', default=hb.DATA_OU)
    common.add_argument('--H', type=int, default=64)
    common.add_argument('--layers', type=int, default=1)
    common.add_argument('--iters', type=int, default=3000)
    common.add_argument('--batch', type=int, default=32)
    common.add_argument('--L', type=int, default=1000)
    common.add_argument('--lr', type=float, default=1e-3)
    common.add_argument('--noise', type=float, default=0.0)
    common.add_argument('--wsub', type=float, default=1.0)
    common.add_argument('--seed', type=int, default=0)
    common.add_argument('--threads', type=int, default=2)
    common.add_argument('--log-every', type=int, default=200)
    common.add_argument('--init', default=None)
    pb = sub.add_parser('block', parents=[common])
    pb.add_argument('--block', type=int, required=True, choices=[1, 2, 3])
    pb.add_argument('--feat', action='store_true', help='積の特徴量を入力に足す（RNN2, RNN3）')
    pb.add_argument('--whh0', action='store_true', help='再帰の重みを 0 から始める')
    pb.add_argument('--whh-pen', type=float, default=0.0, help='再帰の重みの 2 乗の罰則')
    for k in (1, 2, 3):
        pb.add_argument(f'--b{k}', default=os.path.join(BASE, f'block{k}.npz'),
                        help='閉ループ評価で組み合わせるほかのブロック')
    ps = sub.add_parser('single', parents=[common])
    ps.add_argument('--aux', type=float, default=0.0)
    ps.add_argument('--leak', type=float, default=0.0, help='漏れ積分型の隠れ状態（例 0.05）')
    ps.add_argument('--burn', type=int, default=300)
    ps.add_argument('--rollout', type=int, default=0)
    ps.add_argument('--eval-every', type=int, default=0)
    pc = sub.add_parser('chain', parents=[common])
    for k in (1, 2, 3):
        pc.add_argument(f'--b{k}', default=os.path.join(BASE, f'block{k}.npz'))
    pc.add_argument('--train', default='123', help='微調整するブロック')
    pc.add_argument('--K', type=int, default=100, help='閉ループで回すステップ数')
    pc.add_argument('--burn', type=int, default=200)
    pc.add_argument('--K-sched', default='', help="K の段階 'K:回数,K:回数,...'（指定すると --K, --iters より優先）")
    pc.add_argument('--wtraj', type=float, default=1.0)
    pc.add_argument('--onpol', type=float, default=1.0)
    pc.add_argument('--eval-every', type=int, default=200)
    pc.add_argument('--wthr', type=float, default=1.0, help='教師 V が -65〜-45 mV の軌道損失の重み')
    pc.add_argument('--local2', action='store_true', help='RNN2 の on-policy 教師を予測した rate から計算する')
    pl = sub.add_parser('single_cl', parents=[common])
    pl.add_argument('--state', action='store_true', help='m, h, n も予測して次の入力に戻す')
    pl.add_argument('--phys', action='store_true', help='ゲートの g∞ と q だけを学習し、電流の式は HH のものを使う')
    pl.add_argument('--pre-iters', type=int, default=0, help='摂動した状態での 1 ステップ事前学習の回数')
    pl.add_argument('--pre-lr', type=float, default=1e-3)
    pl.add_argument('--pre-batch', type=int, default=512)
    pl.add_argument('--pre-sv', type=float, default=2.0, help='事前学習の V の摂動 [mV]')
    pl.add_argument('--pre-sg', type=float, default=0.02, help='事前学習のゲートの摂動')
    pl.add_argument('--pre-rel', type=float, default=0.0, help='>0 なら ΔV を相対誤差で学習（値は下限 [mV]）')
    pl.add_argument('--onpol', type=float, default=1.0, help='--state のとき、式で計算した 1 ステップの正解との損失の重み')
    pl.add_argument('--cell', default='rnn', choices=['rnn', 'gru', 'mlp'])
    pl.add_argument('--aux', type=float, default=0.0)
    pl.add_argument('--ep-ms', type=float, default=100.0, help='1 本の学習系列の長さ [ms]')
    pl.add_argument('--seg', type=int, default=500, help='truncated BPTT の区切り（ステップ）')
    pl.add_argument('--rest-ms', type=float, default=20.0)
    pl.add_argument('--alpha-iters', type=int, default=300, help='α を 1 から 0 へ下げる回数')
    pl.add_argument('--eval-every', type=int, default=100)
    pl.add_argument('--bank', type=int, default=512, help='教師の軌道の在庫の本数')
    pl.add_argument('--bank-every', type=int, default=200, help='在庫を作り直す間隔（回）')
    pe = sub.add_parser('eval3')
    for k in (1, 2, 3):
        pe.add_argument(f'--b{k}', default=os.path.join(BASE, f'block{k}.npz'))
    pe.add_argument('--label', default='eval3')
    pe.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.cmd == 'block':
        train_block(a)
    elif a.cmd == 'single':
        train_single(a)
    elif a.cmd == 'chain':
        train_chain(a)
    elif a.cmd == 'single_cl':
        train_single_cl(a)
    else:
        eval3(a.b1, a.b2, a.b3, a.label)


if __name__ == '__main__':
    main()
