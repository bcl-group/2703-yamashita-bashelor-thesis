"""
モデル2（RNN 3 個）を、3 つをつないだ閉ループのまま学習する（多段の損失 / rollout 学習）。

これまでの学習（hh_modular.py train）はブロックごとの教師強制で、損失は「正解の状態から
1 ステップ進めたときの誤差」の 2 乗平均だった。しかし閉ループで効くのは、誤差が何千ステップも
積み重なった結果（発火周期のずれなど）で、1 ステップの損失を下げても閉ループが良くなるとは限らない
（追加学習で教師強制の損失は下がったのに、const 7 の周期誤差は +0.95% → -1.83% と符号が反転した）。

そこで、評価と同じ閉ループを PyTorch で組み、K ステップ回した軌道と正解（HH）の差を損失にする。

    1. 教師データ（teacher_ou.npz）からランダムな位置 s を選ぶ
    2. s - W から s までは正解の V, m, h, n を入れて回し（教師強制、勾配なし）、隠れ状態を作る
    3. s から K ステップ、予測した V, m, h, n を次の入力に戻して回す（閉ループ、hm.run_closed_loop と同じ式）
    4. 損失 = (1/K) Σ_k [ ((V̂_k - V_k)/σ_V)^2 + Σ_x ((x̂_k - x_k)/σ_x)^2 ] / 4    x = m, h, n
       σ は教師データでの各変数の標準偏差

重み・標準化はブロックごとの npz（hh_modular.save_block の形式）から読み、学習後も同じ形式で書き出す。
途中で検証用の電流（テストの 26 電流とは別）を numpy の閉ループで回し、スパイク時刻の一致率が
最もよい重みを残す。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_rollout.py \
        --init results/hh_modular_ou --out results/hh_modular_rollout
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_modular as hm   # src を sys.path に入れる
from neuron.hh import simulate, V_REST
from neuron import current as cur
from test_currents import spike_times, match_rate

DT = 0.01
ROOT = os.path.join(hm.BASE_DIR, '..', '..')


class Block(nn.Module):
    """NumPy 版 SimpleRNN 1 個 + 入出力の標準化。1 ステップずつ回す。"""

    def __init__(self, model, in_s, out_s):
        super().__init__()
        H = model.hidden_dim
        # float64 で作ってから重みを写す（float32 で作ると写した時点で丸められる）
        self.cell = nn.RNNCell(model.input_dim, H, nonlinearity='tanh', dtype=torch.float64)
        self.out = nn.Linear(H, model.output_dim, dtype=torch.float64)
        self.target = model.target
        with torch.no_grad():
            self.cell.weight_ih.copy_(torch.from_numpy(model.W_xh.T.copy()))
            self.cell.weight_hh.copy_(torch.from_numpy(model.W_hh.T.copy()))
            self.cell.bias_ih.copy_(torch.from_numpy(model.b_h.copy()))
            self.cell.bias_hh.zero_()
            self.out.weight.copy_(torch.from_numpy(model.W_hy.T.copy()))
            self.out.bias.copy_(torch.from_numpy(model.b_y.copy()))
        self.cell.bias_hh.requires_grad_(False)
        for name, a in [('im', in_s.mean), ('is_', in_s.std), ('om', out_s.mean), ('os_', out_s.std)]:
            self.register_buffer(name, torch.from_numpy(np.asarray(a, dtype=float)))

    def forward(self, x, h):
        h = self.cell((x - self.im) / self.is_, h)
        return self.out(h) * self.os_ + self.om, h

    def write_back(self, model):
        f = lambda p: p.detach().double().numpy()
        model.W_xh = f(self.cell.weight_ih).T.copy()
        model.W_hh = f(self.cell.weight_hh).T.copy()
        model.b_h = f(self.cell.bias_ih) + f(self.cell.bias_hh)
        model.W_hy = f(self.out.weight).T.copy()
        model.b_y = f(self.out.bias).copy()


class Chain(nn.Module):
    """3 ブロックの閉ループ。式は hh_modular.run_closed_loop(clip=True) と同じ。"""

    def __init__(self, blocks):
        super().__init__()
        self.b1, self.b2, self.b3 = (Block(*blocks[k]) for k in (1, 2, 3))

    def init_hidden(self, B):
        return [torch.zeros(B, b.cell.hidden_size, dtype=torch.float64) for b in (self.b1, self.b2, self.b3)]

    def step(self, V, g, I, hs):
        # V: (B,), g: (B, 3), I: (B,) -> 次の V, g
        h1, h2, h3 = hs
        ab, h1 = self.b1(V[:, None], h1)
        ab = torch.clamp(ab, min=0.0)
        dg, h2 = self.b2(torch.cat([ab, g], 1), h2)
        dv, h3 = self.b3(torch.cat([g, V[:, None], I[:, None]], 1), h3)
        g_next = torch.clamp(g + dg, 0.0, 1.0)
        V_next = V + dv[:, 0]
        return V_next, g_next, [h1, h2, h3]


def load_blocks(d):
    return {b: hm.load_block(os.path.join(d, 'block{}.npz'.format(b)))[:3] for b in (1, 2, 3)}


# --- 検証用の電流（test_currents.py の 26 電流とは別にする） ---
VALID = [
    ('const 6.5', lambda t: cur.constant_current(t, 6.5), 300.0),
    ('const 8',   lambda t: cur.constant_current(t, 8.0), 300.0),
    ('sin 8Hz 7±7', lambda t: cur.sin_current(t, 7.0, 8.0, 7.0), 500.0),
    ('OU 8±3 seed 11', lambda t: cur.ou_current(t, 8.0, 3.0, 1.0, seed=11), 500.0),
    ('random step seed 5', lambda t: cur.random_pulse_current(t, -5.0, 20.0, 20.0, seed=5), 500.0),
    ('step -3', lambda t: cur.step_current(t, -3.0, 50.0, 150.0), 300.0),
]


def make_valid():
    cases = []
    for name, f, T in VALID:
        t = np.arange(0, T, DT)
        I = f(t)
        V = simulate(t, I, V0=V_REST)[0]
        cases.append((name, I, V))
    return cases


def validate(blocks, cases):
    rates = []
    for name, I, V in cases:
        Vp = hm.run_closed_loop(blocks, I, V0=V[0], clip=True)[0]
        rates.append(match_rate(spike_times(V[1:], DT), spike_times(Vp, DT)))
    return float(np.mean(rates)), rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', default=os.path.join(ROOT, 'results', 'hh_modular_ou'))
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'hh_modular_rollout'))
    p.add_argument('--data', default=os.path.join(ROOT, 'data', 'hh', 'teacher_ou.npz'))
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--K', type=int, default=1000, help='閉ループで回すステップ数（勾配を流す長さ）')
    p.add_argument('--warmup', type=int, default=500, help='隠れ状態を作る教師強制のステップ数')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--valid-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    blocks = load_blocks(a.init)
    chain = Chain(blocks).double()
    d = np.load(a.data)
    I_all, V_all = d['I_ext'], d['V']
    G_all = np.stack([d['m'], d['h'], d['n']], axis=-1)        # (n_traj, N, 3)
    sig = torch.tensor([V_all.std()] + list(G_all.reshape(-1, 3).std(0)))
    n_traj, N = V_all.shape
    # 初期状態がランダムなので、最初の 50 ms（過渡応答）は始点に選ばない
    s_min = max(a.warmup, 5000)

    cases = make_valid()
    best, rates = validate(blocks, cases)
    print('init  valid match = {:.3f}  {}'.format(best, np.round(rates, 2)), flush=True)
    log = {'args': vars(a), 'valid': [(0, best, rates)], 'loss': []}
    hm_save = lambda tag: [hm.save_block({
        'block': b, 'target': blocks[b][0].target, 'seed': a.seed, 'lr': a.lr, 'chunk_ms': 10.0,
        'data_path': a.data, 'hist': {'loss': []}, 'model': blocks[b][0],
        'in_scaler': blocks[b][1], 'out_scaler': blocks[b][2]}, a.out) for b in (1, 2, 3)]
    hm_save('init')

    params = [q for q in chain.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, amsgrad=True)
    start = time.time()
    for it in range(1, a.iters + 1):
        ti = rng.integers(n_traj, size=a.batch)
        si = rng.integers(s_min, N - a.K - 1, size=a.batch)
        rows = lambda X, lo, hi: torch.from_numpy(np.stack([X[i, s + lo:s + hi] for i, s in zip(ti, si)]))
        I_w, V_w, G_w = rows(I_all, -a.warmup, 0), rows(V_all, -a.warmup, 0), rows(G_all, -a.warmup, 0)
        I_k, V_k, G_k = rows(I_all, 0, a.K), rows(V_all, 0, a.K + 1), rows(G_all, 0, a.K + 1)

        hs = chain.init_hidden(a.batch)
        with torch.no_grad():                       # 教師強制で隠れ状態を作る
            for k in range(a.warmup):
                _, _, hs = chain.step(V_w[:, k], G_w[:, k], I_w[:, k], hs)
        V, g = V_k[:, 0], G_k[:, 0]
        loss = 0.0
        for k in range(a.K):                        # 閉ループ
            V, g, hs = chain.step(V, g, I_k[:, k], hs)
            e = torch.cat([((V - V_k[:, k + 1]) / sig[0])[:, None], (g - G_k[:, k + 1]) / sig[1:]], 1)
            loss = loss + (e ** 2).mean()
        loss = loss / a.K
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log['loss'].append(float(loss.detach()))

        if it % 10 == 0:
            print('iter {:5d}  loss {:.4e}  ({:.0f} s)'.format(
                it, np.mean(log['loss'][-10:]), time.time() - start), flush=True)
        if it % a.valid_every == 0:
            for b, blk in zip((1, 2, 3), (chain.b1, chain.b2, chain.b3)):
                blk.write_back(blocks[b][0])
            score, rates = validate(blocks, cases)
            log['valid'].append((it, score, rates))
            flag = ''
            if score >= best:
                best = score
                hm_save('best')
                flag = '  <- best（保存）'
            print('valid match = {:.3f}  {}{}'.format(score, np.round(rates, 2), flag), flush=True)
            json.dump(log, open(os.path.join(a.out, 'rollout_log.json'), 'w'), indent=0)
    print('done. best valid match = {:.3f}'.format(best))


if __name__ == '__main__':
    main()
