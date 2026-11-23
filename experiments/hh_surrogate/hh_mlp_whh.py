"""
Markov MLP（mm_rel）に再帰の重み W_hh を足した RNN で、「隠れ状態（記憶）が要るか」を調べる。

モデル（cell='mlpr'、Elman 型）:
    z1(t) = SiLU(W1 u(t) + W_hh z1(t-1) + b1)      u(t) = 標準化した [V, m, h, n, I]、z1 は 1 段目の出力（H=128）
    z2(t) = SiLU(W2 z1(t) + b2)
    [ΔV, Δm, Δh, Δn] = W_out z2(t) + b_out
    W_hh = 0 なら mm_rel とまったく同じ。hh_improve.py の Net を受け継ぎ、1 段目の Linear の入力を H 列ぶん広げて作る。

実験:
    sweep : 学習せずに W_hh = ε R（R の成分は N(0, 1/H)、つまり ε は再帰のおよその利得）として ε を大きくし、26 本で採点
    ft    : W_hh = 0（mm_rel と同じ出力）から閉ループで微調整し、W_hh が育つか・精度が上がるかを見る。
            対照として、W_hh のない mm_rel を同じ条件で微調整したものも回す（--ctrl）。微調整は hh_improve.train_single_cl を使う

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_mlp_whh.py check
    uv run python -X utf8 experiments/hh_surrogate/hh_mlp_whh.py sweep --eps 0 0.01 0.1
    uv run python -X utf8 experiments/hh_surrogate/hh_mlp_whh.py ft --tag whh_ft
    uv run python -X utf8 experiments/hh_surrogate/hh_mlp_whh.py ft --tag whh_ft_ctrl --ctrl
結果: results/hh_whh/（sweep_<ε>.json、ft の W_hh の推移）、微調整したモデルは results/hh_improve/<tag>.pt と .json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_improve as hi   # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(ROOT, 'results', 'hh_improve', 'mm_rel.pt')
OUT = os.path.join(ROOT, 'results', 'hh_whh')
_NETS = []   # 作った NetR（微調整中のネットの W_hh を記録するため）


class NetR(hi.Net):
    """hi.Net に cell='mlpr'（1 段目に z1(t-1) を戻す MLP）を足したもの。ほかの cell は hi.Net のまま。"""

    def __init__(self, *args, cell='rnn', **kw):
        super().__init__(*args, cell='mlp' if cell == 'mlpr' else cell, **kw)
        if cell == 'mlpr':
            self.cfg['cell'] = 'mlpr'
            l0 = self.mlp[0]
            self.mlp[0] = nn.Linear(l0.in_features + self.cfg['H'], l0.out_features)
        _NETS.append(self)

    def _step(self, u, hs):
        if self.cfg['cell'] != 'mlpr':
            return super()._step(u, hs)
        # hs = [z1(t-1), z2(t-1)] -> [z1(t), z2(t)]。読み出しは hs[-1]
        z1 = self.mlp[1](self.mlp[0](torch.cat([u, hs[0]], -1)))
        return [z1, self.mlp[2:](z1)]

    def zeros(self, B, dtype):
        if self.cfg['cell'] != 'mlpr':
            return super().zeros(B, dtype)
        return [torch.zeros(B, self.cfg['H'], dtype=dtype) for _ in range(2)]

    def whh(self):
        # W_hh (H, H) = 1 段目の重みの右側 H 列
        return self.mlp[0].weight[:, -self.cfg['H']:]


def from_mlp(src=SRC, eps=0.0, seed=0):
    """mm_rel から NetR（mlpr）を作る。W_hh = eps R、R の成分は N(0, 1/H)。float64 で返す。"""
    d = torch.load(src, weights_only=False, map_location='cpu')
    assert d['cfg']['cell'] == 'mlp'
    net = NetR(**dict(d['cfg'], cell='mlpr'))
    st = dict(d['state'])
    W1 = st['mlp.0.weight']
    H = d['cfg']['H']
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(H, H, generator=g, dtype=W1.dtype, device='cpu') / H ** 0.5
    st['mlp.0.weight'] = torch.cat([W1, eps * R], 1)
    net.load_state_dict(st)
    return net.double()


def norms(net):
    W = net.whh().detach().double()
    return {'fro': float(W.norm()), 'spec': float(torch.linalg.matrix_norm(W, 2)),
            'W1_fro': float(net.mlp[0].weight[:, :-net.cfg['H']].detach().norm())}


def check(a):
    """W_hh = 0 の NetR が mm_rel と同じ V を出すか（閉ループ 100 ms、const 10 / const 7）。"""
    t = np.arange(0, 100.0, hi.DT)
    I = np.stack([np.full_like(t, 10.0), np.full_like(t, 7.0)])
    ref = hi.run_state(hi.load_net(SRC), I)
    got = hi.run_state(from_mlp(eps=0.0), I)
    print('W_hh = 0 と mm_rel の最大差: {:.2e} mV'.format(np.abs(got - ref).max()))
    got1 = hi.run_state(from_mlp(eps=0.1), I)
    print('W_hh = 0.1 R と mm_rel の最大差: {:.2e} mV（0 でないことの確認）'.format(np.abs(got1 - ref).max()))


def sweep(a):
    os.makedirs(OUT, exist_ok=True)
    for eps in a.eps:
        for seed in a.seeds:
            path = os.path.join(OUT, 'sweep_{:g}_s{}.json'.format(eps, seed))
            if os.path.exists(path):
                print('済:', path)
                continue
            net = from_mlp(eps=eps, seed=seed)
            r = hi.hb.score(lambda I: hi.run_state(net, I), 'W_hh = {:g} R (seed {})'.format(eps, seed))
            r.update(eps=eps, seed=seed, norms=norms(net))
            json.dump(r, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)


def ft(a, rest):
    """
    hh_improve.train_single_cl で閉ループの微調整をする。hi.Net を NetR に差し替え、
    検証で採点するたびに W_hh の大きさを results/hh_whh/<tag>_whh.json に記録する。
    """
    os.makedirs(OUT, exist_ok=True)
    init = os.path.join(OUT, a.tag + '_init.pt')
    src = a.src or SRC
    cell0 = torch.load(src, weights_only=False, map_location='cpu')['cfg']['cell']
    if a.ctrl or cell0 == 'mlpr':
        # 対照（mm_rel をそのまま）か、すでに RNN 版のモデル（例: sp_r_scr_e01）から始める
        hi.Net = NetR
        net0 = hi.load_net(src)
    else:
        net0 = from_mlp(src, eps=0.0)
    hi.save_net(net0, init, {'from': os.path.basename(src), 'ctrl': a.ctrl})
    hi.Net = NetR          # train_single_cl と load_net が NetR を使うように
    trace = []
    score0 = hi.hb.score

    def score(run, label=None, *args, **kw):
        net = _NETS[-1]
        r = score0(run, label, *args, **kw)
        if kw.get('val'):
            row = {'label': label, 'n100': r['n100'], 'gamma': r['gamma'], 'mean': r['mean']}
            if net.cfg['cell'] == 'mlpr':
                row.update(norms(net))
                print('  ||W_hh||_F = {fro:.4f}  ||W_hh||_2 = {spec:.4f}  （||W1||_F = {W1_fro:.2f}）'.format(**row),
                      flush=True)
            trace.append(row)
            json.dump(trace, open(os.path.join(OUT, a.tag + '_whh.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=0)
        return r
    hi.hb.score = score
    # --cell は最初に作るネットの種類。すぐに --init のモデル（cfg に cell='mlpr' を含む）で置き換わるので mlp でよい
    sys.argv = ['hh_improve.py', 'single_cl', '--tag', a.tag, '--state', '--cell', 'mlp', '--H', '128', '--layers', '2',
                '--init', init, '--pre-iters', '0', '--iters', str(a.iters), '--lr', str(a.lr),
                '--alpha-iters', '1', '--eval-every', str(a.eval_every), '--threads', str(a.threads),
                '--device', a.device, '--seed', str(a.seed)] + rest
    hi.main()


def new_net(cell, H=128, layers=2):
    """0 から学習する NetR。標準化は train_single_cl と同じく教師の軌道 32 本 × 200 ms から決める。"""
    rng = np.random.default_rng(12345)
    I0 = hi.gen_current(32, 20000, rng)
    V0, G0 = hi.hh_sim(I0)
    X0 = torch.cat([V0[:, :-1, None], G0[:, :-1], I0[..., None]], -1)
    net = NetR(5, H, layers, 4, cell=cell)
    net.set_scalers(X0.reshape(-1, 5).float(), hi.exact_step(X0).reshape(-1, 4).float())
    return net


def pre(a):
    """
    系列版の摂動事前学習（mm_rel の事前学習を、隠れ状態を持つモデルに広げたもの）。
      1. 教師の軌道の在庫（bank 本 × ep_ms）から、長さ L の窓を batch 本切り出す
      2. 状態 [V, m, h, n] にずれを足す。半分の系列に、系列全体で一定のずれ（V: sv mV、ゲート: sg）と、
         毎ステップの小さな雑音（その 1/4）を足す（閉ループでじわじわずれた状態を模す）
      3. ずれた状態の系列を入力に、隠れ状態を 0 から流してモデルを回す（教師強制。予測は戻さない）
      4. 最初の burn ステップを除いた各ステップで、ずれた状態での HH の正しい 1 ステップに合わせる
         （ΔV は |ΔV| で割った相対誤差の Huber、下限 rel mV。ゲートは標準化した 2 乗誤差。mm_rel と同じ）
    --init mm_rel: mm_rel の重みから始める（mlpr なら W_hh = eps0 R）。--init scratch: 0 から
    検証の 17 本で eval_every ごとに採点し、最良を results/hh_improve/<tag>.pt に保存。最後に 26 本で採点する。
    """
    import time
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.init == 'mm_rel':
        net = from_mlp(eps=a.eps0, seed=a.seed) if a.cell == 'mlpr' else hi.load_net(SRC)
        net = net.float()
    else:
        net = new_net(a.cell, a.H, a.layers)
    if a.cell == 'mlpr' and a.init == 'scratch' and a.eps0 >= 0:
        with torch.no_grad():   # 0 からのとき W_hh の初期値を eps0 R にそろえる（既定の一様乱数を使わない）
            H = a.H
            R = torch.randn(H, H, generator=torch.Generator().manual_seed(a.seed), device='cpu') / H ** 0.5
            net.mlp[0].weight[:, -H:] = a.eps0 * R.to(net.mlp[0].weight.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(hi.OUT_DIR, a.tag + '.pt')
    log = {'args': vars(a), 'loss': [], 'eval': []}
    best = (-1, -1.0, 0)
    t0 = time.time()

    def evaluate(it):
        nonlocal best
        net.double()
        r = hi.hb.score(lambda I_: hi.run_state(net, I_), '{} it {} val'.format(a.tag, it), val=True)
        net.float()
        row = {'it': it, 'n100': r['n100'], 'mean': r['mean'], 'gamma': r['gamma']}
        if net.cfg['cell'] == 'mlpr':
            row.update(norms(net))
            print('  ||W_hh||_F = {fro:.4f}  ||W_hh||_2 = {spec:.4f}'.format(**row), flush=True)
        log['eval'].append(row)
        if (r['n100'], r['mean']) > best[:2]:
            best = (r['n100'], r['mean'], it)
            hi.save_net(net, path, {'it': it})

    N = int(a.ep_ms / hi.DT)
    evaluate(0)
    for it in range(a.iters):
        if it % a.bank_every == 0:
            bI = hi.gen_current(a.bank, N, rng)
            bV, bG = hi.hh_sim(bI)
            X = torch.cat([bV[:, :-1, None], bG[:, :-1], bI[..., None]], -1)   # (bank, N, 5) float64
            del bV, bG
        for gr in opt.param_groups:
            gr['lr'] = hi.lr_at(it, a.iters, a.lr)
        b = torch.from_numpy(rng.integers(0, a.bank, size=a.batch)).to(hi.DEV)
        s = torch.from_numpy(rng.integers(0, N - a.L, size=a.batch)).to(hi.DEV)
        idx = s[:, None] + torch.arange(a.L, device=hi.DEV)[None, :]
        x = X[b[:, None], idx].clone()                                          # (batch, L, 5)
        mask = (torch.rand(a.batch, 1, 1, dtype=x.dtype) < 0.5).to(x.dtype)
        off_v = a.sv * torch.randn(a.batch, 1, 1, dtype=x.dtype)
        off_g = a.sg * torch.randn(a.batch, 1, 3, dtype=x.dtype)
        x[..., :1] += mask * (off_v + 0.25 * a.sv * torch.randn(a.batch, a.L, 1, dtype=x.dtype))
        x[..., 1:4] = (x[..., 1:4] + mask * (off_g + 0.25 * a.sg * torch.randn(a.batch, a.L, 3, dtype=x.dtype))).clamp(0, 1)
        yt = hi.exact_step(x).float()                                           # (batch, L, 4)
        xf = x.float()
        hs = net.zeros(a.batch, torch.float32)
        ys = []
        for k in range(a.L):
            y, hs, _ = hi.state_step(net, xf[:, k], hs)
            ys.append(y)
        y = torch.stack(ys, 1)[:, a.burn:]
        t_ = yt[:, a.burn:]
        e0 = (y[..., 0] - t_[..., 0]) / t_[..., 0].abs().clamp(min=a.rel)
        loss = nn.functional.huber_loss(e0, torch.zeros_like(e0), delta=1.0) \
            + (((y[..., 1:] - t_[..., 1:]) / net.os_[1:]) ** 2).mean()
        if a.whh_pen > 0 and net.cfg['cell'] == 'mlpr':
            loss = loss + a.whh_pen * net.whh().pow(2).sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        log['loss'].append(loss.item())
        if (it + 1) % 500 == 0:
            extra = '  ||W_hh||_F {:.4f}'.format(norms(net)['fro']) if net.cfg['cell'] == 'mlpr' else ''
            print('it {:6d} loss {:.3e}{}  ({:.0f} s)'.format(it + 1, np.mean(log['loss'][-500:]), extra,
                                                              time.time() - t0), flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            evaluate(it + 1)
    hi.Net = NetR
    net = hi.load_net(path)
    log['best'] = best
    log['test'] = hi.hb.score(lambda I_: hi.run_state(net, I_), '{} (val best it {})'.format(a.tag, best[2]))
    if net.cfg['cell'] == 'mlpr':
        log['test_norms'] = norms(net)
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(hi.OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'sweep', 'ft', 'pre'])
    # pre（系列版の摂動事前学習）
    ap.add_argument('--init', default='mm_rel', choices=['mm_rel', 'scratch'])
    ap.add_argument('--cell', default='mlpr', choices=['mlpr', 'mlp'])
    ap.add_argument('--eps0', type=float, default=0.0, help='W_hh の初期値 eps0 R')
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--L', type=int, default=32, help='系列の長さ（ステップ）')
    ap.add_argument('--burn', type=int, default=4, help='損失に入れない最初のステップ数')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--sv', type=float, default=2.0, help='V のずれ [mV]')
    ap.add_argument('--sg', type=float, default=0.02, help='ゲートのずれ')
    ap.add_argument('--rel', type=float, default=0.02, help='ΔV の相対誤差の下限 [mV]')
    ap.add_argument('--whh-pen', type=float, default=0.0, help='W_hh の 2 乗の罰則')
    ap.add_argument('--bank', type=int, default=512)
    ap.add_argument('--bank-every', type=int, default=2000)
    ap.add_argument('--ep-ms', type=float, default=100.0)
    ap.add_argument('--seed', type=int, default=0)
    # 共通
    ap.add_argument('--device', default='cpu', help='cpu か cuda')
    ap.add_argument('--out', default=None, help='出力先（既定: results/hh_improve と results/hh_whh）。Colab では Drive のフォルダ')
    ap.add_argument('--src', default=None, help='ft の出発点の .pt（既定: mm_rel。cell=mlpr のモデルも可）')
    ap.add_argument('--eps', type=float, nargs='+', default=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--tag', default='whh_ft')
    ap.add_argument('--ctrl', action='store_true', help='対照: W_hh のない mm_rel をそのまま同じ条件で微調整')
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--threads', type=int, default=2)
    a, rest = ap.parse_known_args()   # 残りの引数は train_single_cl にそのまま渡す
    torch.set_num_threads(a.threads)
    global OUT
    if a.out:
        OUT = a.out
        hi.OUT_DIR = a.out          # train_single_cl と pre のモデル・ログもここに出す
        os.makedirs(a.out, exist_ok=True)
    if a.cmd in ('pre', 'sweep'):
        hi.set_device(a.device)     # ft は train_single_cl の main が --device で切り替える
    {'check': lambda: check(a), 'sweep': lambda: sweep(a), 'ft': lambda: ft(a, rest), 'pre': lambda: pre(a)}[a.cmd]()


if __name__ == '__main__':
    main()
