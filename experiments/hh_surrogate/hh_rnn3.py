"""
RNN 3 個（分割型。並列・直列の両方）を、Markov MLP で効いた作り方で学習し直す（PyTorch）。

構造（各ブロックは tanh RNN のまま。隠れ状態も持つ）:
    RNN1: V -> α, β(6)
    RNN2: α, β, m, h, n -> Δm, Δh, Δn
    RNN3: g_in, V, I -> ΔV        g_in = g(t)（parallel）か RNN2 が出した g(t+dt)（serial）
閉ループ（評価と同じ）: α, β >= 0、g ∈ [0, 1]、V ∈ [-100, 60] mV に制限。隠れ状態は 0、状態は静止状態から始める。

これまでの作り方（hh_improve.py block → chain、hh_serial.py）との違い:
    1. 学習する軌道: 教師データ teacher_ou の窓ではなく、その場で作った電流（gen_current）と
       苦手な電流（gen_hard_v2: 閾値付近の定常・0 からのランプ・正弦）で HH を回した軌道。
    2. 隠れ状態の扱い: 軌道の頭（静止状態、隠れ状態 0）から seg ステップずつ流し、隠れ状態を次の区切りへ引き継ぐ
       （truncated BPTT の「流れ」）。評価と同じく、t = 0 から積み上がった隠れ状態で学習する。
    3. 正解: どのブロックも入力の厳密な関数（HH のオイラー法の式）なので、入力に摂動を足しても式から計算し直す。
       ΔV は下限付きの相対誤差の Huber（Markov MLP の --pre-rel と同じ）。
    4. DAgger: 今のモデルを閉ループで回して通った軌道（V, g, I の列）を在庫に足し、その上でも学習する。
       軌道の頭から流すので、隠れ状態の履歴も今のモデルで作り直される。
    5. 直列の正解: 教師は陽的オイラー法のまま（ゲートを先に進める積分にすると HH との一致が 20/26 本に落ちる）。
       RNN3 の入力 g_in = g(t+dt) からゲートごとに g(t) = (g_in - dt α) / (1 - dt (α + β)) を逆算して
       ΔV = dt (I - I_ion(g(t), V)) / C を正解にする。これも入力の厳密な関数になる。
       在庫の軌道では g_in = 次の時刻の g（教師の軌道では式の値、DAgger の軌道では RNN2 が実際に出した値）。
    6. 短い閉ループ（--cl-w > 0）: 区切りの頭の HH の状態から cl_k ステップ自分の出力で進め、HH の軌道との差を足す。
       摂動した区切りでは軌道の正解にならないので、摂動前の区切りで、DAgger 以外の流れだけに使う。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_rnn3.py train --conn parallel --tag r3p_test --iters 200 --bank 64 --hard 16
    uv run python -X utf8 experiments/hh_surrogate/hh_rnn3.py eval --conn serial --b1 ... --b2 ... --b3 ...
結果は results/hh_rnn3/<tag>_b{1,2,3}.pt と <tag>.json。
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
import hh_improve as hi
import hh_bench as hb
from neuron import hh as H_

OUT_DIR = os.path.join(hb.ROOT, 'results', 'hh_rnn3')
DT = hi.DT
V_LO, V_HI = -100.0, 60.0


# ---------------- 式 ----------------

def gate_inverse(g_in, V):
    # g_in = g + dt (α (1 - g) - β g) を g について解く（ゲートごと）。V: (...)、g_in: (..., 3) -> (..., 3)
    ab = hi.rates(V)
    a, b = ab[..., 0::2], ab[..., 1::2]
    return ((g_in - DT * a) / (1 - DT * (a + b))).clamp(0.0, 1.0)


def targets(conn, V, g, g_in, I):
    """
    各ブロックの入力と正解（どれも入力の厳密な関数）。
    V: (B, L)、g: (B, L, 3) = g(t)、g_in: (B, L, 3) = RNN3 に渡すゲート、I: (B, L)
    Returns: (x1, y1), (x2, y2), (x3, y3)
    """
    ab = hi.rates(V)
    x1, y1 = V[..., None], ab
    x2, y2 = torch.cat([ab, g], -1), hi.d_gates(ab, g)
    g_old = gate_inverse(g_in, V) if conn == 'serial' else g_in
    x3, y3 = torch.cat([g_in, V[..., None], I[..., None]], -1), hi.d_V(g_old, V, I)[..., None]
    return (x1, y1), (x2, y2), (x3, y3)


# ---------------- 閉ループ ----------------

def step(nets, conn, V, g, I_k, hs):
    # 閉ループの 1 ステップ。V: (B,)、g: (B, 3)、I_k: (B,)、hs: 3 つの隠れ状態 -> V_next, g_next, hs
    n1, n2, n3 = nets
    ab, h1 = n1.cell(V[:, None], hs[0])
    dg, h2 = n2.cell(torch.cat([ab.clamp(min=0.0), g], 1), hs[1])
    g_next = (g + dg).clamp(0.0, 1.0)
    g_in = g_next if conn == 'serial' else g
    dv, h3 = n3.cell(torch.cat([g_in, V[:, None], I_k[:, None]], 1), hs[2])
    V_next = (V + dv[:, 0]).clamp(V_LO, V_HI)
    return V_next, g_next, [h1, h2, h3]


@torch.no_grad()
def run(nets, conn, I_np, record=False):
    """
    hb.score の約束どおり: I (B, N) -> V (B, N-1)。record=True なら通った軌道 (B, N, 5) = [V, m, h, n, I] も返す
    （DAgger 用。最後の時刻の I は 0 とする）。
    """
    I = torch.as_tensor(I_np).to(hi.DEV)
    B, N = I.shape
    dt = next(nets[0].parameters()).dtype
    I = I.to(dt)
    V = torch.full((B,), H_.V_REST, dtype=dt)
    g = torch.tensor(H_.steady_state_gates(H_.V_REST), dtype=dt).repeat(B, 1)
    hs = [n.zeros(B, dt) for n in nets]
    out = torch.empty(B, N - 1, dtype=torch.float64)
    seq = torch.empty(B, N, 5, dtype=torch.float32, device='cpu') if record else None
    for k in range(N - 1):
        if record:
            seq[:, k] = torch.cat([V[:, None], g, I[:, k, None]], 1).float().cpu()
        V, g, hs = step(nets, conn, V, g, I[:, k], hs)
        out[:, k] = V.double()
    if record:
        seq[:, N - 1] = torch.cat([V[:, None], g, torch.zeros(B, 1, dtype=dt)], 1).float().cpu()
        return out.cpu().numpy(), seq
    return out.cpu().numpy()


# ---------------- 在庫（軌道）と流れ ----------------

@torch.no_grad()
def hh_traj(I):
    # 静止状態から HH の陽的オイラー法。I: (B, N) -> (B, N, 5) float32 = [V, m, h, n, I]（I[N-1] はそのまま）
    V, G = hi.hh_sim(I)                         # (B, N+1), (B, N+1, 3)
    return torch.cat([V[:, :-1, None], G[:, :-1], I[..., None]], -1).float().cpu()


def make_pool(n, ms, rng, kind, chunk=512, hard_set='v2'):
    N = int(ms / DT)
    out = []
    for c0 in range(0, n, chunk):
        b = min(chunk, n - c0)
        I = (hi.gen_current(b, N, rng).to(hi.DEV) if kind == 'norm'
             else hi.gen_hard_set(b, N, rng, hard_set))
        out.append(hh_traj(I))
    return torch.cat(out)


def pool_path(n, ms, seed, kind):
    return os.path.join(OUT_DIR, 'pools', f'{kind}_{n}x{ms:.0f}ms_s{seed}.pt')


def cached_pool(n, ms, seed, kind):
    """
    在庫を作るか、同じ条件（種類・本数・長さ・seed）で前に作ったものを読む。results/hh_rnn3/pools/ に置く。
    在庫ごとに専用の乱数（seed と種類から決める）を使うので、読み込んでも作っても中身は同じ。
    """
    path = pool_path(n, ms, seed, kind)
    if os.path.exists(path):
        try:
            return torch.load(path)
        except Exception as e:                       # 書きかけなどで読めなければ作り直す
            print('在庫の読み込みに失敗:', path, e, flush=True)
    pool = make_pool(n, ms, np.random.default_rng([seed, {'norm': 1, 'hard': 2}[kind]]), kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    torch.save(pool, tmp)
    os.replace(tmp, path)                            # 同時に作った別のジョブと競合しても壊れたファイルを残さない
    return pool


class Streams:
    """
    B 本の流れ。各流れは在庫の 1 本の軌道を頭から seg ステップずつ進み、終わったら別の軌道に移って隠れ状態を 0 に戻す。
    在庫は pools = {'norm': (n, N, 5), 'hard': ..., 'dagger': ...}。新しい軌道は probs の割合で在庫を選ぶ。
    """

    def __init__(self, B, seg, rng):
        self.B, self.seg, self.rng = B, seg, rng
        self.src = [None] * B          # (在庫の名前, 軌道の番号)
        self.pos = np.zeros(B, dtype=int)

    def pick(self, pools, probs):
        names = [k for k in pools if pools[k] is not None and len(pools[k]) and probs.get(k, 0) > 0]
        p = np.array([probs[k] for k in names], float)
        k = names[self.rng.choice(len(names), p=p / p.sum())]
        return k, int(self.rng.integers(0, len(pools[k])))

    def next(self, pools, probs, hs):
        """
        次の区切り (B, seg+1, 5) を返す。軌道を替えた流れは隠れ状態を 0 にする（hs をその場で書き換える）。
        軌道はいつも頭（静止状態）から始める（評価と同じ条件）。在庫ごとに長さが違うので、替わる時期は自然にばらける。
        """
        segs = []
        for b in range(self.B):
            k = self.src[b]
            if k is None or self.pos[b] + self.seg + 1 > pools[k[0]].shape[1]:
                self.src[b] = k = self.pick(pools, probs)
                self.pos[b] = 0
                for l in range(len(hs)):
                    for h in hs[l]:
                        h[b] = 0.0
            p = self.pos[b]
            segs.append(pools[k[0]][k[1], p:p + self.seg + 1])
            self.pos[b] = p + self.seg
        return torch.stack(segs).to(hi.DEV)


def perturb_seg(x, sv, sg, rng_t):
    """
    区切りの半分の流れに、区切りの中で一定のずれ（V: sv mV、g: sg）と毎ステップの揺らぎ（その 1/5）を足す。
    x: (B, L, 5)。正解はあとで摂動後の入力から式で計算し直す。
    """
    B, L, _ = x.shape
    r = lambda *sh: torch.randn(*sh, generator=rng_t, device='cpu').to(x.device, x.dtype)
    on = (torch.rand(B, 1, generator=rng_t, device='cpu') < 0.5).to(x.device, x.dtype)
    x = x.clone()
    x[..., 0] += on * sv * (r(B, 1) + 0.2 * r(B, L))
    x[..., 1:4] = (x[..., 1:4] + on[..., None] * sg * (r(B, 1, 3) + 0.2 * r(B, L, 3))).clamp(0.0, 1.0)
    return x


# ---------------- 学習 ----------------

def build_nets(a, sample):
    """新しく作るか（--init なし）、既存のブロックから始める（--init-b1/b2/b3）。sample: (n, 5) で標準化を決める"""
    nets = []
    V, g, I = sample[:, 0], sample[:, 1:4], sample[:, 4]
    g_in = (g + hi.d_gates(hi.rates(V), g)).clamp(0, 1) if a.conn == 'serial' else g
    xy = targets(a.conn, V, g, g_in, I)
    for k, (d_in, d_out) in enumerate([(1, 6), (9, 3), (5, 1)]):
        init = getattr(a, f'init_b{k + 1}')
        if init:
            net = hi.load_net(init)
        else:
            net = hi.Net(d_in, a.H, a.layers, d_out, cell=a.cell)
            net.set_scalers(xy[k][0].float(), xy[k][1].float())
            if a.whh0:
                with torch.no_grad():
                    for l in range(a.layers):
                        getattr(net.rnn, f'weight_hh_l{l}').zero_()
        nets.append(net.float().to(hi.DEV))
    return nets


def seg_loss(nets, a, x, hs):
    """
    区切り x (B, seg+1, 5) を 3 ブロックに教師強制で流す（隠れ状態 hs を引き継ぐ）。
    Returns: loss, (l1, l2, l3), 新しい隠れ状態（切り離し済み）
    """
    V, g, I = x[:, :-1, 0], x[:, :-1, 1:4], x[:, :-1, 4]
    g_in = x[:, 1:, 1:4] if a.conn == 'serial' else g          # serial: 次の時刻の g（RNN2 の出力に当たる）
    (x1, y1), (x2, y2), (x3, y3) = targets(a.conn, V.double(), g.double(), g_in.double(), I.double())
    outs, new = [], []
    for k, (n, xx, h) in enumerate(zip(nets, (x1, x2, x3), hs)):
        if k == 2 and a.conn == 'serial' and a.s_gin == 'model':
            # 直列: RNN3 には RNN2 が実際に出した g(t+dt)（勾配は切る）を渡し、正解は今の g(t) からのオイラー法の ΔV
            g_pred = (g + outs[1].detach()).clamp(0.0, 1.0)
            xx = torch.cat([g_pred, V[..., None], I[..., None]], -1)
            y3 = hi.d_V(g.double(), V.double(), I.double())[..., None]
        y, _, _, hN = n(xx.float(), torch.stack(h))             # h: 段ごとの (B, H) -> (layers, B, H)
        outs.append(y)
        new.append([z.detach() for z in hN])
    l1 = (((outs[0] - y1.float()) / nets[0].os_) ** 2).mean()
    l2 = (((outs[1] - y2.float()) / nets[1].os_) ** 2).mean()
    t3 = y3[..., 0].float()
    if a.rel > 0:
        e3 = (outs[2][..., 0] - t3) / t3.abs().clamp(min=a.rel)
        l3 = nn.functional.huber_loss(e3, torch.zeros_like(e3), delta=1.0)
    else:
        l3 = (((outs[2][..., 0] - t3) / nets[2].os_[0]) ** 2).mean()
    loss = a.w1 * l1 + a.w2 * l2 + a.w3 * l3
    if a.whh_pen > 0:
        loss = loss + a.whh_pen * sum(getattr(n.rnn, f'weight_hh_l{l}').pow(2).mean()
                                      for n in nets for l in range(a.layers) if hasattr(n, 'rnn'))
    return loss, (l1, l2, l3), new


def cl_loss(nets, a, x, hs, mask):
    """
    短い閉ループの展開誤差（--cl-w > 0 のとき）。区切りの頭の HH の状態から、隠れ状態 hs（教師強制で積み上げたもの）を
    引き継いで cl_k ステップ自分の出力で進め、各ステップの V, g を HH の軌道と比べる。
    x は摂動前の区切り (B, seg+1, 5)。軌道の正解が要るので、HH の軌道（在庫 norm / hard）の流れだけ使う（mask: (B,) bool）。
    """
    K = min(a.cl_k, x.shape[1] - 1)
    x = x[mask]
    hs_ = [[h[mask] for h in hn] for hn in hs]
    V, g = x[:, 0, 0], x[:, 0, 1:4]
    lv, lg = 0.0, 0.0
    for k in range(K):
        V, g, hs_ = step(nets, a.conn, V, g, x[:, k, 4], hs_)
        ev = (V - x[:, k + 1, 0]) / a.cl_sv
        lv = lv + nn.functional.huber_loss(ev, torch.zeros_like(ev), delta=1.0)
        lg = lg + (((g - x[:, k + 1, 1:4]) / a.cl_sg) ** 2).mean()
    return lv / K, lg / K


def save_nets(nets, a, tag, info):
    for k, n in enumerate(nets):
        hi.save_net(n, os.path.join(OUT_DIR, f'{tag}_b{k + 1}.pt'), dict(info, conn=a.conn, block=k + 1))


def load_nets(tag_or_paths, dev_dtype=torch.float64):
    if isinstance(tag_or_paths, str):
        tag_or_paths = [os.path.join(OUT_DIR, f'{tag_or_paths}_b{k}.pt') for k in (1, 2, 3)]
    return [hi.load_net(p).to(dev_dtype).to(hi.DEV) for p in tag_or_paths]


def train(a):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rng_t = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    if a.pools_only:                                 # 在庫だけ先に作る（同時に走るジョブが同じ在庫を作り合わないように）
        for n, ms, kind in [(a.bank, a.bank_ms, 'norm'), (a.hard, a.hard_ms, 'hard')]:
            if n > 0 and not os.path.exists(pool_path(n, ms, a.seed, kind)):
                cached_pool(n, ms, a.seed, kind)
                print('在庫を作成:', pool_path(n, ms, a.seed, kind), '({:.0f} s)'.format(time.time() - t0), flush=True)
        return
    pools = {'norm': cached_pool(a.bank, a.bank_ms, a.seed, 'norm'),
             'hard': cached_pool(a.hard, a.hard_ms, a.seed, 'hard') if a.hard > 0 else None,
             'dagger': None}
    print('在庫: 通常 {} 本 × {:.0f} ms、苦手 {} 本 × {:.0f} ms  ({:.0f} s)'.format(
        a.bank, a.bank_ms, a.hard, a.hard_ms, time.time() - t0), flush=True)
    flat = pools['norm'].reshape(-1, 5)
    sample = flat[torch.from_numpy(rng.integers(0, len(flat), 200000))].double()
    if pools['hard'] is not None:
        fh = pools['hard'].reshape(-1, 5)
        sample = torch.cat([sample, fh[torch.from_numpy(rng.integers(0, len(fh), 100000))].double()])
    nets = build_nets(a, sample)
    params = [p for n in nets for p in n.parameters()]
    opt = torch.optim.Adam(params, lr=a.lr)
    streams = Streams(a.batch, a.seg, rng)
    hs = [[torch.zeros(a.batch, n.cfg['H'], device=hi.DEV) for _ in range(n.cfg['layers'])] for n in nets]
    log = {'args': vars(a), 'loss': [], 'eval': []}
    best = (-1, -1.0, 0)

    def evaluate(it):
        nonlocal best
        for n in nets:
            n.eval()
        r = hb.score(lambda I_np: run(nets, a.conn, I_np), '{} it {} val'.format(a.tag, it), val=True)
        log['eval'].append((it, r['n100'], r['mean'], r['gamma']))
        if (r['n100'], r['mean']) > best[:2]:
            best = (r['n100'], r['mean'], it)
            save_nets(nets, a, a.tag, {'it': it, 'val': r['n100']})
        return r

    probs = {'norm': 1.0 - a.hard_frac - (a.dagger_frac if a.dagger else 0), 'hard': a.hard_frac,
             'dagger': a.dagger_frac}
    dagger_at = set(int(a.iters * (r + 1) / (a.dagger + 1)) for r in range(a.dagger))
    evaluate(0)
    for it in range(a.iters):
        if it in dagger_at:
            Nd = int(a.dagger_ms / DT)
            Id = torch.cat([hi.gen_current(a.dagger_n // 2, Nd, rng).to(hi.DEV),
                            hi.gen_hard_set(a.dagger_n - a.dagger_n // 2, Nd, rng, 'v2')])
            _, seq = run(nets, a.conn, Id, record=True)
            pools['dagger'] = seq if pools['dagger'] is None else torch.cat([pools['dagger'], seq])
            if len(pools['dagger']) > a.dagger_keep:                 # 古い訪問軌道から捨てる
                pools['dagger'] = pools['dagger'][-a.dagger_keep:]
            print('DAgger: 閉ループの軌道 {} 本を追加 → {} 本  ({:.0f} s)'.format(
                len(seq), len(pools['dagger']), time.time() - t0), flush=True)
        for gr in opt.param_groups:
            gr['lr'] = hi.lr_at(it, a.iters, a.lr)
        for n in nets:
            n.train()
        hs_prev = hs
        x0 = streams.next(pools, probs, hs)                           # 軌道を替えた流れは hs が 0 に戻る
        x = perturb_seg(x0, a.sv, a.sg, rng_t)
        loss, ls, hs = seg_loss(nets, a, x, hs)
        if a.cl_w > 0 and it >= a.cl_start * a.iters:
            mask = torch.tensor([s[0] != 'dagger' for s in streams.src], device=x0.device)
            if mask.any():
                lv, lg = cl_loss(nets, a, x0, hs_prev, mask)
                loss = loss + a.cl_w * (lv + lg)
                ls = (*ls, lv, lg)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log['loss'].append(([float(l.detach()) for l in ls] + [float('nan')] * 2)[:5])
        if (it + 1) % a.log_every == 0:
            m = np.nanmean(np.array(log['loss'][-a.log_every:], float), 0) if any(
                not np.isnan(r[3]) for r in log['loss'][-a.log_every:]) else np.r_[
                np.mean(np.array(log['loss'][-a.log_every:], float)[:, :3], 0), np.nan, np.nan]
            print('it {:6d} loss RNN1 {:.2e} RNN2 {:.2e} RNN3 {:.2e}  閉ループ V {:.2e} g {:.2e}  ({:.0f} s)'.format(
                it + 1, *m, time.time() - t0), flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            evaluate(it + 1)
    log['best'] = best
    best_nets = load_nets(a.tag)
    log['test'] = hb.score(lambda I_np: run(best_nets, a.conn, I_np), '{} (val best it {})'.format(a.tag, best[2]))
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


def evaluate_cmd(a):
    nets = load_nets([a.b1, a.b2, a.b3])
    r = hb.score(lambda I_np: run(nets, a.conn, I_np), a.tag)
    if a.val:
        hb.score(lambda I_np: run(nets, a.conn, I_np), a.tag + ' val', val=True)
    return r


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pt = sub.add_parser('train')
    pt.add_argument('--tag', default='r3_test')
    pt.add_argument('--conn', default='parallel', choices=['parallel', 'serial'])
    pt.add_argument('--cell', default='rnn', choices=['rnn', 'gru'])
    pt.add_argument('--H', type=int, default=64)
    pt.add_argument('--layers', type=int, default=1)
    pt.add_argument('--iters', type=int, default=20000)
    pt.add_argument('--batch', type=int, default=256, help='流れの本数')
    pt.add_argument('--seg', type=int, default=200, help='1 回の更新で進めるステップ数（truncated BPTT の区切り）')
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--rel', type=float, default=0.02, help='>0 なら ΔV を相対誤差の Huber で（値は下限 [mV]）')
    pt.add_argument('--w1', type=float, default=1.0)
    pt.add_argument('--w2', type=float, default=1.0)
    pt.add_argument('--w3', type=float, default=1.0)
    pt.add_argument('--sv', type=float, default=2.0, help='V の摂動 [mV]')
    pt.add_argument('--sg', type=float, default=0.02, help='ゲートの摂動')
    pt.add_argument('--bank', type=int, default=1024, help='通常の電流の本数')
    pt.add_argument('--bank-ms', type=float, default=300.0)
    pt.add_argument('--hard', type=int, default=256, help='苦手な電流（v2）の本数')
    pt.add_argument('--hard-ms', type=float, default=1000.0)
    pt.add_argument('--hard-frac', type=float, default=0.4, help='新しい流れを苦手な電流から取る割合')
    pt.add_argument('--dagger', type=int, default=3, help='DAgger の回数（学習の途中で等間隔に）')
    pt.add_argument('--dagger-n', type=int, default=128)
    pt.add_argument('--dagger-ms', type=float, default=1000.0)
    pt.add_argument('--dagger-frac', type=float, default=0.3, help='新しい流れを DAgger の軌道から取る割合')
    pt.add_argument('--dagger-keep', type=int, default=384, help='DAgger の軌道を何本まで残すか（古いものから捨てる）')
    pt.add_argument('--cl-w', type=float, default=0.0, help='>0 なら短い閉ループの展開誤差を足す（その重み）')
    pt.add_argument('--cl-k', type=int, default=50, help='閉ループで進めるステップ数')
    pt.add_argument('--cl-start', type=float, default=0.0, help='閉ループの誤差を足し始める時期（iters に対する割合）')
    pt.add_argument('--cl-sv', type=float, default=1.0, help='閉ループの V の誤差の目盛り [mV]（Huber の幅）')
    pt.add_argument('--cl-sg', type=float, default=0.01, help='閉ループの g の誤差の目盛り')
    pt.add_argument('--s-gin', default='traj', choices=['traj', 'model'],
                    help='直列の RNN3 の入力 g(t+dt): traj = 軌道の次の時刻の g（正解は逆算した g(t) から）、'
                         'model = RNN2 が出した値（正解は今の g(t) から）')
    pt.add_argument('--whh0', action='store_true', help='再帰の重みを 0 から始める')
    pt.add_argument('--whh-pen', type=float, default=0.0, help='再帰の重みの 2 乗の罰則')
    for k in (1, 2, 3):
        pt.add_argument(f'--init-b{k}', default=None, help='既存のブロック（.pt / .npz）から始める')
    pt.add_argument('--eval-every', type=int, default=2000)
    pt.add_argument('--log-every', type=int, default=200)
    pt.add_argument('--seed', type=int, default=0)
    pt.add_argument('--threads', type=int, default=2)
    pt.add_argument('--device', default='cpu')
    pt.add_argument('--pools-only', action='store_true', help='在庫だけ作って終わる')
    pe = sub.add_parser('eval')
    pe.add_argument('--tag', default='eval')
    pe.add_argument('--conn', default='parallel', choices=['parallel', 'serial'])
    for k in (1, 2, 3):
        pe.add_argument(f'--b{k}', required=True)
    pe.add_argument('--val', action='store_true')
    pe.add_argument('--threads', type=int, default=2)
    pe.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    hi.set_device(a.device)
    if a.cmd == 'train':
        train(a)
    else:
        evaluate_cmd(a)


if __name__ == '__main__':
    main()
