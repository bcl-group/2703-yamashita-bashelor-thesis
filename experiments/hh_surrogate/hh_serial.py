"""
モデル2（RNN 3 個）を、ホワイトボードの図どおり「RNN1 → RNN2 → RNN3 を同じ時刻の中で直列」につないだ版。

これまでの分割型（hh_modular.py / hh_improve.py）との違い:
    これまで: RNN3 には更新前の m, h, n(t) を渡す（hh.py のオイラー法と同じ順番）。RNN3 は RNN1・RNN2 と同時に計算できる
    この版  : RNN3 には RNN2 が出した m, h, n(t+dt) を同じ時刻のうちに渡す（図の RNN2 → RNN3 の矢印）

1 ステップ（閉ループ、clip つき）:
    α, β      = max( RNN1(V(t) [, I(t)]), 0 )
    g(t+dt)   = clip( g(t) + RNN2(α, β, g(t)), 0, 1 )                g = (m, h, n)
    ΔV [, G]  = RNN3( g(t+dt), V(t), I(t) )                         ← serial（図どおり）
              = RNN3( g(t),    V(t), I(t) )                         ← parallel（これまでの接続。比較用）
    V(t+dt)   = V(t) + ΔV

比べる設定（ホワイトボードで読み方が分かれた 2 か所を両方作る）:
    --i1   : I(t) を RNN1 にも入れる（RNN1 の入力が [V, I] になる）
    --gout : RNN3 にコンダクタンス G = (g_Na m^3 h, g_K n^4) も出させる（補助の出力。閉ループでは使わない）

正解は教師データの差分ではなく、HH のオイラー法の式（hh.py と同じ）から計算する:
    RNN1: α, β(V)
    RNN2: Δg = dt (α (1 - g) - β g)
    RNN3: ΔV = dt (I - I_ion(g(t), V)) / C_m   G = (g_Na m^3 h, g_K n^4) at g(t)
    教師モデルは hh.py の陽的オイラー法のまま（これまでの結果と比べられるように）。serial では RNN3 は g(t+dt) を見て、
    オイラー法で使われた g(t) での ΔV を当てる。g(t+dt) - g(t) は O(dt) なので、その差は RNN3 が吸収する。

学習の手順（hh_improve.py と同じ道具）:
    1. block : ブロックごとに教師強制で学習（窓の長さ L、Adam + 余弦減衰）
    2. eval  : 3 つをつないで 26 本のテスト電流で評価（スパイク時刻の一致率）
    3. chain : 3 つをつないだ閉ループで微調整（軌道の誤差 + ずれた状態での式の値との誤差）。
               検証用の電流（hh_bench.VAL）で最良の時点を残し、26 本は最後に 1 回だけ使う

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_serial.py block --block 1 --tag b1
    uv run python -X utf8 experiments/hh_surrogate/hh_serial.py block --block 1 --i1 --tag b1_i
    uv run python -X utf8 experiments/hh_surrogate/hh_serial.py block --block 2 --tag b2
    uv run python -X utf8 experiments/hh_surrogate/hh_serial.py block --block 3 --conn serial --tag b3_s
    uv run python -X utf8 experiments/hh_surrogate/hh_serial.py eval --b1 b1 --b2 b2 --b3 b3_s --tag e_s
    uv run python -X utf8 experiments/hh_surrogate/hh_serial.py chain --b1 b1 --b2 b2 --b3 b3_s --tag c_s
    --b1 などはタグ（results/hh_serial/<tag>.pt）かファイルのパス。
    サーバーでまとめて回すときは results/hh_serial/run_jobs.sh と jobs_*.txt を使う。
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
import hh_improve as hi   # Net, Windows, perturb, rates, d_gates, d_V, lr_at（src も sys.path に入る）
import hh_bench as hb
from neuron import hh as H_

OUT_DIR = os.path.join(hb.ROOT, 'results', 'hh_serial')
DT = hi.DT


# ---------------- 式（正解） ----------------

def conductances(g):
    # g: (..., 3) -> (..., 2) = g_Na m^3 h, g_K n^4  [mS/cm^2]
    m, h, n = g[..., 0], g[..., 1], g[..., 2]
    return torch.stack([H_.g_Na * m ** 3 * h, H_.g_K * n ** 4], -1)


def gate_step(V, g):
    # 正しいオイラー法の 1 ステップでの g(t+dt)。RNN3 の入力（serial）を正解の状態から作るときに使う
    return (g + hi.d_gates(hi.rates(V), g)).clamp(0.0, 1.0)


def block_xy(block, I, V, G, i1=False, conn='serial', gout=False):
    """
    窓 (B, L+1) からブロックの入力 X (B, L, d_in) と、式で計算した正解 Y (B, L, d_out) を作る。

    Args:
        I, V: (B, L+1)、G: (B, L+1, 3)。物理単位。
        i1: RNN1 に I も入れるか。conn: RNN3 の入力のゲート変数を g(t+dt)（serial）にするか g(t)（parallel）にするか。
        gout: RNN3 の正解にコンダクタンス 2 個を足すか。
    """
    I, V, G = I[:, :-1], V[:, :-1], G[:, :-1]
    if block == 1:
        X = torch.stack([V, I], -1) if i1 else V[..., None]
        return X, hi.rates(V)
    if block == 2:
        ab = hi.rates(V)
        return torch.cat([ab, G], -1), hi.d_gates(ab, G)
    g_in = gate_step(V, G) if conn == 'serial' else G
    X = torch.cat([g_in, V[..., None], I[..., None]], -1)
    Y = hi.d_V(G, V, I)[..., None]
    if gout:
        Y = torch.cat([Y, conductances(G)], -1)
    return X, Y


# ---------------- 保存・読み込み ----------------

def resolve(p):
    # タグだけなら results/hh_serial/<tag>.pt
    if os.path.exists(p):
        return p
    return os.path.join(OUT_DIR, p + '.pt')


def save_net(net, path, info):
    torch.save({'cfg': net.cfg, 'state': net.state_dict(), 'info': info}, path)


def load_net(p):
    d = torch.load(resolve(p), weights_only=False)
    net = hi.Net(**d['cfg'])
    net.load_state_dict(d['state'])
    net = net.double()
    net.info = d.get('info', {})
    return net


def chain_cfg(nets):
    # 3 つのネットの形から設定を読み取る
    n1, _, n3 = nets
    return {'i1': n1.cfg['d_in'] == 2,
            'conn': n3.info.get('conn', 'serial'),
            'gout': n3.cfg['d_out'] == 3}


# ---------------- 1 ステップと閉ループ ----------------

def step(nets, cfg, v, g, I_k, hs):
    """
    3 ブロックの 1 ステップ。v: (B,), g: (B, 3), I_k: (B,), hs: 各ブロックの隠れ状態のリスト。

    Returns:
        v_next, g_next, hs, (ab, dg, y3)   y3 は RNN3 の出力全体（gout なら [ΔV, G_Na, G_K]）
    """
    n1, n2, n3 = nets
    x1 = torch.stack([v, I_k], 1) if cfg['i1'] else v[:, None]
    ab, h1 = n1.cell(x1, hs[0])
    dg, h2 = n2.cell(torch.cat([ab.clamp(min=0.0), g], 1), hs[1])
    g_next = (g + dg).clamp(0.0, 1.0)
    g_in = g_next if cfg['conn'] == 'serial' else g
    y3, h3 = n3.cell(torch.cat([g_in, v[:, None], I_k[:, None]], 1), hs[2])
    return v + y3[:, 0], g_next, [h1, h2, h3], (ab, dg, y3)


def run(nets, I_np):
    """hb.score の約束どおり: I (B, N) -> V (B, N-1)。V(0) = V_REST、ゲートは定常値から。"""
    cfg = chain_cfg(nets)
    B, N = I_np.shape
    I = torch.from_numpy(np.asarray(I_np, dtype=np.float64))
    v = torch.full((B,), H_.V_REST, dtype=torch.float64)
    g = torch.tensor(H_.steady_state_gates(H_.V_REST), dtype=torch.float64).repeat(B, 1)
    hs = [n.zeros(B, torch.float64) for n in nets]
    out = torch.empty(B, N - 1, dtype=torch.float64)
    with torch.no_grad():
        for k in range(N - 1):
            v, g, hs, _ = step(nets, cfg, v, g, I[:, k], hs)
            out[:, k] = v
    return out.numpy()


def score(nets, label, val=False):
    return hb.score(lambda I: run(nets, I), label, val=val)


# ---------------- 1. ブロックごとの学習（教師強制） ----------------

@torch.no_grad()
def block_valid(net, data, a):
    # 検証用の軌道（学習に使わない最後の 10 本）での RMSE（物理単位）。出力ごと
    st = data.rng.bit_generator.state
    data.rng = np.random.default_rng(12345)
    I, V, G = data.sample(64, a.L, valid=True)
    data.rng.bit_generator.state = st
    X, Y = block_xy(a.block, I, V, G, a.i1, a.conn, a.gout)
    y, _, _, _ = net(X.float())
    return (y.double() - Y).pow(2).mean((0, 1)).sqrt().tolist()


def train_block(a):
    torch.manual_seed(a.seed)
    rng_t = torch.Generator().manual_seed(a.seed)
    data = hi.Windows(a.data, a.seed)
    d_in = {1: 2 if a.i1 else 1, 2: 9, 3: 5}[a.block]
    d_out = {1: 6, 2: 3, 3: 3 if a.gout else 1}[a.block]
    net = hi.Net(d_in, a.H, a.layers, d_out)
    I, V, G = data.sample(256, a.L)
    X, Y = block_xy(a.block, I, V, G, a.i1, a.conn, a.gout)
    net.set_scalers(X.reshape(-1, d_in).float(), Y.reshape(-1, d_out).float())
    # 出力ごとの損失の重み（gout の補助出力は --aux 倍）
    w_out = torch.ones(d_out)
    if a.block == 3 and a.gout:
        w_out[1:] = a.aux
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    info = {'block': a.block, 'i1': a.i1, 'conn': a.conn, 'gout': a.gout}
    log = {'args': vars(a), 'info': info, 'loss': [], 'valid': []}
    print('RNN{}: d_in = {}, d_out = {}, {}'.format(a.block, d_in, d_out, info), flush=True)
    t0 = time.time()
    for it in range(a.iters):
        for gr in opt.param_groups:
            gr['lr'] = hi.lr_at(it, a.iters, a.lr)
        I, V, G = data.sample(a.batch, a.L)
        V, G = hi.perturb(V, G, a.noise, rng_t)
        X, Y = block_xy(a.block, I, V, G, a.i1, a.conn, a.gout)
        _, ys, _, _ = net(X.float())
        e2 = ((ys - (Y.float() - net.om) / net.os_) ** 2 * w_out).sum(-1) / w_out.sum()
        loss = e2.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        log['loss'].append(loss.item())
        if (it + 1) % a.log_every == 0 or it + 1 == a.iters:
            v = block_valid(net, data, a)
            log['valid'].append((it + 1, v))
            print('it {:5d} loss {:.3e}  valid RMSE {}  ({:.0f} s)'.format(
                it + 1, np.mean(log['loss'][-a.log_every:]), ' '.join('{:.3e}'.format(x) for x in v),
                time.time() - t0), flush=True)
    save_net(net, os.path.join(OUT_DIR, a.tag + '.pt'), info)
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)
    print('saved:', os.path.normpath(os.path.join(OUT_DIR, a.tag + '.pt')))


# ---------------- 2. 評価 ----------------

def evaluate(a):
    nets = [load_net(p) for p in (a.b1, a.b2, a.b3)]
    cfg = chain_cfg(nets)
    print('設定:', cfg, flush=True)
    res = {'blocks': [a.b1, a.b2, a.b3], 'cfg': cfg,
           'val': score(nets, a.tag + ' val', val=True),
           'test': score(nets, a.tag + ' test')}
    json.dump(res, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


# ---------------- 3. 閉ループでの微調整 ----------------

def chain_loss(nets, cfg, I, V, G, a):
    """
    窓 (B, burn+K+1)。最初の burn ステップは正解の状態を入れて隠れ状態を作り（勾配なし）、残り K ステップを閉ループで回す。
      traj : 予測した軌道と正解の差 ((V̂-V)/20)^2 + ((ĝ-g)/0.1)^2
      onpol: 予測した状態での各ブロックの出力と、その状態での HH の式の値との差（標準化した単位）
    """
    B = I.shape[0]
    hs = [n.zeros(B, torch.float64) for n in nets]
    with torch.no_grad():
        for k in range(a.burn):
            _, _, hs, _ = step_teacher(nets, cfg, V[:, k], G[:, k], I[:, k], hs)
    n1, n2, n3 = nets
    v, g = V[:, a.burn], G[:, a.burn]
    traj = onpol = 0.0
    for k in range(a.burn, a.burn + a.K):
        vd, gd = v.detach(), g.detach()
        v, g, hs, (ab, dg, y3) = step(nets, cfg, v, g, I[:, k], hs)
        if a.onpol > 0:
            ab_t = hi.rates(vd)
            y3_t = hi.d_V(gd, vd, I[:, k])[:, None]
            if cfg['gout']:
                y3_t = torch.cat([y3_t, conductances(gd)], 1)
            onpol = onpol + (((ab - ab_t) / n1.os_) ** 2).mean() \
                + (((dg - hi.d_gates(ab_t, gd)) / n2.os_) ** 2).mean() \
                + (((y3 - y3_t) / n3.os_) ** 2).mean()
        Vt = V[:, k + 1]
        w = 1.0 + (a.wthr - 1.0) * ((Vt > -65) & (Vt < -45)).double()
        traj = traj + (w * ((v - Vt) / 20.0) ** 2).mean() + (((g - G[:, k + 1]) / 0.1) ** 2).mean()
    return traj / a.K, (onpol / a.K if a.onpol > 0 else torch.zeros(()))


def step_teacher(nets, cfg, v, g, I_k, hs):
    # 教師強制の 1 ステップ（隠れ状態を作るだけ）。各ブロックに正解の入力を入れる
    n1, n2, n3 = nets
    x1 = torch.stack([v, I_k], 1) if cfg['i1'] else v[:, None]
    _, h1 = n1.cell(x1, hs[0])
    _, h2 = n2.cell(torch.cat([hi.rates(v), g], 1), hs[1])
    g_in = gate_step(v, g) if cfg['conn'] == 'serial' else g
    _, h3 = n3.cell(torch.cat([g_in, v[:, None], I_k[:, None]], 1), hs[2])
    return None, None, [h1, h2, h3], None


def train_chain(a):
    torch.manual_seed(a.seed)
    data = hi.Windows(a.data, a.seed)
    nets = [load_net(p) for p in (a.b1, a.b2, a.b3)]
    cfg = chain_cfg(nets)
    infos = [n.info for n in nets]
    train = [int(c) for c in a.train]
    params = [p for i in train for p in nets[i - 1].parameters()]
    for i, n in enumerate(nets):
        n.requires_grad_((i + 1) in train)
    opt = torch.optim.Adam(params, lr=a.lr)
    log = {'args': vars(a), 'cfg': cfg, 'traj': [], 'onpol': [], 'eval': []}
    print('設定:', cfg, flush=True)
    t0 = time.time()
    best = (-1, -1.0, 0)

    def evaluate_val(it):
        nonlocal best
        r = score(nets, '{} it {} val'.format(a.tag, it), val=True)
        log['eval'].append((it, r['n100'], r['mean']))
        if (r['n100'], r['mean']) > best:
            best = (r['n100'], r['mean'], it)
            for i in train:
                save_net(nets[i - 1], os.path.join(OUT_DIR, '{}_b{}.pt'.format(a.tag, i)), infos[i - 1])

    sched = [tuple(int(x) for x in s.split(':')) for s in a.K_sched.split(',')] if a.K_sched \
        else [(a.K, a.iters)]
    Ks = [K for K, n in sched for _ in range(n)]
    evaluate_val(0)
    for it in range(len(Ks)):
        for gr in opt.param_groups:
            gr['lr'] = hi.lr_at(it, len(Ks), a.lr)
        a.K = Ks[it]
        I, V, G = data.sample(a.batch, a.burn + a.K)
        traj, onpol = chain_loss(nets, cfg, I, V, G, a)
        loss = a.wtraj * traj + a.onpol * onpol
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log['traj'].append(traj.item())
        log['onpol'].append(float(onpol.detach()) if torch.is_tensor(onpol) else float(onpol))
        if (it + 1) % a.log_every == 0:
            print('it {:5d} K {} traj {:.3e} onpol {:.3e}  ({:.0f} s)'.format(
                it + 1, a.K, np.mean(log['traj'][-a.log_every:]), np.mean(log['onpol'][-a.log_every:]),
                time.time() - t0), flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == len(Ks):
            evaluate_val(it + 1)
            json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=0)
    log['best'] = best
    # 検証で最良だった時点に戻して 26 本で評価（微調整しなかったブロックは元のまま）
    for i in train:
        p = os.path.join(OUT_DIR, '{}_b{}.pt'.format(a.tag, i))
        if os.path.exists(p):
            nets[i - 1] = load_net(p)
    log['test'] = score(nets, '{} test (val best it {})'.format(a.tag, best[2]))
    log['sec'] = time.time() - t0
    json.dump(log, open(os.path.join(OUT_DIR, a.tag + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


# ---------------- コマンド ----------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--tag', default='test')
    common.add_argument('--data', default=hb.DATA_OU)
    common.add_argument('--seed', type=int, default=0)
    common.add_argument('--threads', type=int, default=2)
    common.add_argument('--log-every', type=int, default=200)

    pb = sub.add_parser('block', parents=[common])
    pb.add_argument('--block', type=int, required=True, choices=[1, 2, 3])
    pb.add_argument('--i1', action='store_true', help='RNN1 に I(t) も入れる')
    pb.add_argument('--conn', default='serial', choices=['serial', 'parallel'],
                    help='RNN3 の入力のゲート変数: serial = RNN2 の出力 g(t+dt)、parallel = g(t)')
    pb.add_argument('--gout', action='store_true', help='RNN3 にコンダクタンス 2 個も出させる')
    pb.add_argument('--aux', type=float, default=1.0, help='--gout の補助出力の損失の重み')
    pb.add_argument('--H', type=int, default=64)
    pb.add_argument('--layers', type=int, default=1)
    pb.add_argument('--iters', type=int, default=3000)
    pb.add_argument('--batch', type=int, default=32)
    pb.add_argument('--L', type=int, default=1000)
    pb.add_argument('--lr', type=float, default=1e-3)
    pb.add_argument('--noise', type=float, default=0.0)

    for name in ('eval', 'chain'):
        q = sub.add_parser(name, parents=[common])
        for k in (1, 2, 3):
            q.add_argument(f'--b{k}', required=True, help='タグ（results/hh_serial/<tag>.pt）かパス')
        if name == 'chain':
            q.add_argument('--train', default='123', help='微調整するブロック')
            q.add_argument('--K', type=int, default=100)
            q.add_argument('--K-sched', default='100:200,500:200,2000:200',
                           help="'K:回数,K:回数,...'。空にすると --K を --iters 回")
            q.add_argument('--iters', type=int, default=600)
            q.add_argument('--burn', type=int, default=200)
            q.add_argument('--batch', type=int, default=16)
            q.add_argument('--lr', type=float, default=1e-6)
            q.add_argument('--wtraj', type=float, default=1.0)
            q.add_argument('--onpol', type=float, default=1.0)
            q.add_argument('--wthr', type=float, default=3.0)
            q.add_argument('--eval-every', type=int, default=100)

    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    {'block': train_block, 'eval': evaluate, 'chain': train_chain}[a.cmd](a)


if __name__ == '__main__':
    main()
