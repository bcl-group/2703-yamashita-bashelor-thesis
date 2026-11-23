"""
Traub（19 区画）の代理モデル: HH で効いた作り方（Markov MLP ＋ DAgger）を区画ごとに移したもの。

モデル（全区画で重みを共有する Markov MLP。HH の Markov MLP を区画ごとに使う形）:
    区画 i の入力 u_i (33) = [V_i/50, log1p(χ_i/100), ゲート 9 個, (V_{i-1}-V_i)/10, (V_{i+1}-V_i)/10, I·[i = 細胞体], 区画の番号の one-hot 19]
    出力 y_i (11)          = 区画 i の 1 ステップの増分 [ΔV, Δχ, Δゲート 9 個] を、学習データの標準偏差で割ったもの
    閉ループ: V += ΔV（[-100, 60] に切る）、χ = max(χ + Δχ, 0)、ゲート = clip(ゲート + Δ, 0, 1)
    行列積は (B×19, 33) @ (33, W) なので、B 個の細胞を同時に計算すると行が B×19 本になる

正解: 状態 s と電流 I から traub_torch.step で 1 ステップ進めた増分（fp64）。状態だけで決まるので、
      教師データには 1 ms ごとの状態だけを保存し、正解はここでその場で作る。

学習:
    1. 事前学習: 教師データ（学習用の軌道）の状態から一様に取って学習する
    （--noise-v / --noise-g: 学習の状態にノイズを足し、その状態で正解を作り直す）
    2. DAgger を --dagger 回: 今のモデルを学習用の軌道の電流で閉ループで回し、通った状態を 10 ステップごとに集めて足す
    検証: 検証用の軌道の先頭の状態から --val-ms 回し、細胞体の V の Γ（Δ = 2 ms）で比べる。最良の重みを残す
    テスト用の軌道は、ここでは使わない（`eval --split test` で最後に 1 回だけ）

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/traub_surrogate/traub_mlp.py train --data teacher_v2 --tag t_mlp128 --iters 20000
    uv run python -X utf8 experiments/traub_surrogate/traub_mlp.py eval --tag t_mlp128 --split val
結果: results/traub_mlp/<tag>.pt と <tag>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import traub_torch as tt   # noqa: E402
import traub_eval as te    # noqa: E402

NC, SOMA = tt.NC, tt.SOMA
OUT = os.path.join(ROOT, 'results', 'traub_mlp')
D_IN, D_OUT = 33, 11


def unflat(X):
    # (B, 209) -> (V (B, 19), χ (B, 19), G (B, 19, 9))
    B = X.shape[0]
    return X[:, :NC], X[:, NC:2 * NC], X[:, 2 * NC:].reshape(B, NC, 9)


def flat(s):
    V, xi, G = s
    return torch.cat([V, xi, G.reshape(V.shape[0], -1)], 1)


def features(s, I):
    """s = (V, χ, G)、I: (B,) [nA] -> (B×19, 33)"""
    V, xi, G = s
    B = V.shape[0]
    VL = torch.cat([V[:, :1], V[:, :-1]], 1)
    VR = torch.cat([V[:, 1:], V[:, -1:]], 1)
    Ic = torch.zeros_like(V)
    Ic[:, SOMA] = I
    eye = torch.eye(NC, device=V.device, dtype=V.dtype).expand(B, NC, NC)
    u = torch.cat([(V / 50.0)[..., None], torch.log1p(xi.clamp(min=0.0) / 100.0)[..., None], G,
                   ((VL - V) / 10.0)[..., None], ((VR - V) / 10.0)[..., None], Ic[..., None], eye], -1)
    return u.reshape(B * NC, D_IN)


class Model(nn.Module):
    def __init__(self, W=128, L=2):
        super().__init__()
        layers, d = [], D_IN
        for _ in range(L):
            layers += [nn.Linear(d, W), nn.SiLU()]
            d = W
        layers.append(nn.Linear(d, D_OUT))
        self.net = nn.Sequential(*layers)
        self.register_buffer('scale', torch.ones(D_OUT))   # 増分の標準偏差（学習データから）
        self.cfg = {'W': W, 'L': L}

    def delta(self, s, I):
        B = s[0].shape[0]
        return (self.net(features(s, I)) * self.scale).reshape(B, NC, D_OUT)

    def step(self, s, I):
        V, xi, G = s
        d = self.delta(s, I)
        return ((V + d[..., 0]).clamp(-100.0, 60.0), (xi + d[..., 1]).clamp(min=0.0),
                (G + d[..., 2:]).clamp(0.0, 1.0))


def exact_delta(s, I, P):
    n = tt.step(s, I, P)
    return torch.cat([(n[0] - s[0])[..., None], (n[1] - s[1])[..., None], n[2] - s[2]], -1)   # (B, 19, 11)


def load_data(name, dev):
    d = np.load(os.path.join(ROOT, 'data', 'traub', name + '.npz'))
    meta = json.loads(str(d['meta']))
    return {k: d[k] for k in d.files if k != 'meta'}, meta


@torch.no_grad()
def rollout(model, s, I, stride=None):
    """I: (B, N) を閉ループで回す。V_soma (B, N) と χ_soma (B, N)、stride を指定すると通った状態 (B·N/stride, 209) と電流も返す"""
    B, N = I.shape
    Vs = torch.empty(B, N, device=I.device, dtype=I.dtype)
    Xs = torch.empty_like(Vs)
    keep, keepI = [], []
    for k in range(N):
        if stride and k % stride == 0:
            keep.append(flat(s))
            keepI.append(I[:, k])
        s = model.step(s, I[:, k])
        Vs[:, k] = s[0][:, SOMA]
        Xs[:, k] = s[1][:, SOMA]
    if stride:
        return Vs, Xs, torch.cat(keep), torch.cat(keepI)
    return Vs, Xs


def validate(model, data, meta, split, ms, dev, dtype, n_max=30):
    idx = np.nonzero(data['split'] == split)[0][:n_max]
    N = int(round(ms / meta['dt']))
    s = unflat(torch.as_tensor(data['X'][idx, 0], device=dev, dtype=dtype))
    I = torch.as_tensor(data['I_nA'][idx, :N], device=dev, dtype=dtype)
    Vp, Xp = rollout(model, s, I)
    # 時刻 0 の値を先頭に足す（0 → dt で閾値を越えたスパイクも数えるため）。列 k は時刻 k dt
    Vp = torch.cat([s[0][:, SOMA:SOMA + 1], Vp], 1)
    Xp = torch.cat([s[1][:, SOMA:SOMA + 1], Xp], 1)
    Vt = data['V_soma'][idx, :N + 1]
    Xt = data['xi_soma'][idx, :N + 1]
    r = te.score_traces(Vt, Vp.float().cpu().numpy(), Xt, Xp.float().cpu().numpy(), meta['dt'])
    r['idx'] = idx.tolist()
    return r


def train(a):
    dev = torch.device(a.device)
    dtype = torch.float32
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data, meta = load_data(a.data, dev)
    st = meta['stride']
    tr = np.nonzero(data['split'] == 0)[0]
    # 学習用の状態と、その時刻の電流
    # X[:, j] は時刻 j·st·dt の状態で、その 1 ステップには I_nA[:, j·st] を使う。最後の状態には対応する電流がないので除く
    J = data['X'].shape[1]
    Xtr = data['X'][tr][:, :J - 1].reshape(-1, NC * 11)
    Itr = data['I_nA'][tr][:, ::st][:, :J - 1].reshape(-1)
    assert len(Xtr) == len(Itr)
    # 状態のプールは CPU に float32 で置く（DAgger で増えると GPU のメモリが足りなくなるため）。使う分だけ GPU に送って fp64 にする
    pool_X = torch.as_tensor(Xtr, dtype=torch.float32)
    pool_I = torch.as_tensor(Itr, dtype=torch.float32)
    to64 = lambda x: x.to(dev, torch.float64)   # noqa: E731
    P64 = tt.Params(dev, torch.float64)
    model = Model(a.W, a.L).to(dev)
    with torch.no_grad():
        j = torch.as_tensor(rng.integers(0, len(pool_X), 20000))
        d = exact_delta(unflat(to64(pool_X[j])), to64(pool_I[j]), P64).reshape(-1, D_OUT)
        model.scale.copy_(d.std(0).clamp(min=1e-8).float())
    print('状態の数', len(pool_X), '増分の標準偏差', model.scale.cpu().numpy().round(6), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    total = a.iters + a.dagger * a.dagger_iters
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=a.lr * 0.01)
    log = {'args': vars(a), 'meta': meta, 'val': []}
    best = (-10 ** 9, -1, -1e9, -1)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    it = 0
    for rnd in range(a.dagger + 1):
        if rnd > 0:
            # DAgger: 学習用の軌道の電流で閉ループで回し、通った状態を足す
            nd = min(a.dagger_n, len(tr))
            pick = rng.choice(tr, nd, replace=False)
            Nd = int(round(a.dagger_ms / meta['dt']))
            N_rec = data['I_nA'].shape[1]
            j0 = rng.integers(0, (N_rec - Nd) // st + 1, nd)      # 上限は含まないので +1（末尾までちょうど取れる位置も選ぶ）
            s = unflat(torch.as_tensor(data['X'][pick, j0], device=dev, dtype=dtype))
            I = torch.as_tensor(np.stack([data['I_nA'][p, j * st:j * st + Nd] for p, j in zip(pick, j0)]),
                                device=dev, dtype=dtype)
            _, _, Xd, Id = rollout(model, s, I, stride=a.dagger_stride)
            ok = torch.isfinite(Xd).all(1)
            pool_X = torch.cat([pool_X, Xd[ok].float().cpu()])
            pool_I = torch.cat([pool_I, Id[ok].float().cpu()])
            print(f'DAgger {rnd}: 閉ループの状態 {int(ok.sum())} 個を追加 → 計 {len(pool_X)} 個', flush=True)
            del Xd, Id, ok
        n_it = a.iters if rnd == 0 else a.dagger_iters
        for _ in range(n_it):
            j = torch.as_tensor(rng.integers(0, len(pool_X), a.batch))
            s64 = unflat(to64(pool_X[j]))
            I64 = to64(pool_I[j])
            if a.noise_v > 0 or a.noise_g > 0:
                # 状態をずらし、ずらした状態で正解を作り直す（閉ループで少し外れた状態からも戻れるように）
                V, xi, G = s64
                s64 = ((V + a.noise_v * torch.randn_like(V)).clamp(-100.0, 60.0), xi,
                       (G + a.noise_g * torch.randn_like(G)).clamp(0.0, 1.0))
            with torch.no_grad():
                y = (exact_delta(s64, I64, P64) / model.scale.double()).float()
            s32 = tuple(x.float() for x in s64)
            pred = model.net(features(s32, I64.float())).reshape(-1, NC, D_OUT)
            w = torch.ones(NC, 1, device=dev)
            w[SOMA] = a.soma_w
            loss = (((pred - y) ** 2) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % a.eval_every == 0 or it == total:
                r = validate(model, data, meta, 1, a.val_ms, dev, dtype)
                # 発散・Γ が定義できない軌道が少ないものを優先し、次に Γ = 1 の本数、平均 Γ で比べる
                key = (-(r['n_div'] + r['n_nan']), r['n_gamma1'], r['gamma_mean'], it)
                log['val'].append({'it': it, 'loss': float(loss.detach()), 'gamma_mean': r['gamma_mean'], 'n_gamma1': r['n_gamma1'],
                                   'gamma_burst_mean': r['gamma_burst_mean'], 'ca_err_median': r['ca_err_median'],
                                   'n_div': r['n_div'], 'sec': time.time() - t0})
                print('it {:6d} loss {:.3e}  検証 Γ = 1: {:2d}/{}  平均 Γ {:.3f}  バースト Γ {:.3f}  χ 誤差 {:.3f}  発散 {}  ({:.0f} s)'.format(
                    it, float(loss.detach()), r['n_gamma1'], r['n'], r['gamma_mean'], r['gamma_burst_mean'],
                    r['ca_err_median'], r['n_div'], time.time() - t0), flush=True)
                if key[:3] > best[:3]:
                    best = key
                    torch.save({'cfg': model.cfg, 'state': model.state_dict(), 'it': it},
                               os.path.join(OUT, a.tag + '.pt'))
                log['best'] = list(best)
                json.dump(log, open(os.path.join(OUT, a.tag + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('最良（検証）:', best, flush=True)


def load_model(tag, dev):
    d = torch.load(os.path.join(OUT, tag + '.pt'), map_location=dev, weights_only=False)
    m = Model(**d['cfg']).to(dev)
    m.load_state_dict(d['state'])
    return m.eval()


def evaluate(a):
    dev = torch.device(a.device)
    data, meta = load_data(a.data, dev)
    m = load_model(a.tag, dev)
    split = {'train': 0, 'val': 1, 'test': 2}[a.split]
    r = validate(m, data, meta, split, a.val_ms, dev, torch.float32, n_max=1000)
    print('{} [{}]: Γ = 1 {}/{}  平均 Γ {:.3f}  バースト Γ {:.3f}  χ 誤差 {:.3f}  発散 {}'.format(
        a.tag, a.split, r['n_gamma1'], r['n'], r['gamma_mean'], r['gamma_burst_mean'], r['ca_err_median'], r['n_div']))
    json.dump(r, open(os.path.join(OUT, f'{a.tag}_{a.split}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval'])
    ap.add_argument('--data', default='teacher_v2')
    ap.add_argument('--tag', default='t_mlp128')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--W', type=int, default=128)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=1024, help='1 回の更新で使う細胞の状態の数（行は × 19）')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--dagger', type=int, default=0)
    ap.add_argument('--dagger-iters', type=int, default=10000)
    ap.add_argument('--dagger-n', type=int, default=64)
    ap.add_argument('--dagger-ms', type=float, default=500.0)
    ap.add_argument('--dagger-stride', type=int, default=10)
    ap.add_argument('--noise-v', type=float, default=0.0, help='学習時に V に足すノイズの標準偏差 [mV]')
    ap.add_argument('--noise-g', type=float, default=0.0, help='学習時にゲートに足すノイズの標準偏差')
    ap.add_argument('--soma-w', type=float, default=5.0, help='細胞体の区画の損失の重み')
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--val-ms', type=float, default=500.0)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else evaluate(a)


if __name__ == '__main__':
    main()
