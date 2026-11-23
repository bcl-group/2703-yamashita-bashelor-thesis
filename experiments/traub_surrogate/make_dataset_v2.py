"""
Traub（19 区画）の教師データ（第 2 版、評価手法（計画）.md 1 章の案）。

今の teacher_modular.npz（12 軌道 × 150 ms、毎ステップ保存）は動作確認用。ここでは
200 軌道 × T ms を作り、軌道ごとに 学習 / 検証 / テスト = 70 / 15 / 15 % に分ける。

保存の仕方（全区画の状態を毎ステップ保存すると 200 軌道 × 5 s で約 80 GB になるため）:
    - 状態（209 個）は stride ステップごと（既定 100 = 1 ms）に保存する。
      1 ステップの正解（増分）は状態だけで決まるので、学習では traub_torch.step でその場で作る（HH の DAgger と同じ考え方）
    - 細胞体の膜電位 V_soma と、細胞体・尖端樹状突起の χ は毎ステップ保存する（評価で使う）
    - 電流 I_nA [nA] は毎ステップ保存する（float32）

電流（細胞体だけ、-0.4〜1.5 nA に収める）。軌道ごとに次の 4 種から 1 つ選び、区間ごとに作り直す:
    const : 一定値 U[-0.2, 1.2]
    step  : 5〜500 ms ごとに U[-0.4, 1.5] へ切り替わる段階電流
    pulse : 幅 1〜20 ms、高さ U[0.5, 1.5] のパルスを、平均間隔 20〜200 ms で（基準 U[-0.2, 0.3]）
    ou    : 基準レベル U[-0.2, 1.2] を 20〜500 ms 保ち、OU の揺らぎ（σ ~ U[0, 0.3]、τ ~ U[1, 50] ms）を足す

初期状態: initial_state() から、軌道ごとに U[500, 2000] ms の助走（同じ種類の電流）をしてから記録を始める
（K_AHP の q や χ のような遅い変数を軌道ごとにばらつかせるため）。

保存するファイル data/traub/teacher_v2.npz:
    I_nA      : (n_traj, N)        float32
    V_soma    : (n_traj, N + 1)    float32。[:, k] は時刻 k dt の値
    xi_soma, xi_apic : (n_traj, N + 1) float32（区画 8 と 13 の χ）
    X         : (n_traj, N // stride + 1, 209) float32。時刻 j·stride·dt の状態 [V 19, χ 19, ゲート 19×9]
    split     : (n_traj,) int8。0 = 学習、1 = 検証、2 = テスト
    kind      : (n_traj,) int8。電流の種類（KINDS の番号）
    meta      : 条件の JSON
使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/traub_surrogate/make_dataset_v2.py --n 10 --T 2000 --name timing   # 時間の見積もり
    uv run python -X utf8 experiments/traub_surrogate/make_dataset_v2.py --n 200 --T 2000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import traub_torch as tt   # noqa: E402

KINDS = ['const', 'step', 'pulse', 'ou']
DT = 0.01
I_MIN, I_MAX = -0.4, 1.5


def ou_unit(n, tau, rng):
    # 平均 0・分散 1 の OU 過程（厳密な離散化）
    a = np.exp(-DT / tau)
    z = rng.standard_normal(n) * np.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for k in range(1, n):
        x[k] = a * x[k - 1] + z[k]
    return x


def make_current(kind, n, rng):
    if kind == 'const':
        I = np.full(n, rng.uniform(-0.2, 1.2))
    elif kind == 'step':
        I = np.empty(n)
        k = 0
        while k < n:
            L = int(rng.uniform(5, 500) / DT)
            I[k:k + L] = rng.uniform(I_MIN, I_MAX)
            k += L
    elif kind == 'pulse':
        I = np.full(n, rng.uniform(-0.2, 0.3))
        mean_gap = rng.uniform(20, 200)
        t = rng.exponential(mean_gap)
        while t < n * DT:
            w = rng.uniform(1, 20)
            I[int(t / DT):int((t + w) / DT)] = rng.uniform(0.5, 1.5)
            t += w + rng.exponential(mean_gap)
    elif kind == 'ou':
        I = np.empty(n)
        k = 0
        while k < n:
            L = int(rng.uniform(20, 500) / DT)
            I[k:k + L] = rng.uniform(-0.2, 1.2)
            k += L
        I = I + rng.uniform(0, 0.3) * ou_unit(n, rng.uniform(1, 50), rng)
    else:
        raise ValueError(kind)
    return np.clip(I, I_MIN, I_MAX)


def flat(s):
    V, xi, G = s
    return torch.cat([V, xi, G.reshape(V.shape[0], -1)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--T', type=float, default=2000.0, help='記録する長さ [ms]')
    ap.add_argument('--stride', type=int, default=100)
    ap.add_argument('--warm-min', type=float, default=500.0)
    ap.add_argument('--warm-max', type=float, default=2000.0)
    ap.add_argument('--seed', type=int, default=20261125)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--name', default='teacher_v2')
    a = ap.parse_args()
    dev, dtype = torch.device(a.device), torch.float64
    rng = np.random.default_rng(a.seed)
    B, N, st = a.n, int(round(a.T / DT)), a.stride
    assert N % st == 0
    kind = np.arange(B) % len(KINDS)
    rng.shuffle(kind)
    split = np.zeros(B, np.int8)
    perm = rng.permutation(B)
    split[perm[int(0.70 * B):int(0.85 * B)]] = 1
    split[perm[int(0.85 * B):]] = 2
    P = tt.Params(dev, dtype)
    t0 = time.time()

    # 1. 助走（軌道ごとに長さを変える。終わった軌道は止める）
    n_warm = (rng.uniform(a.warm_min, a.warm_max, B) / DT).astype(int)
    Iw = torch.as_tensor(np.stack([make_current(KINDS[k], max(int(n_warm.max()), 1), rng) for k in kind]), device=dev, dtype=dtype)
    nw = torch.as_tensor(n_warm, device=dev)
    s = tt.init_state(B, P)
    for k in range(int(n_warm.max())):
        new = tt.step(s, Iw[:, k], P)
        act = (k < nw)
        s = tuple(torch.where(act.view(-1, *([1] * (x.dim() - 1))), y, x) for x, y in zip(s, new))
    print('助走: {:.0f} s'.format(time.time() - t0), flush=True)

    # 2. 記録
    I_np = np.stack([make_current(KINDS[k], N, rng) for k in kind])
    I = torch.as_tensor(I_np, device=dev, dtype=dtype)
    V_soma = torch.empty(B, N + 1, device=dev, dtype=torch.float32)
    xi_s = torch.empty_like(V_soma)
    xi_a = torch.empty_like(V_soma)
    X = torch.empty(B, N // st + 1, tt.NC * 11, device=dev, dtype=torch.float32)
    for k in range(N + 1):
        V_soma[:, k] = s[0][:, tt.SOMA]
        xi_s[:, k] = s[1][:, tt.SOMA]
        xi_a[:, k] = s[1][:, 13]
        if k % st == 0:
            X[:, k // st] = flat(s).float()
        if k < N:
            s = tt.step(s, I[:, k], P)
    sec = time.time() - t0
    print('記録: {:.0f} s（{} 軌道 × {:.0f} ms、助走込み）'.format(sec, B, a.T), flush=True)
    assert torch.isfinite(X).all(), '状態に NaN / inf がある'

    out = os.path.join(ROOT, 'data', 'traub', a.name + '.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = dict(vars(a), dt=DT, kinds=KINDS, sec=sec, state='[V 19, xi 19, gates 19x9 (m h s r n a b c q)]')
    np.savez(out, I_nA=I_np.astype(np.float32), V_soma=V_soma.cpu().numpy(), xi_soma=xi_s.cpu().numpy(),
             xi_apic=xi_a.cpu().numpy(), X=X.cpu().numpy(), split=split, kind=kind.astype(np.int8),
             warm_ms=n_warm * DT, meta=json.dumps(meta))
    print('保存:', out, '({:.0f} MB)'.format(os.path.getsize(out) / 2 ** 20))
    # スパイクの本数（-20 mV を上向きに越えた回数）を種類ごとに表示する（発火様式が偏っていないかの確認）
    v = V_soma.cpu().numpy()
    nsp = ((v[:, :-1] < -20) & (v[:, 1:] >= -20)).sum(1)
    for j, kname in enumerate(KINDS):
        m = kind == j
        print('  {:5s}: {:3d} 本、スパイク数 中央値 {:.0f}、0 本の軌道 {}'.format(kname, m.sum(), np.median(nsp[m]),
                                                                   int((nsp[m] == 0).sum())))


if __name__ == '__main__':
    main()
