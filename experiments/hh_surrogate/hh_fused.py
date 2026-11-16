"""
モデル2（RNN 3 個）を Tensor Core 向けにまとめ直した版（FusedHH）。式は hh_modular.run_closed_loop(clip=True) と同じ。

変更点（どれも式を変えない書き換え）:
    1. 依存関係: RNN1 は V(k)、RNN3 は m, h, n, V, I(k) だけを使うので、この 2 つは同時に計算できる。
       1 ステップの行列積を 3 回 → 2 回（[RNN1 ‖ RNN3] を bmm 1 回、RNN2 を matmul 1 回）にする。
    2. 標準化の畳み込み:  ((x - μ)/σ) W + b = x (W/σ) + (b - (μ/σ) W)、出力側 (h W + b) s + m も同様。
    3. [x, h] をつないで 1 回の行列積にする。隠れ状態 H=50 は 64 に、入力は 16 に 0 で埋める
       （Tensor Core の 16 単位にそろえる。0 の行・列なので結果は変わらない）。
    4. 複数のニューロン（ここでは 26 個のテスト電流）をバッチの行に並べて同時に回す。

精度の模擬（GPU がなくても Tensor Core の丸めを再現する）:
    Tensor Core は「入力を低精度に丸め、積和は fp32 で足す」。そこで、行列積の 2 つの入力（重み、[x, h]）を
    bf16 / fp16 / TF32 に丸めてから fp32 で行列積を計算する。状態の足し込み V + ΔV、g + Δg は fp32 で行う。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_fused.py check          # NumPy 版との一致
    uv run python -X utf8 experiments/hh_surrogate/hh_fused.py eval           # 精度ごとに 26 電流
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_modular as hm   # src を sys.path に入れる
import test_currents as tc
from neuron.hh import simulate, V_REST

ROOT = os.path.join(hm.BASE_DIR, '..', '..')
DT = 0.01
HP = 64     # 隠れ状態の幅（0 埋め後）
KX = 16     # 入力の幅（0 埋め後）
OUT = 8     # 出力の幅（0 埋め後）
MODES = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16']


def round_to(x, mode):
    """行列積の入力を Tensor Core の精度に丸める（値は fp32 のまま返す）。"""
    if mode == 'bf16':
        return x.to(torch.bfloat16).to(torch.float32)
    if mode == 'fp16':
        return x.to(torch.float16).to(torch.float32)
    if mode == 'tf32':
        # fp32 の仮数部 23 bit を 10 bit に丸める（最近接丸め、0.5 は切り上げ）
        b = x.contiguous().view(torch.int32)
        b = (b + 0x1000) & ~0x1FFF
        return b.view(torch.float32)
    return x


def fold(model, in_s, out_s):
    """
    標準化を重みに畳み込み、[x, h] -> h の重み (KX+HP, HP) と出力の重み (HP, OUT) を作る。

    Returns:
        W (KX+HP, HP), b (HP,), Wy (HP, OUT), by (OUT,)。float64 の numpy 配列。
    """
    d, H, o = model.input_dim, model.hidden_dim, model.output_dim
    assert d <= KX and H <= HP and o <= OUT
    W = np.zeros((KX + HP, HP))
    W[:d, :H] = model.W_xh / in_s.std[:, None]
    W[KX:KX + H, :H] = model.W_hh
    b = np.zeros(HP)
    b[:H] = model.b_h - (in_s.mean / in_s.std) @ model.W_xh
    Wy = np.zeros((HP, OUT))
    Wy[:H, :o] = model.W_hy * out_s.std[None, :]
    by = np.zeros(OUT)
    by[:o] = model.b_y * out_s.std + out_s.mean
    return W, b, Wy, by


class FusedHH:
    def __init__(self, blocks, mode='fp64'):
        # RNN1 の出力は target によらず α, β そのもの（hm.run_closed_loop と同じ扱い）
        assert blocks[2][0].target == 'delta' and blocks[3][0].target == 'delta'
        self.mode = mode
        self.dtype = torch.float64 if mode == 'fp64' else torch.float32
        f = {k: fold(*blocks[k]) for k in (1, 2, 3)}
        T = lambda a: round_to(torch.tensor(np.asarray(a), dtype=self.dtype), mode)
        # ステージ A: ブロック 0 = RNN1、ブロック 1 = RNN3
        self.WA = T(np.stack([f[1][0], f[3][0]]))              # (2, KX+HP, HP)
        self.bA = torch.tensor(np.stack([f[1][1], f[3][1]])[:, None, :], dtype=self.dtype)  # (2, 1, HP)
        self.WyA = T(np.stack([f[1][2], f[3][2]]))             # (2, HP, OUT)
        self.byA = torch.tensor(np.stack([f[1][3], f[3][3]])[:, None, :], dtype=self.dtype)
        # ステージ B: RNN2
        self.WB, self.WyB = T(f[2][0]), T(f[2][2])
        self.bB = torch.tensor(f[2][1], dtype=self.dtype)
        self.byB = torch.tensor(f[2][3], dtype=self.dtype)

    def mm(self, a, b):
        # Tensor Core の模擬: 入力 a を丸め（b は丸め済みの重み）、積和は fp32（fp64）
        return torch.matmul(round_to(a, self.mode), b)

    def run(self, I, V0=V_REST):
        """
        Args:
            I (np.ndarray): (B, N) の電流 [uA/cm^2]。B 個のニューロンを同時に回す。
        Returns:
            np.ndarray: (B, N-1) の V。hm.run_closed_loop(...)[0] と同じ並び。
        """
        B, N = I.shape
        dt = self.dtype
        It = torch.tensor(I, dtype=dt)
        V = torch.full((B,), float(V0), dtype=dt)
        g = torch.tensor(hm.steady_state_gates(V0), dtype=dt).repeat(B, 1)   # (B, 3)
        hA = torch.zeros(2, B, HP, dtype=dt)
        hB = torch.zeros(B, HP, dtype=dt)
        xA = torch.zeros(2, B, KX, dtype=dt)
        xB = torch.zeros(B, KX, dtype=dt)
        out = torch.empty(B, N - 1, dtype=dt)
        with torch.no_grad():
            for k in range(N - 1):
                # ステージ A: RNN1（入力 V）と RNN3（入力 m, h, n, V, I）を bmm 1 回で
                xA[0, :, 0] = V
                xA[1, :, :3] = g
                xA[1, :, 3] = V
                xA[1, :, 4] = It[:, k]
                hA = torch.tanh(self.mm(torch.cat([xA, hA], 2), self.WA) + self.bA)
                yA = self.mm(hA, self.WyA) + self.byA                 # (2, B, OUT)
                ab = torch.clamp(yA[0, :, :6], min=0.0)
                dV = yA[1, :, 0]
                # ステージ B: RNN2（入力 α, β, m, h, n）
                xB[:, :6] = ab
                xB[:, 6:9] = g
                hB = torch.tanh(self.mm(torch.cat([xB, hB], 1), self.WB) + self.bB)
                dg = (self.mm(hB, self.WyB) + self.byB)[:, :3]
                # 状態の足し込みは fp32 以上
                V = V + dV
                g = torch.clamp(g + dg, 0.0, 1.0)
                out[:, k] = V
        return out.double().numpy()


def check(d):
    blocks = tc.load_blocks(d)
    t = np.arange(0, 300.0, DT)
    I = np.stack([np.full_like(t, 10.0), np.full_like(t, 7.0)])
    t0 = time.time()
    ref = np.stack([hm.run_closed_loop(blocks, Ii, V0=V_REST, clip=True)[0] for Ii in I])
    t1 = time.time()
    got = FusedHH(blocks, 'fp64').run(I)
    t2 = time.time()
    err = np.abs(got - ref)
    print('NumPy 版との差（fp64、300 ms、const 10 / const 7）')
    for n_ in (1, 10, 100, 1000, 10000, len(t) - 1):
        print('  最初の {:6d} ステップの最大差: {:.3e} mV'.format(n_, err[:, :n_].max()))
    print('時間: NumPy 版 {:.1f} s（1 本ずつ）, FusedHH {:.1f} s（2 本同時）'.format(t1 - t0, t2 - t1))


def evaluate(d, out_dir, modes):
    blocks = tc.load_blocks(d)
    os.makedirs(out_dir, exist_ok=True)
    # 26 電流を長さをそろえて（短いものは末尾を 0 で埋めて）同時に回す
    Ts = [T for *_, T in tc.TESTS]
    N = int(round(max(Ts) / DT))
    t_full = np.arange(N) * DT
    I = np.zeros((len(tc.TESTS), N))
    Vt, lens = [], []
    for i, (name, func, params, T) in enumerate(tc.TESTS):
        t = np.arange(0, T, DT)
        I[i, :len(t)] = func(t, **params)
        Vt.append(simulate(t, I[i, :len(t)], V0=V_REST)[0][1:])
        lens.append(len(t))
    res = {'tests': [n for n, *_ in tc.TESTS], 'modes': {}}
    for mode in modes:
        t0 = time.time()
        Vp = FusedHH(blocks, mode).run(I)
        sec = time.time() - t0
        rows = []
        for i, name in enumerate(res['tests']):
            vp = Vp[i, :lens[i] - 1]
            st, sp = tc.spike_times(Vt[i], DT), tc.spike_times(vp, DT)
            rows.append({'spikes': (len(st), len(sp)), 'match': tc.match_rate(st, sp),
                         'rmse': float(np.sqrt(np.mean((vp - Vt[i]) ** 2)))})
        n100 = sum(r['match'] == 1.0 for r in rows)
        res['modes'][mode] = {'rows': rows, 'n100': n100, 'sec': sec}
        print('{:5s}  一致率 100% {:2d}/26   平均一致率 {:.3f}   {:.0f} s'.format(
            mode, n100, np.mean([r['match'] for r in rows]), sec), flush=True)
    json.dump(res, open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    plot(res, out_dir)


def plot(res, out_dir):
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    modes = list(res['modes'])
    M = np.array([[r['match'] for r in res['modes'][m]['rows']] for m in modes]).T  # (26, モード)
    fig, ax = plt.subplots(figsize=(1.3 * len(modes) + 3, 9))
    ax.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, '{:.0%}'.format(M[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 0.6 else '#1f1f1f')
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(['{}\n100%: {}/26'.format(m, res['modes'][m]['n100']) for m in modes])
    ax.set_yticks(range(len(res['tests'])))
    ax.set_yticklabels(res['tests'], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_title('FusedHH: 行列積の精度ごとのスパイク時刻の一致率（閉ループ、26 電流）', pad=40)
    fig.tight_layout()
    p = os.path.join(out_dir, 'precision_match.png')
    fig.savefig(p, dpi=110)
    print('saved:', os.path.normpath(p))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['check', 'eval'])
    p.add_argument('--dir', default=os.path.join(ROOT, 'results', 'hh_modular_ou'))
    p.add_argument('--out', default=None)
    p.add_argument('--modes', nargs='+', default=MODES)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.cmd == 'check':
        check(a.dir)
    else:
        evaluate(a.dir, a.out or os.path.join(a.dir, 'fused_precision'), a.modes)


if __name__ == '__main__':
    main()
