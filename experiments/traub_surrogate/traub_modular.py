"""
Traub モデル（19 コンパートメント）の分割型代理モデル。HH のモデル2（hh_modular.py）と同じ分け方で、
陽的オイラー法（dt = 0.01 ms）の 1 ステップを RNN 3 個で置き換える。

各ブロックは全コンパートメントで重みを共有する（α, β の式やゲートの更新式はコンパートメントによらず
同じなので）。コンパートメントの違い（最大コンダクタンス、隣との結合）は RNN3 の入力 P_i で与える。
19 個のコンパートメントは RNN のバッチ次元に並べ、それぞれが自分の隠れ状態を持つ。

    RNN1: V_i(t)                                          -> α, β (t)            16 個（8 ゲート × 2）
    RNN2: α, β(t), ゲート 8 個(t), q(t), χ(t)              -> Δ(ゲート 8 個, q)
    RNN3: ゲート 8 個(t), q(t), χ(t), V_{i-1}, V_i, V_{i+1}(t), I_i(t), P_i -> Δ(V_i, χ_i)

    ゲート: m, h（Na）, s, r（Ca）, n（K_DR）, a, b（K_A）, c（K_C）。q（K_AHP）は χ の関数なので RNN2 で扱う
    P_i   : g_Na, g_K_DR, g_K_A, g_K_C, g_K_AHP, g_Ca, φ, g_{i-1,i}/area_i, g_{i,i+1}/area_i（9 個）
    I_i   : 注入電流密度 [uA/cm^2]。細胞体（8）以外は 0

閉ループ:
    αβ      = max(RNN1(V), 0)
    x(t+dt) = clip(x + RNN2(...), 0, 1)        x = ゲート 8 個と q
    V(t+dt) = V + ΔV,  χ(t+dt) = max(χ + Δχ, 0)

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/traub_surrogate/make_dataset.py
    uv run python -X utf8 experiments/traub_surrogate/traub_modular.py train --block 1
    uv run python -X utf8 experiments/traub_surrogate/traub_modular.py train --block 2
    uv run python -X utf8 experiments/traub_surrogate/traub_modular.py train --block 3
    uv run python -X utf8 experiments/traub_surrogate/traub_modular.py eval
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))

from rnn import SimpleRNN
from neuron import traub as T
from neuron import current as cur
# 保存・読み込み・標準化・1 ステップの計算は HH の分割型と共通
from hh_modular import ColumnStandardizer, save_block, load_block, rnn_step

DATA_PATH = os.path.join(ROOT, 'data', 'traub', 'teacher_modular.npz')
RESULT_DIR = os.path.join(ROOT, 'results', 'traub_modular')

GATES = ['m', 'h', 's', 'r', 'n', 'a', 'b', 'c']
AB_NAMES = [f'{p}_{g}' for g in GATES for p in ('alpha', 'beta')]
P_NAMES = ['g_Na', 'g_K_DR', 'g_K_A', 'g_K_C', 'g_K_AHP', 'g_Ca', 'phi', 'gL/area', 'gR/area']
BLOCKS = {
    1: {'inputs': ['V'], 'outputs': AB_NAMES},
    2: {'inputs': AB_NAMES + GATES + ['q', 'xi'], 'outputs': ['d' + g for g in GATES + ['q']]},
    3: {'inputs': GATES + ['q', 'xi', 'V_left', 'V', 'V_right', 'I'] + P_NAMES,
        'outputs': ['dV', 'dxi']},
}
NA_TO_DENSITY = 1e-3 / T.area[T.SOMA]
DT = 0.01


def comp_params():
    # P: (NC, 9)。隣との結合は i-1 側と i+1 側を分け、両端は 0
    g = T.compartment_conductance()
    gL = np.r_[0.0, g] / T.area
    gR = np.r_[g, 0.0] / T.area
    return np.stack([T.g_Na, T.g_K_DR, T.g_K_A, T.g_K_C, T.g_K_AHP, T.g_Ca,
                     T.phi / 1e4, gL, gR], axis=1)


def rates(V):
    # V: (..., NC) -> (..., NC, 16)。列の順は AB_NAMES
    u = V - T.V_LEAK
    cols = []
    for g in GATES:
        alpha, beta = T.VOLTAGE_GATES[g]
        cols += [alpha(u), beta(u)]
    return np.stack(cols, axis=-1)


def neighbors(V):
    # V: (..., NC) -> 左隣、右隣の電位。両端は自分自身（結合は 0 なので値は効かない）
    VL = np.concatenate([V[..., :1], V[..., :-1]], axis=-1)
    VR = np.concatenate([V[..., 1:], V[..., -1:]], axis=-1)
    return VL, VR


def block_xy(block, d, P):
    """
    教師データ d（make_dataset.py の npz）から、ブロックの入力 X と正解 Y を作る。

    Returns:
        X: (n_traj * NC, N, d_in)、Y: (n_traj * NC, N, d_out)。float32、物理単位。
            1 行が「ある軌道のあるコンパートメント」の系列。正解の変化量は float64 で引いてから float32 にする
    """
    V = d['V']                                   # (B, N+1, NC)
    B, N1, NC = V.shape
    N = N1 - 1
    G = np.stack([d[g] for g in GATES], axis=-1)  # (B, N+1, NC, 8)
    q, xi = d['q'], d['xi']

    if block == 1:
        X = V[:, :N, :, None]
        Y = rates(V[:, :N])
    elif block == 2:
        X = np.concatenate([rates(V[:, :N]), G[:, :N], q[:, :N, :, None], xi[:, :N, :, None]], axis=-1)
        Y = np.concatenate([G[:, 1:] - G[:, :N], (q[:, 1:] - q[:, :N])[..., None]], axis=-1)
    elif block == 3:
        VL, VR = neighbors(V[:, :N])
        I = np.zeros((B, N, NC))
        I[..., T.SOMA] = d['I_nA'][:, :N] * NA_TO_DENSITY
        Pb = np.broadcast_to(P, (B, N, NC, P.shape[1]))
        X = np.concatenate([G[:, :N], q[:, :N, :, None], xi[:, :N, :, None], VL[..., None],
                            V[:, :N, :, None], VR[..., None], I[..., None], Pb], axis=-1)
        Y = np.stack([V[:, 1:] - V[:, :N], xi[:, 1:] - xi[:, :N]], axis=-1)
    else:
        raise ValueError(block)
    # (B, N, NC, d) -> (B*NC, N, d)
    f = lambda A: np.ascontiguousarray(A.transpose(0, 2, 1, 3).reshape(B * NC, N, A.shape[-1]),
                                       dtype=np.float32)
    return f(X), f(Y)


def train(block, epochs, lr, batch_size, chunk_ms, hidden, seed, out_dir, threads, init=None):
    np.random.seed(seed)
    spec = BLOCKS[block]
    d = dict(np.load(DATA_PATH))
    X, Y = block_xy(block, d, comp_params())
    del d
    rows, N, d_in = X.shape
    in_s = ColumnStandardizer.fit(X.reshape(-1, d_in).astype(np.float64))
    out_s = ColumnStandardizer.fit(Y.reshape(-1, Y.shape[2]).astype(np.float64))
    if init:
        _, in_s, out_s, _ = load_block(init)
    # 標準化も float32 のまま行い、メモリを節約する
    Xs = ((X - in_s.mean.astype(np.float32)) / in_s.std.astype(np.float32))
    Ys = ((Y - out_s.mean.astype(np.float32)) / out_s.std.astype(np.float32))
    L = int(round(chunk_ms / DT))
    n_chunk = N // L
    x = Xs[:, :n_chunk * L].reshape(rows * n_chunk, L, d_in)
    t = Ys[:, :n_chunk * L].reshape(rows * n_chunk, L, Y.shape[2])

    print('Traub RNN{}: {} -> {}'.format(block, spec['inputs'], spec['outputs']))
    print('x.shape = {}, t.shape = {}'.format(x.shape, t.shape))
    model = SimpleRNN(input_dim=d_in, hidden_dim=hidden, output_dim=Y.shape[2],
                      init='xavier', return_sequences=True)
    prev = []
    if init:
        m0 = load_block(init)[0]
        for name in model.param_names():
            setattr(model, name, getattr(m0, name).copy())
        prev = list(np.load(init)['loss'])

    import torch
    from rnn.torch_backend import train_chunks_torch
    torch.set_num_threads(threads)
    start = time.time()
    model.target = 'abs' if block == 1 else 'delta'

    def checkpoint(h):
        # 途中経過も block{n}.npz に上書き保存する（止まっても --init で続きから学習できる）
        save_block({'block': block, 'target': model.target, 'seed': seed, 'lr': lr,
                    'chunk_ms': chunk_ms, 'data_path': DATA_PATH,
                    'hist': {'loss': prev + list(h['loss'])}, 'model': model,
                    'in_scaler': in_s, 'out_scaler': out_s}, out_dir)

    hist = train_chunks_torch(model, x, t, epochs=epochs, batch_size=batch_size, lr=lr,
                              verbose_every=10, checkpoint_every=50, on_checkpoint=checkpoint)
    elapsed = time.time() - start
    hist['loss'] = prev + list(hist['loss'])
    model.target = 'abs' if block == 1 else 'delta'

    # 教師強制の誤差（物理単位）。最初の軌道の 19 コンパートメントを系列全体で流す
    y = model.forward(Xs[:T.NC].astype(np.float64))
    Yp = out_s.inverse(y.reshape(-1, Y.shape[2]))
    Yt = Y[:T.NC].reshape(-1, Y.shape[2]).astype(np.float64)
    rmse = np.sqrt(np.mean((Yp - Yt) ** 2, axis=0))
    print('\nepochs: {}, time: {:.0f} s'.format(len(hist['loss']), elapsed))
    print('教師強制の RMSE（軌道 0、全コンパートメント）:')
    for n_, v, s in zip(spec['outputs'], rmse, out_s.std):
        print('  {:10s} {:.4g}   （正解の標準偏差 {:.4g}）'.format(n_, v, s))

    save_block({'block': block, 'target': model.target, 'seed': seed, 'lr': lr,
                'chunk_ms': chunk_ms, 'data_path': DATA_PATH, 'hist': hist, 'model': model,
                'in_scaler': in_s, 'out_scaler': out_s}, out_dir)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(hist['loss'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (標準化した MSE)')
    ax.set_title('Traub RNN{}'.format(block))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'block{}_learning_curve.png'.format(block)), dpi=110)


# --- 閉ループ ---

def run_closed_loop(blocks, I_nA, state0, clip=True):
    """
    I_nA (N,) [nA] と初期状態 state0（traub.initial_state() と同じ形、各 (NC,)）から、
    3 ブロックをつないで全コンパートメントの V を逐次計算する。

    Returns:
        V: (N, NC)。V[k] は k+1 ステップ目の値（traub の V_hist[1:] に対応）
    """
    (m1, i1, o1), (m2, i2, o2), (m3, i3, o3) = blocks[1], blocks[2], blocks[3]
    P = comp_params()
    NC = T.NC
    V = state0['V'].astype(float).copy()
    G = np.stack([state0[g] for g in GATES], axis=-1).astype(float)
    q = state0['q'].astype(float).copy()
    xi = state0['xi'].astype(float).copy()
    h1, h2, h3 = (np.zeros((NC, m.hidden_dim)) for m in (m1, m2, m3))
    I = np.zeros(NC)
    out = np.empty((len(I_nA), NC))
    for k in range(len(I_nA)):
        y, h1 = rnn_step(m1, i1.transform(V[:, None]), h1)
        ab = o1.inverse(y)
        if clip:
            ab = np.maximum(ab, 0.0)
        y, h2 = rnn_step(m2, i2.transform(np.concatenate([ab, G, q[:, None], xi[:, None]], 1)), h2)
        dg = o2.inverse(y)
        VL, VR = neighbors(V)
        I[T.SOMA] = I_nA[k] * NA_TO_DENSITY
        y, h3 = rnn_step(m3, i3.transform(np.concatenate(
            [G, q[:, None], xi[:, None], VL[:, None], V[:, None], VR[:, None], I[:, None], P], 1)), h3)
        dv = o3.inverse(y)
        G = G + dg[:, :8]
        q = q + dg[:, 8]
        V = V + dv[:, 0]
        xi = xi + dv[:, 1]
        if clip:
            G = np.clip(G, 0.0, 1.0)
            q = np.clip(q, 0.0, 1.0)
            xi = np.maximum(xi, 0.0)
        out[k] = V
    return out


def simulate_traub(I_nA, state0):
    g_comp = T.compartment_conductance()
    st = {k: v.copy() for k, v in state0.items()}
    out = np.empty((len(I_nA), T.NC))
    inj = np.zeros(T.NC)
    for k in range(len(I_nA)):
        inj[T.SOMA] = I_nA[k] * NA_TO_DENSITY
        st = T.step(st, inj, DT, g_comp)
        out[k] = st['V']
    return out


def spike_times(V, thr=-20.0):
    up = np.where((V[1:] >= thr) & (V[:-1] < thr))[0]
    return (up + 1) * DT


def match_rate(t_true, t_pred, window=2.0):
    if len(t_true) == 0 and len(t_pred) == 0:
        return 1.0
    if len(t_true) == 0 or len(t_pred) == 0:
        return 0.0
    used = np.zeros(len(t_pred), bool)
    hit = 0
    for s in t_true:
        j = np.argmin(np.abs(t_pred - s) + used * 1e9)
        if not used[j] and abs(t_pred[j] - s) <= window:
            used[j] = True
            hit += 1
    return hit / max(len(t_true), len(t_pred))


def const(a):
    return lambda t: np.full_like(t, a)


TESTS = [
    # (名前, 電流を作る関数 t -> I [nA], 長さ [ms])
    ('const 0 nA',          const(0.0), 300.0),
    ('const 0.2 nA',        const(0.2), 300.0),
    ('const 0.5 nA',        const(0.5), 300.0),
    ('const 1.0 nA',        const(1.0), 300.0),
    ('step 1 nA 50-100ms',  lambda t: cur.step_current(t, 1.0, 50.0, 100.0), 300.0),
    ('step -0.3 nA',        lambda t: cur.step_current(t, -0.3, 50.0, 150.0), 300.0),
    ('pulse 3 nA/1ms/20ms', lambda t: cur.pulse_train_current(t, 3.0, 1.0, 20.0, 20.0), 300.0),
    ('OU 0.6±0.2',          lambda t: 0.6 + 0.2 * cur._unit_ou(len(t), DT, 1.0, np.random.default_rng(0)), 300.0),
    ('level+OU',            lambda t: cur.level_ou_current(t, -0.2, 1.2, 20.0, 150.0, 0.2, 1.0, -0.4, 1.5, seed=7), 500.0),
]


def cost_table(blocks):
    # 1 ステップあたりの演算回数。RNN は 1 コンパートメントあたり H(d+H+o) 回の積とほぼ同数の和、
    # tanh H 回、標準化（入力・出力）を数える。19 コンパートメント分を掛ける
    tot = {'mul': 0, 'pm': 0, 'tanh': 0, 'div': 0}
    for b, (m, _, _) in blocks.items():
        H, di, do = m.hidden_dim, m.input_dim, m.output_dim
        tot['mul'] += H * (di + H + do) + do
        tot['pm'] += H * (di + H + do) + di + do
        tot['tanh'] += H
        tot['div'] += di
    return {k: v * T.NC for k, v in tot.items()}


def evaluate(d_blocks, out_dir):
    blocks = {b: load_block(os.path.join(d_blocks, 'block{}.npz'.format(b)))[:3] for b in (1, 2, 3)}
    os.makedirs(out_dir, exist_ok=True)
    state0 = T.initial_state()
    rows = []
    for name, func, Tms in TESTS:
        t = np.arange(0, Tms, DT)
        I = func(t)
        t0 = time.time()
        Vt = simulate_traub(I, state0)
        t1 = time.time()
        Vp = run_closed_loop(blocks, I, state0)
        t2 = time.time()
        st, sp = spike_times(Vt[:, T.SOMA]), spike_times(Vp[:, T.SOMA])
        rmse = float(np.sqrt(np.mean((Vp[:, T.SOMA] - Vt[:, T.SOMA]) ** 2)))
        if not np.isfinite(rmse):
            rmse = float('inf')
        r = {'name': name, 't': t, 'I': I, 'Vt': Vt, 'Vp': Vp, 'spikes': (len(st), len(sp)),
             'match': match_rate(st, sp), 'rmse': rmse, 'sec': (t1 - t0, t2 - t1)}
        rows.append(r)
        print('{:22s} spikes Traub {:3d} / 代理 {:3d}  一致 {:4.0%}  RMSE(soma) {:7.2f} mV'.format(
            name, *r['spikes'], r['match'], rmse))

    # 波形の図：細胞体と尖端樹状突起 13
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 2.0 * len(rows)), sharey=True, squeeze=False)
    for i, r in enumerate(rows):
        for j, (c, lab) in enumerate([(T.SOMA, '細胞体 (8)'), (13, '尖端樹状突起 (13)')]):
            ax = axes[i, j]
            ax.plot(r['t'], r['Vt'][:, c], color='#888780', ls='--', lw=0.9, label='Traub')
            ax.plot(r['t'], np.clip(r['Vp'][:, c], -150, 100), color='#2a78d6', lw=0.8, label='代理モデル')
            ax.set_ylim(-100, 60)
            ax.grid(True, color='#e6e5df', lw=0.5)
            if i == 0:
                ax.set_title(lab)
                ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=2)
            if j == 0:
                ax.set_ylabel('{}\nV [mV]'.format(r['name']), fontsize=8)
                ax.text(0.99, 0.04, 'スパイク {} / {}　一致 {:.0%}'.format(*r['spikes'], r['match']),
                        transform=ax.transAxes, ha='right', fontsize=8, color='#52514e')
    for ax in axes[-1]:
        ax.set_xlabel('時刻 [ms]')
    fig.suptitle('Traub（19 区画）の分割型代理モデル：閉ループ（灰色の破線が Traub）')
    fig.tight_layout()
    p = os.path.join(out_dir, 'closedloop_tests.png')
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print('saved:', os.path.normpath(p))

    cost = cost_table(blocks)
    print('\n1 ステップあたりの演算回数（代理モデル、19 区画分）:', cost, '合計', sum(cost.values()))
    json.dump({
        'tests': [{k: r[k] for k in ('name', 'spikes', 'match', 'rmse', 'sec')} for r in rows],
        'cost_surrogate': cost,
    }, open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def main():
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pt = sub.add_parser('train')
    pt.add_argument('--block', type=int, required=True, choices=[1, 2, 3])
    pt.add_argument('--epochs', type=int, default=400)
    pt.add_argument('--lr', type=float, default=0.001)
    pt.add_argument('--batch-size', type=int, default=100)
    pt.add_argument('--chunk-ms', type=float, default=10.0)
    pt.add_argument('--hidden', type=int, default=50)
    pt.add_argument('--seed', type=int, default=123)
    pt.add_argument('--threads', type=int, default=1)
    pt.add_argument('--init', default=None)
    pt.add_argument('--out', default=RESULT_DIR)
    pe = sub.add_parser('eval')
    pe.add_argument('--dir', default=RESULT_DIR)
    a = p.parse_args()
    if a.cmd == 'train':
        os.makedirs(a.out, exist_ok=True)
        train(a.block, a.epochs, a.lr, a.batch_size, a.chunk_ms, a.hidden, a.seed, a.out,
              a.threads, a.init)
    else:
        evaluate(a.dir, a.dir)


if __name__ == '__main__':
    main()
