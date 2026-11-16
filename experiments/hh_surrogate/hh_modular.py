"""
モデル2: HH の式の構造に合わせて RNN を3つに分けた代理モデル。

    RNN1: V(t)                        -> α_m, β_m, α_h, β_h, α_n, β_n (t)
    RNN2: α, β(t), m(t), h(t), n(t)   -> m, h, n(t+dt)
    RNN3: m, h, n(t), V(t), I(t)      -> V(t+dt)

各ブロックは入力を1時刻ずつ受け取り、自分の隠れ状態を時刻をまたいで持ち越す
（SimpleRNN(return_sequences=True)）。

学習: ブロックごとに、正解（npz の V, m, h, n と、V から hh.py で計算した α, β）を
      入力・出力にして別々に学習する（教師強制）。系列は 10 ms ずつ区切り、各区間は
      h=0 から始め、区間をシャッフルしてミニバッチにする（hh_surrogate.train_chunks）。
推論: 3つをつなぎ、出した V, m, h, n を次の時刻の入力に戻して回す。
      外から与えるのは I(t) と初期状態（静止電位とそのときのゲート変数の定常値）だけ。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_modular.py train --block 1
    uv run python -X utf8 experiments/hh_surrogate/hh_modular.py train --block 2
    uv run python -X utf8 experiments/hh_surrogate/hh_modular.py train --block 3
    uv run python -X utf8 experiments/hh_surrogate/hh_modular.py eval
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

# hh_surrogate.py が src/ を sys.path に追加する
from hh_surrogate import (
    DATA_PATH,
    EVAL_T,
    EVAL_CURRENTS,
    make_eval_current,
    count_spikes,
    train_chunks,
    plot_case,
)
from rnn import SimpleRNN, Adam, EarlyStopping, numerical_gradient_check
from neuron.hh import (
    simulate, steady_state_gates, V_REST,
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n,
)
from evaluation import compare_waveforms, format_metrics, HH_COST, rnn_cost, format_cost


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'hh_modular')

AB_NAMES = ['alpha_m', 'beta_m', 'alpha_h', 'beta_h', 'alpha_n', 'beta_n']
GATE_NAMES = ['m', 'h', 'n']

BLOCKS = {
    1: {'inputs': ['V'], 'outputs': AB_NAMES},
    2: {'inputs': AB_NAMES + GATE_NAMES, 'outputs': GATE_NAMES},
    3: {'inputs': GATE_NAMES + ['V', 'I'], 'outputs': ['V']},
}


def load_hh_full(path=DATA_PATH):
    # path はカンマ区切りで複数指定できる（軌道の向きにつなげる。どれも複数軌道の npz で同じ dt・長さ）
    paths = path.split(',')
    if len(paths) > 1:
        parts = [load_hh_full(p) for p in paths]
        t, meta = parts[0][0], parts[0][6]
        meta = dict(meta, sources=[os.path.normpath(p) for p in paths])
        return (t, *[np.concatenate([q[i] for q in parts]) for i in range(1, 6)], meta)
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    return d['t'], d['I_ext'], d['V'], d['m'], d['h'], d['n'], meta


def rate_constants(V):
    # V: (N,) -> (N, 6)。列の順は AB_NAMES
    return np.stack([alpha_m(V), beta_m(V), alpha_h(V), beta_h(V),
                     alpha_n(V), beta_n(V)], axis=1)


TARGETS = ('abs', 'delta')
# RNN2, RNN3 の入力 X のうち、出力と同じ変数（更新前の値）が入っている列
STATE_COLS = {2: slice(6, 9), 3: slice(3, 4)}


def block_data(block, I, V, m, h, n, target='abs'):
    """
    ブロックの入力 X(t) と正解 Y を、物理単位のまま作る。

    Args:
        block (int): 1, 2, 3。
        I, V, m, h, n (np.ndarray): shape (N,)。
        target (str): RNN2, RNN3 の正解の形。
            'abs'  : 次の時刻の値 x(t+dt)
            'delta': 1ステップの変化量 Δx = x(t+dt) - x(t)
            RNN1 の α, β は V の同時刻の関数なので、どちらでも同じ（値そのもの）。

    Returns:
        tuple[np.ndarray, np.ndarray]: X (N-1, 入力次元), Y (N-1, 出力次元)。
            k 番目の行が時刻 t_k の入力と、それに対応する正解。
    """
    X, Y = _block_data_abs(block, I, V, m, h, n)
    if target == 'delta' and block in STATE_COLS:
        Y = Y - X[:, STATE_COLS[block]]
    return X, Y


def to_next_value(block, X, Y, target):
    # ブロックの出力 Y を「次の時刻の値」に直す。delta なら更新前の値 x(t) を足す
    if target == 'delta' and block in STATE_COLS:
        return Y + X[:, STATE_COLS[block]]
    return Y


def _block_data_abs(block, I, V, m, h, n):
    AB = rate_constants(V)                       # (N, 6)
    G = np.stack([m, h, n], axis=1)              # (N, 3)

    if block == 1:
        # α, β は V の同時刻の関数なので、正解も時刻 t
        X = V[:-1, None]
        Y = AB[:-1]
    elif block == 2:
        X = np.concatenate([AB[:-1], G[:-1]], axis=1)
        Y = G[1:]
    elif block == 3:
        X = np.concatenate([G[:-1], V[:-1, None], I[:-1, None]], axis=1)
        Y = V[1:, None]
    else:
        raise ValueError(f'block は 1, 2, 3 のどれか: {block}')
    return X, Y


class ColumnStandardizer:
    """列（変数）ごとに標準化する。mean, std は shape (次元,)。"""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, X):
        return cls(X.mean(axis=0), X.std(axis=0))

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


def make_chunks(X, chunk_len):
    # X: (N, d) -> (区間数, chunk_len, d)。重ならないように切り、端数は捨てる
    n_chunks = len(X) // chunk_len
    return X[:n_chunks * chunk_len].reshape(n_chunks, chunk_len, X.shape[1])


def forward_sequence(model, in_scaler, out_scaler, X):
    # 系列全体を h=0 から1本で流す（教師強制）。X: (N, d_in) -> (N, d_out)、物理単位
    y = model.forward(in_scaler.transform(X)[None])
    return out_scaler.inverse(y[0])


def block_rmse(Y_true, Y_pred, names):
    return {name: float(np.sqrt(np.mean((Y_pred[:, i] - Y_true[:, i]) ** 2)))
            for i, name in enumerate(names)}


# --- 学習 ---

def train_block(block, epochs=1000, patience=None, lr=0.001, batch_size=10,
                chunk_ms=10.0, hidden_dim=50, seed=123, data_path=DATA_PATH,
                check_grad=True, verbose_every=10, target='abs', init_path=None,
                backend='numpy', device='auto', threads=None, checkpoint_dir=None):
    # backend: 'numpy'（src/rnn の手書き BPTT）か 'torch'（同じ式の RNN を PyTorch で学習し、
    #          重みを NumPy 版に書き戻す。src/rnn/torch_backend.py）
    # init_path: 保存済みの block{n}.npz。指定すると、その重みと標準化の値から学習を続ける
    np.random.seed(seed)
    spec = BLOCKS[block]

    t_ms, I, V, m, h, n, meta = load_hh_full(data_path)
    dt = meta['dt_ms']

    # 軌道が1本なら shape (N,)、複数本（teacher_multi.npz）なら (本数, N)。
    # どちらも (本数, N) にそろえ、軌道ごとに X, Y を作る（軌道をまたいで1ステップを作らない）
    if V.ndim == 1:
        I, V, m, h, n = (a[None] for a in (I, V, m, h, n))
    n_traj = len(V)
    xy = lambda i: block_data(block, I[i], V[i], m[i], h[i], n[i], target=target)

    # out_scaler は正解 Y（delta なら変化量）の平均・標準偏差で標準化する。全軌道で1つ
    # 続きから学習するときは、元のモデルと同じ標準化を使う（変えると重みの意味がずれる）
    if init_path is not None:
        init_model, in_scaler, out_scaler, _ = load_block(init_path)
        if init_model.target != target:
            raise ValueError('init の target ({}) と --target ({}) が違う'.format(
                init_model.target, target))
    else:
        # 全軌道をつなげた配列を作らずに、軌道ごとの和・2乗和から平均と標準偏差を求める（メモリ節約）
        cnt, sx, sxx, sy, syy = 0, 0.0, 0.0, 0.0, 0.0
        for i in range(n_traj):
            X, Y = xy(i)
            cnt += len(X)
            sx, sxx = sx + X.sum(0), sxx + (X ** 2).sum(0)
            sy, syy = sy + Y.sum(0), syy + (Y ** 2).sum(0)
        std = lambda s, ss: np.sqrt(np.maximum(ss / cnt - (s / cnt) ** 2, 0.0))
        in_scaler = ColumnStandardizer(sx / cnt, std(sx, sxx))
        out_scaler = ColumnStandardizer(sy / cnt, std(sy, syy))

    # 区間は軌道ごとに切り、全軌道の区間をまとめる。torch で学習するときは float32 で持つ
    # （torch 側も float32 で計算するので精度は変わらない。メモリが半分になる）
    chunk_len = int(round(chunk_ms / dt))
    dtype = np.float32 if backend == 'torch' else np.float64
    XY = []   # 学習後の誤差の確認に使う最初の 20 本だけ残す
    x = t = None
    for i in range(n_traj):
        X, Y = xy(i)
        cx = make_chunks(in_scaler.transform(X), chunk_len)
        cy = make_chunks(out_scaler.transform(Y), chunk_len)
        if x is None:
            nc = len(cx)
            x = np.empty((n_traj * nc, chunk_len, cx.shape[2]), dtype=dtype)
            t = np.empty((n_traj * nc, chunk_len, cy.shape[2]), dtype=dtype)
        x[i * nc:(i + 1) * nc] = cx
        t[i * nc:(i + 1) * nc] = cy
        if i < 20:
            XY.append((X, Y))
    del I, V, m, h, n, xy   # 元の軌道はもう使わない

    print('RNN{}: {} -> {}  (target = {})'.format(block, spec['inputs'], spec['outputs'],
                                                 target))
    print('data = {}'.format(os.path.normpath(data_path)))
    print('dt = {} ms, chunk = {} ms ({} steps)'.format(dt, chunk_ms, chunk_len))
    print('x.shape = {}, t.shape = {}'.format(x.shape, t.shape))
    print()

    model = SimpleRNN(input_dim=x.shape[2], hidden_dim=hidden_dim,
                      output_dim=t.shape[2], init='xavier', return_sequences=True)
    prev_loss = []
    if init_path is not None:
        for name in model.param_names():
            setattr(model, name, getattr(init_model, name).copy())
        prev_loss = list(np.load(init_path)['loss'])
        print('init = {}（これまで {} エポック）'.format(os.path.normpath(init_path), len(prev_loss)))
        check_grad = False

    if check_grad and backend == 'numpy':
        numerical_gradient_check(model, x[:2, :200], t[:2, :200])

    optimizer = Adam(model, lr=lr, beta1=0.9, beta2=0.999, amsgrad=True)
    es = EarlyStopping(patience=patience) if patience is not None else None

    start = time.time()
    if backend == 'torch':
        import torch
        from rnn.torch_backend import train_chunks_torch
        if threads:
            torch.set_num_threads(threads)
        model.target = target

        def checkpoint(h):
            # 途中経過も block{n}.npz に上書き保存する（止まっても --init で続きから学習できる）
            if checkpoint_dir:
                save_block({'block': block, 'target': target, 'seed': seed, 'lr': lr,
                            'chunk_ms': chunk_ms, 'data_path': data_path,
                            'hist': {'loss': prev_loss + list(h['loss'])}, 'model': model,
                            'in_scaler': in_scaler, 'out_scaler': out_scaler}, checkpoint_dir)

        hist = train_chunks_torch(model, x, t, epochs=epochs, batch_size=batch_size,
                                  lr=lr, verbose_every=verbose_every,
                                  early_stopping=es, device=device,
                                  checkpoint_every=10, on_checkpoint=checkpoint)
    else:
        hist = train_chunks(model, optimizer, x, t, epochs=epochs,
                            batch_size=batch_size, verbose_every=verbose_every,
                            early_stopping=es)
    elapsed = time.time() - start
    hist['loss'] = prev_loss + list(hist['loss'])   # 学習曲線は前回の分からつなげる

    # 学習データの軌道を1本ずつ流したときの誤差（教師強制、物理単位）。軌道が多いときは最初の 20 本。
    # 次の時刻の値に直してから比べる（入力が正解なので、delta でも変化量の誤差と同じ値になる）
    Y_pred = np.concatenate([
        to_next_value(block, X, forward_sequence(model, in_scaler, out_scaler, X), target)
        for X, _ in XY[:20]])
    Y = np.concatenate([to_next_value(block, X, Y, target) for X, Y in XY[:20]])
    rmse = block_rmse(Y, Y_pred, spec['outputs'])
    model.target = target

    return {
        'block': block,
        'target': target,
        'seed': seed,
        'lr': lr,
        'chunk_ms': chunk_ms,
        'data_path': data_path,
        'epochs_run': len(hist['loss']),
        'elapsed': elapsed,
        'hist': hist,
        'model': model,
        'in_scaler': in_scaler,
        'out_scaler': out_scaler,
        't_ms': t_ms,
        'Y_true': Y,
        'Y_pred': Y_pred,
        'rmse': rmse,
    }


def save_block(r, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model = r['model']
    params = {name: getattr(model, name) for name in model.param_names()}
    path = os.path.join(save_dir, 'block{}.npz'.format(r['block']))
    np.savez(path,
             in_mean=r['in_scaler'].mean, in_std=r['in_scaler'].std,
             out_mean=r['out_scaler'].mean, out_std=r['out_scaler'].std,
             block=r['block'], target=r['target'], seed=r['seed'], lr=r['lr'],
             chunk_ms=r['chunk_ms'],
             data_path=r['data_path'], loss=np.array(r['hist']['loss']), **params)
    print('saved:', os.path.normpath(path))


def load_block(path):
    d = np.load(path)
    input_dim, hidden_dim = d['W_xh'].shape
    output_dim = d['W_hy'].shape[1]
    model = SimpleRNN(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, return_sequences=True)
    for name in model.param_names():
        setattr(model, name, d[name].copy())
    # target がない古いファイルは、次の時刻の値を直接出すモデル
    model.target = str(d['target']) if 'target' in d.files else 'abs'
    in_scaler = ColumnStandardizer(d['in_mean'], d['in_std'])
    out_scaler = ColumnStandardizer(d['out_mean'], d['out_std'])
    return model, in_scaler, out_scaler, str(d['data_path'])


def plot_block(r, save_dir=RESULT_DIR):
    # 教師強制の予測の図は eval で描く（block{n}_tf_train.png）。ここでは学習曲線だけ
    os.makedirs(save_dir, exist_ok=True)

    fig1 = plt.figure()
    plt.plot(r['hist']['loss'], color='black', linewidth=1, label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE, standardized)')
    plt.yscale('log')
    plt.legend()
    plt.title('RNN{} learning curve'.format(r['block']))
    fig1.savefig(os.path.join(save_dir, 'block{}_learning_curve.png'.format(r['block'])),
                 dpi=120)
    plt.close('all')


# --- 3つをつないで回す ---

def rnn_step(model, x, h):
    # 1時刻だけ進める。x: (d_in,), h: (hidden,) -> y: (d_out,), h
    h = np.tanh(x @ model.W_xh + h @ model.W_hh + model.b_h)
    return h @ model.W_hy + model.b_y, h


def run_closed_loop(blocks, I, V0=V_REST, clip=False):
    """
    3つのブロックをつなぎ、I(t) だけを与えて V を逐次計算する。

    Args:
        blocks (dict[int, tuple]): {1: (model, in_scaler, out_scaler), 2: ..., 3: ...}。
            model.target が 'delta' のブロックは変化量を出すので、更新前の値に足す。
        I (np.ndarray): 入力電流 [uA/cm^2]。shape (N,)。
        V0 (float): 膜電位の初期値 [mV]。ゲート変数はその定常値から始める（HH と同じ）。
        clip (bool): True なら出力を物理的にとりうる範囲に収める。
            α, β >= 0（速度定数は正）、0 <= m, h, n <= 1（ゲート変数は開いている割合）。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            V (N-1,): V(t_1) … V(t_{N-1})、AB (N-1, 6): 各時刻 t_k で出した α, β、
            G (N-1, 3): m, h, n(t_1) … (t_{N-1})。
    """
    (m1, in1, out1), (m2, in2, out2), (m3, in3, out3) = blocks[1], blocks[2], blocks[3]
    N = len(I)

    V = float(V0)
    g = np.array(steady_state_gates(V), dtype=float)       # (3,) = m, h, n
    h1 = np.zeros(m1.hidden_dim)
    h2 = np.zeros(m2.hidden_dim)
    h3 = np.zeros(m3.hidden_dim)

    V_out = np.empty(N - 1)
    AB_out = np.empty((N - 1, 6))
    G_out = np.empty((N - 1, 3))

    for k in range(N - 1):
        y1, h1 = rnn_step(m1, in1.transform(np.array([V])), h1)
        ab = out1.inverse(y1)
        if clip:
            ab = np.maximum(ab, 0.0)

        y2, h2 = rnn_step(m2, in2.transform(np.concatenate([ab, g])), h2)
        g_next = out2.inverse(y2)
        if m2.target == 'delta':
            g_next = g + g_next
        if clip:
            g_next = np.clip(g_next, 0.0, 1.0)

        # RNN3 には更新前の m, h, n(t) を渡す（hh.py のオイラー法と同じ順番）
        y3, h3 = rnn_step(m3, in3.transform(np.concatenate([g, [V, I[k]]])), h3)
        dV_or_V = float(out3.inverse(y3)[0])
        V = V + dV_or_V if m3.target == 'delta' else dV_or_V
        g = g_next

        V_out[k] = V
        AB_out[k] = ab
        G_out[k] = g

    return V_out, AB_out, G_out


def predict_case(blocks, t_ms, I, V, m, h, n, clip=False):
    V_pred, AB_pred, G_pred = run_closed_loop(blocks, I, V0=V[0], clip=clip)
    V_true = V[1:]
    err = V_pred - V_true
    dt = float(t_ms[1] - t_ms[0])

    # どのブロックで誤差が出るかを見るため、正解を入れたとき（教師強制）の誤差も求める
    tf_rmse = {}
    tf = {}
    for b, (model, in_s, out_s) in blocks.items():
        # 図や RMSE は「次の時刻の値」で揃える（delta のブロックも x(t) を足して直す）
        X, Y = block_data(b, I, V, m, h, n, target=model.target)
        Y_pred = to_next_value(b, X, forward_sequence(model, in_s, out_s, X), model.target)
        Y = to_next_value(b, X, Y, model.target)
        tf[b] = {'X': X, 'Y_true': Y, 'Y_pred': Y_pred}
        tf_rmse['RNN{}'.format(b)] = block_rmse(Y, Y_pred, BLOCKS[b]['outputs'])

    # 閉ループでの内部の変数。AB は時刻 t_k、G は t_{k+1} の値（run_closed_loop の返り値と同じ）
    closed_loop = {
        'AB_pred': AB_pred,
        'AB_true': rate_constants(V)[:-1],
        'G_pred': G_pred,
        'G_true': np.stack([m, h, n], axis=1)[1:],
    }

    return {
        't_ms': t_ms,
        'I': I,
        't_target': t_ms[1:],
        'V_true': V_true,
        'V_pred': V_pred,
        'rmse_mV': float(np.sqrt(np.mean(err ** 2))),
        'mae_mV': float(np.mean(np.abs(err))),
        'spikes_true': count_spikes(V_true),
        'spikes_pred': count_spikes(V_pred),
        'metrics': compare_waveforms(V_true, V_pred, dt),
        'teacher_forcing_rmse': tf_rmse,
        'teacher_forcing': tf,
        'closed_loop': closed_loop,
    }


def evaluate(blocks, data_path, clip=False):
    t_ms, I, V, m, h, n, meta = load_hh_full(data_path)
    dt = meta['dt_ms']
    cases = {'train': predict_case(blocks, t_ms, I, V, m, h, n, clip=clip)}

    t_eval = np.arange(0, EVAL_T, dt)
    for name, func, params in EVAL_CURRENTS:
        I_eval = make_eval_current(t_eval, func, params)
        V_e, m_e, h_e, n_e = simulate(t_eval, I_eval, V0=V_REST)
        cases[name] = predict_case(blocks, t_eval, I_eval, V_e, m_e, h_e, n_e, clip=clip)
    return cases


def modular_cost(blocks):
    cost = None
    for model, _, _ in blocks.values():
        c = rnn_cost(input_dim=model.input_dim, hidden_dim=model.hidden_dim,
                     output_dim=model.output_dim)
        cost = c if cost is None else cost + c
    return cost


def report(cases, blocks):
    for name, c in cases.items():
        print(format_metrics(name, c['metrics']))
        print('  教師強制のときの RMSE（ブロック単体）:')
        for b, rm in c['teacher_forcing_rmse'].items():
            print('    {}: {}'.format(b, ', '.join(
                '{} {:.4g}'.format(k, v) for k, v in rm.items())))
        print()

    print('1ステップあたりの演算回数（Euler の更新は除く。先輩と同じ数え方）')
    print(format_cost('HH', HH_COST, 'RNN x3', modular_cost(blocks)))


def save_metrics(cases, blocks, block_dir, save_dir, clip=False):
    os.makedirs(save_dir, exist_ok=True)
    out = {
        'condition': {
            'block_dir': os.path.normpath(block_dir),
            'target': {'RNN{}'.format(b): blk[0].target for b, blk in blocks.items()},
            'clip': clip,
            'hidden_dim': {'RNN{}'.format(b): blk[0].hidden_dim
                           for b, blk in blocks.items()},
        },
        'opcost': {'HH': HH_COST.to_dict(), 'RNN x3': modular_cost(blocks).to_dict()},
        'metrics': {name: c['metrics'] for name, c in cases.items()},
        'teacher_forcing_rmse': {name: c['teacher_forcing_rmse']
                                 for name, c in cases.items()},
    }
    path = os.path.join(save_dir, 'metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved:', os.path.normpath(path))


def plot_eval(cases, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig1, ax = plt.subplots(figsize=(12, 3.5))
    plot_case(ax, 'train', cases['train'])
    ax.legend(loc='upper right')
    ax.set_xlabel('Time [ms]')
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'prediction_train.png'), dpi=120)

    eval_names = [name for name in cases if name != 'train']
    fig2, axes = plt.subplots(len(eval_names), 1, figsize=(10, 2.2 * len(eval_names)),
                              sharex=True)
    for ax, name in zip(axes, eval_names):
        plot_case(ax, name, cases[name])
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Time [ms]')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'prediction_eval.png'), dpi=120)
    plt.close('all')


def tf_time(b, t_ms):
    # 教師強制の出力の時刻。RNN1 は入力と同じ時刻 t_k、RNN2・RNN3 は次の時刻 t_{k+1}
    return t_ms[:-1] if b == 1 else t_ms[1:]


def plot_blocks_tf(cases, save_dir):
    """各ブロック単体（教師強制）の図: 学習電流全体と、評価用電流ごと。"""
    eval_names = [name for name in cases if name != 'train']

    for b, spec in BLOCKS.items():
        names = spec['outputs']

        # 学習電流 900 ms 全体
        c = cases['train']
        tf = c['teacher_forcing'][b]
        t = tf_time(b, c['t_ms'])
        fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.8 * len(names)),
                                 sharex=True, squeeze=False)
        for i, (ax, name) in enumerate(zip(axes[:, 0], names)):
            ax.plot(t, tf['Y_true'][:, i], color='gray', linestyle='--', linewidth=0.8,
                    label='HH')
            ax.plot(t, tf['Y_pred'][:, i], color='black', linewidth=0.6,
                    label='RNN{}'.format(b))
            ax.set_ylabel(name)
            ax.set_xlim(t[0], t[-1])
            ax.set_title('RMSE = {:.4g}'.format(c['teacher_forcing_rmse']['RNN{}'.format(b)][name]),
                         fontsize=8)
        axes[0, 0].legend(loc='upper right')
        axes[-1, 0].set_xlabel('Time [ms]')
        fig.suptitle('RNN{} teacher forcing (train current)'.format(b))
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'block{}_tf_train.png'.format(b)), dpi=120)
        plt.close(fig)

        # 評価用電流: 行 = 電流、列 = 出力
        fig, axes = plt.subplots(len(eval_names), len(names),
                                 figsize=(max(9.0, 3.2 * len(names)), 1.9 * len(eval_names)),
                                 sharex=True, squeeze=False)
        for r, case_name in enumerate(eval_names):
            c = cases[case_name]
            tf = c['teacher_forcing'][b]
            t = tf_time(b, c['t_ms'])
            for i, name in enumerate(names):
                ax = axes[r, i]
                ax.plot(t, tf['Y_true'][:, i], color='gray', linestyle='--', linewidth=0.8)
                ax.plot(t, tf['Y_pred'][:, i], color='black', linewidth=0.6)
                ax.set_xlim(t[0], t[-1])
                ax.text(0.98, 0.95, 'RMSE {:.3g}'.format(
                            c['teacher_forcing_rmse']['RNN{}'.format(b)][name]),
                        transform=ax.transAxes, ha='right', va='top', fontsize=7)
                if r == 0:
                    ax.set_title(name, fontsize=9)
                if i == 0:
                    ax.set_ylabel(case_name, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel('Time [ms]')
        fig.suptitle('RNN{} teacher forcing (eval currents)  gray: HH, black: RNN{}'.format(b, b))
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'block{}_tf_eval.png'.format(b)), dpi=110)
        plt.close(fig)


def plot_blocks_parity(cases, save_dir, n_points=5000, seed=0):
    """
    予測と正解の対応図（教師強制）。点が多すぎるので各条件から n_points 点を間引いて描く。
      RNN1: 横軸 V、縦軸 α, β。HH の式の曲線に RNN の点が乗っているか
      RNN2, RNN3: 横軸 正解の変化量 x(t+dt) - x(t)、縦軸 RNN の変化量。
                  閉ループでは変化量の誤差が積み重なるので、値そのものではなく変化量で比べる
    """
    rng = np.random.default_rng(seed)
    eval_names = [name for name in cases if name != 'train']

    def sample(n):
        return rng.choice(n, size=min(n, n_points), replace=False)

    # RNN1
    names = BLOCKS[1]['outputs']
    V_grid = np.linspace(-100.0, 50.0, 500)
    AB_grid = rate_constants(V_grid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for i, (ax, name) in enumerate(zip(axes.ravel(), names)):
        ax.plot(V_grid, AB_grid[:, i], color='gray', linestyle='--', linewidth=1.0,
                label='HH (formula)')
        for case_name, color in [('train', 'black')] + [(n, 'tab:orange') for n in eval_names]:
            tf = cases[case_name]['teacher_forcing'][1]
            idx = sample(len(tf['X']))
            ax.scatter(tf['X'][idx, 0], tf['Y_pred'][idx, i], s=1, color=color, alpha=0.3,
                       label='RNN1 train' if case_name == 'train' else None)
        ax.scatter([], [], s=4, color='tab:orange', label='RNN1 eval')
        ax.set_xlabel('V [mV]')
        ax.set_title(name, fontsize=9)
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.suptitle('RNN1: rate constants vs V (teacher forcing)')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'block1_parity.png'), dpi=120)
    plt.close(fig)

    # RNN2, RNN3: 変化量どうしの対応
    for b, state_cols in [(2, slice(6, 9)), (3, slice(3, 4))]:
        names = BLOCKS[b]['outputs']
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), squeeze=False)
        for i, (ax, name) in enumerate(zip(axes[0], names)):
            all_d = []
            for case_name, color in [('train', 'black')] + [(n, 'tab:orange') for n in eval_names]:
                tf = cases[case_name]['teacher_forcing'][b]
                now = tf['X'][:, state_cols][:, i]
                d_true = tf['Y_true'][:, i] - now
                d_pred = tf['Y_pred'][:, i] - now
                idx = sample(len(now))
                ax.scatter(d_true[idx], d_pred[idx], s=1, color=color, alpha=0.3,
                           label='train' if case_name == 'train' else None)
                all_d.append(np.stack([d_true, d_pred], axis=1))
            # ごく一部の外れ値で軸が広がらないよう、0.1〜99.9 パーセンタイルの範囲を描く
            all_d = np.concatenate(all_d)
            lo, hi = np.percentile(all_d, [0.1, 99.9])
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            n_out = int(np.sum(np.any((all_d < lo) | (all_d > hi), axis=1)))
            ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=1.0,
                    label='y = x')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.scatter([], [], s=4, color='tab:orange', label='eval')
            ax.set_xlabel('true  {0}(t+dt) - {0}(t)'.format(name))
            ax.set_ylabel('RNN{}  {}(t+dt) - {}(t)'.format(b, name, name))
            ax.set_title('{}  ({} of {} points outside)'.format(name, n_out, len(all_d)),
                         fontsize=9)
        axes[0, 0].legend(fontsize=7, markerscale=3)
        fig.suptitle('RNN{}: one-step change, RNN vs HH (teacher forcing)'.format(b))
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'block{}_parity.png'.format(b)), dpi=120)
        plt.close(fig)


def plot_closed_loop_states(cases, save_dir):
    """3つをつないで回したときの内部の変数（V, m, h, n, α, β）を HH と比べる。"""
    # 学習電流: 全変数を縦に並べる
    c = cases['train']
    cl = c['closed_loop']
    t_ab = c['t_ms'][:-1]
    t_next = c['t_ms'][1:]
    rows = ([('V [mV]', t_next, c['V_true'], c['V_pred'])]
            + [(name, t_next, cl['G_true'][:, i], cl['G_pred'][:, i])
               for i, name in enumerate(GATE_NAMES)]
            + [(name, t_ab, cl['AB_true'][:, i], cl['AB_pred'][:, i])
               for i, name in enumerate(AB_NAMES)])
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.5 * len(rows)), sharex=True)
    for ax, (label, t, y_true, y_pred) in zip(axes, rows):
        ax.plot(t, y_true, color='gray', linestyle='--', linewidth=0.8, label='HH')
        ax.plot(t, y_pred, color='black', linewidth=0.6, label='RNN x3 (closed loop)')
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlim(t[0], t[-1])
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Closed loop internal states (train current)')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'closedloop_states_train.png'), dpi=110)
    plt.close(fig)

    # 評価用電流: 行 = 電流、列 = V, m, h, n
    eval_names = [name for name in cases if name != 'train']
    cols = ['V [mV]'] + GATE_NAMES
    fig, axes = plt.subplots(len(eval_names), len(cols),
                             figsize=(3.4 * len(cols), 1.9 * len(eval_names)),
                             sharex=True, squeeze=False)
    for r, case_name in enumerate(eval_names):
        c = cases[case_name]
        cl = c['closed_loop']
        t = c['t_ms'][1:]
        series = [(c['V_true'], c['V_pred'])] + [
            (cl['G_true'][:, i], cl['G_pred'][:, i]) for i in range(3)]
        for j, (y_true, y_pred) in enumerate(series):
            ax = axes[r, j]
            ax.plot(t, y_true, color='gray', linestyle='--', linewidth=0.8)
            ax.plot(t, y_pred, color='black', linewidth=0.6)
            ax.set_xlim(t[0], t[-1])
            if r == 0:
                ax.set_title(cols[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(case_name, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time [ms]')
    fig.suptitle('Closed loop V, m, h, n (eval currents)  gray: HH, black: RNN x3')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'closedloop_states_eval.png'), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='ブロックを1つ学習して保存する')
    p_train.add_argument('--block', type=int, required=True, choices=[1, 2, 3])
    p_train.add_argument('--epochs', type=int, default=1000)
    p_train.add_argument('--patience', type=int, default=None,
                         help='指定しなければ EarlyStopping を使わない（train loss で判定）')
    p_train.add_argument('--lr', type=float, default=0.001)
    p_train.add_argument('--batch-size', type=int, default=10, help='1バッチの区間数')
    p_train.add_argument('--chunk-ms', type=float, default=10.0,
                         help='学習で切る区間の長さ [ms]')
    p_train.add_argument('--hidden', type=int, default=50)
    p_train.add_argument('--seed', type=int, default=123)
    p_train.add_argument('--data', default=DATA_PATH)
    p_train.add_argument('--out', default=RESULT_DIR, help='block{n}.npz の保存先')
    p_train.add_argument('--target', default='abs', choices=TARGETS,
                         help='RNN2, RNN3 の正解: abs = 次の時刻の値、delta = 1ステップの変化量')
    p_train.add_argument('--backend', default='numpy', choices=['numpy', 'torch'],
                         help='numpy = 手書き BPTT、torch = PyTorch で学習（速い。保存形式は同じ）')
    p_train.add_argument('--device', default='auto',
                         help='--backend torch のとき: auto（GPU があれば GPU）, cpu, cuda')
    p_train.add_argument('--threads', type=int, default=None,
                         help='--backend torch の CPU スレッド数（複数を並列に回すときは減らす）')
    p_train.add_argument('--init', default=None,
                         help='保存済みの block{n}.npz から学習を続ける（重みと標準化を引き継ぐ）')

    p_eval = sub.add_parser('eval', help='保存した3つのブロックをつないで評価する')
    p_eval.add_argument('--dir', default=RESULT_DIR, help='block1〜3.npz があるフォルダ')
    p_eval.add_argument('--out', default=None, help='結果の保存先（既定は --dir と同じ）')
    p_eval.add_argument('--data', default=None,
                        help='学習電流の npz（既定は block の学習に使ったファイル）')
    p_eval.add_argument('--clip', action='store_true',
                        help='閉ループで α, β >= 0、0 <= m, h, n <= 1 に収める')

    args = parser.parse_args()

    if args.command == 'train':
        r = train_block(args.block, epochs=args.epochs, patience=args.patience,
                        lr=args.lr, batch_size=args.batch_size, chunk_ms=args.chunk_ms,
                        hidden_dim=args.hidden, seed=args.seed, data_path=args.data,
                        target=args.target, init_path=args.init,
                        backend=args.backend, device=args.device, threads=args.threads,
                        checkpoint_dir=args.out)
        print()
        print('epochs: {}, time: {:.0f} s'.format(r['epochs_run'], r['elapsed']))
        print('教師強制のときの RMSE（学習電流全体）:')
        for name, v in r['rmse'].items():
            print('  {:8s} {:.4g}'.format(name, v))
        save_block(r, args.out)
        plot_block(r, args.out)
    else:
        out = args.out or args.dir
        blocks = {}
        data_path = None
        for b in (1, 2, 3):
            model, in_s, out_s, data_path = load_block(
                os.path.join(args.dir, 'block{}.npz'.format(b)))
            blocks[b] = (model, in_s, out_s)

        if args.data is not None:
            data_path = args.data
        elif not os.path.exists(data_path):
            # 別のマシン（サーバー）で学習したときは絶対パスが合わないので、
            # このリポジトリの data/hh/ にある同じ名前のファイルを使う
            local = os.path.join(os.path.dirname(DATA_PATH), os.path.basename(data_path))
            print('学習時のデータが見つからないので、こちらを使う: {}'.format(
                os.path.normpath(local)))
            data_path = local

        start = time.time()
        cases = evaluate(blocks, data_path, clip=args.clip)
        print('評価にかかった時間: {:.0f} s'.format(time.time() - start))
        print()
        report(cases, blocks)
        save_metrics(cases, blocks, args.dir, out, clip=args.clip)
        plot_eval(cases, out)
        plot_blocks_tf(cases, out)
        plot_blocks_parity(cases, out)
        plot_closed_loop_states(cases, out)


if __name__ == '__main__':
    main()
