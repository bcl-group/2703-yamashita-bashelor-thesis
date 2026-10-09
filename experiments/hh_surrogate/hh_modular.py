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
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    return d['t'], d['I_ext'], d['V'], d['m'], d['h'], d['n'], meta


def rate_constants(V):
    # V: (N,) -> (N, 6)。列の順は AB_NAMES
    return np.stack([alpha_m(V), beta_m(V), alpha_h(V), beta_h(V),
                     alpha_n(V), beta_n(V)], axis=1)


def block_data(block, I, V, m, h, n):
    """
    ブロックの入力 X(t) と正解 Y を、物理単位のまま作る。

    Args:
        block (int): 1, 2, 3。
        I, V, m, h, n (np.ndarray): shape (N,)。

    Returns:
        tuple[np.ndarray, np.ndarray]: X (N-1, 入力次元), Y (N-1, 出力次元)。
            k 番目の行が時刻 t_k の入力と、それに対応する正解。
    """
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
                check_grad=True, verbose_every=10):
    np.random.seed(seed)
    spec = BLOCKS[block]

    t_ms, I, V, m, h, n, meta = load_hh_full(data_path)
    dt = meta['dt_ms']

    X, Y = block_data(block, I, V, m, h, n)
    in_scaler = ColumnStandardizer.fit(X)
    out_scaler = ColumnStandardizer.fit(Y)

    chunk_len = int(round(chunk_ms / dt))
    x = make_chunks(in_scaler.transform(X), chunk_len)
    t = make_chunks(out_scaler.transform(Y), chunk_len)

    print('RNN{}: {} -> {}'.format(block, spec['inputs'], spec['outputs']))
    print('data = {}'.format(os.path.normpath(data_path)))
    print('dt = {} ms, chunk = {} ms ({} steps)'.format(dt, chunk_ms, chunk_len))
    print('x.shape = {}, t.shape = {}'.format(x.shape, t.shape))
    print()

    model = SimpleRNN(input_dim=X.shape[1], hidden_dim=hidden_dim,
                      output_dim=Y.shape[1], init='xavier', return_sequences=True)

    if check_grad:
        numerical_gradient_check(model, x[:2, :200], t[:2, :200])

    optimizer = Adam(model, lr=lr, beta1=0.9, beta2=0.999, amsgrad=True)
    es = EarlyStopping(patience=patience) if patience is not None else None

    start = time.time()
    hist = train_chunks(model, optimizer, x, t, epochs=epochs,
                        batch_size=batch_size, verbose_every=verbose_every,
                        early_stopping=es)
    elapsed = time.time() - start

    # 学習電流全体を1本で流したときの誤差（教師強制、物理単位）
    Y_pred = forward_sequence(model, in_scaler, out_scaler, X)
    rmse = block_rmse(Y, Y_pred, spec['outputs'])

    return {
        'block': block,
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
             block=r['block'], seed=r['seed'], lr=r['lr'], chunk_ms=r['chunk_ms'],
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
    in_scaler = ColumnStandardizer(d['in_mean'], d['in_std'])
    out_scaler = ColumnStandardizer(d['out_mean'], d['out_std'])
    return model, in_scaler, out_scaler, str(d['data_path'])


def plot_block(r, save_dir=RESULT_DIR, show_ms=100.0):
    os.makedirs(save_dir, exist_ok=True)
    names = BLOCKS[r['block']]['outputs']

    fig1 = plt.figure()
    plt.plot(r['hist']['loss'], color='black', linewidth=1, label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE, standardized)')
    plt.yscale('log')
    plt.legend()
    plt.title('RNN{} learning curve'.format(r['block']))
    fig1.savefig(os.path.join(save_dir, 'block{}_learning_curve.png'.format(r['block'])),
                 dpi=120)

    # 細かい形が見えるように先頭 show_ms だけ描く
    t_target = r['t_ms'][:len(r['Y_true'])]
    show = t_target < show_ms
    fig2, axes = plt.subplots(len(names), 1, figsize=(10, 1.8 * len(names)),
                              sharex=True, squeeze=False)
    for i, (ax, name) in enumerate(zip(axes[:, 0], names)):
        ax.plot(t_target[show], r['Y_true'][show, i], color='gray', linestyle='--',
                linewidth=0.8, label='HH')
        ax.plot(t_target[show], r['Y_pred'][show, i], color='black', linewidth=0.8,
                label='RNN{}'.format(r['block']))
        ax.set_ylabel(name)
        ax.set_title('RMSE = {:.4g}'.format(r['rmse'][name]), fontsize=8)
    axes[0, 0].legend(loc='upper right')
    axes[-1, 0].set_xlabel('Time [ms]')
    fig2.suptitle('RNN{} (teacher forcing, first {:.0f} ms)'.format(r['block'], show_ms))
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'block{}_prediction.png'.format(r['block'])),
                 dpi=120)
    plt.close('all')


# --- 3つをつないで回す ---

def rnn_step(model, x, h):
    # 1時刻だけ進める。x: (d_in,), h: (hidden,) -> y: (d_out,), h
    h = np.tanh(x @ model.W_xh + h @ model.W_hh + model.b_h)
    return h @ model.W_hy + model.b_y, h


def run_closed_loop(blocks, I, V0=V_REST):
    """
    3つのブロックをつなぎ、I(t) だけを与えて V を逐次計算する。

    Args:
        blocks (dict[int, tuple]): {1: (model, in_scaler, out_scaler), 2: ..., 3: ...}。
        I (np.ndarray): 入力電流 [uA/cm^2]。shape (N,)。
        V0 (float): 膜電位の初期値 [mV]。ゲート変数はその定常値から始める（HH と同じ）。

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

        y2, h2 = rnn_step(m2, in2.transform(np.concatenate([ab, g])), h2)
        g_next = out2.inverse(y2)

        # RNN3 には更新前の m, h, n(t) を渡す（hh.py のオイラー法と同じ順番）
        y3, h3 = rnn_step(m3, in3.transform(np.concatenate([g, [V, I[k]]])), h3)
        V = float(out3.inverse(y3)[0])
        g = g_next

        V_out[k] = V
        AB_out[k] = ab
        G_out[k] = g

    return V_out, AB_out, G_out


def predict_case(blocks, t_ms, I, V, m, h, n):
    V_pred, _, _ = run_closed_loop(blocks, I, V0=V[0])
    V_true = V[1:]
    err = V_pred - V_true
    dt = float(t_ms[1] - t_ms[0])

    # どのブロックで誤差が出るかを見るため、正解を入れたとき（教師強制）の誤差も求める
    tf_rmse = {}
    for b, (model, in_s, out_s) in blocks.items():
        X, Y = block_data(b, I, V, m, h, n)
        tf_rmse['RNN{}'.format(b)] = block_rmse(
            Y, forward_sequence(model, in_s, out_s, X), BLOCKS[b]['outputs'])

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
    }


def evaluate(blocks, data_path):
    t_ms, I, V, m, h, n, meta = load_hh_full(data_path)
    dt = meta['dt_ms']
    cases = {'train': predict_case(blocks, t_ms, I, V, m, h, n)}

    t_eval = np.arange(0, EVAL_T, dt)
    for name, func, params in EVAL_CURRENTS:
        I_eval = make_eval_current(t_eval, func, params)
        V_e, m_e, h_e, n_e = simulate(t_eval, I_eval, V0=V_REST)
        cases[name] = predict_case(blocks, t_eval, I_eval, V_e, m_e, h_e, n_e)
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


def save_metrics(cases, blocks, block_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    out = {
        'condition': {
            'block_dir': os.path.normpath(block_dir),
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

    p_eval = sub.add_parser('eval', help='保存した3つのブロックをつないで評価する')
    p_eval.add_argument('--dir', default=RESULT_DIR, help='block1〜3.npz があるフォルダ')
    p_eval.add_argument('--out', default=None, help='結果の保存先（既定は --dir と同じ）')

    args = parser.parse_args()

    if args.command == 'train':
        r = train_block(args.block, epochs=args.epochs, patience=args.patience,
                        lr=args.lr, batch_size=args.batch_size, chunk_ms=args.chunk_ms,
                        hidden_dim=args.hidden, seed=args.seed, data_path=args.data)
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

        start = time.time()
        cases = evaluate(blocks, data_path)
        print('評価にかかった時間: {:.0f} s'.format(time.time() - start))
        print()
        report(cases, blocks)
        save_metrics(cases, blocks, args.dir, out)
        plot_eval(cases, out)


if __name__ == '__main__':
    main()
