import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# src/（neuron, rnn）を import できるようにする
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from rnn import (
    SimpleRNN,
    Adam,
    mean_squared_error,
    EarlyStopping,
    numerical_gradient_check,
)
from neuron.hh import simulate, V_REST
from neuron.current import (
    constant_current,
    ramp_current,
    sin_current,
    pulse_train_current,
)
from evaluation import compare_waveforms, format_metrics, HH_COST, rnn_cost, format_cost


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'hh', 'teacher_thesis.npz')
RESULT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'hh_surrogate')

# 評価用の電流（棟近先輩に倣う）
#   定常 5, 10, 15 : 卒論 p.17 §4.4.1
#   ramp / sin / pulse の値、長さ 300 ms、前後 10 ms の無入力:
#     SINDyNeuroSurrogate の _current_catalog.py の既定値と scripts/catalog.py の _STIM
EVAL_T = 300.0
EVAL_SILENCE = 10.0
EVAL_CURRENTS = [
    ('steady 5',   constant_current,    {'amplitude': 5.0}),
    ('steady 10',  constant_current,    {'amplitude': 10.0}),
    ('steady 15',  constant_current,    {'amplitude': 15.0}),
    ('ramp 0-30',  ramp_current,        {'i_start': 0.0, 'i_end': 30.0}),
    ('sin 10Hz',   sin_current,         {'amplitude': 7.5, 'freq': 10.0, 'offset': 7.5}),
    ('pulse 20Hz', pulse_train_current, {'amplitude': 20.0, 'width': 25.0, 'interval': 50.0}),
]


def load_hh(path=DATA_PATH):
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    return d['t'], d['I_ext'], d['V'], meta


def make_eval_current(t, func, params, silence=EVAL_SILENCE):
    I = np.zeros_like(t)
    active = (t >= t[0] + silence) & (t < t[-1] + (t[1] - t[0]) - silence)
    I[active] = func(t[active] - (t[0] + silence), **params)
    return I


def create_hh_sequence(I, V):
    # I(t) → V(t+dt) を全時刻で1点ずつ対応させた1本の系列。shape は (1, 系列長, 1)
    x = I[:-1].reshape(1, -1, 1)
    t = V[1:].reshape(1, -1, 1)
    return x, t


def create_hh_chunks(I, V, chunk_len):
    # 系列を重ならない区間に切る（端数は捨てる）。各区間は学習時に h=0 から始める
    # x: (区間数, chunk_len, 1)  t: (区間数, chunk_len, 1)
    x, t = create_hh_sequence(I, V)
    n_chunks = x.shape[1] // chunk_len
    n = n_chunks * chunk_len
    x = x[0, :n].reshape(n_chunks, chunk_len, 1)
    t = t[0, :n].reshape(n_chunks, chunk_len, 1)
    return x, t


class Standardizer:

    def __init__(self, data):
        self.mean = float(np.mean(data))
        self.std = float(np.std(data))

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


def count_spikes(V):
    return int(np.sum((V[:-1] < 0.0) & (V[1:] >= 0.0)))


def train_chunks(model, optimizer, x, t, epochs=1000, batch_size=10,
                 verbose_every=10, early_stopping=None):
    # 区間をシャッフルしてミニバッチにする（教科書の sin 波と同じ流れ）。
    # 1つのミニバッチ = batch_size 個の区間を h=0 から流し、区間の中だけ BPTT する
    n = len(x)
    hist = {'loss': []}

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            y = model.forward(x[idx])
            loss += mean_squared_error(t[idx], y)
            grads = model.backward(t[idx])
            optimizer.step(model, grads)
            n_batches += 1

        loss /= n_batches
        hist['loss'].append(loss)

        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print('epoch: {:4d}, loss: {:.6f}'.format(epoch + 1, loss))

        if early_stopping is not None and early_stopping(loss):
            if verbose_every:
                print('epoch: {:4d}, loss: {:.6f}'.format(epoch + 1, loss))
            break

    return hist


def predict(model, scalers, t_ms, I, V):
    # 区切らず、h=0 から電流全体を1本で流す
    I_scaler, V_scaler = scalers
    x, _ = create_hh_sequence(I_scaler.transform(I), V_scaler.transform(V))
    V_pred = V_scaler.inverse(model.forward(x)).ravel()
    V_true = V[1:]
    err = V_pred - V_true
    dt = float(t_ms[1] - t_ms[0])
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
        # 棟近先輩（SINDyNeuroSurrogate）と同じ指標一式
        'metrics': compare_waveforms(V_true, V_pred, dt),
    }


def evaluate(model, scalers, t_ms, I, V, dt):
    # 学習電流（train）と評価用の電流すべてで HH と RNN の波形を比べる
    cases = {'train': predict(model, scalers, t_ms, I, V)}

    t_eval = np.arange(0, EVAL_T, dt)
    for name, func, params in EVAL_CURRENTS:
        I_eval = make_eval_current(t_eval, func, params)
        V_eval, _, _, _ = simulate(t_eval, I_eval, V0=V_REST)
        cases[name] = predict(model, scalers, t_eval, I_eval, V_eval)
    return cases


def run_experiment(seed=123, epochs=1000, patience=None, lr=0.001, batch_size=10,
                   chunk_ms=10.0, check_grad=True, verbose_every=10,
                   data_path=DATA_PATH):
    np.random.seed(seed)

    t_ms, I, V, meta = load_hh(data_path)
    dt = meta['dt_ms']

    # 標準化の平均・標準偏差は学習電流の全区間から求める
    I_scaler = Standardizer(I)
    V_scaler = Standardizer(V)

    chunk_len = int(round(chunk_ms / dt))
    x, t = create_hh_chunks(I_scaler.transform(I), V_scaler.transform(V), chunk_len)

    print('data = {}'.format(os.path.normpath(data_path)))
    print('dt = {} ms, T = {:.0f} ms, chunk = {} ms ({} steps)'.format(
        dt, t_ms[-1] + dt, chunk_ms, chunk_len))
    print('x.shape = {}, t.shape = {}'.format(x.shape, t.shape))
    print('I: mean = {:.3f}, std = {:.3f} / V: mean = {:.3f}, std = {:.3f}'.format(
        I_scaler.mean, I_scaler.std, V_scaler.mean, V_scaler.std))
    print()

    # 入力は1時刻ずつ。隠れ状態 h を時刻をまたいで持ち越し、各時刻で V(t+dt) を出す
    model = SimpleRNN(input_dim=1, hidden_dim=50, output_dim=1, init='xavier',
                      return_sequences=True)

    if check_grad:
        # 区間全体だと数値微分が重いので、2区間の先頭 200 点で検算する
        numerical_gradient_check(model, x[:2, :200], t[:2, :200])

    optimizer = Adam(model, lr=lr, beta1=0.9, beta2=0.999, amsgrad=True)

    es = EarlyStopping(patience=patience) if patience is not None else None

    start = time.time()
    hist = train_chunks(model, optimizer, x, t, epochs=epochs,
                        batch_size=batch_size, verbose_every=verbose_every,
                        early_stopping=es)
    elapsed = time.time() - start

    scalers = (I_scaler, V_scaler)
    cases = evaluate(model, scalers, t_ms, I, V, dt)

    return {
        'seed': seed,
        'lr': lr,
        'data_path': data_path,
        'epochs_run': len(hist['loss']),
        'elapsed': elapsed,
        'hist': hist,
        'model': model,
        'scalers': scalers,
        'cases': cases,
    }


def save_model(r, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model = r['model']
    I_scaler, V_scaler = r['scalers']
    params = {name: getattr(model, name) for name in model.param_names()}
    np.savez(os.path.join(save_dir, 'model.npz'),
             I_mean=I_scaler.mean, I_std=I_scaler.std,
             V_mean=V_scaler.mean, V_std=V_scaler.std,
             seed=r['seed'], lr=r['lr'], data_path=r['data_path'],
             loss=np.array(r['hist']['loss']), **params)


def load_model(path):
    # save_model で保存した model.npz から、学習済みの RNN と標準化の値を読み込む
    d = np.load(path)
    input_dim, hidden_dim = d['W_xh'].shape
    output_dim = d['W_hy'].shape[1]
    model = SimpleRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                      return_sequences=True)
    for name in model.param_names():
        setattr(model, name, d[name].copy())

    I_scaler = Standardizer(np.zeros(1))
    I_scaler.mean, I_scaler.std = float(d['I_mean']), float(d['I_std'])
    V_scaler = Standardizer(np.zeros(1))
    V_scaler.mean, V_scaler.std = float(d['V_mean']), float(d['V_std'])

    info = {'seed': int(d['seed']), 'lr': float(d['lr']), 'data_path': str(d['data_path']),
            'epochs_run': len(d['loss']), 'elapsed': float('nan'),
            'hist': {'loss': list(d['loss'])}}
    return model, (I_scaler, V_scaler), info


def report(r):
    # 先輩と同じ指標を、評価条件ごとに表で表示する
    for name, c in r['cases'].items():
        print(format_metrics(name, c['metrics']))
        print()

    model = r['model']
    cost = rnn_cost(input_dim=model.input_dim, hidden_dim=model.hidden_dim,
                    output_dim=model.output_dim)
    print('1ステップあたりの演算回数（Euler の更新は除く。先輩と同じ数え方）')
    print(format_cost('HH', HH_COST, 'RNN', cost))


def save_metrics(r, save_dir=RESULT_DIR):
    # 条件（seed, lr, 学習データ, 隠れ層の次元）と指標をまとめて JSON に保存する
    os.makedirs(save_dir, exist_ok=True)
    model = r['model']
    out = {
        'condition': {
            'seed': r['seed'],
            'lr': r['lr'],
            'data_path': os.path.normpath(r['data_path']),
            'epochs_run': r['epochs_run'],
            'hidden_dim': model.hidden_dim,
        },
        'opcost': {
            'HH': HH_COST.to_dict(),
            'RNN': rnn_cost(input_dim=model.input_dim, hidden_dim=model.hidden_dim,
                            output_dim=model.output_dim).to_dict(),
        },
        'metrics': {name: c['metrics'] for name, c in r['cases'].items()},
    }
    path = os.path.join(save_dir, 'metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved:', os.path.normpath(path))


def plot_case(ax, name, c):
    ax.plot(c['t_target'], c['V_true'], color='gray', linestyle='--',
            linewidth=0.8, label='HH')
    ax.plot(c['t_target'], c['V_pred'], color='black', linewidth=0.8, label='RNN')
    ax.set_ylabel('V [mV]')
    ax.set_xlim(c['t_ms'][0], c['t_ms'][-1])
    ax.set_title('{}  (RMSE = {:.1f} mV, spikes HH/RNN = {}/{})'.format(
        name, c['rmse_mV'], c['spikes_true'], c['spikes_pred']), fontsize=9)

    ax_i = ax.twinx()
    ax_i.plot(c['t_ms'], c['I'], color='red', linewidth=0.6, alpha=0.4)
    ax_i.set_ylabel(r'$I_{ext}$ [$\mu$A/cm$^2$]', color='red', fontsize=8)


def plot_results(r, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    hist = r['hist']

    fig1 = plt.figure()
    plt.plot(hist['loss'], color='black', linewidth=1, label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE, standardized)')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning curve')
    fig1.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=120)

    fig2, ax = plt.subplots(figsize=(12, 3.5))
    plot_case(ax, 'train', r['cases']['train'])
    ax.legend(loc='upper right')
    ax.set_xlabel('Time [ms]')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'prediction_train.png'), dpi=120)

    eval_names = [name for name in r['cases'] if name != 'train']
    fig3, axes = plt.subplots(len(eval_names), 1, figsize=(10, 2.2 * len(eval_names)),
                              sharex=True)
    for ax, name in zip(axes, eval_names):
        plot_case(ax, name, r['cases'][name])
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Time [ms]')
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'prediction_eval.png'), dpi=120)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=None,
                        help='指定しなければ EarlyStopping を使わない（train loss で判定）')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=10, help='1バッチの区間数')
    parser.add_argument('--chunk-ms', type=float, default=10.0, help='学習で切る区間の長さ [ms]')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out', default=RESULT_DIR, help='結果の保存先')
    parser.add_argument('--data', default=DATA_PATH,
                        help='学習に使う npz（例: data/hh/munechika_train.npz）')
    parser.add_argument('--model', default=None,
                        help='学習済みの model.npz を指定すると、学習せずに評価だけを行う')
    args = parser.parse_args()

    if args.model is not None:
        model, scalers, r = load_model(args.model)
        t_ms, I, V, meta = load_hh(r['data_path'])
        r['model'] = model
        r['scalers'] = scalers
        r['cases'] = evaluate(model, scalers, t_ms, I, V, meta['dt_ms'])
    else:
        r = run_experiment(seed=args.seed, epochs=args.epochs,
                           patience=args.patience, lr=args.lr,
                           batch_size=args.batch_size, chunk_ms=args.chunk_ms,
                           check_grad=True, verbose_every=10, data_path=args.data)
        print()
        print('epochs: {}, time: {:.0f} s'.format(r['epochs_run'], r['elapsed']))
        save_model(r, args.out)
    print()

    report(r)
    save_metrics(r, args.out)
    plot_results(r, args.out)


if __name__ == '__main__':
    main()
