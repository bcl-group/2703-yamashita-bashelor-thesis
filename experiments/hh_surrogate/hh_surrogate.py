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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'hh', 'teacher_thesis.npz')
RESULT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'hh_surrogate')

# 正解 V の時刻で分ける（訓練 / 検証 / テスト = 60 / 20 / 20）
SPLIT_MS = (540.0, 720.0)


def load_hh(path=DATA_PATH):
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    return d['t'], d['I_ext'], d['V'], meta


def create_hh_sequence(I, V):
    # I(t) → V(t+dt) を全時刻で1点ずつ対応させる。shape は (1, 系列長, 1)
    x = I[:-1].reshape(1, -1, 1)
    t = V[1:].reshape(1, -1, 1)
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


def train_sequence(model, optimizer, x, t, n_train, idx_val,
                   epochs=1000, verbose_every=10, early_stopping=None):
    # 訓練区間（先頭 n_train 点）を h=0 から流し、全体で1回 BPTT して1回更新する。
    # 検証は 0 から系列全体を流して、検証区間の誤差を測る（h を途切れさせないため）
    x_train = x[:, :n_train]
    t_train = t[:, :n_train]
    hist = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        y = model.forward(x_train)
        loss = mean_squared_error(t_train, y)
        grads = model.backward(t_train)
        optimizer.step(model, grads)

        y_all = model.forward(x)
        val_loss = mean_squared_error(t[:, idx_val], y_all[:, idx_val])

        hist['loss'].append(loss)
        hist['val_loss'].append(val_loss)

        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print('epoch: {:4d}, loss: {:.6f}, val_loss: {:.6f}'.format(
                epoch + 1, loss, val_loss))

        if early_stopping is not None and early_stopping(val_loss):
            if verbose_every:
                print('epoch: {:4d}, loss: {:.6f}, val_loss: {:.6f}'.format(
                    epoch + 1, loss, val_loss))
            break

    return hist


def run_experiment(seed=123, epochs=1000, patience=None, lr=0.001,
                   check_grad=True, verbose_every=10):
    np.random.seed(seed)

    t_ms, I, V, meta = load_hh()

    # 標準化の平均・標準偏差は訓練区間だけから求める
    train_mask = t_ms < SPLIT_MS[0]
    I_scaler = Standardizer(I[train_mask])
    V_scaler = Standardizer(V[train_mask])

    x, t = create_hh_sequence(I_scaler.transform(I), V_scaler.transform(V))
    t_target = t_ms[1:]

    idx_train = t_target < SPLIT_MS[0]
    idx_val = (t_target >= SPLIT_MS[0]) & (t_target < SPLIT_MS[1])
    idx_test = t_target >= SPLIT_MS[1]
    n_train = int(idx_train.sum())

    print('dt = {} ms'.format(meta['dt_ms']))
    print('x.shape = {}, t.shape = {}'.format(x.shape, t.shape))
    print('train: {}, val: {}, test: {}'.format(
        n_train, idx_val.sum(), idx_test.sum()))
    print('I: mean = {:.3f}, std = {:.3f} / V: mean = {:.3f}, std = {:.3f}'.format(
        I_scaler.mean, I_scaler.std, V_scaler.mean, V_scaler.std))
    print()

    model = SimpleRNN(input_dim=1, hidden_dim=50, output_dim=1, init='xavier',
                      return_sequences=True)

    if check_grad:
        # 系列全体だと数値微分が重いので、先頭 200 点で検算する
        numerical_gradient_check(model, x[:, :200], t[:, :200])

    optimizer = Adam(model, lr=lr, beta1=0.9, beta2=0.999, amsgrad=True)

    es = EarlyStopping(patience=patience) if patience is not None else None

    start = time.time()
    hist = train_sequence(model, optimizer, x, t, n_train, idx_val,
                          epochs=epochs, verbose_every=verbose_every,
                          early_stopping=es)
    elapsed = time.time() - start

    V_pred = V_scaler.inverse(model.forward(x)).ravel()
    V_true = V[1:]

    scores = {}
    for name, idx in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        err = V_pred[idx] - V_true[idx]
        scores[name] = {
            'rmse_mV': float(np.sqrt(np.mean(err ** 2))),
            'mae_mV': float(np.mean(np.abs(err))),
            'spikes_true': count_spikes(V_true[idx]),
            'spikes_pred': count_spikes(V_pred[idx]),
        }

    return {
        'seed': seed,
        'lr': lr,
        'epochs_run': len(hist['loss']),
        'elapsed': elapsed,
        'hist': hist,
        'model': model,
        'scalers': (I_scaler, V_scaler),
        'scores': scores,
        't_ms': t_ms,
        'I': I,
        't_target': t_target,
        'V_true': V_true,
        'V_pred': V_pred,
    }


def save_model(r, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model = r['model']
    I_scaler, V_scaler = r['scalers']
    params = {name: getattr(model, name) for name in model.param_names()}
    np.savez(os.path.join(save_dir, 'model.npz'),
             I_mean=I_scaler.mean, I_std=I_scaler.std,
             V_mean=V_scaler.mean, V_std=V_scaler.std,
             seed=r['seed'], lr=r['lr'],
             loss=np.array(r['hist']['loss']),
             val_loss=np.array(r['hist']['val_loss']), **params)


def plot_results(r, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    hist = r['hist']

    fig1 = plt.figure()
    plt.plot(hist['loss'], color='black', linewidth=1, label='train loss')
    plt.plot(hist['val_loss'], color='gray', linewidth=1, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE, standardized)')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning curve')
    fig1.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=120)

    fig2, axes = plt.subplots(3, 1, figsize=(12, 8),
                              gridspec_kw={'height_ratios': [1, 2, 2]})

    axes[0].plot(r['t_ms'], r['I'], color='red', linewidth=0.8)
    axes[0].set_ylabel(r'$I_{ext}$ [$\mu$A/cm$^2$]')

    for ax in axes[:2]:
        ax.axvspan(SPLIT_MS[0], SPLIT_MS[1], color='tab:blue', alpha=0.08)
        ax.axvspan(SPLIT_MS[1], r['t_ms'][-1], color='tab:orange', alpha=0.08)
        ax.set_xlim(0, r['t_ms'][-1])

    axes[1].plot(r['t_target'], r['V_true'], color='gray', linestyle='--',
                 linewidth=0.8, label='HH')
    axes[1].plot(r['t_target'], r['V_pred'], color='black', linewidth=0.8,
                 label='RNN')
    axes[1].set_ylabel('V [mV]')
    axes[1].legend(loc='upper right')
    axes[1].set_title('train | val (blue) | test (orange)', fontsize=9)

    test = r['t_target'] >= SPLIT_MS[1]
    axes[2].plot(r['t_target'][test], r['V_true'][test], color='gray',
                 linestyle='--', linewidth=0.8, label='HH')
    axes[2].plot(r['t_target'][test], r['V_pred'][test], color='black',
                 linewidth=0.8, label='RNN')
    axes[2].set_xlabel('Time [ms]')
    axes[2].set_ylabel('V [mV]')
    axes[2].legend(loc='upper right')
    axes[2].set_title('test region', fontsize=9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'prediction.png'), dpi=120)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=None,
                        help='指定しなければ EarlyStopping を使わない')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out', default=RESULT_DIR, help='結果の保存先')
    args = parser.parse_args()

    r = run_experiment(seed=args.seed, epochs=args.epochs,
                       patience=args.patience, lr=args.lr,
                       check_grad=True, verbose_every=10)
    print()
    print('epochs: {}, time: {:.0f} s'.format(r['epochs_run'], r['elapsed']))
    for name, s in r['scores'].items():
        print('{:5s}: RMSE = {:6.2f} mV, MAE = {:6.2f} mV, spikes HH/RNN = {}/{}'.format(
            name, s['rmse_mV'], s['mae_mV'], s['spikes_true'], s['spikes_pred']))

    save_model(r, args.out)
    plot_results(r, args.out)


if __name__ == '__main__':
    main()
