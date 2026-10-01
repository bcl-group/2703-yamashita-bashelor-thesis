import json
import os

import numpy as np
import matplotlib.pyplot as plt

from rnn import (
    SimpleRNN,
    Adam,
    train,
    EarlyStopping,
    numerical_gradient_check,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'HH', 'data', 'teacher_thesis.npz')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'hh')

WINDOW_MS = 20.0
# 正解 V の時刻で分ける（訓練 / 検証 / テスト = 60 / 20 / 20）
SPLIT_MS = (540.0, 720.0)


def load_hh(path=DATA_PATH):
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    return d['t'], d['I_ext'], d['V'], meta


def create_hh_dataset(I, V, maxlen):
    # I[i : i+maxlen] → V[i+maxlen]
    x = []
    t = []

    for i in range(len(I) - maxlen):
        x.append(I[i:i + maxlen])
        t.append(V[i + maxlen])

    x = np.array(x).reshape(-1, maxlen, 1)
    t = np.array(t).reshape(-1, 1)

    return x, t


class Standardizer:

    def __init__(self, data):
        self.mean = float(np.mean(data))
        self.std = float(np.std(data))

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


def predict(model, x, batch_size=1000):
    # 窓が多いので、まとめて forward するとメモリが足りなくなる
    y = [model.forward(x[s:s + batch_size]) for s in range(0, len(x), batch_size)]
    return np.concatenate(y, axis=0)


def count_spikes(V):
    return int(np.sum((V[:-1] < 0.0) & (V[1:] >= 0.0)))


def run_experiment(seed=123, epochs=1000, patience=10, check_grad=True,
                   verbose_every=1):
    np.random.seed(seed)

    t_ms, I, V, meta = load_hh()
    dt = meta['dt_ms']
    maxlen = int(round(WINDOW_MS / dt))

    # 標準化の平均・標準偏差は訓練区間だけから求める
    train_mask = t_ms < SPLIT_MS[0]
    I_scaler = Standardizer(I[train_mask])
    V_scaler = Standardizer(V[train_mask])

    x, t = create_hh_dataset(I_scaler.transform(I), V_scaler.transform(V),
                             maxlen)
    t_target = t_ms[maxlen:]

    idx_train = t_target < SPLIT_MS[0]
    idx_val = (t_target >= SPLIT_MS[0]) & (t_target < SPLIT_MS[1])
    idx_test = t_target >= SPLIT_MS[1]

    x_train, t_train = x[idx_train], t[idx_train]
    x_val, t_val = x[idx_val], t[idx_val]

    print('dt = {} ms, maxlen = {} ({} ms)'.format(dt, maxlen, WINDOW_MS))
    print('x.shape = {}, t.shape = {}'.format(x.shape, t.shape))
    print('train: {}, val: {}, test: {}'.format(
        idx_train.sum(), idx_val.sum(), idx_test.sum()))
    print('I: mean = {:.3f}, std = {:.3f} / V: mean = {:.3f}, std = {:.3f}'.format(
        I_scaler.mean, I_scaler.std, V_scaler.mean, V_scaler.std))
    print()

    model = SimpleRNN(input_dim=1, hidden_dim=50, output_dim=1, init='xavier')

    if check_grad:
        numerical_gradient_check(model, x_train[:8], t_train[:8])

    optimizer = Adam(model, lr=0.001, beta1=0.9, beta2=0.999, amsgrad=True)

    es = EarlyStopping(patience=patience) if patience is not None else None

    hist = train(model, optimizer, x_train, t_train, x_val, t_val,
                 epochs=epochs, batch_size=100, verbose_every=verbose_every,
                 early_stopping=es)

    V_pred = V_scaler.inverse(predict(model, x)).ravel()
    V_true = V[maxlen:]

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
        'epochs_run': len(hist['loss']),
        'hist': hist,
        'model': model,
        'scalers': (I_scaler, V_scaler),
        'scores': scores,
        't_ms': t_ms,
        'I': I,
        't_target': t_target,
        'V_true': V_true,
        'V_pred': V_pred,
        'maxlen': maxlen,
    }


def save_model(r, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model = r['model']
    I_scaler, V_scaler = r['scalers']
    params = {name: getattr(model, name) for name in model.param_names()}
    np.savez(os.path.join(save_dir, 'model.npz'),
             I_mean=I_scaler.mean, I_std=I_scaler.std,
             V_mean=V_scaler.mean, V_std=V_scaler.std,
             maxlen=r['maxlen'], seed=r['seed'], **params)


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
    r = run_experiment(seed=123, epochs=1000, patience=10, check_grad=True,
                       verbose_every=1)
    print()
    print('epochs: {}'.format(r['epochs_run']))
    for name, s in r['scores'].items():
        print('{:5s}: RMSE = {:6.2f} mV, MAE = {:6.2f} mV, spikes HH/RNN = {}/{}'.format(
            name, s['rmse_mV'], s['mae_mV'], s['spikes_true'], s['spikes_pred']))

    save_model(r)
    plot_results(r)


if __name__ == '__main__':
    main()
