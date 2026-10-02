import os

import numpy as np
import matplotlib.pyplot as plt

from rnn import (
    SimpleRNN,
    SGD,
    Adam,
    create_dataset,
    split_in_order,
    train,
    EarlyStopping,
    numerical_gradient_check,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'sin_wave')


def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)


def toy_problem(T=100, ampl=0.05):

    x = np.arange(0, 2 * T + 1)

    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))

    return sin(x, T) + noise


def generate_sequence(model, x_seed, n_steps, maxlen):
    gen = [None for _ in range(maxlen)]

    z = x_seed.copy()

    for _ in range(n_steps):
        pred = model.forward(z)

        z = np.concatenate([z[:, 1:, :], pred[:, np.newaxis, :]], axis=1)

        gen.append(pred[0, 0])

    return gen


def gen_mse(gen, sin_true, maxlen):
    # 生成した区間（先頭 maxlen 点より後ろ）だけで、ノイズなしのsin波と比べる
    return np.mean((np.array(gen[maxlen:]) - sin_true[maxlen:]) ** 2)


def run_experiment(seed=123, init='simple', epochs=1000, patience=None,
                   check_grad=False, verbose_every=100, amsgrad_raw=True):
    np.random.seed(seed)

    T = 100
    maxlen = 25

    f = toy_problem(T)
    length_of_sequences = len(f)

    x, t = create_dataset(f, maxlen)
    x_train, x_val, t_train, t_val = split_in_order(x, t, test_size=0.2)

    model = SimpleRNN(input_dim=1, hidden_dim=50, output_dim=1, init=init)

    if check_grad:
        numerical_gradient_check(model, x_train[:8], t_train[:8])

    optimizer = Adam(model, lr=0.001, beta1=0.9, beta2=0.999, amsgrad=True,
                     amsgrad_raw=amsgrad_raw)

    # patience=None のときは EarlyStopping を使わず epochs 回し切る
    es = EarlyStopping(patience=patience) if patience is not None else None

    hist = train(model, optimizer, x_train, t_train, x_val, t_val,
                 epochs=epochs, batch_size=100, verbose_every=verbose_every,
                 early_stopping=es)

    sin_true = toy_problem(T, ampl=0.0)
    gen = generate_sequence(model, x[:1], length_of_sequences - maxlen, maxlen)

    return {
        'seed': seed,
        'epochs_run': len(hist['loss']),
        'val_loss': hist['val_loss'][-1],
        'gen_mse': gen_mse(gen, sin_true, maxlen),
        'hist': hist,
        'f': f,
        'sin_true': sin_true,
        'gen': gen,
        'T': T,
        'maxlen': maxlen,
        'n_train': len(x_train),
    }


def plot_data_split(f, sin_true, maxlen, n_train, save_dir=RESULT_DIR):
    # 学習に使ったノイズ入りsin波を、訓練／検証で色分けして描く
    #   訓練の問題 i（0〜n_train-1）は f[i]〜f[i+maxlen-1] を見て f[i+maxlen] を当てる
    #   → 訓練で使う点は f[0]〜f[n_train-1+maxlen]
    #   検証の問題は f[n_train]〜f[200]（答えは f[n_train+maxlen]〜f[200]）
    os.makedirs(save_dir, exist_ok=True)

    time = np.arange(len(f))
    train_end = n_train - 1 + maxlen            # 訓練で使う最後の点（=164）

    fig = plt.figure(figsize=(10, 4))
    plt.plot(time[:train_end + 1], f[:train_end + 1],
             color='tab:blue', linewidth=1, marker='o', markersize=2,
             label='train (t=0-{})'.format(train_end))
    plt.plot(time[train_end:], f[train_end:],
             color='tab:orange', linewidth=1, marker='o', markersize=2,
             label='val (t={}-{})'.format(train_end + 1, len(f) - 1))
    plt.xlim([0, len(f) - 1])
    plt.ylim([-1.5, 1.5])
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'data_split.png'), dpi=120)


def plot_results(hist, f, sin_true, gen, T, save_dir=RESULT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    fig1 = plt.figure()
    plt.plot(hist['loss'], color='black', linewidth=1, label='train loss')
    plt.plot(hist['val_loss'], color='gray', linewidth=1, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning curve')
    fig1.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=120)

    fig2 = plt.figure()
    plt.rc('font', family='serif')
    plt.xlim([0, 2 * T])
    plt.ylim([-1.5, 1.5])
    plt.plot(range(len(f)), sin_true,
             color='gray', linestyle='--', linewidth=0.5,
             label='true sin')
    plt.plot(range(len(f)), gen,
             color='black', linewidth=1,
             marker='o', markersize=1,
             markerfacecolor='black', markeredgecolor='black',
             label='predicted')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.title('Sin wave prediction by NumPy RNN')
    fig2.savefig(os.path.join(save_dir, 'prediction.png'), dpi=120)

    plt.show()


def main():
    # 教科書（資料 p.270）と同じく EarlyStopping(patience=10) を使う
    r = run_experiment(seed=123, init='xavier', epochs=1000, patience=10,
                       check_grad=True, verbose_every=10)
    print()
    print('epochs: {}, val_loss: {:.6f}, gen_mse: {:.4f}'.format(
        r['epochs_run'], r['val_loss'], r['gen_mse']))

    plot_data_split(r['f'], r['sin_true'], r['maxlen'], r['n_train'])
    plot_results(r['hist'], r['f'], r['sin_true'], r['gen'], r['T'])


if __name__ == '__main__':
    main()
