"""
教科書（GitHub yusugomori/deeplearning-keras-tf2-torch/5）の
01_sin_rnn_keras.py と 03_sin_rnn_torch.py を、ライブラリを使ったまま複数シードで回し、
NumPy版（sin_wave_eval.py）と同じ gen_mse で比べる。

教科書のコードからの変更点は次だけ:
  - seed を引数にした（教科書は 123 固定）
  - EarlyStopping を使うかどうかを引数にした
  - print・描画をやめ、gen_mse を返すようにした
  - Keras の fit / predict に verbose=0 を付けた（表示を消すだけ）
"""

import os
import itertools
from multiprocessing import Pool

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

from sin_wave_eval import SEEDS


def _data(seed):
    np.random.seed(seed)

    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2*T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0,
                                         size=len(x))
        return sin(x) + noise

    T = 100
    f = toy_problem(T).astype(np.float32)
    length_of_sequences = len(f)
    maxlen = 25

    x = []
    t = []

    for i in range(length_of_sequences - maxlen):
        x.append(f[i:i+maxlen])
        t.append(f[i+maxlen])

    x = np.array(x).reshape(-1, maxlen, 1)
    t = np.array(t).reshape(-1, 1)

    # 正解のsin波は学習の後に作る（ampl=0 でも乱数を消費するので、
    # 学習前に作るとシャッフルの順番が教科書と変わってしまう）
    make_sin_true = lambda: toy_problem(T, ampl=0.)
    return f, x, t, make_sin_true, length_of_sequences, maxlen


def _gen_mse(gen, sin_true, maxlen):
    return float(np.mean((np.array(gen[maxlen:], dtype=np.float64)
                          - sin_true[maxlen:]) ** 2))


def run_torch(seed, use_es):
    import torch
    import torch.nn as nn
    import torch.optim as optimizers
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from rnn import EarlyStopping      # 教科書の callbacks/EarlyStopping.py と同一のコード

    class RNN(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.l1 = nn.RNN(1, hidden_dim,
                             nonlinearity='tanh',
                             batch_first=True)
            self.l2 = nn.Linear(hidden_dim, 1)

            nn.init.xavier_normal_(self.l1.weight_ih_l0)
            nn.init.orthogonal_(self.l1.weight_hh_l0)

        def forward(self, x):
            h, _ = self.l1(x)
            y = self.l2(h[:, -1])
            return y

    # 教科書は np.random.seed → torch.manual_seed → データ生成 の順
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')

    f, x, t, make_sin_true, length_of_sequences, maxlen = _data(seed)

    x_train, x_val, t_train, t_val = \
        train_test_split(x, t, test_size=0.2, shuffle=False)

    model = RNN(50).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)

    def train_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.train()
        preds = model(x)
        loss = criterion(preds, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, preds

    def val_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = criterion(preds, t)
        return loss, preds

    epochs = 1000
    batch_size = 100
    n_batches_train = x_train.shape[0] // batch_size + 1
    n_batches_val = x_val.shape[0] // batch_size + 1
    es = EarlyStopping(patience=10, verbose=0)

    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.
        x_, t_ = shuffle(x_train, t_train)

        for batch in range(n_batches_train):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = train_step(x_[start:end], t_[start:end])
            train_loss += loss.item()

        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = val_step(x_val[start:end], t_val[start:end])
            val_loss += loss.item()

        train_loss /= n_batches_train
        val_loss /= n_batches_val

        if use_es and es(val_loss):
            break

    model.eval()
    sin_true = make_sin_true()
    gen = [None for i in range(maxlen)]
    z = x[:1]
    for i in range(length_of_sequences - maxlen):
        z_ = torch.Tensor(z[-1:]).to(device)
        preds = model(z_).data.cpu().numpy()
        z = np.append(z, preds)[1:]
        z = z.reshape(-1, maxlen, 1)
        gen.append(preds[0, 0])

    return epoch + 1, val_loss, _gen_mse(gen, sin_true, maxlen), gen


def run_keras(seed, use_es):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(seed)
    tf.random.set_seed(seed)

    f, x, t, make_sin_true, length_of_sequences, maxlen = _data(seed)

    x_train, x_val, t_train, t_val = \
        train_test_split(x, t, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(SimpleRNN(50, activation='tanh',
                        kernel_initializer='glorot_normal',
                        recurrent_initializer='orthogonal'))
    model.add(Dense(1, activation='linear'))

    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss',
                       patience=10,
                       verbose=0)

    hist = model.fit(x_train, t_train,
                     epochs=1000, batch_size=100,
                     verbose=0,
                     validation_data=(x_val, t_val),
                     callbacks=[es] if use_es else [])

    sin_true = make_sin_true()
    gen = [None for i in range(maxlen)]
    z = x[:1]
    for i in range(length_of_sequences - maxlen):
        preds = model.predict(z[-1:], verbose=0)
        z = np.append(z, preds)[1:]
        z = z.reshape(-1, maxlen, 1)
        gen.append(preds[0, 0])

    n_ep = len(hist.history['loss'])
    return n_ep, hist.history['val_loss'][-1], _gen_mse(gen, sin_true, maxlen), gen


RUNNERS = {'torch': run_torch, 'keras': run_keras}


def _job(args):
    lib, use_es, seed = args
    n_ep, val_loss, g, _ = RUNNERS[lib](seed, use_es)
    return lib, use_es, seed, n_ep, val_loss, g


def main(libs=('torch', 'keras'), seeds=SEEDS, processes=None):
    jobs = list(itertools.product(libs, (True, False), seeds))
    with Pool(processes) as pool:
        rows = pool.map(_job, jobs, chunksize=1)

    print('{:6s} {:5s} {:>4s} {:>6s} {:>9s} {:>8s}'.format(
        'lib', 'stop', 'seed', 'epochs', 'val_loss', 'gen_mse'))
    for lib, use_es, seed, n_ep, val_loss, g in rows:
        print('{:6s} {:5s} {:4d} {:6d} {:9.6f} {:8.4f}'.format(
            lib, 'ES' if use_es else '1000', seed, n_ep, val_loss, g))
    print()

    print('{:13s} {:>7s} {:>9s} {:>8s} {:>8s} {:>8s} {:>8s}'.format(
        'condition', 'epochs', 'val_loss', 'gen_mean', 'gen_med', 'gen_max', '<0.05'))
    for lib, use_es in itertools.product(libs, (True, False)):
        sub = [r for r in rows if r[0] == lib and r[1] == use_es]
        n_ep = np.array([r[3] for r in sub])
        val = np.array([r[4] for r in sub])
        g = np.array([r[5] for r in sub])
        print('{:13s} {:7.0f} {:9.6f} {:8.4f} {:8.4f} {:8.4f} {:5d}/{}'.format(
            '{} / {}'.format(lib, 'ES' if use_es else '1000'),
            n_ep.mean(), val.mean(), g.mean(), np.median(g), g.max(),
            int(np.sum(g < 0.05)), len(g)))


def plot_seed(seed=123):
    # 同じ seed・EarlyStopping あり（教科書と同じ条件）で3つの実装の生成波形を並べる
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sin_wave import run_experiment, RESULT_DIR

    results = []
    for name, fn in [('PyTorch (textbook)', run_torch), ('Keras (textbook)', run_keras)]:
        n_ep, val_loss, g, gen = fn(seed, True)
        results.append((name, n_ep, val_loss, g, gen))

    r = run_experiment(seed=seed, init='xavier', epochs=1000, patience=10,
                       verbose_every=0)
    results.append(('NumPy', r['epochs_run'], r['val_loss'], r['gen_mse'], r['gen']))

    T = 100
    x = np.arange(0, 2 * T + 1)
    sin_true = np.sin(2.0 * np.pi * x / T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, n_ep, val_loss, g, gen) in zip(axes, results):
        print('{:18s} epochs={:4d}, val_loss={:.6f}, gen_mse={:.4f}'.format(
            name, n_ep, val_loss, g))
        ax.set_xlim([0, 2 * T])
        ax.set_ylim([-1.5, 1.5])
        ax.plot(x, sin_true, color='gray', linestyle='--', linewidth=0.5)
        ax.plot(x, gen, color='black', linewidth=1, marker='o', markersize=1)
        ax.set_title('{}\nstopped at epoch {}, val_loss={:.4f}, gen_mse={:.4f}'.format(
            name, n_ep, val_loss, g), fontsize=9)
        ax.set_xlabel('time')

    fig.suptitle('Sin wave generation (seed={}, EarlyStopping patience=10)'.format(seed))
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, 'textbook_vs_numpy_seed{}.png'.format(seed))
    fig.savefig(path, dpi=120)
    print('saved:', path)


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['plot']:
        plot_seed()
    else:
        main(libs=tuple(sys.argv[1:]) or ('torch', 'keras'))
