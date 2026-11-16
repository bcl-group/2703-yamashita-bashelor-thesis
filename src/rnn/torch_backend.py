"""
SimpleRNN（NumPy 版）の学習を PyTorch で速く回す。

NumPy 版の forward / backward は時刻ごとの Python ループで、1区間 1,000 ステップの
BPTT が遅い。ここでは同じ式の RNN を torch.nn.RNN で組み、学習だけを PyTorch に任せる。
学習が終わったら重みを NumPy 版の SimpleRNN に書き戻すので、保存（npz）・評価・閉ループは
これまでのコードがそのまま使える。

式の対応（NumPy 版 → torch.nn.RNN, nonlinearity='tanh'）:
    h(t) = tanh(x(t) W_xh + h(t-1) W_hh + b_h)
         = tanh(x(t) W_ih^T + b_ih + h(t-1) W_hh'^T + b_hh)
    y(t) = h(t) W_hy + b_y = Linear(h(t))

    W_xh (d_in, H)  = W_ih^T      （torch の weight_ih_l0 は (H, d_in)）
    W_hh (H, H)     = W_hh'^T     （torch の weight_hh_l0 は (H, H)）
    b_h  (H,)       = b_ih + b_hh （torch はバイアスを2つ持つ。b_hh を 0 に固定して b_ih = b_h とする）
    W_hy (H, d_out) = W^T         （torch の Linear.weight は (d_out, H)）
    b_y  (d_out,)   = Linear.bias

損失は NumPy 版と同じ「全バッチ・全時刻・全出力の平均二乗誤差」、最適化は Adam（amsgrad）。
NumPy 版の Adam は PyTorch の amsgrad と同じ更新式（optimizers.py の amsgrad_raw=True）。
"""
import time

import numpy as np
import torch
from torch import nn


class TorchSimpleRNN(nn.Module):
    """NumPy 版 SimpleRNN(return_sequences=True) と同じ式の RNN。"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, nonlinearity='tanh', batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        # b_hh は使わない（b_h は b_ih だけで表す）。0 に固定して学習させない
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.bias_hh_l0.requires_grad_(False)

    def forward(self, x):
        # x: (batch, maxlen, d_in) -> y: (batch, maxlen, d_out)。h(0) = 0
        h_seq, _ = self.rnn(x)
        return self.out(h_seq)

    @torch.no_grad()
    def load_from_numpy(self, model):
        self.rnn.weight_ih_l0.copy_(torch.from_numpy(model.W_xh.T.copy()))
        self.rnn.weight_hh_l0.copy_(torch.from_numpy(model.W_hh.T.copy()))
        self.rnn.bias_ih_l0.copy_(torch.from_numpy(model.b_h.copy()))
        self.rnn.bias_hh_l0.zero_()
        self.out.weight.copy_(torch.from_numpy(model.W_hy.T.copy()))
        self.out.bias.copy_(torch.from_numpy(model.b_y.copy()))

    @torch.no_grad()
    def save_to_numpy(self, model):
        # NumPy 版は float64 で持つ
        f = lambda p: p.detach().cpu().double().numpy()
        model.W_xh = f(self.rnn.weight_ih_l0).T.copy()
        model.W_hh = f(self.rnn.weight_hh_l0).T.copy()
        model.b_h = f(self.rnn.bias_ih_l0) + f(self.rnn.bias_hh_l0)
        model.W_hy = f(self.out.weight).T.copy()
        model.b_y = f(self.out.bias).copy()


def pick_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def train_chunks_torch(model, x, t, epochs=1000, batch_size=10, lr=0.001,
                       verbose_every=10, early_stopping=None, device='auto',
                       dtype=torch.float32, checkpoint_every=None, on_checkpoint=None):
    """
    hh_surrogate.train_chunks と同じ流れ（区間をシャッフルしてミニバッチ、区間ごとに h=0）を
    PyTorch で回す。終わったら model（NumPy 版 SimpleRNN）の重みを書き換える。

    Args:
        model (rnn.SimpleRNN): 初期の重みを持つ NumPy 版モデル。学習後の重みで上書きされる。
        x (np.ndarray): 入力の区間。shape (区間数, chunk_len, d_in)、標準化済み。
        t (np.ndarray): 正解の区間。shape (区間数, chunk_len, d_out)、標準化済み。
        epochs, batch_size, lr: NumPy 版と同じ意味。
        early_stopping: EarlyStopping（train loss で判定）か None。
        device (str): 'auto'（GPU があれば GPU）、'cpu'、'cuda' など。
        dtype: 計算の精度。既定は float32（NumPy 版は float64）。
        checkpoint_every, on_checkpoint: checkpoint_every エポックごとに model の重みを書き換え、
            on_checkpoint(hist) を呼ぶ（途中で止まっても重みが残るように保存するのに使う）。

    Returns:
        dict: {'loss': [エポックごとの平均損失]}。NumPy 版の train_chunks と同じ形。
    """
    dev = pick_device(device)
    net = TorchSimpleRNN(model.input_dim, model.hidden_dim, model.output_dim)
    net.load_from_numpy(model)
    net.to(device=dev, dtype=dtype)

    # 区間はまとめて device に載せる（teacher_ou の RNN2 で float32 なら約 0.5 GB）
    xt = torch.as_tensor(x, dtype=dtype, device=dev)
    tt = torch.as_tensor(t, dtype=dtype, device=dev)

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
    loss_fn = nn.MSELoss()   # 全要素の平均。NumPy 版の mean_squared_error と同じ

    print('backend = torch {}, device = {}, dtype = {}, threads = {}'.format(
        torch.__version__, dev, str(dtype).replace('torch.', ''), torch.get_num_threads()))

    n = len(x)
    hist = {'loss': []}
    for epoch in range(epochs):
        # シャッフルは NumPy の乱数で（seed をそろえれば NumPy 版と同じ順番になる）
        perm = torch.from_numpy(np.random.permutation(n)).to(dev)
        total = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[idx]), tt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1

        loss_epoch = total / n_batches
        hist['loss'].append(loss_epoch)

        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print('epoch: {:4d}, loss: {:.6f}'.format(epoch + 1, loss_epoch), flush=True)

        if checkpoint_every and on_checkpoint and (epoch + 1) % checkpoint_every == 0:
            net.save_to_numpy(model)
            on_checkpoint(hist)

        if early_stopping is not None and early_stopping(loss_epoch):
            if verbose_every:
                print('epoch: {:4d}, loss: {:.6f}'.format(epoch + 1, loss_epoch))
            break

    net.save_to_numpy(model)
    return hist
