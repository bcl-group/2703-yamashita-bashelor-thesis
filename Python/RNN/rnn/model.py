import numpy as np


class SimpleRNN:

    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1,
                 init='simple'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if init in ('orthogonal', 'xavier'):
            # 'xavier' は教科書（PyTorch xavier_normal_ / Keras glorot_normal）と同じ
            # sqrt(2 / (n_in + n_out))。'orthogonal' は以前の sqrt(1 / n_in)
            if init == 'xavier':
                scale = np.sqrt(2.0 / (input_dim + hidden_dim))
            else:
                scale = np.sqrt(1.0 / input_dim)
            self.W_xh = np.random.normal(
                loc=0.0, scale=scale,
                size=(input_dim, hidden_dim))

            a = np.random.normal(size=(hidden_dim, hidden_dim))
            q, r = np.linalg.qr(a)
            self.W_hh = q * np.sign(np.diag(r))

            self.W_hy = np.random.normal(
                loc=0.0, scale=np.sqrt(1.0 / hidden_dim),
                size=(hidden_dim, output_dim))
        else:
            self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.01
            self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
            self.W_hy = np.random.randn(hidden_dim, output_dim) * 0.01

        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

        self.cache = None

    def forward(self, x):
        batch_size, maxlen, _ = x.shape

        h = np.zeros((batch_size, self.hidden_dim))

        h_list = [h]

        for t in range(maxlen):
            x_t = x[:, t, :]

            p = x_t @ self.W_xh + h @ self.W_hh + self.b_h

            h = np.tanh(p)

            h_list.append(h)

        h_last = h_list[-1]

        y = h_last @ self.W_hy + self.b_y

        self.cache = (x, h_list, y)

        return y

    def backward(self, t_true):
        x, h_list, y = self.cache
        batch_size, maxlen, _ = x.shape

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        e_o = 2.0 * (y - t_true) / (batch_size * self.output_dim)

        h_last = h_list[-1]

        dW_hy = h_last.T @ e_o

        db_y = np.sum(e_o, axis=0)

        e_h = (e_o @ self.W_hy.T) * (1.0 - h_last ** 2)

        for t in reversed(range(maxlen)):
            x_t = x[:, t, :]
            h_prev = h_list[t]

            dW_xh += x_t.T @ e_h

            dW_hh += h_prev.T @ e_h

            db_h += np.sum(e_h, axis=0)

            e_h = (e_h @ self.W_hh.T) * (1.0 - h_prev ** 2)

        grads = {
            'W_xh': dW_xh,
            'W_hh': dW_hh,
            'b_h': db_h,
            'W_hy': dW_hy,
            'b_y': db_y,
        }
        return grads

    def param_names(self):
        return ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']
