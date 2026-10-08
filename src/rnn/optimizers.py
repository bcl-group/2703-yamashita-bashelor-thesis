import numpy as np


class SGD:

    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, model, grads):
        for name, grad in grads.items():
            param = getattr(model, name)
            param -= self.lr * grad


class Adam:

    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, amsgrad=True, amsgrad_raw=True):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        # True : PyTorch と同じ。補正前の v の最大値を保持し、あとで補正する
        # False: 補正後の v_hat の最大値を保持する（以前の実装）
        self.amsgrad_raw = amsgrad_raw

        names = model.param_names()
        self.m = {k: np.zeros_like(getattr(model, k)) for k in names}
        self.v = {k: np.zeros_like(getattr(model, k)) for k in names}
        self.v_max = {k: np.zeros_like(getattr(model, k)) for k in names}

        self.t = 0

    def step(self, model, grads):
        self.t += 1

        for name, grad in grads.items():
            param = getattr(model, name)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            if self.amsgrad and self.amsgrad_raw:
                self.v_max[name] = np.maximum(self.v_max[name], self.v[name])
                v_hat = self.v_max[name] / (1 - self.beta2 ** self.t)
            elif self.amsgrad:
                self.v_max[name] = np.maximum(self.v_max[name], v_hat)
                v_hat = self.v_max[name]

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
