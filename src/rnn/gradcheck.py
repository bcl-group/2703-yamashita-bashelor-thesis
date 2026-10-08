import numpy as np

from .losses import mean_squared_error


def numerical_gradient_check(model, x, t_true, n_check=3, eps=1e-5):
    model.forward(x)
    grads = model.backward(t_true)

    print('--- 勾配の検算（BPTT vs 数値微分） ---')
    for name in model.param_names():
        param = getattr(model, name)
        flat = param.ravel()
        grad_flat = grads[name].ravel()

        idx_list = np.random.choice(flat.size, min(n_check, flat.size),
                                    replace=False)

        max_rel_err = 0.0
        for idx in idx_list:
            original = flat[idx]

            flat[idx] = original + eps
            loss_plus = mean_squared_error(t_true, model.forward(x))

            flat[idx] = original - eps
            loss_minus = mean_squared_error(t_true, model.forward(x))

            flat[idx] = original

            numerical = (loss_plus - loss_minus) / (2.0 * eps)
            analytic = grad_flat[idx]

            denom = max(1e-12, abs(numerical) + abs(analytic))
            rel_err = abs(numerical - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)

        print('  {:5s} : 最大相対誤差 = {:.3e}'.format(name, max_rel_err))
    print()
