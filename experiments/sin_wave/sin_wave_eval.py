import os
import itertools
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sin_wave import run_experiment, RESULT_DIR


SEEDS = list(range(10))

# (表示名, init, patience, amsgrad_raw)
#   patience=None     : EarlyStopping なしで 1000 エポック
#   amsgrad_raw=True  : PyTorch と同じ AMSGrad / False: 以前の実装（補正後の v の最大値）
CONDITIONS = [
    ('orthogonal / old',      'orthogonal', None, False),
    ('orthogonal / torch',    'orthogonal', None, True),
    ('xavier / old',          'xavier',     None, False),
    ('xavier / torch',        'xavier',     None, True),
    ('xavier / torch / ES',   'xavier',     10,   True),
    ('simple / torch',        'simple',     None, True),
]


def _run(args):
    label, init, patience, amsgrad_raw, seed = args
    r = run_experiment(seed=seed, init=init, epochs=1000, patience=patience,
                       verbose_every=0, amsgrad_raw=amsgrad_raw)
    return label, seed, r['epochs_run'], r['val_loss'], r['gen_mse']


def main():
    jobs = [cond + (seed,) for cond, seed in itertools.product(CONDITIONS, SEEDS)]

    with Pool() as pool:
        rows = pool.map(_run, jobs)

    w = max(len(c[0]) for c in CONDITIONS) + 1

    print('{:{w}s} {:>4s} {:>6s} {:>9s} {:>8s}'.format(
        'condition', 'seed', 'epochs', 'val_loss', 'gen_mse', w=w))
    for label, seed, n_ep, val_loss, g in rows:
        print('{:{w}s} {:4d} {:6d} {:9.6f} {:8.4f}'.format(
            label, seed, n_ep, val_loss, g, w=w))
    print()

    print('{:{w}s} {:>7s} {:>9s} {:>8s} {:>8s} {:>8s} {:>8s}'.format(
        'condition', 'epochs', 'val_loss', 'gen_mean', 'gen_med', 'gen_max', '<0.05', w=w))
    summary = {}
    for label, *_ in CONDITIONS:
        sub = [r for r in rows if r[0] == label]
        n_ep = np.array([r[2] for r in sub])
        val = np.array([r[3] for r in sub])
        g = np.array([r[4] for r in sub])
        summary[label] = g
        print('{:{w}s} {:7.0f} {:9.6f} {:8.4f} {:8.4f} {:8.4f} {:5d}/{}'.format(
            label, n_ep.mean(), val.mean(), g.mean(), np.median(g), g.max(),
            int(np.sum(g < 0.05)), len(g), w=w))

    # 条件ごとの gen_mse の分布（1点が1シード）
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(9, 4))
    for i, (label, g) in enumerate(summary.items()):
        plt.scatter(np.full(len(g), i), g, color='black', s=12)
        plt.hlines(np.median(g), i - 0.25, i + 0.25, color='gray')
    plt.xticks(range(len(summary)), list(summary.keys()), rotation=20, ha='right')
    plt.yscale('log')
    plt.ylabel('gen_mse')
    plt.title('Generation error over {} seeds (line: median)'.format(len(SEEDS)))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'gen_mse_by_condition.png'), dpi=120)


if __name__ == '__main__':
    main()
