"""
モデル2（RNN 3個）の閉ループを、改善前（次の時刻の値を出す）、変化量＋値域の制約（teacher_thesis で学習）、
改訂版（teacher_ou で学習）で比べる図を描く。図は results/hh_modular_ou/ に保存する。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/compare_modular.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hh_modular as hm
from hh_surrogate import EVAL_T, EVAL_CURRENTS, make_eval_current, count_spikes
from neuron.hh import simulate, V_REST

ROOT = os.path.join(hm.BASE_DIR, '..', '..')
OUT_DIR = os.path.join(ROOT, 'results', 'hh_modular_delta')
FIG_DIR = os.path.join(ROOT, 'results', 'hh_modular_ou')

CONDITIONS = [
    # (表示名, ブロックのフォルダ, clip, 色)
    ('改善前（値を出力）', os.path.join(ROOT, 'results', 'hh_modular'), False, '#eb6834'),
    ('変化量＋制約（teacher_thesis）', OUT_DIR, True, '#2a78d6'),
    ('改訂版（teacher_ou）', os.path.join(ROOT, 'results', 'hh_modular_ou'), True, '#1baf7a'),
]
METRIC_DIRS = [
    ('改善前', 'results/hh_modular'),
    ('制約だけ', 'results/hh_modular_clip'),
    ('変化量だけ', 'results/hh_modular_delta/noclip'),
    ('変化量＋制約', 'results/hh_modular_delta/clip'),
    ('改訂版（OUデータ）', 'results/hh_modular_ou'),
]
TRAIN_SHOW_MS = 200.0

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']


def load_blocks(d):
    return {b: hm.load_block(os.path.join(d, 'block{}.npz'.format(b)))[:3] for b in (1, 2, 3)}


def make_cases():
    t, I, V, m, h, n, meta = hm.load_hh_full()
    cases = [('学習電流（最初の {:.0f} ms）'.format(TRAIN_SHOW_MS), t, I, V)]
    t_eval = np.arange(0, EVAL_T, meta['dt_ms'])
    for name, func, params in EVAL_CURRENTS:
        I_e = make_eval_current(t_eval, func, params)
        V_e = simulate(t_eval, I_e, V0=V_REST)[0]
        cases.append((name, t_eval, I_e, V_e))
    return cases


def plot_traces(cases, preds, path):
    fig, axes = plt.subplots(len(cases), len(CONDITIONS), figsize=(17, 1.9 * len(cases)),
                             sharey=True, squeeze=False)
    for r, (name, t, I, V) in enumerate(cases):
        for c, (label, _, _, color) in enumerate(CONDITIONS):
            ax = axes[r, c]
            Vp = preds[c][r]
            ax.plot(t[1:], V[1:], color='#888780', linestyle='--', linewidth=0.9, label='HH')
            ax.plot(t[1:], Vp, color=color, linewidth=0.8, label=label)
            ax.set_xlim(t[0], TRAIN_SHOW_MS if r == 0 else t[-1])
            ax.set_ylim(-130, 60)
            ax.grid(True, color='#e6e5df', linewidth=0.5)
            ax.text(0.99, 0.04, 'スパイク HH {} / 予測 {}'.format(
                        count_spikes(V[1:]), count_spikes(Vp)),
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    color='#52514e')
            if r == 0:
                ax.set_title(label, fontsize=11)
                ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
            if c == 0:
                ax.set_ylabel('{}\nV [mV]'.format(name), fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('時刻 [ms]')
    fig.suptitle('モデル2（RNN 3個）の閉ループ：灰色の破線が HH', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metrics(path):
    ms = [(lab, json.load(open(os.path.join(ROOT, d, 'metrics.json'), encoding='utf-8'))['metrics'])
          for lab, d in METRIC_DIRS]
    names = list(ms[0][1].keys())
    colors = ['#eb6834', '#eda100', '#e87ba4', '#2a78d6', '#1baf7a']
    x = np.arange(len(names))
    w = 0.8 / (len(ms) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    hh_spk = [ms[0][1][n]['spike_count'][0] for n in names]
    ax1.bar(x - 0.4 + w / 2, hh_spk, w * 0.92, color='#888780', label='HH（正解）')
    for i, ((lab, m), col) in enumerate(zip(ms, colors)):
        xs = x - 0.4 + w * (i + 1.5)
        ax1.bar(xs, [m[n]['spike_count'][1] for n in names], w * 0.92, color=col, label=lab)
        ax2.bar(xs, [m[n]['rmse'] for n in names], w * 0.92, color=col, label=lab)
    ax1.set_ylabel('スパイク数')
    ax1.set_yscale('symlog', linthresh=10)
    ax1.legend(fontsize=9, frameon=False, ncol=5, loc='upper left')
    ax1.set_title('スパイク数（縦軸は 10 以上で対数）', fontsize=10)
    ax2.set_ylabel('V の RMSE [mV]')
    ax2.set_title('閉ループでの膜電位の RMSE（小さいほどよい）', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    for ax in (ax1, ax2):
        ax.grid(True, axis='y', color='#e6e5df', linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cases = make_cases()
    preds = []
    for _, d, clip, _ in CONDITIONS:
        blocks = load_blocks(d)
        preds.append([hm.run_closed_loop(blocks, I, V0=V[0], clip=clip)[0]
                      for _, _, I, V in cases])
    os.makedirs(FIG_DIR, exist_ok=True)
    p1 = os.path.join(FIG_DIR, 'compare_closedloop.png')
    p2 = os.path.join(FIG_DIR, 'compare_metrics.png')
    plot_traces(cases, preds, p1)
    plot_metrics(p2)
    print('saved:', os.path.normpath(p1))
    print('saved:', os.path.normpath(p2))


if __name__ == '__main__':
    main()
