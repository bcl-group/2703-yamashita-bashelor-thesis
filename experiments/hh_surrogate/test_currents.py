"""
src/neuron/current.py にあるすべての電流関数で、モデル2（RNN 3個）の閉ループをテストする。

各テスト電流について、
    1. HH を dt = 0.01 ms の陽的オイラー法で解く（V(0) = -65 mV、ゲート変数は定常値）
    2. モデル2を同じ初期状態から閉ループで回す（外から与えるのは I(t) だけ）
    3. 膜電位を比べる（スパイク数、スパイク時刻の一致、RMSE など）
を行い、図と metrics を保存する。

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/test_currents.py
    uv run python -X utf8 experiments/hh_surrogate/test_currents.py \
        --dirs results/hh_modular_ou results/hh_modular_delta --out results/hh_modular_ou/test_currents
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hh_modular as hm
from neuron.hh import simulate, V_REST
from neuron import current as cur
from evaluation import compare_waveforms

ROOT = os.path.join(hm.BASE_DIR, '..', '..')
DT = 0.01
MATCH_MS = 2.0   # スパイク時刻がこの範囲 [ms] 内なら一致とみなす

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']

# (名前, 電流関数, 引数, 長さ [ms])。current.py の関数をすべて使う
TESTS = [
    # step_current：一定時間だけ流す
    ('step 10 (50-150ms)',      cur.step_current, {'amplitude': 10.0, 't_start': 50.0, 't_end': 150.0}, 300.0),
    ('step 5 (50-250ms)',       cur.step_current, {'amplitude': 5.0, 't_start': 50.0, 't_end': 250.0}, 300.0),
    ('step 30 1ms',             cur.step_current, {'amplitude': 30.0, 't_start': 50.0, 't_end': 51.0}, 150.0),
    ('step -5 (50-150ms)',      cur.step_current, {'amplitude': -5.0, 't_start': 50.0, 't_end': 150.0}, 300.0),
    # constant_current：一定電流
    ('const 2.5',               cur.constant_current, {'amplitude': 2.5}, 300.0),
    ('const 5',                 cur.constant_current, {'amplitude': 5.0}, 300.0),
    ('const 7',                 cur.constant_current, {'amplitude': 7.0}, 300.0),
    ('const 10',                cur.constant_current, {'amplitude': 10.0}, 300.0),
    ('const 20',                cur.constant_current, {'amplitude': 20.0}, 300.0),
    # ramp_current：直線的に増やす
    ('ramp 0-20',               cur.ramp_current, {'i_start': 0.0, 'i_end': 20.0}, 500.0),
    ('ramp 0-30',               cur.ramp_current, {'i_start': 0.0, 'i_end': 30.0}, 500.0),
    # sin_current
    ('sin 5Hz 5±5',             cur.sin_current, {'amplitude': 5.0, 'freq': 5.0, 'offset': 5.0}, 500.0),
    ('sin 10Hz 7.5±7.5',        cur.sin_current, {'amplitude': 7.5, 'freq': 10.0, 'offset': 7.5}, 500.0),
    ('sin 50Hz 10±10',          cur.sin_current, {'amplitude': 10.0, 'freq': 50.0, 'offset': 10.0}, 300.0),
    # pulse_train_current：周期的なパルス
    ('pulse 10/1ms/10ms',       cur.pulse_train_current, {'amplitude': 10.0, 'width': 1.0, 'interval': 10.0}, 300.0),
    ('pulse 20/25ms/50ms',      cur.pulse_train_current, {'amplitude': 20.0, 'width': 25.0, 'interval': 50.0}, 300.0),
    ('pulse 40/0.5ms/20ms',     cur.pulse_train_current, {'amplitude': 40.0, 'width': 0.5, 'interval': 20.0}, 300.0),
    # random_pulse_current：一定間隔で値が変わる階段状の電流
    ('random step -5~20/20ms',  cur.random_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'interval': 20.0, 'seed': 1}, 500.0),
    ('random step 0~20/10ms',   cur.random_pulse_current, {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0, 'seed': 1}, 500.0),
    # random_timing_pulse_current：ランダムな時刻のパルス
    ('random pulse 5ms',        cur.random_timing_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'width': 5.0, 'rate': 0.05, 'seed': 1}, 500.0),
    ('random pulse 1ms 0~40',   cur.random_timing_pulse_current, {'i_min': 0.0, 'i_max': 40.0, 'width': 1.0, 'rate': 0.02, 'seed': 2}, 500.0),
    # ou_current：OU 過程で揺らぐ電流
    ('OU 5±3',                  cur.ou_current, {'mean': 5.0, 'sigma': 3.0, 'tau': 1.0, 'seed': 1}, 500.0),
    ('OU 10±3',                 cur.ou_current, {'mean': 10.0, 'sigma': 3.0, 'tau': 1.0, 'seed': 1}, 500.0),
    ('OU 7±5 (tau 5ms)',        cur.ou_current, {'mean': 7.0, 'sigma': 5.0, 'tau': 5.0, 'seed': 1}, 500.0),
    # level_ou_current：学習データ（teacher_ou）と同じ種類。seed は学習に使っていない値
    ('level+OU',                cur.level_ou_current, {'seed': 99}, 1000.0),
    # munechika_train_current：棟近さんの学習用電流（最初の 2000 ms）
    ('munechika train',         cur.munechika_train_current, {}, 2000.0),
]

COLORS = ['#1baf7a', '#2a78d6', '#eb6834', '#e87ba4']


def spike_times(V, dt):
    # 0 mV を下から上に横切った時刻 [ms]
    idx = np.where((V[:-1] < 0.0) & (V[1:] >= 0.0))[0] + 1
    return idx * dt


def match_rate(t_true, t_pred, window=MATCH_MS):
    """
    スパイク時刻の一致率。正解の各スパイクに、±window ms 以内の予測スパイクを1つずつ対応させ、
    対応できた数 / max(正解の数, 予測の数) を返す。両方 0 発なら 1。
    """
    if len(t_true) == 0 and len(t_pred) == 0:
        return 1.0
    used = np.zeros(len(t_pred), dtype=bool)
    n_match = 0
    for s in t_true:
        cand = np.where(~used & (np.abs(t_pred - s) <= window))[0]
        if len(cand):
            used[cand[np.argmin(np.abs(t_pred[cand] - s))]] = True
            n_match += 1
    return n_match / max(len(t_true), len(t_pred))


def load_blocks(d):
    return {b: hm.load_block(os.path.join(d, 'block{}.npz'.format(b)))[:3] for b in (1, 2, 3)}


def run_tests(models, clip):
    results = []
    for name, func, params, T in TESTS:
        t = np.arange(0, T, DT)
        I = func(t, **params)
        V = simulate(t, I, V0=V_REST)[0]
        V_true = V[1:]
        row = {'name': name, 'func': func.__name__, 'params': params, 'T_ms': T,
               't': t, 'I': I, 'V_true': V_true, 'models': {}}
        st_true = spike_times(V_true, DT)
        for label, blocks in models:
            V_pred = hm.run_closed_loop(blocks, I, V0=V[0], clip=clip)[0]
            m = compare_waveforms(V_true, V_pred, DT)
            row['models'][label] = {
                'V_pred': V_pred,
                'spikes': (len(st_true), len(spike_times(V_pred, DT))),
                'match': match_rate(st_true, spike_times(V_pred, DT)),
                'rmse': m['rmse'],
                'metrics': m,
            }
        results.append(row)
        print('{:24s} HH {:3d} | '.format(name, len(st_true)) + ' | '.join(
            '{} {:3d} 一致 {:4.0%} RMSE {:5.1f}'.format(
                label, r['spikes'][1], r['match'], r['rmse'])
            for label, r in row['models'].items()))
    return results


def plot_results(results, labels, out_dir, per_fig=7):
    paths = []
    for f, start in enumerate(range(0, len(results), per_fig)):
        rows = results[start:start + per_fig]
        fig, axes = plt.subplots(2 * len(rows), 1, figsize=(13, 2.6 * len(rows)),
                                 gridspec_kw={'height_ratios': [1, 3] * len(rows)})
        for i, r in enumerate(rows):
            ax_i, ax_v = axes[2 * i], axes[2 * i + 1]
            t = r['t']
            ax_i.plot(t, r['I'], color='#52514e', linewidth=0.6)
            ax_i.set_ylabel('I', fontsize=8)
            ax_i.set_xlim(t[0], t[-1])
            ax_i.set_xticklabels([])
            ax_i.set_title('{}（{}）'.format(r['name'], r['func']), fontsize=10, loc='left')
            ax_v.plot(t[1:], r['V_true'], color='#888780', linestyle='--', linewidth=0.9, label='HH')
            texts = []
            for label, color in zip(labels, COLORS):
                mr = r['models'][label]
                ax_v.plot(t[1:], mr['V_pred'], color=color, linewidth=0.7, label=label)
                texts.append('{}: {} 発, 一致 {:.0%}'.format(label, mr['spikes'][1], mr['match']))
            ax_v.text(0.995, 0.03, 'HH: {} 発\n'.format(r['models'][labels[0]]['spikes'][0])
                      + '\n'.join(texts), transform=ax_v.transAxes, ha='right', va='bottom',
                      fontsize=7.5, color='#52514e',
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
            ax_v.set_xlim(t[0], t[-1])
            ax_v.set_ylim(-110, 60)
            ax_v.set_ylabel('V [mV]', fontsize=8)
            ax_v.grid(True, color='#e6e5df', linewidth=0.5)
            if i == 0:
                ax_v.legend(loc='upper left', fontsize=8, frameon=False, ncol=len(labels) + 1)
        axes[-1].set_xlabel('時刻 [ms]')
        fig.tight_layout()
        path = os.path.join(out_dir, 'test_currents_{}.png'.format(f + 1))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)

    # まとめ：ケースごとのスパイク一致率
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(results))
    w = 0.8 / len(labels)
    for k, (label, color) in enumerate(zip(labels, COLORS)):
        ax.bar(x - 0.4 + w * (k + 0.5), [r['models'][label]['match'] for r in results],
               w * 0.92, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('スパイク時刻の一致率（±{:.0f} ms）'.format(MATCH_MS))
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', color='#e6e5df', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.12), ncol=len(labels))
    fig.tight_layout()
    path = os.path.join(out_dir, 'test_currents_summary.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
    return paths


def save_metrics(results, labels, out_dir, dirs, clip):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return clean(v.item())
        return v

    out = {
        'condition': {'models': dict(zip(labels, [os.path.normpath(d) for d in dirs])),
                      'clip': clip, 'dt_ms': DT, 'match_window_ms': MATCH_MS},
        'tests': [{
            'name': r['name'], 'func': r['func'], 'params': r['params'], 'T_ms': r['T_ms'],
            'models': {label: {k: v for k, v in mr.items() if k != 'V_pred'}
                       for label, mr in r['models'].items()},
        } for r in results],
    }
    path = os.path.join(out_dir, 'metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean(out), f, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+',
                        default=['results/hh_modular_ou', 'results/hh_modular_delta'],
                        help='block1〜3.npz があるフォルダ（複数なら並べて比べる）')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='図の凡例（既定はフォルダ名）')
    parser.add_argument('--out', default='results/hh_modular_ou/test_currents')
    parser.add_argument('--no-clip', action='store_true', help='値域の制約を入れない')
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.dirs]
    models = [(label, load_blocks(d)) for label, d in zip(labels, args.dirs)]
    clip = not args.no_clip
    os.makedirs(args.out, exist_ok=True)

    start = time.time()
    results = run_tests(models, clip)
    print('テストにかかった時間: {:.0f} s'.format(time.time() - start))
    for p in plot_results(results, labels, args.out):
        print('saved:', os.path.normpath(p))
    print('saved:', os.path.normpath(save_metrics(results, labels, args.out, args.dirs, clip)))


if __name__ == '__main__':
    main()
