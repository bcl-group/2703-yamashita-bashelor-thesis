"""
flow function（論文 Learning flow functions of spiking systems のコード）で学習したモデルを、
RNN 型と同じ指標（スパイク時刻の一致率と coincidence factor Γ、±2 ms）で採点する。

正解は論文の HH（HodgkinHuxleyFS、電流 0〜1 uA、solve_ivp）で、RNN 型の HH（Euler 法、dt = 0.01 ms）とは
モデルも電流の範囲も違う。そのため数値は RNN 型と同じ土俵ではなく、「flow function が自分の対象をどこまで
時刻まで再現できるか」の参考値として扱う。

採点する入力（どちらも学習に使っていない乱数・電流）:
    current_py : predict_current_py.py と同じ 7 種類の電流（0〜1 uA に縮めた step, constant, ramp, sin,
                 pulse train, random pulse, random timing pulse）。静止状態から 1000 ms
    paper      : 論文と同じ分布（初期状態ランダム、100 ms ごとに U[0, 1] の階段電流）の軌道を --n_paper 本。
                 学習に使った乱数とは別の seed

時刻の刻み:
    モデルの出力（t_eval）は --res_ms ごと（既定 0.05 ms）。スパイクは 0 mV を下から上に横切った時刻で数える

実行方法（学内サーバ、卒論リポジトリのルートで。論文コード用の .venv を使う）:
    PY=external/flow-learning-l4dc/.venv/bin/python
    $PY -X utf8 experiments/flow_learning/score_flow_gamma.py external/flow-learning-l4dc/outputs/hhfs/*.pth
結果:
    results/flow_learning/gamma/<モデル名>.json（入力ごとの行）と results/flow_learning/hhfs_gamma.csv（モデルごとの要約）
"""
import csv
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import predict_current_py as pc   # noqa: E402  論文コードの読み込み・入力の整形・正解の計算をそのまま使う

WINDOW_MS = 2.0


# ---- 指標（experiments/hh_surrogate/hh_bench.py と同じ定義。論文コードの環境で動くように写してある） ----

def spike_times(V, t_ms):
    # 0 mV を下から上に横切った時刻 [ms]。V, t_ms: (n,)
    idx = np.where((V[:-1] < 0.0) & (V[1:] >= 0.0))[0] + 1
    return t_ms[idx]


def n_coincident(t_true, t_pred, window=WINDOW_MS):
    used = np.zeros(len(t_pred), dtype=bool)
    n = 0
    for s in t_true:
        cand = np.where(~used & (np.abs(t_pred - s) <= window))[0]
        if len(cand):
            used[cand[np.argmin(np.abs(t_pred[cand] - s))]] = True
            n += 1
    return n


def match_rate(t_true, t_pred, window=WINDOW_MS):
    # 対応できた数 / max(正解の数, 予測の数)。両方 0 発なら 1
    if len(t_true) == 0 and len(t_pred) == 0:
        return 1.0
    return n_coincident(t_true, t_pred, window) / max(len(t_true), len(t_pred))


def coincidence_factor(t_true, t_pred, T, window=WINDOW_MS):
    # Γ（Kistler et al. 1997; Jolivet et al. 2008）。両方 0 発なら 1
    n1, n2 = len(t_true), len(t_pred)
    if n1 == 0 and n2 == 0:
        return 1.0
    nu = n2 / T
    norm = 1.0 - 2.0 * nu * window
    if norm <= 0:
        return float('nan')
    return (n_coincident(t_true, t_pred, window) - 2.0 * nu * window * n1) / (0.5 * (n1 + n2)) / norm


# ---- 採点 ----

def score_model(exp_path, args):
    experiment, model, pack_model_inputs = pc.load_experiment(exp_path, args.l4dc_root)
    sampler = experiment.generator.sampler
    dyn, method, delta = sampler._dyn, sampler._ode_method, sampler._delta
    ms = dyn.time_scale                                   # モデル時間 1 あたりの ms
    delta_ms = delta * ms
    mask = np.array([bool(v) for v in dyn.mask])
    T = args.t_end / ms
    n_control = 1 + int(np.floor(T / delta))
    t_fine = np.arange(0.0, n_control * delta_ms, 0.01)   # 電流を作る刻み [ms]
    t_eval = np.arange(0.0, T, args.res_ms / ms)          # 出力の時刻（モデル時間）。shape (n_eval,)
    t_ms = t_eval * ms

    sampler._seq_gen._rng = np.random.default_rng(args.seed)
    sampler.state_generator.rng = np.random.default_rng(args.seed + 1000)
    cases = []
    x_rest = pc.resting_state(dyn, sampler.state_generator.sample(), method)
    for name, make_I in pc.CURRENTS.items():
        I = make_I(t_fine)
        cases.append(('current_py', name, x_rest, pc.to_control_sequence(I, 0.01, delta_ms, n_control)))
    for i in range(args.n_paper):
        x0 = sampler.state_generator.sample()
        u = sampler._seq_gen.sample(time_range=(0.0, T), delta=delta)[:n_control]
        cases.append(('paper', f'paper {i}', x0, u))

    rows = []
    for kind, name, x0, u in cases:
        V_true = dyn.v_scale * pc.simulate_true(dyn, x0, u, delta, t_eval, method)[:, mask][:, 0]
        u_feed = np.vstack([u, u[-1:]]).astype(np.float32)
        x0_feed, _, rnn_input, deltas = pack_model_inputs(x0, t_eval, u_feed, delta)
        with torch.no_grad():
            V_pred = dyn.v_scale * np.flip(experiment.predict(model, x0_feed, rnn_input, deltas), 0)[:, 0]
        st, sp = spike_times(V_true, t_ms), spike_times(V_pred, t_ms)
        rows.append({'kind': kind, 'name': name, 'spikes': [len(st), len(sp)],
                     'match': match_rate(st, sp), 'gamma': coincidence_factor(st, sp, args.t_end),
                     'rmse': float(np.sqrt(np.mean((V_true - V_pred) ** 2)))})
        print('  {:>20s}: spikes {:3d}/{:3d}  一致率 {:.3f}  Γ {:.3f}  RMSE {:.1f} mV'.format(
            name, len(st), len(sp), rows[-1]['match'], rows[-1]['gamma'], rows[-1]['rmse']), flush=True)
    return rows


def summarize(rows, kind):
    r = [x for x in rows if x['kind'] == kind]
    return {'n': len(r), 'n100': sum(x['match'] == 1.0 for x in r),
            'mean': float(np.mean([x['match'] for x in r])), 'gamma': float(np.nanmean([x['gamma'] for x in r])),
            'rmse': float(np.mean([x['rmse'] for x in r]))}


def main():
    ap = ArgumentParser()
    ap.add_argument('exp_paths', nargs='+', help='論文コードが保存した学習結果の .pth（複数可）')
    ap.add_argument('--l4dc_root', default=os.path.join(ROOT, 'external', 'flow-learning-l4dc'))
    ap.add_argument('--t_end', type=float, default=1000.0, help='シミュレーションの長さ [ms]')
    ap.add_argument('--res_ms', type=float, default=0.05, help='出力の時刻の刻み [ms]')
    ap.add_argument('--n_paper', type=int, default=20, help='論文と同じ分布の軌道の本数')
    ap.add_argument('--seed', type=int, default=12345, help='採点用の乱数の seed（学習とは別）')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results', 'flow_learning'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / 'gamma').mkdir(parents=True, exist_ok=True)
    summary = []
    for p in args.exp_paths:
        name = Path(p).stem
        print(f'=== {name} ===', flush=True)
        rows = score_model(p, args)
        res = {'model': name, 'path': p, 'args': vars(args), 'rows': rows,
               'current_py': summarize(rows, 'current_py'), 'paper': summarize(rows, 'paper')}
        json.dump(res, open(out_dir / 'gamma' / f'{name}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        c, q = res['current_py'], res['paper']
        print('[{}] current_py: 100% {}/{}  平均 {:.3f}  Γ {:.3f}  |  paper: 100% {}/{}  平均 {:.3f}  Γ {:.3f}'.format(
            name, c['n100'], c['n'], c['mean'], c['gamma'], q['n100'], q['n'], q['mean'], q['gamma']), flush=True)
        summary.append({'model': name, **{f'cur_{k}': v for k, v in c.items()},
                        **{f'paper_{k}': v for k, v in q.items()}})
    with open(out_dir / 'hhfs_gamma.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print('保存:', os.path.normpath(out_dir / 'hhfs_gamma.csv'))


if __name__ == '__main__':
    main()
