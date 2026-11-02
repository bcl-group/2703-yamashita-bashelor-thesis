"""
論文（Learning flow functions of spiking systems, L4DC 2024）のコードで学習したモデルに電流を入れ、
正解の HH モデルと膜電位を並べてグラフにする。

論文のコード（flow-learning の 2407-l4dc ブランチ）は書き換えず、学習結果の .pth を読み込むだけ。
このスクリプトで足しているのは「モデルに入れる電流」だけである。

    --currents paper      : 論文 §4（図2・図4）と同じ作り方。学習データと同じ分布の電流と初期状態
                            （100 ms ごとに U[0, 1] uA から引き直す電流、V(0) ~ U[-100, 100] mV など）
    --currents current_py : src/neuron/current.py の電流（入力 0 で落ち着かせた静止状態から始める）

時間と単位の対応（論文コードの dynamics.py）:
    モデル時間 1 = time_scale [ms] = 100 ms
    入力の刻み Δ = 0.1（モデル時間）= 10 ms。入力は Δ ごとに一定値
    膜電位の状態 v = V / v_scale（v_scale = 100、V は mV）
    電流の単位は uA/cm^2（current.py と同じ）

current.py の電流は dt_ms ごとの値なので、10 ms ごとに平均して階段状の入力にする。
正解の HH も、モデルと同じ階段状の入力で計算する。

実行方法（卒論リポジトリのルートから。論文コード用の .venv を使う）:
    PY=external/flow-learning-l4dc/.venv/bin/python
    $PY -X utf8 experiments/flow_learning/predict_current_py.py <学習結果の .pth> --currents paper
    $PY -X utf8 experiments/flow_learning/predict_current_py.py <学習結果の .pth> --currents current_py
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# src/（neuron）を import できるようにする
sys.path.insert(0, os.path.join(ROOT, 'src'))

from neuron.current import (
    step_current,
    constant_current,
    ramp_current,
    sin_current,
    pulse_train_current,
    random_pulse_current,
    random_timing_pulse_current,
)

# 学習データの電流は 0〜1 uA/cm^2 なので、振幅はその範囲に収める
# 値: t [ms] を受け取り、同じ shape の電流 [uA/cm^2] を返す関数
CURRENTS = {
    'step':                lambda t: step_current(t, amplitude=0.8, t_start=100.0, t_end=700.0),
    'constant':            lambda t: constant_current(t, amplitude=0.5),
    'ramp':                lambda t: ramp_current(t, i_start=0.0, i_end=1.0),
    'sin':                 lambda t: sin_current(t, amplitude=0.5, freq=2.0, offset=0.5),
    'pulse train':         lambda t: pulse_train_current(t, amplitude=1.0, width=50.0, interval=200.0),
    'random pulse':        lambda t: random_pulse_current(t, i_min=0.0, i_max=1.0, interval=20.0, seed=0),
    'random timing pulse': lambda t: random_timing_pulse_current(t, i_min=0.0, i_max=1.0, width=30.0,
                                                                 rate=0.005, seed=0),
}


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('exp_path', type=str, help='論文コード（load_data.py）が保存した学習結果の .pth')
    ap.add_argument('--l4dc_root', type=str, default=os.path.join(ROOT, 'external', 'flow-learning-l4dc'),
                    help='論文コード（2407-l4dc ブランチ）のフォルダ')
    ap.add_argument('--currents', choices=['paper', 'current_py'], default='paper')
    ap.add_argument('--n_paper', type=int, default=2, help='paper のときに描く軌道の本数（論文の図は 2 本）')
    ap.add_argument('--seed', type=int, default=0, help='paper のときの乱数の seed')
    ap.add_argument('--t_end', type=float, default=None,
                    help='シミュレーションの長さ [ms]。省略すると論文の図と同じ（1 細胞 1000 ms、2 細胞 1400 ms）')
    ap.add_argument('--dt_ms', type=float, default=0.01, help='current.py に渡す時刻の刻み [ms]')
    ap.add_argument('--n_eval', type=int, default=3000, help='膜電位を評価する時刻の数')
    ap.add_argument('--out', type=str, default=None,
                    help='保存先。省略すると results/flow_learning/<学習名>_<currents>.png')
    return ap.parse_args()


def load_experiment(exp_path, l4dc_root):
    """
    論文コードの学習結果を読み込む。

    .pth には Experiment オブジェクトがまるごと pickle されていて、学習データの生成器
    （dynamics.py などのクラス）も入っている。読み込むには論文コードを import できる必要がある。

    Returns:
        experiment (flow_model.Experiment): 学習結果。generator.sampler に学習時のサンプラーがある
        model (CausalFlowModel): 検証損失が最小のときの重みを入れたモデル（推論モード）
        pack_model_inputs (function): 論文コードの入力整形関数
    """
    l4dc_root = os.path.abspath(l4dc_root)
    sys.path.insert(0, os.path.join(l4dc_root, 'experiments'))
    sys.path.insert(0, l4dc_root)
    from flow_model import pack_model_inputs

    experiment = torch.load(exp_path, map_location='cpu', weights_only=False)
    model = experiment.load_model()
    model.eval()
    return experiment, model, pack_model_inputs


def to_control_sequence(I, dt_ms, delta_ms, n_control):
    """
    dt_ms 刻みの電流を、delta_ms ごとの平均値の列（モデルの入力 u）に直す。

    Args:
        I (np.ndarray): 電流 [uA/cm^2]。shape (N,)
        dt_ms (float): I の時刻の刻み [ms]
        delta_ms (float): モデルの入力の刻み [ms]（10 ms）
        n_control (int): 作る入力の個数

    Returns:
        np.ndarray: u。shape (n_control, 1)。u[k] は [kΔ, (k+1)Δ) の I の平均
    """
    n_per = int(round(delta_ms / dt_ms))
    u = np.zeros((n_control, 1))
    for k in range(n_control):
        seg = I[k * n_per:(k + 1) * n_per]
        u[k, 0] = seg.mean() if len(seg) > 0 else I[-1]
    return u


def simulate_true(dyn, x0, u, delta, t_eval, method):
    """
    階段状の入力 u で HH を数値積分する（論文コードの TrajectorySampler.get_example と同じ式）。

    Args:
        dyn: 論文コードの Dynamics（HodgkinHuxleyFS / HodgkinHuxleyFFE など）
        x0 (np.ndarray): 初期状態。shape (n,)
        u (np.ndarray): 入力。shape (n_control, 1)
        delta (float): 入力の刻み（モデル時間）
        t_eval (np.ndarray): 状態を出す時刻（モデル時間）。shape (n_eval,)

    Returns:
        np.ndarray: 状態。shape (n_eval, n)
    """
    def f(t, y):
        return dyn(y, u[int(np.floor(t / delta))])

    # max_step を入れないと、静止中に刻みが大きくなって電流の切り替わりを飛び越えることがある
    sol = solve_ivp(f, (0.0, t_eval[-1]), x0, t_eval=t_eval, method=method, max_step=delta)
    return sol.y.T


def resting_state(dyn, x_init, method, t_rest=30.0):
    """
    入力 0 で t_rest（モデル時間。30 = 3 s）だけ動かし、最後の状態を静止状態とする。
    2 細胞モデルの RSA 型は適応電流が遅い（数百 ms）ので、長めに動かす。

    Returns:
        np.ndarray: 静止状態。shape (n,)
    """
    sol = solve_ivp(lambda t, y: dyn(y, np.zeros(1)), (0.0, t_rest), x_init, method=method)
    return sol.y[:, -1]


def main():
    args = parse_args()
    experiment, model, pack_model_inputs = load_experiment(args.exp_path, args.l4dc_root)

    sampler = experiment.generator.sampler
    dyn = sampler._dyn
    method = sampler._ode_method
    delta = sampler._delta                              # モデル時間（0.1）
    ms = dyn.time_scale                                 # モデル時間 1 あたりの ms（100）
    delta_ms = delta * ms                               # 10 ms
    mask = np.array([bool(v) for v in dyn.mask])        # 出力する状態（各細胞の v）
    n_out = int(mask.sum())
    two_cells = n_out > 1

    t_end = args.t_end if args.t_end else (1400.0 if two_cells else 1000.0)
    T = t_end / ms                                      # モデル時間での長さ
    n_control = 1 + int(np.floor(T / delta))
    t_ms = np.arange(0.0, n_control * delta_ms, args.dt_ms)
    t_eval = np.linspace(0.0, T, args.n_eval)           # モデル時間。shape (n_eval,)

    # 初期状態と入力の乱数を固定する（論文コードの reset_rngs は seed を取らないので、生成器を直接差し替える）
    sampler._seq_gen._rng = np.random.default_rng(args.seed)
    sampler.state_generator.rng = np.random.default_rng(args.seed + 1000)

    # 各条件: (名前, 初期状態 x0 shape (n,), 細かい電流 I shape (N,) か None, 入力 u shape (n_control, 1))
    cases = []
    if args.currents == 'current_py':
        x0 = resting_state(dyn, sampler.state_generator.sample(), method)
        print(f'静止状態 V = {dyn.v_scale * x0[mask]} mV')
        for name, make_I in CURRENTS.items():
            I = make_I(t_ms)
            cases.append((name, x0, I, to_control_sequence(I, args.dt_ms, delta_ms, n_control)))
    else:
        for i in range(args.n_paper):
            x0 = sampler.state_generator.sample()
            u = sampler._seq_gen.sample(time_range=(0.0, T), delta=delta)[:n_control]
            cases.append((f'paper {i}', x0, None, u))

    n_rows = n_out + 1
    fig = plt.figure(figsize=(9, (1.6 * n_out + 0.9) * len(cases) + 0.5))
    subfigs = np.atleast_1d(fig.subfigures(len(cases), 1))

    for sf, (name, x0, I, u) in zip(subfigs, cases):
        y_true = simulate_true(dyn, x0, u, delta, t_eval, method)[:, mask]   # shape (n_eval, n_out)

        # 論文コードの入力整形 → 予測。予測は時刻の降順で返るので反転する
        # 最後の時刻で入力の添字が 1 つはみ出さないよう、最後の値を 1 つ足しておく
        u_feed = np.vstack([u, u[-1:]]).astype(np.float32)
        x0_feed, _, rnn_input, deltas = pack_model_inputs(x0, t_eval, u_feed, delta)
        y_pred = np.flip(experiment.predict(model, x0_feed, rnn_input, deltas), 0)  # shape (n_eval, n_out)

        V_true = dyn.v_scale * y_true
        V_pred = dyn.v_scale * y_pred
        rmse = np.sqrt(np.mean((V_true - V_pred) ** 2, axis=0))             # shape (n_out,)
        v0 = dyn.v_scale * x0[mask]
        print(f'{name:>20s}: V(0) = {np.round(v0, 1)} mV, RMSE = {np.round(rmse, 2)} mV')

        axes = sf.subplots(n_rows, 1, sharex=True, height_ratios=[2] * n_out + [1])
        t_plot = t_eval * ms
        for k in range(n_out):
            ax = axes[k]
            ax.plot(t_plot, V_pred[:, k], c='orange', lw=1.0, label='Prediction')
            ax.plot(t_plot, V_true[:, k], 'b--', lw=1.0, label='True')
            ax.set_ylabel(f'$V_{k + 1}$ [mV]' if two_cells else '$V$ [mV]')
            ax.text(0.99, 0.95, f'RMSE {rmse[k]:.1f} mV', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8)
        ax = axes[-1]
        if I is not None:
            ax.plot(t_ms, I, c='gray', lw=0.8)
        ax.step(np.arange(n_control) * delta_ms, u[:, 0], where='post')
        ax.set_ylabel('$u$ [$\\mu$A]')
        ax.set_xlabel('$t$ [ms]')
        sf.suptitle(f'{name}', fontsize=10, x=0.02, ha='left')
        sf.subplots_adjust(hspace=0)

    axes_first = subfigs[0].axes[0]
    axes_first.legend(loc='upper center', fontsize=8, ncol=2)

    stem = Path(args.exp_path).parent.name
    out = Path(args.out or os.path.join(ROOT, 'results', 'flow_learning', f'{stem}_{args.currents}.png'))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'保存: {os.path.normpath(out)}')


if __name__ == '__main__':
    main()
