"""
flow-learning（Learning flow functions of spiking systems）の学習済みモデルに
src/neuron/current.py の電流を入れ、正解の HH FS と並べてグラフにする。

元の experiments/interactive_test.py は semble の乱数電流しか使えないので、
電流の作り方だけを current.py に差し替えたもの。

時間と単位の対応（semble の HodgkinHuxleyFS）:
    モデル時間 1 = 100 ms（time_scale = 100）
    入力の刻み Δ = 0.1（モデル時間）= 10 ms。入力は Δ ごとに一定値
    膜電位の状態 v = V / 100（V は mV）
    電流の単位は uA/cm^2（current.py と同じ）

current.py の電流は dt_ms ごとの値なので、10 ms ごとに平均して階段状の入力にする。
正解の HH FS も、モデルと同じ階段状の入力で計算する。

実行方法（リポジトリのルートから。flow-learning の .venv を使う）:
    external/flow-learning/.venv/Scripts/python -X utf8 experiments/flow_learning/predict_current_py.py <モデルのフォルダ>
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from scipy.integrate import solve_ivp

# src/（neuron）を import できるようにする
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from flumen import CausalFlowModel
from flumen.utils import pack_model_inputs
from semble import make_trajectory_sampler

from neuron.current import (
    step_current,
    constant_current,
    ramp_current,
    sin_current,
    pulse_train_current,
    random_pulse_current,
    random_timing_pulse_current,
)

MS_PER_MODEL_TIME = 100.0  # モデル時間 1 が何 ms か（HodgkinHuxleyFS.time_scale）

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
    ap.add_argument('model_dir', type=str, help='state_dict.pth と metadata.yaml があるフォルダ')
    ap.add_argument('--t_end', type=float, default=1000.0, help='シミュレーションの長さ [ms]')
    ap.add_argument('--dt_ms', type=float, default=0.01, help='current.py に渡す時刻の刻み [ms]')
    ap.add_argument('--n_eval', type=int, default=3000, help='膜電位を評価する時刻の数')
    ap.add_argument('--out', type=str, default='results/flow_learning/predict_current_py.png')
    return ap.parse_args()


def load_model(model_dir):
    """
    学習済みモデルと、学習時の設定を読み込む。

    Returns:
        model (CausalFlowModel): 推論モードのモデル
        metadata (dict): metadata.yaml の中身（data_settings に dynamics と control_delta がある）
    """
    model_dir = Path(model_dir)
    with open(model_dir / 'metadata.yaml', 'r') as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    state_dict = torch.load(model_dir / 'state_dict.pth', weights_only=False, map_location='cpu')

    model = CausalFlowModel(**metadata['args'])
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


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
    階段状の入力 u で HH FS を数値積分する（semble の TrajectorySampler.get_example と同じ）。

    Args:
        dyn: semble の Dynamics（HodgkinHuxleyFS）
        x0 (np.ndarray): 初期状態 (v, n, m, h)。shape (4,)
        u (np.ndarray): 入力。shape (n_control, 1)
        delta (float): 入力の刻み（モデル時間）
        t_eval (np.ndarray): 状態を出す時刻（モデル時間）。shape (n_eval,)

    Returns:
        np.ndarray: 状態。shape (n_eval, 4)
    """
    def f(t, y):
        return dyn(y, u[int(np.floor(t / delta))])

    # max_step を入れないと、静止中に刻みが大きくなって電流の切り替わりを飛び越えることがある
    sol = solve_ivp(f, (0.0, t_eval[-1]), x0, t_eval=t_eval, method=method, max_step=delta)
    return sol.y.T


def resting_state(dyn, method, t_rest=10.0):
    """
    入力 0 で t_rest（モデル時間）だけ動かし、最後の状態を静止状態とする。

    Returns:
        np.ndarray: 静止状態 (v, n, m, h)。shape (4,)
    """
    x_init = np.array([-0.7, 0.0, 0.0, 1.0])
    sol = solve_ivp(lambda t, y: dyn(y, np.zeros(1)), (0.0, t_rest), x_init, method=method)
    return sol.y[:, -1]


def main():
    args = parse_args()
    model, metadata = load_model(args.model_dir)

    sampler = make_trajectory_sampler(metadata['data_settings'])
    dyn = sampler._dyn
    method = sampler._ode_method
    delta = sampler._delta                          # モデル時間（0.1）
    delta_ms = delta * MS_PER_MODEL_TIME            # 10 ms
    mask = np.array([bool(v) for v in dyn.mask])    # 出力する状態（v だけ）

    T = args.t_end / MS_PER_MODEL_TIME              # モデル時間での長さ
    n_control = 1 + int(np.floor(T / delta))
    t_ms = np.arange(0.0, n_control * delta_ms, args.dt_ms)
    t_eval = np.linspace(0.0, T, args.n_eval)       # モデル時間。shape (n_eval,)

    x0 = resting_state(dyn, method)
    print(f'静止状態 x0 = {x0}（V = {100 * x0[0]:.2f} mV）')

    fig, axes = plt.subplots(len(CURRENTS), 2, figsize=(13, 2.2 * len(CURRENTS)),
                             sharex=True, gridspec_kw={'width_ratios': [3, 1]})

    for row, (name, make_I) in enumerate(CURRENTS.items()):
        I = make_I(t_ms)                                           # shape (N,)
        u = to_control_sequence(I, args.dt_ms, delta_ms, n_control)  # shape (n_control, 1)

        y_true = simulate_true(dyn, x0, u, delta, t_eval, method)[:, mask]  # shape (n_eval, 1)

        with torch.no_grad():
            x0_feed, u_feed, skips, tau = pack_model_inputs(x0, t_eval.reshape(-1, 1), u, delta)
            y_pred = model.forward_trajectory(x0_feed, u_feed, skips, tau).numpy().reshape(-1, 1)

        V_true = 100.0 * y_true[:, 0]
        V_pred = 100.0 * y_pred[:, 0]
        rmse = np.sqrt(np.mean((V_true - V_pred) ** 2))
        print(f'{name:>20s}: RMSE = {rmse:.2f} mV')

        t_plot = t_eval * MS_PER_MODEL_TIME
        ax = axes[row, 0]
        ax.plot(t_plot, V_true, 'b--', lw=1.0, label='HH FS（正解）')
        ax.plot(t_plot, V_pred, c='orange', lw=1.0, label='flow-learning')
        ax.set_ylabel(f'{name}\nV [mV]', fontsize=9)
        ax.set_title(f'RMSE = {rmse:.2f} mV', fontsize=9, loc='right')
        ax.grid(True, ls='--', alpha=0.5)

        ax = axes[row, 1]
        ax.plot(t_ms, I, c='gray', lw=0.8, label='current.py')
        ax.step(np.arange(n_control) * delta_ms, u[:, 0], where='post', c='k', lw=1.0,
                label='入力 u（10 ms 平均）')
        ax.set_ylabel('I [uA/cm²]', fontsize=9)
        ax.grid(True, ls='--', alpha=0.5)

    axes[0, 0].legend(fontsize=8, loc='upper left')
    axes[0, 1].legend(fontsize=8, loc='upper left')
    axes[-1, 0].set_xlabel('Time [ms]')
    axes[-1, 1].set_xlabel('Time [ms]')
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'保存: {out}')


if __name__ == '__main__':
    # Windows と Linux サーバーのどちらでも日本語が出るように、候補を並べておく
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'Noto Sans CJK JP', 'IPAexGothic', 'sans-serif']
    main()
