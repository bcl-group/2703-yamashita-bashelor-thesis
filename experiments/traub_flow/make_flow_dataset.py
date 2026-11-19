"""
flow function（Learning flow functions of spiking systems, L4DC 2024）の論文コードで Traub モデル
（19 コンパートメント）を学習するための教師データを作る。

論文コード（flow-learning の 2407-l4dc ブランチ）は書き換えない。HH の hhfs_flow.py と同じ形式の
.pth（generate_data.ExperimentData を torch.save したもの）を書き出すので、学習は HH と同じく
experiments/load_data.py にこのファイルを渡せばよい。

HH（hhfs_flow.py）との違い:
    状態     : 19 区画 × 11 変数 = 209 次元。並びは変数ごと（V の 19 個、χ の 19 個、m の 19 個、…）
               χ は log1p(χ) にしてから、全 209 次元を学習データの平均・標準偏差で標準化する。
               学習のときに --whiten_data は付けない（ここで標準化済み）
    出力     : 既定は細胞体の V だけ（--outputs soma）。--outputs all_v で 19 区画の V
    初期状態 : 一様乱数ではなく、静止状態から warmup_ms だけランダムな電流で助走した後の状態
    入力電流 : 細胞体への電流 [nA]。Δ = 10 ms ごとの区分定数で、値は U[i_min, i_max]、
               同じ値を hold_min〜hold_max ms（Δ の倍数）保つ
    数値解法 : solve_ivp（BDF）ではなく、src/neuron/traub.py の陽的オイラー法（dt = 0.01 ms）を
               全軌道まとめて回す（他のパターンの教師と同じモデルにするため）
    棄却抽出 : 軌道ごとの min/max ではなく、固定の電位の範囲で正規化した細胞体の V で受理確率を決める。
               受理確率 p = p_max · (p_floor + (1 − p_floor) · clip((V − v_lo)/(v_hi − v_lo), 0, 1))
               計算しながら 1 ステップずつ判定できるので、全状態の密な時系列を持たなくてよい。
               細胞体の V の極大（peak_min mV 以上）と t = 0 は必ず残す（論文の find_peaks に相当）

時間の単位は論文コードの HH と同じにする:
    モデル時間 1 = time_scale [ms] = 100 ms、Δ = 10 ms = 0.1（モデル時間）
    max_seq_len = 20 は 200 ms の窓になる

書き出すファイル:
    <l4dc>/outputs/<name>.pth          : 論文コードが読む教師データ（train/val/test = 60/20/20 % が既定）
    results/traub_flow/<name>.npz      : 条件・標準化の値・確認用の密な波形（最初の数軌道）
    results/traub_flow/<name>.png      : 確認用の図（細胞体の V、残したサンプル、電流）

使い方（卒論リポジトリのルートで。論文コード用の .venv を使う）:
    PY=external/flow-learning-l4dc/.venv/bin/python
    $PY -X utf8 experiments/traub_flow/make_flow_dataset.py traub_fs.pth
    $PY -X utf8 experiments/traub_flow/make_flow_dataset.py traub_try.pth --n_trajectories 10 --time_horizon_ms 300   # 動作確認

学習（HH の run_paper.sh と同じ引数。--control_rnn_size は HH の 24 より大きくする）:
    cd external/flow-learning-l4dc
    PYTHONPATH=. $PY ../../experiments/flow_learning/l4dc_compat.py experiments/load_data.py \
        outputs/traub_fs.pth --control_rnn_size 64 --encoder_depth 3 --decoder_depth 3 \
        --lr 1e-3 --sched_patience 5 --sched_factor 10 --es_patience 15 \
        --max_seq_len 20 --samples_per_state 5 --experiment_id traub_fs

注意:
    .pth の generator は SimpleNamespace（条件と標準化の値だけ）で、論文コードの TrajectorySampler では
    ない。そのため generator.sampler を使う論文の作図スクリプト（l4dc_plot_predictions.py など）は
    そのままでは動かない。予測の図は別に作る。
"""
import json
import os
import sys
import time
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from neuron.traub import step, initial_state, compartment_conductance, NC, SOMA, area

STATE_NAMES = ['V', 'xi', 'm', 'h', 's', 'r', 'n', 'a', 'b', 'c', 'q']
N_STATE = len(STATE_NAMES) * NC  # 209
# 1 nA = 1e-3 uA。traub.step() は電流密度 [uA/cm^2] を受け取るので細胞体の面積で割る
NA_TO_DENSITY = 1e-3 / area[SOMA]
# 確認用に密な波形を保存する軌道の数
N_PREVIEW = 4


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('name', help='出力ファイル名（例: traub_fs.pth）。<l4dc>/outputs/ に書く')
    ap.add_argument('--l4dc', default=os.path.join(ROOT, 'external', 'flow-learning-l4dc'),
                    help='論文コード（2407-l4dc ブランチ）のフォルダ')
    ap.add_argument('--n_trajectories', type=int, default=200)
    ap.add_argument('--time_horizon_ms', type=float, default=2000.0, help='1 軌道の長さ [ms]')
    ap.add_argument('--warmup_ms', type=float, default=500.0, help='記録前の助走 [ms]')
    ap.add_argument('--dt', type=float, default=0.01, help='オイラー法の刻み [ms]')
    ap.add_argument('--control_delta_ms', type=float, default=10.0, help='入力の刻み Δ [ms]')
    ap.add_argument('--time_scale', type=float, default=100.0, help='モデル時間 1 に当たる ms')
    ap.add_argument('--i_min', type=float, default=-0.4, help='電流の最小値 [nA]')
    ap.add_argument('--i_max', type=float, default=1.5, help='電流の最大値 [nA]')
    ap.add_argument('--hold_min_ms', type=float, default=10.0, help='同じ値を保つ最短時間 [ms]')
    ap.add_argument('--hold_max_ms', type=float, default=500.0, help='同じ値を保つ最長時間 [ms]')
    ap.add_argument('--p_max', type=float, default=0.1, help='受理確率の最大値（1 ステップあたり）')
    ap.add_argument('--p_floor', type=float, default=0.02, help='静止区間の受理確率の割合（p_max に対して）')
    ap.add_argument('--v_lo', type=float, default=-65.0, help='受理確率を正規化する電位の下限 [mV]')
    ap.add_argument('--v_hi', type=float, default=0.0, help='受理確率を正規化する電位の上限 [mV]')
    ap.add_argument('--peak_min', type=float, default=-40.0, help='必ず残す極大の電位の下限 [mV]')
    ap.add_argument('--outputs', choices=['soma', 'all_v'], default='soma')
    ap.add_argument('--batch', type=int, default=100, help='同時に計算する軌道の数')
    ap.add_argument('--data_split', nargs=2, type=int, default=[20, 20],
                    help='検証・テストに使う軌道の割合 [%%]（残りが学習）')
    ap.add_argument('--samples_per_state', type=int, default=5,
                    help='学習時に使う値（メモリの見積もりだけに使う）')
    ap.add_argument('--seed', type=int, default=2026)
    return ap.parse_args()


def make_control(rng, n_ctrl, args):
    """
    区分定数の電流列を作る。

    Returns:
        np.ndarray: 電流 [nA]。shape (n_ctrl,)。要素 k が [kΔ, (k+1)Δ) の電流
    """
    hold_lo = max(1, int(round(args.hold_min_ms / args.control_delta_ms)))
    hold_hi = max(hold_lo, int(round(args.hold_max_ms / args.control_delta_ms)))
    u = np.empty(n_ctrl)
    k = 0
    while k < n_ctrl:
        h = int(rng.integers(hold_lo, hold_hi + 1))
        u[k:k + h] = rng.uniform(args.i_min, args.i_max)
        k += h
    return u


def to_inj(I_nA):
    # (B,) [nA] -> (B, NC) [uA/cm^2]。細胞体以外は 0
    inj = np.zeros((len(I_nA), NC))
    inj[:, SOMA] = I_nA * NA_TO_DENSITY
    return inj


def stack_state(state, rows):
    """
    状態の dict から、指定した軌道の 209 次元ベクトルを作る。χ は log1p をとる。

    Args:
        state (dict[str, np.ndarray]): 値は shape (B, NC)。
        rows (np.ndarray): 取り出す軌道の番号。shape (R,)。

    Returns:
        np.ndarray: shape (R, 209)、float32。並びは STATE_NAMES の順に 19 区画ずつ
    """
    cols = []
    for name in STATE_NAMES:
        x = state[name][rows]
        if name == 'xi':
            x = np.log1p(np.maximum(x, 0.0))
        cols.append(x)
    return np.concatenate(cols, axis=1).astype(np.float32)


def accept_prob(V_soma, args):
    """細胞体の V [mV]（shape (B,)）から 1 ステップの受理確率（shape (B,)）を求める。"""
    v = np.clip((V_soma - args.v_lo) / (args.v_hi - args.v_lo), 0.0, 1.0)
    return args.p_max * (args.p_floor + (1.0 - args.p_floor) * v)


def run_batch(rng, U_warm, U, args, g_comp):
    """
    B 本の軌道を助走させてから記録し、棄却抽出で残したサンプルを返す。

    Args:
        U_warm (np.ndarray): 助走の電流 [nA]。shape (B, n_ctrl_warm)。
        U (np.ndarray): 記録区間の電流 [nA]。shape (B, n_ctrl)。

    Returns:
        tuple:
            samples (list[tuple[np.ndarray, np.ndarray]]): 軌道ごとの (時刻のステップ番号 (n,), 状態 (n, 209))
            V_dense (np.ndarray): 細胞体の V [mV]。shape (B, N+1)、float32（確認用）
    """
    B = U.shape[0]
    dt = args.dt
    per_ctrl = int(round(args.control_delta_ms / dt))
    N = int(round(args.time_horizon_ms / dt))
    all_rows = np.arange(B)

    # 1. 助走
    state = initial_state((B,))
    for k in range(int(round(args.warmup_ms / dt))):
        state = step(state, to_inj(U_warm[:, k // per_ctrl]), dt, g_comp)

    # 2. 記録。ステップ k の状態（更新前）を時刻 k·dt のサンプルとして判定する
    idx_b, idx_k, rows = [], [], []
    V_dense = np.empty((B, N + 1), dtype=np.float32)
    prev, V_prev2 = None, None
    kept_prev = np.zeros(B, dtype=bool)
    for k in range(N + 1):
        V = state['V'][:, SOMA]
        V_dense[:, k] = V

        # 1 つ前のステップが極大なら、まだ残していない軌道だけ追加する
        if V_prev2 is not None:
            V_prev = prev['V'][:, SOMA]
            peak = (V_prev > V_prev2) & (V_prev >= V) & (V_prev >= args.peak_min) & ~kept_prev
            if peak.any():
                b = all_rows[peak]
                idx_b.append(b)
                idx_k.append(np.full(len(b), k - 1))
                rows.append(stack_state(prev, b))

        keep = rng.random(B) < accept_prob(V, args)
        if k == 0:
            keep[:] = True
        if keep.any():
            b = all_rows[keep]
            idx_b.append(b)
            idx_k.append(np.full(len(b), k))
            rows.append(stack_state(state, b))

        V_prev2 = prev['V'][:, SOMA] if prev is not None else None
        prev, kept_prev = state, keep
        if k < N:
            state = step(state, to_inj(U[:, k // per_ctrl]), dt, g_comp)

    idx_b = np.concatenate(idx_b)
    idx_k = np.concatenate(idx_k)
    rows = np.concatenate(rows)
    order = np.lexsort((idx_k, idx_b))  # 軌道ごと・時刻順に並べる
    idx_b, idx_k, rows = idx_b[order], idx_k[order], rows[order]
    samples = []
    for j in range(B):
        sel = idx_b == j
        samples.append((idx_k[sel], rows[sel]))
    return samples, V_dense


def count_spikes(V, thr=-20.0):
    """細胞体の V（shape (N,)）で thr を下から上に横切った回数。"""
    return int(np.sum((V[1:] >= thr) & (V[:-1] < thr)))


def main():
    args = parse_args()
    sys.path.insert(0, args.l4dc)
    sys.path.insert(0, os.path.join(args.l4dc, 'experiments'))
    import torch
    from flow_model import RawTrajectoryDataset
    from generate_data import ExperimentData  # pickle で 'generate_data.ExperimentData' として保存される

    rng = np.random.default_rng(args.seed)
    g_comp = compartment_conductance()
    n_traj = args.n_trajectories
    n_ctrl = 1 + int(np.floor(args.time_horizon_ms / args.control_delta_ms))
    n_ctrl_warm = 1 + int(np.ceil(args.warmup_ms / args.control_delta_ms))
    U_warm = np.stack([make_control(rng, n_ctrl_warm, args) for _ in range(n_traj)])
    U = np.stack([make_control(rng, n_ctrl, args) for _ in range(n_traj)])  # (n_traj, n_ctrl) [nA]

    # 1. 全軌道をバッチごとに計算する
    start = time.time()
    samples, spikes, V_preview = [], [], None
    for s in range(0, n_traj, args.batch):
        e = min(s + args.batch, n_traj)
        smp, V_dense = run_batch(rng, U_warm[s:e], U[s:e], args, g_comp)
        samples += smp
        spikes += [count_spikes(v) for v in V_dense]
        if V_preview is None:
            V_preview = V_dense[:N_PREVIEW]
        print('trajectories {}-{} done: {:.0f} s'.format(s, e - 1, time.time() - start), flush=True)

    # 2. 学習・検証・テストに軌道単位で分ける
    n_val = int(n_traj * args.data_split[0] / 100)
    n_test = int(n_traj * args.data_split[1] / 100)
    n_train = n_traj - n_val - n_test
    splits = {'train': range(0, n_train), 'val': range(n_train, n_train + n_val),
              'test': range(n_train + n_val, n_traj)}

    # 3. 標準化の値を学習データだけで求める
    train_rows = np.concatenate([samples[i][1] for i in splits['train']]).astype(np.float64)
    mean = train_rows.mean(axis=0)
    std = np.maximum(train_rows.std(axis=0), 1e-6)

    if args.outputs == 'soma':
        out_idx = [SOMA]  # V は先頭の 19 個なので、細胞体の V は 8 番目
    else:
        out_idx = list(range(NC))
    mask = tuple(int(i in out_idx) for i in range(N_STATE))
    delta = args.control_delta_ms / args.time_scale

    generator = SimpleNamespace(
        noise_std=0.0,
        time_horizon=args.time_horizon_ms / args.time_scale,
        control_delta=delta,
        cfg=vars(args),
        state_names=['{}_{}'.format(n, c) for n in STATE_NAMES for c in range(NC)],
        state_mean=mean, state_std=std, output_index=out_idx,
        note='Traub 19 区画。状態は χ を log1p してから state_mean/state_std で標準化。'
             '時刻はモデル時間（1 = time_scale ms）、電流は細胞体への nA',
    )

    def raw(split):
        data = []
        for i in splits[split]:
            k, rows = samples[i]
            x = ((rows - mean) / std).astype(np.float32)
            data.append({
                'init_state': x[0],
                'time': (k * args.dt / args.time_scale).reshape(-1, 1),
                'state': x,
                'control': U[i].reshape(-1, 1).astype(np.float32),
            })
        return RawTrajectoryDataset(data, N_STATE, 1, len(out_idx), delta=delta,
                                    output_mask=mask, noise_std=0.0, generator=generator)

    exp_data = ExperimentData.__new__(ExperimentData)
    exp_data.generator = generator
    exp_data.train_data, exp_data.val_data, exp_data.test_data = raw('train'), raw('val'), raw('test')

    out_dir = os.path.join(args.l4dc, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_pth = os.path.join(out_dir, args.name)
    torch.save(exp_data, out_pth)
    print('saved:', os.path.normpath(out_pth), '({:.0f} MB)'.format(os.path.getsize(out_pth) / 2**20))

    # 4. 要約。窓付き損失のデータの大きさは、1 サンプルあたり (209 + 出力 + 2·n_ctrl) 個の float32
    n_kept = np.array([len(s[0]) for s in samples])
    spikes = np.array(spikes)
    n_train_kept = n_kept[list(splits['train'])].sum()
    mem = n_train_kept * args.samples_per_state * (N_STATE + len(out_idx) + 2 * n_ctrl) * 4
    print('残したサンプル数/軌道: 平均 {:.0f}, 最小 {}, 最大 {}'.format(n_kept.mean(), n_kept.min(), n_kept.max()))
    print('細胞体のスパイク数/軌道: 平均 {:.1f}, 0 本の軌道 {} / {}'.format(
        spikes.mean(), int((spikes == 0).sum()), n_traj))
    print('学習時の窓付きデータの見積もり: {:.1f} GB（samples_per_state = {}）'.format(
        mem / 2**30, args.samples_per_state))

    res_dir = os.path.join(ROOT, 'results', 'traub_flow')
    os.makedirs(res_dir, exist_ok=True)
    stem = os.path.join(res_dir, os.path.splitext(args.name)[0])
    np.savez(stem + '.npz', cfg=json.dumps(vars(args)), state_mean=mean, state_std=std,
             state_names=np.array(generator.state_names), n_kept=n_kept, spikes=spikes,
             U_preview=U[:N_PREVIEW], V_preview=V_preview)

    # 5. 確認用の図：細胞体の V（密な波形）、残したサンプル、電流
    t = np.arange(V_preview.shape[1]) * args.dt
    fig, axes = plt.subplots(len(V_preview), 1, figsize=(12, 2.4 * len(V_preview)), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        k, rows = samples[j]
        ax.plot(t, V_preview[j], color='#2a78d6', lw=0.7, label='細胞体の V')
        ax.plot(k * args.dt, rows[:, SOMA], '.', color='#e8590c', ms=2,
                label='残したサンプル（{} 点）'.format(len(k)))
        ax.set_ylabel('V [mV]')
        ax2 = ax.twinx()
        t_u = np.arange(n_ctrl) * args.control_delta_ms
        ax2.step(t_u, U[j], where='post', color='#888780', lw=0.8)
        ax2.set_ylabel('I [nA]', color='#888780')
        ax2.set_ylim(args.i_min - 0.1, args.i_max * 4)
        ax.set_title('軌道 {}（スパイク {} 本）'.format(j, spikes[j]), fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc='upper left')
    np.atleast_1d(axes)[-1].set_xlabel('時刻 [ms]（助走の後）')
    fig.tight_layout()
    fig.savefig(stem + '.png', dpi=110)
    print('saved:', os.path.normpath(stem + '.png'))


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'sans-serif']
    main()
