"""
論文コード（flow-learning の 2407-l4dc ブランチ）が作った教師データ（outputs/hhfs.pth）から、
学習に使った入力電流 u(t) と膜電位 V(t) をグラフにする。

教師データの中身（論文コードの ExperimentData。train_data / val_data / test_data の 3 つ）:
    d = data.train_data                      # RawTrajectoryDataset
    d.init_state  : Tensor (軌道数, 4)        初期状態 (V/100, n, m, h)
    d.time[i]     : Tensor (K_i, 1)           サンプル時刻（モデル時間。1 = 100 ms）
    d.state[i]    : Tensor (K_i, 4)           その時刻の状態。0 列目が V/100
    d.control_seq[i] : Tensor (51, 1)         入力電流 ω_0, ω_1, ...（Δ = 0.1 = 10 ms ごと）[µA]
K_i は棄却サンプリング後に残ったサンプル数で、軌道ごとに違う。サンプルはスパイクの近くに密に残る。
そのため膜電位は「点のつながり」で描き、点も重ねて見せる。

出力:
    <out>.png       : 軌道を n 本、膜電位（上）と入力電流（下）の組で並べた図
    <out>_dist.png  : 学習用の全軌道について、電流値・初期膜電位・残ったサンプル数の分布
    標準出力        : 分割ごとの軌道数と、1 軌道あたりに残ったサンプル数

実行方法（卒論リポジトリのルートから。論文コード用の .venv を使う）:
    PY=external/flow-learning-l4dc/.venv/bin/python
    $PY -X utf8 experiments/flow_learning/plot_training_data.py \
        external/flow-learning-l4dc/outputs/hhfs.pth --out results/flow_learning/hhfs_training_data
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
L4 = os.path.join(ROOT, 'external', 'flow-learning-l4dc')

# 論文コードの単位（experiments/dynamics.py の HodgkinHuxleyFS）
TIME_SCALE = 100.   # モデル時間 1 = 100 ms
V_SCALE = 100.      # 状態の 0 列目 = V / 100


def load_trajectories(path, split):
    """教師データを読み、numpy の配列のリストにして返す。

    返り値: list of dict。各要素が 1 本の軌道で、
        't_ms'  : (K_i,)  サンプル時刻 [ms]
        'V'     : (K_i,)  膜電位 [mV]
        'u'     : (M,)    入力電流 ω_0..ω_{M-1} [µA]
        'delta_ms' : float  入力の刻み Δ [ms]
    """
    # 保存されたオブジェクトは論文コードのクラス（generate_data.ExperimentData など）なので、
    # 読み込むには論文コードのフォルダを import できるようにしておく必要がある
    sys.path.insert(0, L4)
    sys.path.insert(0, os.path.join(L4, 'experiments'))
    import torch

    data = torch.load(path, map_location='cpu', weights_only=False)
    d = getattr(data, f'{split}_data')
    delta_ms = float(d.delta) * TIME_SCALE

    trajs = []
    for t, y, u in zip(d.time, d.state, d.control_seq):
        trajs.append({
            't_ms': t.numpy().ravel() * TIME_SCALE,
            'V': y.numpy()[:, 0] * V_SCALE,
            'u': u.numpy().ravel(),
            'delta_ms': delta_ms,
        })
    counts = {s: len(getattr(data, f'{s}_data').time) for s in ('train', 'val', 'test')}
    return trajs, counts


def step_xy(u, delta_ms, t_end):
    """区分定数の入力 ω_k（区間 [kΔ, (k+1)Δ)）を、階段の折れ線の座標にする。"""
    edges = np.arange(len(u) + 1) * delta_ms        # (M+1,) 区間の端
    x = np.repeat(edges, 2)[1:-1]                    # (2M,)
    y = np.repeat(u, 2)                              # (2M,)
    return np.minimum(x, t_end), y                   # 軌道の終わり（500 ms）より先は描かない


def plot_trajectories(trajs, idxs, out_png, title):
    n = len(idxs)
    fig, axes = plt.subplots(2 * n, 1, figsize=(9, 2.4 * n + 0.6), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1] * n})
    for row, i in enumerate(idxs):
        tr = trajs[i]
        t_end = tr['t_ms'][-1]
        axV, axU = axes[2 * row], axes[2 * row + 1]
        axV.plot(tr['t_ms'], tr['V'], lw=0.8, color='tab:blue')
        axV.plot(tr['t_ms'], tr['V'], '.', ms=1.5, color='tab:blue', alpha=0.4)
        axV.set_ylabel('V [mV]')
        axV.set_title(f'trajectory #{i}  ({len(tr["t_ms"])} samples)', fontsize=9, loc='left')
        x, y = step_xy(tr['u'], tr['delta_ms'], t_end)
        axU.plot(x, y, color='tab:orange', lw=1.2)
        axU.set_ylim(-0.05, 1.05)
        axU.set_ylabel('u [µA]')
    axes[-1].set_xlabel('t [ms]')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'saved: {out_png}')


def plot_distributions(trajs, out_png, title):
    # 電流値は 10 個ずつ同じ値が並ぶので、各段の先頭（100 ms ごと）だけを数える
    period = int(round(100. / trajs[0]['delta_ms']))
    omegas = np.concatenate([tr['u'][::period] for tr in trajs])
    v0 = np.array([tr['V'][0] for tr in trajs])
    n_samples = np.array([len(tr['t_ms']) for tr in trajs])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(omegas, bins=20, range=(0, 1), color='tab:orange')
    axes[0].set_xlabel('current value ω [µA] (one per 100 ms)')
    axes[1].hist(v0, bins=20, range=(-100, 100), color='tab:blue')
    axes[1].set_xlabel('initial V(0) [mV]')
    axes[2].hist(n_samples, bins=20, color='tab:gray')
    axes[2].set_xlabel('samples kept per trajectory')
    for ax in axes:
        ax.set_ylabel('count')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'saved: {out_png}')


def main():
    ap = ArgumentParser()
    ap.add_argument('data', help='論文コードの教師データ（outputs/hhfs.pth）')
    ap.add_argument('--out', required=True, help='出力ファイル名（拡張子なし）')
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    ap.add_argument('--n_plots', type=int, default=4, help='並べる軌道の本数')
    ap.add_argument('--seed', type=int, default=0, help='どの軌道を選ぶかの乱数シード')
    args = ap.parse_args()

    trajs, counts = load_trajectories(args.data, args.split)

    n_samples = np.array([len(tr['t_ms']) for tr in trajs])
    print(f'軌道数: train {counts["train"]}, val {counts["val"]}, test {counts["test"]}')
    print(f'{args.split} の 1 軌道あたりのサンプル数（棄却サンプリング後）: '
          f'min {n_samples.min()}, median {int(np.median(n_samples))}, max {n_samples.max()}')

    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(trajs), size=min(args.n_plots, len(trajs)), replace=False))
    name = os.path.basename(args.data)
    plot_trajectories(trajs, idxs, args.out + '.png', f'{name} ({args.split})')
    plot_distributions(trajs, args.out + '_dist.png',
                       f'{name} ({args.split}, {len(trajs)} trajectories)')


if __name__ == '__main__':
    main()
