"""
論文 図3 と同じ「学習した全モデルの損失の分布」を箱ひげ図にし、論文の値と並べる。

入力: 論文コードの experiments/scripts/compute_stats.py が書く CSV（1 行 = 1 モデル）。
      使う列は train_mse（窓付き学習損失）・val_mse（検証損失）・test_mse（テスト損失）。
      どれも学習時に記録した最良エポックの値（Experiment の train/val/test_loss_best）。
出力: 箱ひげ図の PNG と、標準出力への表（最小・中央値・最大と論文の中央値）。

論文の作図スクリプト l4dc_loss_plot.py は pandas・seaborn・LaTeX を使い、図を保存しない（plt.show のみ）。
そのため同じ図を matplotlib だけで描いて保存する。

論文の値は、PDF p.9 の図3 を目盛りから目で読んだ中央値で、有効数字は 1〜2 桁しかない。
    fs（§4.1, 図3a）: train ≈ 1.3e-2, val ≈ 5.8e-2, test ≈ 6.2e-2

使い方:
    python plot_loss_box.py <stats.csv> --out <出力.png> [--paper fs]
"""
import csv
import sys
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

# (CSV の列名, 図のラベル)
COLUMNS = [('train_mse', 'Train (windowed)'), ('val_mse', 'Validation'), ('test_mse', 'Test')]

# 論文 図3 の目視読み取り（中央値）
PAPER = {
    'fs': {'train_mse': 1.3e-2, 'val_mse': 5.8e-2, 'test_mse': 6.2e-2},
}


def main():
    ap = ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--paper', default=None, help='論文の値を重ねる実験（fs のみ。他は重ねない）')
    args = ap.parse_args()

    with open(args.csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f'{args.csv} にモデルの行がない')

    # data[列名]: shape (モデル数,) の float64 配列
    data = {c: np.array([float(r[c]) for r in rows]) for c, _ in COLUMNS}
    paper = PAPER.get(args.paper, {})

    print(f'モデル数: {len(rows)}')
    print(f'{"":18}{"min":>10}{"median":>10}{"max":>10}{"論文(中央値)":>14}')
    for c, label in COLUMNS:
        v = data[c]
        ref = f'{paper[c]:.1e}' if c in paper else '-'
        print(f'{label:18}{v.min():10.2e}{np.median(v):10.2e}{v.max():10.2e}{ref:>14}')

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([data[c] for c, _ in COLUMNS])
    ax.set_xticks(range(1, len(COLUMNS) + 1))
    ax.set_xticklabels([label for _, label in COLUMNS])
    if paper:
        ax.scatter(range(1, len(COLUMNS) + 1), [paper[c] for c, _ in COLUMNS], marker='x',
                   color='tab:red', zorder=3, label='paper median (read from Fig. 3)')
        ax.legend(fontsize=8, loc='lower right')
    ax.set_yscale('log')
    ax.set_ylabel('Log loss')
    ax.set_title(f'{len(rows)} models')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
