"""
最終テスト（54 本）の Γ を、電流のグループごとの小パネル（6 枚、横一列）にする（発表スライド用）。

    各パネル: 1 つのグループ。横にモデルを並べ、棒 = 学習 seed の平均、誤差棒 = 標準偏差、点 = seed ごとの値
    seed ごとの値: そのグループの電流で Γ を平均したもの（Γ が計算できない電流は除く）

図の約束（研究室の図表の掟と、見やすさのため）:
    - 縦軸は全パネル共通。上は 1（完全一致）で固定し、下は 0 と全データの最小値の小さいほうまで広げる
      （Γ は偶然より悪いと負になるので、0 で切ると見切れる）。0 の線を引く
    - 平均には標準偏差の誤差棒を付け、seed ごとの点も重ねる。seed が 1 本のモデルは誤差棒なしで、凡例に「seed 1 本」と書く
    - 「学習範囲の外」と「長時間（20 s）」は別のことを調べているので、別々の背景色と見出しで区別する
    - パネルの見出しに、電流の本数と、Γ を計算できた本数・発散した本数（0 でなければ）を書く
    - 色はモデルだけに使う（4 色）。参考のモデルは灰色。文字は黒と灰色
    - 4 モデルの並びは 2×2 の比較（記憶なし／あり × 1 個／3 個に分割）の順に固定し、色はモデルに付ける
    - 数値は棒に書かない（表は予備スライドで見せる）

使い方（リポジトリのルートで）:
    uv run python -X utf8 results/final/plot_final.py --sanity     # _hh と _rest で図の体裁を確認
    uv run python -X utf8 results/final/plot_final.py              # MODELS の候補で描く
    uv run python -X utf8 results/final/plot_final.py --src DIR --out x.png --tags A=a_s0,a_s1 B=b_s0   # 任意の JSON で描く（体裁の確認用）
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# 発表で見せるモデル（4 分野の最良）。名前 -> 学習の seed ごとのタグ（results/final/<タグ>.json）。
# 各分野の最良は開発用のデータ（検証 17 本・26 本・追加 125 本）だけで選び、決まってから埋める。順番は変えない（色が付く）
MODELS = {
    # 'A: MLP': [...],                 # Markov MLP（記憶なし、1 個）       例: r9_mlp_v2*_s0〜s2
    # 'B: MLP + 記憶（RNN 化）': [...],  # A に再帰の重み W_hh を足したもの     例: whh_* / sp_r_*
    # 'C: MLP 3 個': [...],            # HH の式の 3 部分を MLP で置き換え   例: m3_v2f05d_s0〜s2
    # 'D: RNN 3 個': [...],            # C の各 MLP を RNN にしたもの（並列） 例: k_3_w3
}
REFERENCE = {}   # 参考として灰色で描くモデル（例: {'参考: HH の式を含む版': ['cl_p_mlp']}）

GROUPS = [('threshold', '閾値付近'), ('transient', '過渡応答'), ('tracking', '追従'),
          ('fluct', '揺らぎ'), ('ood', '強い入力・速い変化'), ('long', 'OU 電流 20 s')]
# 背景で区別するグループ: キー -> (帯の見出し, 背景色)
BANDS = {'ood': ('学習範囲の外', '#f1f0ec'), 'long': ('長時間', '#e9f0f8')}

# 色（dataviz スキルの検証済みの順番の先頭 4 色）。隣り合う棒の組は色覚の違いがあっても区別できる（検証済み）。
# 橙（B）と黄（D）は隣り合わないが近いので、凡例と予備スライドの表で区別を補う。水色・黄は白地との対比が 3:1 未満
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
GRAY = '#9a9893'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif'], 'font.size': 15,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})


def load(src, tag):
    return json.load(open(os.path.join(src, tag + '.json'), encoding='utf-8'))


def group_stats(src, tags):
    """
    1 モデル（学習 seed 違いのタグの並び）を、グループごとに集計する。

    Returns:
        per_seed (n_seed, G): seed ごとに、グループ内で Γ を平均した値（計算できない電流は除く。全部だめなら NaN）
        n_cur (G,): グループの電流の本数
        n_valid (n_seed, G): Γ を計算できた本数
        n_div (n_seed, G): 発散した本数
    """
    keys = [k for k, _ in GROUPS]
    per_seed, n_valid, n_div, n_cur = [], [], [], None
    for tag in tags:
        rows = load(src, tag)['rows']
        by = {k: [r for r in rows if r['name'].split(':', 1)[0] == k] for k in keys}
        g = [np.array([np.nan if r['gamma'] is None else r['gamma'] for r in by[k]], dtype=float) for k in keys]
        per_seed.append([np.nanmean(x) if np.isfinite(x).any() else np.nan for x in g])
        n_valid.append([int(np.isfinite(x).sum()) for x in g])
        n_div.append([int(sum(bool(r.get('diverged')) for r in by[k])) for k in keys])
        n_cur = [len(by[k]) for k in keys]
    return np.array(per_seed), np.array(n_cur), np.array(n_valid), np.array(n_div)


def plot(models, reference, out, src):
    series = [(name, tags, COLORS[i]) for i, (name, tags) in enumerate(models.items())]
    series += [(name, tags, GRAY) for name, tags in reference.items()]
    stats = [group_stats(src, tags) for _, tags, _ in series]
    G, S = len(GROUPS), len(series)

    # 縦軸の範囲（全パネル共通）: 上は 1 で固定、下は 0 と（seed の値・平均 − 標準偏差）の最小値の小さいほう
    lows = [0.0]
    for ps, *_ in stats:
        lows.append(np.nanmin(ps))
        if ps.shape[0] > 1:
            lows.append(np.nanmin(np.nanmean(ps, 0) - np.nanstd(ps, 0, ddof=1)))
    lo = min(lows)
    ymin = 0.0 if lo >= 0 else np.floor(lo / 0.25) * 0.25 - 0.05
    ticks = np.arange(np.ceil(ymin / 0.25) * 0.25, 1.0 + 1e-9, 0.25)

    fig, axes = plt.subplots(1, G, figsize=(15, 5.0), dpi=200, sharey=True)
    x = np.arange(S)
    for gi, ((key, title), ax) in enumerate(zip(GROUPS, axes)):
        if key in BANDS:
            band, face = BANDS[key]
            ax.set_facecolor(face)
            ax.text(0.5, 1.20, band, transform=ax.transAxes, ha='center', va='bottom', color=INK2, fontsize=13,
                    fontweight='bold')
        for j, ((name, tags, color), (ps, n_cur, n_valid, n_div)) in enumerate(zip(series, stats)):
            v = ps[:, gi]
            m = np.nanmean(v) if np.isfinite(v).any() else np.nan
            ax.bar(j, m, 0.72, color=color, zorder=2, lw=0,
                   hatch='//' if color == GRAY else None, edgecolor='white' if color == GRAY else None)
            if len(v) > 1 and np.isfinite(v).sum() > 1:
                ax.errorbar(j, m, yerr=np.nanstd(v, ddof=1), fmt='none', ecolor=INK, elinewidth=1.2, capsize=3,
                            zorder=3)
            if len(v) > 1:   # seed ごとの点（少し横にずらして重なりを避ける）
                off = np.linspace(-0.18, 0.18, len(v))
                ax.scatter(j + off, v, s=16, facecolor='white', edgecolor=INK, lw=0.9, zorder=4)
        # 見出し: グループ名と本数。Γ を計算できなかった・発散した電流があれば添える
        n_cur = stats[0][1][gi]
        notes = []
        miss = max(int(n_cur - st[2][:, gi].min()) for st in stats)
        div = max(int(st[3][:, gi].max()) for st in stats)
        if miss:
            notes.append('Γ なし 最大 {} 本'.format(miss))
        if div:
            notes.append('発散 最大 {} 本'.format(div))
        sub = '（{} 本）'.format(n_cur) + ('\n' + '・'.join(notes) if notes else '')
        ax.set_title('{}\n{}'.format(title, sub), fontsize=13, color=INK, pad=8)
        ax.axhline(0, color=INK2, lw=1.0, zorder=1.5)
        ax.set_xticks([])
        ax.set_xlim(-0.6, S - 0.4)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=1)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if gi:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
    axes[0].set_ylim(ymin, 1.0)
    axes[0].set_yticks(ticks)
    axes[0].set_ylabel('coincidence factor Γ')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch='//' if c == GRAY else None) for _, _, c in series]
    labels = ['{}（seed {} 本）'.format(name, len(tags)) for name, tags, _ in series]
    if any(len(t) > 1 for _, t, _ in series):
        handles.append(plt.Line2D([], [], ls='', marker='o', mfc='white', mec=INK, ms=5))
        labels.append('seed ごとの値')
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.05, 1.0), ncol=min(len(labels), 5),
               frameon=False, fontsize=13, handlelength=1.2, columnspacing=1.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out, bbox_inches='tight')
    print('saved:', os.path.relpath(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--src', help='JSON を読むフォルダ（体裁の確認用）')
    ap.add_argument('--out')
    ap.add_argument('--tags', nargs='+', help='名前=タグ1,タグ2 の並び（--src と使う）')
    a = ap.parse_args()
    if a.sanity:
        src = os.path.join(HERE, 'sanity')
        plot({'HH そのもの': ['_hh'], '静止のまま': ['_rest']}, {}, os.path.join(src, 'final_groups.png'), src)
    elif a.src:
        models = dict((s.split('=', 1)[0], s.split('=', 1)[1].split(',')) for s in a.tags)
        plot(models, {}, a.out, a.src)
    else:
        plot(MODELS, REFERENCE, os.path.join(HERE, 'final_groups.png'), HERE)


if __name__ == '__main__':
    main()
