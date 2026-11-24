"""
学習の様子をまとめて描く: 1 本の学習につき 2 枚
    fig_<タグ>_loss.png   損失の減り方（学習データ、対数軸）と検証 Γ の推移
    fig_<タグ>_iv.png     開発用 26 本の入力電流 i(t) と膜電位 v(t)（HH の正解と代理モデルを重ねる）

使い方（リポジトリのルートで）:
    .venv/Scripts/python.exe -X utf8 results/plot_runs.py <ログ> [<ログ> ...]
    .venv/Scripts/python.exe -X utf8 results/plot_runs.py results/hh_improve/logs/*.log --loss-only

ログとタグの対応:
    results/hh_rnn3/logs/<t>.log  → タグ rnn3:<t>（重み results/hh_rnn3/<t>_b1〜3.pt）
    それ以外の <t>.log            → タグ <t>（m3_* は results/hh_improve/<t>_b1〜3.pt、他は <t>.pt）
                                    ただし hh_improve に重みが無く results/hh_rnn3/<t>_b1.pt があれば rnn3:<t>（hh_eq の D）
    重みがまだ無い（学習中）ときは i・v を飛ばして損失だけ描く。

読むログの行（学習スクリプトごとに形が違うので、"loss" の後ろの「名前 数値」の組を全部拾う）:
    pre   500 loss 2.78e-01  (2306 s)                                   A: Markov MLP（hh_improve）
    pre   500 loss RNN1 8.3e-03 RNN2 5.1e-02 RNN3 1.4e-01  (496 s)       C: MLP 3 個（名前は RNN だが中身は MLP）
    it   5000 loss 1.2e-03 ...  (100 s)                                 B: W_hh（hh_mlp_whh）
    it   5000 loss RNN1 .. RNN2 .. RNN3 ..  閉ループ V .. g ..  (.. s)   D: RNN 3 個（hh_rnn3）
    DAgger k: ...                                                       DAgger で状態を足した位置
    [<タグ> it N val] ... Γ x                                           検証 17 本の Γ

出力: ログと同じ結果フォルダ（logs の 1 つ上）。i・v の閉ループの結果は <出力先>/traces_<タグ>.npz に残し、次からは読み直す。
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'results'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))

plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})
COLS = ['#2a78d6', '#e8673a', '#1baf7a', '#9b59b6', '#c0392b']
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf'


# ---------- ログを読む ----------

def parse_log(path):
    """
    Returns:
        phases: {段階名 ('pre' / 'it' / ...): {'it': list[int], 'series': {系列名: list[float]}}}
        dag:    list[(段階名, iter)]  DAgger の位置（直前の損失の行の iter）
        val:    list[(iter, Γ)]       検証 17 本
    """
    phases, dag, val = {}, [], []
    last = (None, 0)
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*([A-Za-z]\w*)\s+(\d+)\s+(?:α\s+\S+\s+)?loss\s+(.*)$', line)
        if m:
            ph, it, rest = m.group(1), int(m.group(2)), re.sub(r'\(\s*\d+ s\)\s*$', '', m.group(3))
            # "loss" の直後が数値なら名前は "loss"。以降は「名前 数値」の組（日本語の名前も可）
            toks = rest.split()
            pairs, name = [], None
            for tk in toks:
                if re.fullmatch(NUM, tk):
                    pairs.append((name or 'loss', float(tk)))
                    name = None
                else:
                    name = tk if name is None else name + ' ' + tk
            P = phases.setdefault(ph, {'it': [], 'series': {}})
            P['it'].append(it)
            for k, v in pairs:
                if k.startswith('||'):      # B の ||W_hh||_F は損失ではないので描かない
                    continue
                P['series'].setdefault(k, []).append(v)
            # 途中から出てくる系列（閉ループの損失など）は、出る前を nan で埋める
            for k, v in P['series'].items():
                while len(v) < len(P['it']):
                    v.insert(0, np.nan)
            last = (ph, it)
            continue
        if line.startswith('DAgger'):
            dag.append(last)
            continue
        m = re.match(r'\[\S+ it (\d+) val\].*Γ\s+(' + NUM + ')', line)
        if m:
            val.append((int(m.group(1)), float(m.group(2))))
    return phases, dag, val


def loss_extent(phases):
    """損失の正の値の (最小, 最大) と、段階をつないだ iter の最大。図どうしの軸をそろえるのに使う"""
    ys, xmax = [], 0
    for P in phases.values():
        xmax += max(P['it']) if P['it'] else 0
        for y in P['series'].values():
            y = np.array(y, dtype=float)
            ys.append(y[np.isfinite(y) & (y > 0)])
    ys = np.concatenate(ys) if ys else np.array([])
    return (ys.min(), ys.max(), xmax) if len(ys) else (np.nan, np.nan, xmax)


def plot_loss(tag, phases, dag, val, out, ylim=None, xmax=None):
    # 段階（pre → it）は続けて並べる: 後の段階の iter に前の段階の最後の iter を足す
    offs, o = {}, 0
    for ph, P in phases.items():
        offs[ph] = o
        o += max(P['it']) if P['it'] else 0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})
    c = 0
    for ph, P in phases.items():
        x = np.array(P['it']) + offs[ph]
        for name, y in P['series'].items():
            y = np.array(y, dtype=float)
            col = COLS[c % len(COLS)]
            c += 1
            lab = name if len(phases) == 1 else f'{ph}: {name}'
            a1.plot(x, y, color=col, lw=0.6, alpha=0.35)
            # 1 点ずつはミニバッチ 1 回の値でばらつくので、移動平均（5 点）も重ねる
            w = 5
            ok = np.isfinite(y)
            if ok.sum() >= w:
                sm = np.convolve(y[ok], np.ones(w) / w, mode='valid')
                a1.plot(x[ok][w - 1:], sm, color=col, lw=1.8, label=lab)
            else:
                a1.plot(x, y, color=col, lw=1.8, label=lab)
    a1.set_yscale('log')
    if ylim:
        a1.set_ylim(*ylim)
    if xmax:
        a1.set_xlim(-0.02 * xmax, 1.02 * xmax)
    a1.set_ylabel('損失（学習データ）')
    a1.legend(frameon=False, fontsize=9)
    a1.set_title(tag, loc='left', fontsize=12)
    for k, (ph, it) in enumerate(dag):
        xv = it + offs.get(ph, 0)
        for ax in (a1, a2):
            ax.axvline(xv, color='#999999', lw=0.8, ls='--')
        if k == 0:
            a1.text(xv, a1.get_ylim()[1], ' DAgger', va='top', fontsize=9, color='#777777')
    if len(phases) > 1:
        for ph, o in offs.items():
            if o:
                for ax in (a1, a2):
                    ax.axvline(o, color='#333333', lw=1.0)
                a1.text(o, a1.get_ylim()[0], f' {ph} 開始', va='bottom', fontsize=9)
    if val:
        # 検証の iter は最後の段階（評価を出すのは本学習）に合わせる
        last_off = list(offs.values())[-1]
        v = np.array(val)
        a2.plot(v[:, 0] + last_off, v[:, 1], 'o-', color='#222222')
    else:
        a2.text(0.5, 0.5, '検証 Γ はまだ無い', transform=a2.transAxes, ha='center', color='#777777')
    a2.set_ylim(-0.05, 1.05)
    a2.set_ylabel('検証 Γ（17 本）')
    a2.set_xlabel('iteration')
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.09)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('saved', os.path.relpath(out, ROOT), flush=True)


# ---------- i・v（開発用 26 本） ----------

def weights_exist(tag):
    if tag.startswith('rnn3:'):
        return os.path.exists(os.path.join(ROOT, 'results', 'hh_rnn3', tag[5:] + '_b1.pt'))
    if tag.startswith('m3_'):
        return all(os.path.exists(os.path.join(ROOT, 'results', 'hh_improve', f'{tag}_b{k}.pt')) for k in (1, 2, 3))
    return os.path.exists(os.path.join(ROOT, 'results', 'hh_improve', tag + '.pt'))


def traces(tag, cache):
    """開発用 26 本の閉ループ。Returns: names, I (26, N), V_true list, V_pred (26, N-1), lens"""
    import hh_bench as hb
    names, I, Vt, lens = hb.reference(val=False)
    if os.path.exists(cache):
        return names, I, Vt, np.load(cache)['V'], lens
    import torch
    from rescore_gamma import get_model
    torch.set_num_threads(4)
    print(f'{tag}: 開発用 26 本を閉ループで計算', flush=True)
    Vp = np.asarray(get_model(tag)(I), dtype=np.float32)
    np.savez_compressed(cache, V=Vp)
    return names, I, Vt, Vp, lens


def plot_iv(tag, names, I, Vt, Vp, lens, out):
    import hh_bench as hb
    import test_currents as tc
    dt = hb.DT
    nrow, ncol = 13, 2
    fig = plt.figure(figsize=(17, 46))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[3, 1] * nrow, hspace=0.5, wspace=0.12)
    gams = []
    lo = min(I[k, :lens[k] - 1].min() for k in range(len(names)))
    hi = max(I[k, :lens[k] - 1].max() for k in range(len(names)))
    ilim = (10 * np.floor(lo / 10), 10 * np.ceil(hi / 10))
    iticks = np.arange(ilim[0], ilim[1] + 1, 20 if ilim[1] - ilim[0] > 40 else 10)
    for i, name in enumerate(names):
        r, c = i % nrow, i // nrow
        av = fig.add_subplot(gs[2 * r, c])
        ai = fig.add_subplot(gs[2 * r + 1, c], sharex=av)
        n = lens[i]
        t = np.arange(n - 1) * dt
        vp = np.nan_to_num(np.asarray(Vp[i, :n - 1], dtype=float), nan=1e3, posinf=1e3, neginf=-1e3)
        st, sp = tc.spike_times(Vt[i], dt), tc.spike_times(vp, dt)
        g = hb.coincidence_factor(st, sp, (n - 1) * dt)
        gams.append(g)
        av.plot(t, Vt[i], color='#222222', lw=0.8, label='HH（正解）')
        av.plot(t, np.clip(vp, -120, 80), color=COLS[0], lw=0.8, alpha=0.85, label='代理モデル')
        av.set_ylim(-90, 60)
        av.set_yticks([-80, -40, 0, 40])
        av.set_title(f'{name}   スパイク HH {len(st)} / 代理 {len(sp)}   Γ {g:.2f}', loc='left', fontsize=10,
                     color='#222222' if g >= 0.999 else '#c0392b')
        av.tick_params(labelbottom=False, labelsize=8)
        av.set_ylabel('v [mV]', fontsize=8)
        ai.plot(t, I[i, :n - 1], color='#e8673a', lw=0.8)
        # 電流の縦軸は 26 本で共通（パネルごとに自動で拡大すると、一定電流 2.5 と 20 が同じ高さに見える）
        ai.set_ylim(ilim)
        ai.set_yticks(iticks)
        ai.set_ylabel('i [µA/cm²]', fontsize=8)
        ai.tick_params(labelsize=8)
        if r == nrow - 1:
            ai.set_xlabel('t [ms]')
        if i == 0:
            av.legend(frameon=False, fontsize=8, loc='upper right', ncol=2)
    fig.suptitle(f'{tag}   開発用 26 本   平均 Γ {np.nanmean(gams):.3f}   Γ = 1: {sum(g >= 0.999 for g in gams)}/26',
                 x=0.01, ha='left', y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.985, bottom=0.015)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print('saved', os.path.relpath(out, ROOT), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--loss-only', action='store_true')
    ap.add_argument('--tag', help='ログ 1 本のときだけ: 重みのタグを直接指定（例 rnn3:eq_d_h64_s0）')
    ap.add_argument('--own-axes', action='store_true', help='損失の図の軸を 1 本ずつ自動で決める（既定は渡したログ全体で共通）')
    a = ap.parse_args()
    # 損失の図の軸は、渡したログ全体でそろえる（縦: 損失の最小〜最大を 10 の累乗に丸める、横: 0〜最長の iter）
    ylim = xmax = None
    if not a.own_axes:
        ext = [loss_extent(parse_log(os.path.abspath(lp))[0]) for lp in a.logs]
        lo, hi = np.nanmin([e[0] for e in ext]), np.nanmax([e[1] for e in ext])
        if np.isfinite(lo) and np.isfinite(hi):
            ylim = (10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi)))
        xmax = max(e[2] for e in ext) or None
    for lp in a.logs:
        lp = os.path.abspath(lp)
        stem = os.path.splitext(os.path.basename(lp))[0]
        logdir = os.path.dirname(lp)
        outdir = os.path.dirname(logdir) if os.path.basename(logdir) == 'logs' else logdir
        # hh_eq のように重みが別のフォルダにあるときは、hh_rnn3 に重みがあれば RNN 3 個とみなす
        is_rnn3 = os.path.basename(outdir) == 'hh_rnn3' or (
            not weights_exist(stem) and os.path.exists(os.path.join(ROOT, 'results', 'hh_rnn3', stem + '_b1.pt')))
        tag = a.tag or ('rnn3:' + stem if is_rnn3 else stem)
        phases, dag, val = parse_log(lp)
        if phases:
            plot_loss(tag, phases, dag, val, os.path.join(outdir, f'fig_{stem}_loss.png'), ylim, xmax)
        else:
            print(f'{stem}: 損失の行が無い（スキップ）')
        if a.loss_only:
            continue
        if not weights_exist(tag):
            print(f'{stem}: 重み（{tag}）が手元に無いので i・v は飛ばす')
            continue
        names, I, Vt, Vp, lens = traces(tag, os.path.join(outdir, f'traces_{stem}.npz'))
        plot_iv(tag, names, I, Vt, Vp, lens, os.path.join(outdir, f'fig_{stem}_iv.png'))


if __name__ == '__main__':
    main()
