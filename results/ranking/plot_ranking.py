# 精度ランキング（テスト 26 本の Γ）と速度ランキング（hh_speed の結果）をまとめた表と図
# 使い方（リポジトリのルートで、make_ranking.py の後で）: uv run python -X utf8 results/ranking/plot_ranking.py cpu [gpu_all]
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'results'))
from make_ranking import ARCH_SPEED, CATEGORY   # noqa: E402

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'IPAexGothic', 'sans-serif']
D = os.path.join(ROOT, 'results', 'ranking')
acc = json.load(open(os.path.join(D, 'accuracy.json'), encoding='utf-8'))['accuracy']
tags = sys.argv[1:] or ['cpu']


def speed(tag):
    res = json.load(open(os.path.join(ROOT, 'results', 'hh_speed', tag + '.json'), encoding='utf-8'))
    t = {}
    for r in res['rows']:
        if not r['record'] and r.get('mm', 'fp32') == 'fp32':
            t.setdefault(r['model'], {})[(r['runner'], r['B'])] = r['per_step_s'] * 1e6
    # 速度は形だけで決まる。サーバは RNN 1 個を同じ形の sp_r_scr_e01_s1 で測った
    if 'rnn1:sp_r_init' not in t and 'rnn1:sp_r_scr_e01_s1' in t:
        t['rnn1:sp_r_init'] = t['rnn1:sp_r_scr_e01_s1']
        res['flops'].setdefault('rnn1:sp_r_init', res['flops'].get('rnn1:sp_r_scr_e01_s1'))
    return res, t


lines = ['# HH 代理モデルの精度ランキングと速度ランキング', '']
# ---- 精度（上位 25 と、アーキテクチャごとの最良）----
lines += ['## 精度ランキング（テスト 26 本の平均 Γ、上位 25）', '',
          '| 順位 | モデル | アーキテクチャ | 平均 Γ | 100% 一致 | 検証 17 本 |', '|---|---|---|---|---|---|']
for i, r in enumerate(acc[:25], 1):
    lines.append('| {} | {} | {} | {:.3f} | {}/26 | {} |'.format(
        i, r['key'], r['arch'] or '—', r['test_gamma'], r['test_n100'],
        '{}/17'.format(r['val_n100']) if r['val_n100'] is not None else '—'))
lines += ['', '## アーキテクチャごとの最良（テスト Γ で最良 / 検証 17 本で選んだもの）', '',
          '| アーキテクチャ | テストで最良のモデル | Γ | 検証で選んだモデル | その Γ | モデル数 |', '|---|---|---|---|---|---|']
by_arch = {}
for r in acc:
    by_arch.setdefault(r['arch'], []).append(r)
arch_best = {}
for arch, rs in sorted(by_arch.items(), key=lambda kv: -max(x['test_gamma'] for x in kv[1])):
    if arch is None:
        continue
    bt = max(rs, key=lambda x: x['test_gamma'])
    vs = [x for x in rs if x['val_n100'] is not None]
    bv = max(vs, key=lambda x: (x['val_n100'], x['val_mean'] or 0)) if vs else None
    arch_best[arch] = (bt, bv)
    lines.append('| {} | {} | {:.3f} | {} | {} | {} |'.format(
        arch, bt['key'], bt['test_gamma'], bv['key'] if bv else '—',
        '{:.3f}'.format(bv['test_gamma']) if bv else '—', len(rs)))

# ---- 速度 ----
for tag in tags:
    p = os.path.join(ROOT, 'results', 'hh_speed', tag + '.json')
    if not os.path.exists(p):
        continue
    res, t = speed(tag)
    env = res['env']
    dev = env.get('gpu') or env.get('cpu') or 'CPU'
    runners = sorted({rn for v in t.values() for (rn, _) in v}, key=lambda x: (x != 'cudagraph', x))
    rn = runners[0]
    Bs = sorted({B for (r_, B) in t['hh_euler'] if r_ == rn})
    Bbig = Bs[-1]
    hh = t['hh_euler'][(rn, Bbig)]
    order = sorted([a for a in ARCH_SPEED if ARCH_SPEED[a] in t and (rn, Bbig) in t[ARCH_SPEED[a]]],
                   key=lambda a: t[ARCH_SPEED[a]][(rn, Bbig)])
    lines += ['', f'## 速度ランキング（{tag}: {dev}、回し方 {rn}、1 ステップの時間 [µs]）', '',
              '| 順位 | アーキテクチャ | ' + ' | '.join(f'B={B}' for B in Bs) + f' | HH 比（B={Bbig}） | 行列積 FLOP / ニューロン・ステップ |',
              '|---|---|' + '---|' * len(Bs) + '---|---|']
    for i, a in enumerate(order, 1):
        m = ARCH_SPEED[a]
        lines.append('| {} | {} | {} | {:.1f} 倍 | {:,.0f} |'.format(
            i, a, ' | '.join('{:.0f}'.format(t[m][(rn, B)]) if (rn, B) in t[m] else '—' for B in Bs),
            t[m][(rn, Bbig)] / hh, res['flops'].get(m, 0)))

    # 図: 精度（アーキテクチャごとの最良 Γ）と速度（B 最大の 1 ステップの時間）
    fig, ax = plt.subplots(figsize=(9, 6))
    cols = {'Markov MLP': '#2a78d6', 'Markov MLP（アンサンブル）': '#7fb3e8', 'MLP 3 個': '#1baf7a',
            'RNN 1 個': '#e87ba4', 'RNN 3 個': '#eb6834'}
    for a, (bt, bv) in arch_best.items():
        m = ARCH_SPEED.get(a)
        if m not in t or (rn, Bbig) not in t[m]:
            continue
        x = t[m][(rn, Bbig)] / hh
        c = cols.get(CATEGORY.get(a), 'gray')
        ax.scatter([x], [bt['test_gamma']], s=80, color=c, zorder=3)
        if bv:
            ax.scatter([x], [bv['test_gamma']], s=80, facecolors='none', edgecolors=c, zorder=3)
            ax.plot([x, x], [bv['test_gamma'], bt['test_gamma']], color=c, lw=1, alpha=0.5)
        ax.annotate(a, (x, bt['test_gamma']), textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.axvline(1.0, color='k', lw=1, ls=':')
    ax.text(1.02, ax.get_ylim()[0] + 0.005, 'HH と同じ速さ', fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 5, 8]); ax.set_xticklabels(['1', '2', '3', '5', '8'])
    ax.set_xlabel(f'1 ステップの時間（HH の陽的 Euler 法を 1 とした比、{dev}、B={Bbig}）  ← 速い')
    ax.set_ylabel('テスト 26 本の平均 Γ（● テストで最良、○ 検証で選んだもの）')
    ax.set_title('精度と速度（アーキテクチャごと）')
    fig.tight_layout()
    fig.savefig(os.path.join(D, f'ranking_{tag}.png'), dpi=130)

open(os.path.join(D, 'ranking.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
