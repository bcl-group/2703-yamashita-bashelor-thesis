"""
学習・評価の結果を保存する（数値ファイルと図の両方）。train.py と evaluate.py が必ず呼ぶ。

    save_loss(run_dir, loss_rows, val_rows, dagger_its)     loss.csv, val_history.csv, loss.png
    save_responses(run_dir, model, ref, v_lo)               responses_<set>.npz, figs/<set>/*.png, 採点結果を返す

responses_<set>.npz の中身（予測だけ。HH の正解は eval_sets.reference(<set>) で同じ並びに作り直せる）:
    電流ごとに長さが違うので、全電流をつないだ配列と区切り offsets で持つ。電流 i は [offsets[i], offsets[i+1])
    names (n,)        電流の名前
    lens (n,)         各電流の点数 len_i（= 長さ / dt）。offsets (n+1,) = [0, cumsum(lens)]
    dt                0.01 [ms]。時刻 t = k dt
    V (Σ len_i,)      膜電位 [mV]。k = 0 は静止状態（-65 mV）
    g (Σ len_i, 3)    ゲート変数 m, h, n（初期値込み）
    ab (Σ len_i, 6)   速度定数 α_m, β_m, α_h, β_h, α_n, β_n [1/ms]。k 番目は V[k] からブロック 1 が出した値を
                      0 で下から切ったもの（ゲートの更新に使った値）。各電流の最後の点は次の更新に使わないので NaN
    dtype は float32。採点（score_<set>.json）は保存前の float64 の V で行う
read_responses(path, ref) で、正解と並べた電流ごとの dict に戻せる
"""
import csv
import json
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hh
import eval_sets as es

plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'DejaVu Sans']
C_TRUE, C_PRED = 'black', '#eb6834'

# 図の範囲（セットの中の全電流で同じにする。自動の範囲は使わない）
V_LIM = {'val': (-90.0, 75.0), 'test': (-90.0, 75.0), 'final': (-130.0, 75.0)}
I_LIM = {'val': (-10.0, 45.0), 'test': (-10.0, 45.0), 'final': (-25.0, 75.0)}
MAX_PLOT_POINTS = 200_000            # これより長い電流は、図だけ間引く（数値ファイルは間引かない）


def rate_lims(v_lim):
    # 速度定数の縦軸: V が v_lim の範囲を動いたときの HH の式の最大値（速度定数ごとに、全電流で共通）
    ab = hh.rates(torch.linspace(*v_lim, 2001, dtype=torch.float64)).numpy()
    return [(0.0, float(1.05 * ab[:, c].max())) for c in range(6)]


def _safe(name):
    return re.sub(r'[^0-9A-Za-z.+\-]+', '_', name).strip('_')


# ---------------- 損失 ----------------

def save_loss(run_dir, loss_rows, val_rows, dagger_its):
    """
    loss_rows: [(it, lr, loss_b1, loss_b2, loss_b3)]（数百回ごとの平均）
    val_rows : [(it, n100, mean, gamma)]（検証 32 本の採点）
    """
    with open(os.path.join(run_dir, 'loss.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['it', 'lr', 'loss_b1', 'loss_b2', 'loss_b3', 'loss_total'])
        for r in loss_rows:
            w.writerow([*r, r[2] + r[3] + r[4]])
    with open(os.path.join(run_dir, 'val_history.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['it', 'n100', 'mean_match', 'gamma'])
        w.writerows(val_rows)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    if loss_rows:
        L = np.array(loss_rows, dtype=float)
        for j, lab in enumerate(['ブロック 1（αβ）', 'ブロック 2（Δm, Δh, Δn）', 'ブロック 3（ΔV）']):
            ax[0].semilogy(L[:, 0], L[:, 2 + j], label=lab)
        ax[0].semilogy(L[:, 0], L[:, 2:5].sum(1), 'k', lw=1, label='合計')
    ax[0].set_ylabel('学習の損失')
    ax[0].legend(fontsize=8)
    if val_rows:
        Vr = np.array(val_rows, dtype=float)
        ax[1].plot(Vr[:, 0], Vr[:, 3], 'o-', label='検証の平均 Γ')
        ax[1].plot(Vr[:, 0], Vr[:, 2], 's--', label='検証の平均一致率')
    ax[1].set_ylim(0.0, 1.05)
    ax[1].set_xlabel('学習の回数')
    ax[1].legend(fontsize=8, loc='lower right')
    for a in ax:
        for it in dagger_its:
            a.axvline(it, color='gray', lw=0.5, ls=':')
    fig.savefig(os.path.join(run_dir, 'loss.png'), dpi=120)
    plt.close(fig)


# ---------------- 応答 ----------------

def read_responses(path, ref):
    """
    responses_<set>.npz と正解 ref（eval_sets.reference）を電流ごとの dict のリストにする。
    キー: name, I, V_true, V_pred (len_i,), g_true, g_pred (len_i, 3), ab_true, ab_pred (len_i - 1, 6)
    """
    d = np.load(path, allow_pickle=True)
    assert list(d['names']) == ref.names
    off = d['offsets']
    out = []
    for i, name in enumerate(ref.names):
        s, e = off[i], off[i + 1]
        Vt = ref.V[i]
        out.append({'name': name, 'I': ref.I[i, :e - s],
                    'V_true': Vt, 'V_pred': d['V'][s:e],
                    'g_true': np.stack([ref.m[i], ref.h[i], ref.n[i]], 1), 'g_pred': d['g'][s:e],
                    'ab_true': hh.rates(torch.from_numpy(Vt[:-1])).numpy(), 'ab_pred': d['ab'][s:e - 1]})
    return out


def save_responses(run_dir, model, ref, v_lo, label=None):
    """
    model を ref の全電流で閉ループに回し、V・ゲート・速度定数の応答を保存して採点する。
    Returns: eval_sets.score の結果（rows に電流ごとのゲート・速度定数の RMSE も足す）
    """
    rec = model.run(ref.I, v_lo=v_lo, record=True)
    res = es.score(ref, rec['V'][:, 1:], label=label)
    path = os.path.join(run_dir, f'responses_{ref.name}.npz')
    ab = np.concatenate([rec['ab'], np.full_like(rec['ab'][:, :1], np.nan)], 1)       # (B, N, 6)
    cut = lambda x: np.concatenate([x[i, :n] for i, n in enumerate(ref.lens)]).astype(np.float32)
    np.savez_compressed(path, names=np.array(ref.names), lens=ref.lens,
                        offsets=np.concatenate([[0], np.cumsum(ref.lens)]), dt=es.DT,
                        V=cut(rec['V']), g=cut(rec['g']), ab=cut(ab))
    resp = read_responses(path, ref)
    for r, row in zip(resp, res['rows']):
        row['rmse_gate'] = np.sqrt(np.mean((r['g_pred'] - r['g_true']) ** 2, 0)).tolist()       # m, h, n
        row['rmse_rate'] = np.sqrt(np.mean((r['ab_pred'] - r['ab_true']) ** 2, 0)).tolist()     # α_m, β_m, ...
    with open(os.path.join(run_dir, f'score_{ref.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    plot_responses(run_dir, ref.name, resp, res['rows'])
    return res


def plot_responses(run_dir, set_name, resp, rows):
    """
    電流ごとに 3 枚: <番号>_<名前>_V.png（入力電流と膜電位）、_gates.png（m, h, n）、_rates.png（α, β。行が m, h, n、列が α, β）
    黒が HH の正解、橙が MLP 3 個の予測
    """
    out = os.path.join(run_dir, 'figs', set_name)
    os.makedirs(out, exist_ok=True)
    v_lim, i_lim = V_LIM[set_name], I_LIM[set_name]
    r_lim = rate_lims(v_lim)
    for i, (r, row) in enumerate(zip(resp, rows)):
        n = len(r['V_true'])
        s = max(1, -(-n // MAX_PLOT_POINTS))
        t = np.arange(n)[::s] * es.DT
        ta = np.arange(n - 1)[::s] * es.DT
        head = '{}（スパイク 正解 {} / 予測 {}、一致率 {:.2f}、Γ {:.3f}）'.format(
            r['name'], *row['spikes'], row['match'], row['gamma'])
        stem = os.path.join(out, '{:02d}_{}'.format(i, _safe(r['name'])))

        fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, height_ratios=[1, 2.5],
                               constrained_layout=True)
        ax[0].plot(t, r['I'][::s], 'k', lw=0.8)
        ax[0].set_ylim(*i_lim)
        ax[0].set_ylabel('I [µA/cm²]')
        ax[1].plot(t, r['V_true'][::s], color=C_TRUE, lw=0.8, label='HH')
        ax[1].plot(t, r['V_pred'][::s], color=C_PRED, lw=0.8, ls='--', label='MLP 3 個')
        ax[1].set_ylim(*v_lim)
        ax[1].set_ylabel('V [mV]')
        ax[1].set_xlabel('時刻 [ms]')
        ax[1].legend(fontsize=8, loc='upper right')
        ax[1].text(0.01, 0.03, 'RMSE {:.2f} mV'.format(row['rmse']), transform=ax[1].transAxes, fontsize=8)
        ax[0].set_title(head, fontsize=9)
        fig.savefig(stem + '_V.png', dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True, constrained_layout=True)
        for j, g in enumerate('mhn'):
            ax[j].plot(t, r['g_true'][::s, j], color=C_TRUE, lw=0.8, label='HH')
            ax[j].plot(t, r['g_pred'][::s, j], color=C_PRED, lw=0.8, ls='--', label='MLP 3 個')
            ax[j].set_ylim(-0.05, 1.05)
            ax[j].set_yticks([0, 0.5, 1])
            ax[j].set_ylabel(g)
            ax[j].text(0.01, 0.80, 'RMSE {:.2e}'.format(row['rmse_gate'][j]), transform=ax[j].transAxes, fontsize=8)
        ax[0].legend(fontsize=8, loc='upper right')
        ax[0].set_title(head, fontsize=9)
        ax[-1].set_xlabel('時刻 [ms]')
        fig.savefig(stem + '_gates.png', dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(3, 2, figsize=(12, 5.5), sharex=True, constrained_layout=True)
        for j, g in enumerate('mhn'):
            for col, ab_name in enumerate('αβ'):
                c = 2 * j + col
                ax[j, col].plot(ta, r['ab_true'][::s, c], color=C_TRUE, lw=0.8, label='HH')
                ax[j, col].plot(ta, r['ab_pred'][::s, c], color=C_PRED, lw=0.8, ls='--', label='MLP 3 個')
                ax[j, col].set_ylim(*r_lim[c])
                ax[j, col].set_ylabel(f'{ab_name}_{g} [1/ms]')
                ax[j, col].text(0.01, 0.80, 'RMSE {:.2e} 1/ms'.format(row['rmse_rate'][c]),
                                transform=ax[j, col].transAxes, fontsize=8)
        ax[0, 1].legend(fontsize=8, loc='upper right')
        fig.suptitle(head, fontsize=9)
        for col in range(2):
            ax[-1, col].set_xlabel('時刻 [ms]')
        fig.savefig(stem + '_rates.png', dpi=110)
        plt.close(fig)
