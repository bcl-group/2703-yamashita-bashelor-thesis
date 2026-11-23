"""
Traub の代理モデルの評価の部品（評価手法（計画）.md 3.3 節）。

    spike_times(V)            : 細胞体の V が -20 mV を上向きに越えた時刻 [ms]（線形補間）
    gamma(t_true, t_pred, T)  : coincidence factor Γ（Δ = 2 ms。HH と同じ hh_bench.coincidence_factor）
    bursts(t)                 : スパイク間隔が ISI_BURST（10 ms）未満で続くスパイクのまとまり（2 本以上）をバーストとする
    burst_scores(t_true, t_pred, T) : バーストの数、1 バーストあたりのスパイク数、バースト開始時刻の Γ（Δ = 5 ms）
    ca_error(xi_true, xi_pred): χ の誤差。正解の標準偏差で割った RMSE（0 で完全一致、1 で「平均値を出すだけ」と同じ程度）
    score_traces(...)         : 軌道ごとにまとめて dict を返す
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
from hh_bench import coincidence_factor   # noqa: E402

DT = 0.01
THR = -20.0
ISI_BURST = 10.0     # [ms]
WIN_BURST = 5.0      # [ms]


def spike_times(V, dt=DT, thr=THR):
    V = np.asarray(V, dtype=np.float64)
    k = np.nonzero((V[:-1] < thr) & (V[1:] >= thr))[0]
    frac = (thr - V[k]) / (V[k + 1] - V[k])
    return (k + frac) * dt


def gamma(t_true, t_pred, T, window=2.0):
    return coincidence_factor(t_true, t_pred, T, window)


def bursts(t, isi=ISI_BURST):
    """スパイク時刻 t から、バーストの (開始時刻, スパイク数) のリスト"""
    out, i = [], 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] - t[j] < isi:
            j += 1
        if j > i:
            out.append((t[i], j - i + 1))
        i = j + 1
    return out


def burst_scores(t_true, t_pred, T):
    bt, bp = bursts(t_true), bursts(t_pred)
    on_t = np.array([b[0] for b in bt])
    on_p = np.array([b[0] for b in bp])
    return {'n_burst': [len(bt), len(bp)],
            'spk_per_burst': [float(np.mean([b[1] for b in bt])) if bt else 0.0,
                              float(np.mean([b[1] for b in bp])) if bp else 0.0],
            'gamma_burst': coincidence_factor(on_t, on_p, T, WIN_BURST)}


def ca_error(xi_true, xi_pred):
    xi_true, xi_pred = np.asarray(xi_true, float), np.asarray(xi_pred, float)
    sd = xi_true.std()
    return float(np.sqrt(np.mean((xi_pred - xi_true) ** 2)) / sd) if sd > 0 else float('nan')


def score_traces(V_true, V_pred, xi_true=None, xi_pred=None, dt=DT):
    """
    V_true, V_pred: (B, N) 細胞体の膜電位。xi_*: (B, N) か None
    Returns: dict。rows（軌道ごと）と、まとめ（Γ の平均、Γ = 1 の本数、バースト開始の Γ の平均、χ の誤差の中央値、発散の本数）
    """
    rows = []
    for b in range(V_true.shape[0]):
        T = V_true.shape[1] * dt
        div = not np.all(np.isfinite(V_pred[b]))
        tt_, tp = spike_times(V_true[b], dt), spike_times(np.nan_to_num(V_pred[b], nan=-100.0), dt)
        r = {'spikes': [len(tt_), len(tp)], 'gamma': gamma(tt_, tp, T), 'diverged': bool(div)}
        r.update(burst_scores(tt_, tp, T))
        if xi_true is not None:
            r['ca_err'] = ca_error(xi_true[b], xi_pred[b])
        rows.append(r)
    g = np.array([r['gamma'] for r in rows], float)
    gb = np.array([r['gamma_burst'] for r in rows], float)
    out = {'rows': rows, 'gamma_mean': float(np.nanmean(g)), 'n_gamma1': int(np.sum(g > 0.999)),
           'gamma_burst_mean': float(np.nanmean(gb)), 'n_div': int(sum(r['diverged'] for r in rows)), 'n': len(rows)}
    if xi_true is not None:
        out['ca_err_median'] = float(np.nanmedian([r['ca_err'] for r in rows]))
    return out


if __name__ == '__main__':
    # 自己確認: 同じ波形どうしは Γ = 1、χ の誤差 = 0
    import traub_torch as tt
    import torch
    P = tt.Params(torch.device('cpu'), torch.float64)
    s = tt.init_state(2, P)
    I = torch.tensor([0.3, 1.0], dtype=torch.float64)
    V, X = [], []
    for k in range(30000):
        s = tt.step(s, I, P)
        V.append(s[0][:, tt.SOMA].numpy().copy())
        X.append(s[1][:, tt.SOMA].numpy().copy())
    V, X = np.array(V).T, np.array(X).T
    r = score_traces(V, V, X, X)
    print('同じ波形: Γ 平均', r['gamma_mean'], 'χ 誤差', r['ca_err_median'])
    print('スパイク数', [x['spikes'][0] for x in r['rows']], 'バースト', [x['n_burst'][0] for x in r['rows']],
          '1 バーストのスパイク数', [round(x['spk_per_burst'][0], 2) for x in r['rows']])
    r2 = score_traces(V, np.roll(V, 500, axis=1))
    print('5 ms ずらした波形: Γ 平均', round(r2['gamma_mean'], 3), '（2 ms の窓を超えるので 1 より小さいはず）')
