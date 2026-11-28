"""
HH 代理モデルの改良実験で共通に使う道具（データ読み込みと、26 電流の一括閉ループ評価）。

    load_teacher(paths)        教師データ (I, V, m, h, n) を float32 で読む。各 (n_traj, N)
    reference()                26 電流（test_currents.TESTS）の電流と HH の正解 V。初回だけ計算してキャッシュする
    score(run, label=None)     run(I) -> V の閉ループ関数を 26 電流で一括評価する

run の約束:
    入力 I: np.ndarray (26, N)。短いテストは末尾を 0 で埋めてある。V(0) = V_REST、ゲートは定常値から始める
    出力 V: np.ndarray (26, N-1)。V[:, k] は k+1 ステップ目（hm.run_closed_loop(...)[0] と同じ並び）
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_modular as hm   # src を sys.path に入れる
import test_currents as tc
from neuron.hh import simulate, V_REST

ROOT = os.path.normpath(os.path.join(hm.BASE_DIR, '..', '..'))
DT = 0.01
CACHE = os.path.join(ROOT, 'results', 'hh_bench_reference.npz')
DATA_OU = os.path.join(ROOT, 'data', 'hh', 'teacher_ou.npz')
DATA_NEAR = os.path.join(ROOT, 'data', 'hh', 'teacher_near.npz')
CACHE_VAL = os.path.join(ROOT, 'results', 'hh_bench_val.npz')

# 検証用の電流（26 本のテストとは別。最良の時点を選ぶのに使い、26 本は最終評価だけに使う）
# 種類は 26 本と揃え、振幅・周波数・乱数の seed を変える。閾値付近（5〜9 µA/cm²）を多めにする
_c = tc.cur
VAL = [
    ('v step 7 (50-250ms)',   _c.step_current, {'amplitude': 7.0, 't_start': 50.0, 't_end': 250.0}, 300.0),
    ('v const 6',             _c.constant_current, {'amplitude': 6.0}, 300.0),
    ('v const 6.5',           _c.constant_current, {'amplitude': 6.5}, 300.0),
    ('v const 8',             _c.constant_current, {'amplitude': 8.0}, 300.0),
    ('v const 15',            _c.constant_current, {'amplitude': 15.0}, 300.0),
    ('v ramp 0-15',           _c.ramp_current, {'i_start': 0.0, 'i_end': 15.0}, 500.0),
    ('v ramp 0-25',           _c.ramp_current, {'i_start': 0.0, 'i_end': 25.0}, 500.0),
    ('v sin 8Hz 6±6',         _c.sin_current, {'amplitude': 6.0, 'freq': 8.0, 'offset': 6.0}, 500.0),
    ('v sin 20Hz 8±8',        _c.sin_current, {'amplitude': 8.0, 'freq': 20.0, 'offset': 8.0}, 300.0),
    ('v pulse 15/1ms/15ms',   _c.pulse_train_current, {'amplitude': 15.0, 'width': 1.0, 'interval': 15.0}, 300.0),
    ('v random step -5~20',   _c.random_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'interval': 20.0, 'seed': 3}, 500.0),
    ('v random step 0~20',    _c.random_pulse_current, {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0, 'seed': 4}, 500.0),
    ('v random pulse 5ms',    _c.random_timing_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'width': 5.0, 'rate': 0.05, 'seed': 5}, 500.0),
    ('v OU 6±3',              _c.ou_current, {'mean': 6.0, 'sigma': 3.0, 'tau': 1.0, 'seed': 7}, 500.0),
    ('v OU 8±4',              _c.ou_current, {'mean': 8.0, 'sigma': 4.0, 'tau': 1.0, 'seed': 8}, 500.0),
    ('v OU 7±5 (tau 5ms)',    _c.ou_current, {'mean': 7.0, 'sigma': 5.0, 'tau': 5.0, 'seed': 9}, 500.0),
    ('v level+OU',            _c.level_ou_current, {'seed': 7}, 1000.0),
]


def _ramp_hold(t, i_end=20.0, t_ramp=300.0):
    # 0 から i_end まで t_ramp [ms] かけて直線で上げ、その後は i_end のまま保つ。t: (N,) [ms] -> (N,)
    return np.where(t < t_ramp, i_end * t / t_ramp, i_end)


# 2026-11-28 に追加した検証用の電流（15 本）。C が外したのはゆっくり閾値をまたぐ場面（ランプ・低い周波数の正弦波）だったので、
# その場面を増やす。開発用 26 本（ramp 0-20・0-30、sin 5・10・50 Hz、const 2.5・5・7・10・20）と
# 最終テスト 54 本（hh_bench_ext.py、乱数で作る）とは振幅・長さ・周波数をずらしてある
VAL += [
    ('v ramp 0-12 (800ms)',   _c.ramp_current, {'i_start': 0.0, 'i_end': 12.0}, 800.0),
    ('v ramp 0-18 (500ms)',   _c.ramp_current, {'i_start': 0.0, 'i_end': 18.0}, 500.0),
    ('v ramp 0-22 (1000ms)',  _c.ramp_current, {'i_start': 0.0, 'i_end': 22.0}, 1000.0),
    ('v ramp 0-28 (300ms)',   _c.ramp_current, {'i_start': 0.0, 'i_end': 28.0}, 300.0),
    ('v ramp 25-0 (500ms)',   _c.ramp_current, {'i_start': 25.0, 'i_end': 0.0}, 500.0),
    ('v ramp-hold 16 (400ms)', _ramp_hold, {'i_end': 16.0, 't_ramp': 400.0}, 700.0),
    ('v ramp-hold 24 (250ms)', _ramp_hold, {'i_end': 24.0, 't_ramp': 250.0}, 600.0),
    ('v sin 3Hz 8±8',         _c.sin_current, {'amplitude': 8.0, 'freq': 3.0, 'offset': 8.0}, 700.0),
    ('v sin 6Hz 7±5',         _c.sin_current, {'amplitude': 5.0, 'freq': 6.0, 'offset': 7.0}, 600.0),
    ('v sin 12Hz 6±6',        _c.sin_current, {'amplitude': 6.0, 'freq': 12.0, 'offset': 6.0}, 500.0),
    ('v sin 15Hz 8±6',        _c.sin_current, {'amplitude': 6.0, 'freq': 15.0, 'offset': 8.0}, 400.0),
    ('v const 5.5',           _c.constant_current, {'amplitude': 5.5}, 300.0),
    ('v const 7.5',           _c.constant_current, {'amplitude': 7.5}, 300.0),
    ('v const 9',             _c.constant_current, {'amplitude': 9.0}, 300.0),
    ('v step 6.5 (100-400ms)', _c.step_current, {'amplitude': 6.5, 't_start': 100.0, 't_end': 400.0}, 500.0),
]


def load_teacher(paths=DATA_OU, dtype=np.float32):
    """
    Returns:
        dict: 'I', 'V', 'm', 'h', 'n' がそれぞれ (n_traj, N) の配列（dtype）。複数ファイルは軌道の向きにつなぐ
    """
    if isinstance(paths, str):
        paths = paths.split(',')
    out = {k: [] for k in ('I', 'V', 'm', 'h', 'n')}
    for p in paths:
        d = np.load(p)
        out['I'].append(d['I_ext'].astype(dtype))
        for k in ('V', 'm', 'h', 'n'):
            out[k].append(d[k].astype(dtype))
        del d
    return {k: np.concatenate(v) for k, v in out.items()}


def reference(val=False):
    """
    Returns:
        names (list[str]), I (n_cur, N) float64, V_true (list of (len_i - 1,)), lens (list[int])
        val=True なら検証用の電流（VAL）
    """
    tests, cache = (VAL, CACHE_VAL) if val else (tc.TESTS, CACHE)
    names = [n for n, *_ in tests]
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if list(d['names']) == names:
            lens = list(d['lens'])
            return names, d['I'], [d['V{}'.format(i)] for i in range(len(names))], lens
    N = int(round(max(T for *_, T in tests) / DT))
    I = np.zeros((len(names), N))
    Vt, lens = [], []
    for i, (name, func, params, T) in enumerate(tests):
        t = np.arange(0, T, DT)
        I[i, :len(t)] = func(t, **params)
        Vt.append(simulate(t, I[i, :len(t)], V0=V_REST)[0][1:])
        lens.append(len(t))
    np.savez(cache, names=np.array(names), I=I, lens=np.array(lens),
             **{'V{}'.format(i): v for i, v in enumerate(Vt)})
    return names, I, Vt, lens


def n_coincident(t_true, t_pred, window=2.0):
    # 正解の各スパイクに ±window ms 以内の予測スパイクを 1 つずつ対応させた数（近いものから、1 つの予測は 1 回だけ）
    used = np.zeros(len(t_pred), dtype=bool)
    n = 0
    for s in t_true:
        cand = np.where(~used & (np.abs(t_pred - s) <= window))[0]
        if len(cand):
            used[cand[np.argmin(np.abs(t_pred[cand] - s))]] = True
            n += 1
    return n


def coincidence_factor(t_true, t_pred, T, window=2.0):
    """
    coincidence factor Γ（Kistler et al. 1997; Jolivet et al. 2008）。

        Γ = (N_coinc - <N_coinc>) / (½ (N_true + N_pred)) · 1/𝒩
        <N_coinc> = 2 ν Δ N_true      予測が発火率 ν = N_pred / T のポアソン過程だったときの偶然の一致数
        𝒩 = 1 - 2 ν Δ                  完全に一致したとき Γ = 1 にする正規化

    Args:
        t_true, t_pred: スパイク時刻 [ms]。T: 電流の長さ [ms]。window: Δ [ms]
    Returns:
        float。両方 0 発なら 1（このプロジェクトでの約束。論文では定義されない）
    """
    n1, n2 = len(t_true), len(t_pred)
    if n1 == 0 and n2 == 0:
        return 1.0
    nu = n2 / T
    norm = 1.0 - 2.0 * nu * window
    if norm <= 0:
        return float('nan')
    return (n_coincident(t_true, t_pred, window) - 2.0 * nu * window * n1) / (0.5 * (n1 + n2)) / norm


def score(run, label=None, out_json=None, val=False):
    """
    Returns:
        dict: n100（一致率 100% の数）、mean（平均一致率）、rows（電流ごとの spikes, match, rmse）、sec
    """
    names, I, Vt, lens = reference(val)
    t0 = time.time()
    Vp = run(I)
    sec = time.time() - t0
    rows = []
    for i, name in enumerate(names):
        vp = np.asarray(Vp[i, :lens[i] - 1], dtype=float)
        if not np.all(np.isfinite(vp)):
            vp = np.nan_to_num(vp, nan=1e3, posinf=1e3, neginf=-1e3)
        st, sp = tc.spike_times(Vt[i], DT), tc.spike_times(vp, DT)
        rows.append({'name': name, 'spikes': (len(st), len(sp)), 'match': tc.match_rate(st, sp),
                     'gamma': coincidence_factor(st, sp, (lens[i] - 1) * DT),
                     'rmse': float(np.sqrt(np.mean((vp - Vt[i]) ** 2)))})
    res = {'label': label, 'n100': int(sum(r['match'] == 1.0 for r in rows)),
           'mean': float(np.mean([r['match'] for r in rows])),
           'gamma': float(np.mean([r['gamma'] for r in rows])), 'rows': rows, 'sec': sec}
    if label:
        fails = ', '.join('{} {:.0%}'.format(r['name'], r['match']) for r in rows if r['match'] < 1)
        print('[{}] 100%: {}/{}  平均 {:.3f}  Γ {:.3f}  ({:.0f} s)  不一致: {}'.format(
            label, res['n100'], len(rows), res['mean'], res['gamma'], sec, fails), flush=True)
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        json.dump(res, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return res


if __name__ == '__main__':
    # 基準: 今のモデル2（hh_modular_ou）を FusedHH（fp64）で評価
    import torch
    from hh_fused import FusedHH
    torch.set_num_threads(4)
    blocks = tc.load_blocks(os.path.join(ROOT, 'results', 'hh_modular_ou'))
    score(FusedHH(blocks, 'fp64').run, 'hh_modular_ou (fp64)')
