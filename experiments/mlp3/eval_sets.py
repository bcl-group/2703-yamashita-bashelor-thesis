"""
評価に使う電流のセットと、HH の正解・採点。

    val   (32 本)  検証。学習中に最良の時点を選ぶのに使う（旧 hh_bench.VAL と同じ。2026-11-28 に 17 → 32 本）
    test  (26 本)  テスト（開発用）。学習の最後に 1 回採点する（旧 test_currents.TESTS と同じ）
    final (54 本)  最終テスト。最終候補を決めた後に 1 回だけ使う（旧 hh_bench_ext の final と同じ）。
                   学習では自動で使わない（evaluate.py --final でだけ使う）

どのセットも、電流は t = 0 から dt = 0.01 ms 刻み。HH の正解は src/neuron/hh.py の simulate（V(0) = -65 mV、
ゲートは定常値）で、初回だけ計算して results/mlp3/_reference/<set>.npz にキャッシュする（git には入れない）。

    reference(name) -> Ref          電流と正解（V, m, h, n は初期値込みの長さ len_i）
    score(name, ref, V_pred)        スパイク時刻の一致率・Γ・RMSE を電流ごとに計算する
"""
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from neuron import current as cur            # noqa: E402
from neuron.hh import simulate, V_REST       # noqa: E402

DT = 0.01
MATCH_MS = 2.0                                # スパイク時刻がこの範囲 [ms] 内なら一致
CACHE_DIR = os.path.join(ROOT, 'results', 'mlp3', '_reference')

# (名前, 電流関数, 引数, 長さ [ms])
VAL = [
    ('v step 7 (50-250ms)',   cur.step_current, {'amplitude': 7.0, 't_start': 50.0, 't_end': 250.0}, 300.0),
    ('v const 6',             cur.constant_current, {'amplitude': 6.0}, 300.0),
    ('v const 6.5',           cur.constant_current, {'amplitude': 6.5}, 300.0),
    ('v const 8',             cur.constant_current, {'amplitude': 8.0}, 300.0),
    ('v const 15',            cur.constant_current, {'amplitude': 15.0}, 300.0),
    ('v ramp 0-15',           cur.ramp_current, {'i_start': 0.0, 'i_end': 15.0}, 500.0),
    ('v ramp 0-25',           cur.ramp_current, {'i_start': 0.0, 'i_end': 25.0}, 500.0),
    ('v sin 8Hz 6±6',         cur.sin_current, {'amplitude': 6.0, 'freq': 8.0, 'offset': 6.0}, 500.0),
    ('v sin 20Hz 8±8',        cur.sin_current, {'amplitude': 8.0, 'freq': 20.0, 'offset': 8.0}, 300.0),
    ('v pulse 15/1ms/15ms',   cur.pulse_train_current, {'amplitude': 15.0, 'width': 1.0, 'interval': 15.0}, 300.0),
    ('v random step -5~20',   cur.random_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'interval': 20.0, 'seed': 3}, 500.0),
    ('v random step 0~20',    cur.random_pulse_current, {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0, 'seed': 4}, 500.0),
    ('v random pulse 5ms',    cur.random_timing_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'width': 5.0, 'rate': 0.05, 'seed': 5}, 500.0),
    ('v OU 6±3',              cur.ou_current, {'mean': 6.0, 'sigma': 3.0, 'tau': 1.0, 'seed': 7}, 500.0),
    ('v OU 8±4',              cur.ou_current, {'mean': 8.0, 'sigma': 4.0, 'tau': 1.0, 'seed': 8}, 500.0),
    ('v OU 7±5 (tau 5ms)',    cur.ou_current, {'mean': 7.0, 'sigma': 5.0, 'tau': 5.0, 'seed': 9}, 500.0),
    ('v level+OU',            cur.level_ou_current, {'seed': 7}, 1000.0),
]


def _ramp_hold(t, i_end=20.0, t_ramp=300.0):
    # 0 から i_end まで t_ramp [ms] かけて直線で上げ、その後は i_end のまま保つ。t: (N,) [ms] -> (N,)
    return np.where(t < t_ramp, i_end * t / t_ramp, i_end)


# 2026-11-28 に追加した 15 本。C が外したのはゆっくり閾値をまたぐ場面（ランプ・低い周波数の正弦波）だったので増やす。
# テスト 26 本・最終テスト 54 本とは振幅・長さ・周波数をずらしてある
VAL += [
    ('v ramp 0-12 (800ms)',    cur.ramp_current, {'i_start': 0.0, 'i_end': 12.0}, 800.0),
    ('v ramp 0-18 (500ms)',    cur.ramp_current, {'i_start': 0.0, 'i_end': 18.0}, 500.0),
    ('v ramp 0-22 (1000ms)',   cur.ramp_current, {'i_start': 0.0, 'i_end': 22.0}, 1000.0),
    ('v ramp 0-28 (300ms)',    cur.ramp_current, {'i_start': 0.0, 'i_end': 28.0}, 300.0),
    ('v ramp 25-0 (500ms)',    cur.ramp_current, {'i_start': 25.0, 'i_end': 0.0}, 500.0),
    ('v ramp-hold 16 (400ms)', _ramp_hold, {'i_end': 16.0, 't_ramp': 400.0}, 700.0),
    ('v ramp-hold 24 (250ms)', _ramp_hold, {'i_end': 24.0, 't_ramp': 250.0}, 600.0),
    ('v sin 3Hz 8±8',          cur.sin_current, {'amplitude': 8.0, 'freq': 3.0, 'offset': 8.0}, 700.0),
    ('v sin 6Hz 7±5',          cur.sin_current, {'amplitude': 5.0, 'freq': 6.0, 'offset': 7.0}, 600.0),
    ('v sin 12Hz 6±6',         cur.sin_current, {'amplitude': 6.0, 'freq': 12.0, 'offset': 6.0}, 500.0),
    ('v sin 15Hz 8±6',         cur.sin_current, {'amplitude': 6.0, 'freq': 15.0, 'offset': 8.0}, 400.0),
    ('v const 5.5',            cur.constant_current, {'amplitude': 5.5}, 300.0),
    ('v const 7.5',            cur.constant_current, {'amplitude': 7.5}, 300.0),
    ('v const 9',              cur.constant_current, {'amplitude': 9.0}, 300.0),
    ('v step 6.5 (100-400ms)', cur.step_current, {'amplitude': 6.5, 't_start': 100.0, 't_end': 400.0}, 500.0),
]

TEST = [
    ('step 10 (50-150ms)',      cur.step_current, {'amplitude': 10.0, 't_start': 50.0, 't_end': 150.0}, 300.0),
    ('step 5 (50-250ms)',       cur.step_current, {'amplitude': 5.0, 't_start': 50.0, 't_end': 250.0}, 300.0),
    ('step 30 1ms',             cur.step_current, {'amplitude': 30.0, 't_start': 50.0, 't_end': 51.0}, 150.0),
    ('step -5 (50-150ms)',      cur.step_current, {'amplitude': -5.0, 't_start': 50.0, 't_end': 150.0}, 300.0),
    ('const 2.5',               cur.constant_current, {'amplitude': 2.5}, 300.0),
    ('const 5',                 cur.constant_current, {'amplitude': 5.0}, 300.0),
    ('const 7',                 cur.constant_current, {'amplitude': 7.0}, 300.0),
    ('const 10',                cur.constant_current, {'amplitude': 10.0}, 300.0),
    ('const 20',                cur.constant_current, {'amplitude': 20.0}, 300.0),
    ('ramp 0-20',               cur.ramp_current, {'i_start': 0.0, 'i_end': 20.0}, 500.0),
    ('ramp 0-30',               cur.ramp_current, {'i_start': 0.0, 'i_end': 30.0}, 500.0),
    ('sin 5Hz 5±5',             cur.sin_current, {'amplitude': 5.0, 'freq': 5.0, 'offset': 5.0}, 500.0),
    ('sin 10Hz 7.5±7.5',        cur.sin_current, {'amplitude': 7.5, 'freq': 10.0, 'offset': 7.5}, 500.0),
    ('sin 50Hz 10±10',          cur.sin_current, {'amplitude': 10.0, 'freq': 50.0, 'offset': 10.0}, 300.0),
    ('pulse 10/1ms/10ms',       cur.pulse_train_current, {'amplitude': 10.0, 'width': 1.0, 'interval': 10.0}, 300.0),
    ('pulse 20/25ms/50ms',      cur.pulse_train_current, {'amplitude': 20.0, 'width': 25.0, 'interval': 50.0}, 300.0),
    ('pulse 40/0.5ms/20ms',     cur.pulse_train_current, {'amplitude': 40.0, 'width': 0.5, 'interval': 20.0}, 300.0),
    ('random step -5~20/20ms',  cur.random_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'interval': 20.0, 'seed': 1}, 500.0),
    ('random step 0~20/10ms',   cur.random_pulse_current, {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0, 'seed': 1}, 500.0),
    ('random pulse 5ms',        cur.random_timing_pulse_current, {'i_min': -5.0, 'i_max': 20.0, 'width': 5.0, 'rate': 0.05, 'seed': 1}, 500.0),
    ('random pulse 1ms 0~40',   cur.random_timing_pulse_current, {'i_min': 0.0, 'i_max': 40.0, 'width': 1.0, 'rate': 0.02, 'seed': 2}, 500.0),
    ('OU 5±3',                  cur.ou_current, {'mean': 5.0, 'sigma': 3.0, 'tau': 1.0, 'seed': 1}, 500.0),
    ('OU 10±3',                 cur.ou_current, {'mean': 10.0, 'sigma': 3.0, 'tau': 1.0, 'seed': 1}, 500.0),
    ('OU 7±5 (tau 5ms)',        cur.ou_current, {'mean': 7.0, 'sigma': 5.0, 'tau': 5.0, 'seed': 1}, 500.0),
    ('level+OU',                cur.level_ou_current, {'seed': 99}, 1000.0),
    ('munechika train',         cur.munechika_train_current, {}, 2000.0),
]


def _defs(name):
    # [(名前, f(t) -> I, 長さ [ms])]
    if name == 'final':
        # 54 本の定義は乱数の引き方まで固定されているので、旧コードのものをそのまま使う
        sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
        import hh_bench_ext
        return hh_bench_ext._final_defs()
    sets = {'val': VAL, 'test': TEST}
    return [(n, (lambda t, f=f, p=p: f(t, **p)), T) for n, f, p, T in sets[name]]


@dataclass
class Ref:
    """
    names: list[str]、lens: (n,) int（各電流の点数 = 長さ / dt）
    I: (n, N_max) float64。短い電流は末尾を 0 で埋める（まとめて閉ループに入れるため）
    V, m, h, n: list of (len_i,) float64。HH の正解（初期値込み）
    """
    name: str
    names: list
    lens: np.ndarray
    I: np.ndarray
    V: list
    m: list
    h: list
    n: list


def reference(name):
    path = os.path.join(CACHE_DIR, name + '.npz')
    defs = _defs(name)
    names = [n for n, _, _ in defs]
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if list(d['names']) == names:
            cut = np.cumsum(d['lens'])[:-1]
            return Ref(name, names, d['lens'], d['I'], *[np.split(d[k], cut) for k in 'Vmhn'])
    t0 = time.time()
    lens = np.array([len(np.arange(0, T, DT)) for *_, T in defs])
    I = np.zeros((len(defs), lens.max()))
    tr = {k: [] for k in 'Vmhn'}
    for i, (_, f, T) in enumerate(defs):
        t = np.arange(0, T, DT)
        I[i, :len(t)] = f(t)
        for k, x in zip('Vmhn', simulate(t, I[i, :len(t)], V0=V_REST)):
            tr[k].append(x)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, names=np.array(names), lens=lens, I=I, **{k: np.concatenate(v) for k, v in tr.items()})
    print('HH の正解を計算: {} {} 本 ({:.0f} s)'.format(name, len(defs), time.time() - t0), flush=True)
    return Ref(name, names, lens, I, tr['V'], tr['m'], tr['h'], tr['n'])


# ---------------- 採点 ----------------

def spike_times(V, dt=DT):
    # 0 mV を下から上に横切った時刻 [ms]
    return (np.where((V[:-1] < 0.0) & (V[1:] >= 0.0))[0] + 1) * dt


def n_coincident(t_true, t_pred, window=MATCH_MS):
    # 正解の各スパイクに ±window ms 以内の予測スパイクを 1 つずつ対応させた数（近いものから、1 つの予測は 1 回だけ）
    used = np.zeros(len(t_pred), dtype=bool)
    n = 0
    for s in t_true:
        cand = np.where(~used & (np.abs(t_pred - s) <= window))[0]
        if len(cand):
            used[cand[np.argmin(np.abs(t_pred[cand] - s))]] = True
            n += 1
    return n


def match_rate(t_true, t_pred, window=MATCH_MS):
    # 一致率 = 対応できた数 / max(正解の数, 予測の数)。両方 0 発なら 1
    if len(t_true) == 0 and len(t_pred) == 0:
        return 1.0
    return n_coincident(t_true, t_pred, window) / max(len(t_true), len(t_pred))


def coincidence_factor(t_true, t_pred, T, window=MATCH_MS):
    """
    Γ = (N_coinc - <N_coinc>) / (½ (N_true + N_pred)) / 𝒩    （Kistler et al. 1997; Jolivet et al. 2008）
        <N_coinc> = 2 ν Δ N_true   予測が発火率 ν = N_pred / T のポアソン過程だったときの偶然の一致数
        𝒩 = 1 - 2 ν Δ               完全に一致したとき Γ = 1 にする正規化
    両方 0 発なら 1（このプロジェクトの約束）
    """
    n1, n2 = len(t_true), len(t_pred)
    if n1 == 0 and n2 == 0:
        return 1.0
    nu = n2 / T
    norm = 1.0 - 2.0 * nu * window
    if norm <= 0:
        return float('nan')
    return (n_coincident(t_true, t_pred, window) - 2.0 * nu * window * n1) / (0.5 * (n1 + n2)) / norm


def score(ref, V_pred, label=None):
    """
    V_pred: (n, N_max - 1)。V_pred[i, k] は k+1 ステップ目。正解 V[i][1:] と比べる（旧 hh_bench.score と同じ）
    Returns: dict  n100（一致率 100% の本数）, mean（平均一致率）, gamma（平均 Γ）, rows（電流ごと）
    """
    rows = []
    for i, name in enumerate(ref.names):
        vt = ref.V[i][1:]
        vp = np.asarray(V_pred[i, :ref.lens[i] - 1], dtype=float)
        diverged = bool(not np.all(np.isfinite(vp)))
        vp = np.nan_to_num(vp, nan=1e3, posinf=1e3, neginf=-1e3)
        st, sp = spike_times(vt), spike_times(vp)
        rows.append({'name': name, 'spikes': (len(st), len(sp)), 'match': match_rate(st, sp),
                     'gamma': coincidence_factor(st, sp, (ref.lens[i] - 1) * DT),
                     'rmse': float(np.sqrt(np.mean((vp - vt) ** 2))), 'diverged': diverged})
    res = {'set': ref.name, 'n': len(rows), 'n100': int(sum(r['match'] == 1.0 for r in rows)),
           'mean': float(np.mean([r['match'] for r in rows])),
           'gamma': float(np.nanmean([r['gamma'] for r in rows])), 'rows': rows}
    if label:
        fails = ', '.join('{} {:.0%}'.format(r['name'], r['match']) for r in rows if r['match'] < 1)
        print('[{} {}] 100%: {}/{}  平均 {:.3f}  Γ {:.3f}  不一致: {}'.format(
            label, ref.name, res['n100'], res['n'], res['mean'], res['gamma'], fails), flush=True)
    return res
