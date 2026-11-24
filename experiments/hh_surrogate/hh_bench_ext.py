"""
26 本のテスト電流を補う追加のテスト（HH 代理モデル用）。4 つのセットに分ける。

    fi    : f–I 曲線。定常電流 0〜30 µA/cm² を 0.5 刻み（61 本）、各 1000 ms。
            後半 500 ms のスパイク数から発火率 [Hz] を出し、反復発火が始まる電流（後半に 2 発以上）を比べる
    seeds : 系統ごとに seed を 10 本ずつ（5 系統 × 10 = 50 本、各 500 ms）。26 本・検証 17 本とは別の seed
            OU 5±3、OU 10±3、OU 7±5（τ 5 ms）、ランダムステップ −5〜20 / 20 ms、ランダムなタイミングのパルス 5 ms
    long  : 長時間（各 20 s）。OU 7±5（τ 5 ms）を 4 本、段階 + OU を 2 本。発散と、前半 5 s と後半 5 s の Γ の差を見る
    ood   : 学習の範囲の外。強い抑制（−20）、強い入力（60）、チャープ（0.5→200 Hz）、とても遅いランプ（0→10 を 2 s、
            0→15 を 3 s）、下りランプ（20→0）、白色雑音に近い速い OU（τ 0.1 ms、σ 10）
    final : 最終テスト（54 本、2026-11-22 に固定）。上の 4 セットと 26 本は試行錯誤に使ったので「開発用」とし、
            最終候補の比較にはこれだけを使う。6 グループ（threshold / transient / tracking / fluct / ood / long）。
            SUITES には入れていない（うっかり途中で見ないため）。詳細は _final_defs()

正解は src/neuron/hh.py の陽的オイラー法（dt = 0.01 ms、静止状態から）。初回だけ計算して
results/hh_bench_ext/<セット>.npz にキャッシュする（大きいので git には入れない）。

使い方:
    import hh_bench_ext as hx
    res = hx.score_suite(run, 'fi', label='mm_rel')     # run(I (B, N)) -> V (B, N-1)。hh_bench.score と同じ約束
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_bench as hb          # noqa: E402
import test_currents as tc     # noqa: E402
import clamp_track as ct       # noqa: E402
from neuron.hh import simulate, V_REST   # noqa: E402

DT = hb.DT
CUR = tc.cur
CACHE_DIR = os.path.join(hb.ROOT, 'results', 'hh_bench_ext')
FI_AMPS = np.round(np.arange(0.0, 30.0 + 1e-9, 0.5), 2)
FI_MS = 1000.0
LONG_MS = 20000.0


def _chirp(t, f0=0.5, f1=200.0, amp=10.0, offset=10.0):
    # 周波数が f0 から f1 [Hz] まで指数的に上がる正弦波。t [ms]
    T = t[-1] if t[-1] > 0 else 1.0
    k = np.log(f1 / f0) / T
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k / 1000.0
    return offset + amp * np.sin(phase)


def _ramp(t, a, b):
    return a + (b - a) * t / t[-1]


def suite_defs(name):
    """
    Returns:
        list of (名前, 関数 f(t) -> I, 長さ [ms])
    """
    if name == 'fi':
        return [('const {:.1f}'.format(a), (lambda t, a=a: CUR.constant_current(t, amplitude=a)), FI_MS)
                for a in FI_AMPS]
    if name == 'seeds':
        fams = [
            ('OU 5±3', lambda t, s: CUR.ou_current(t, mean=5.0, sigma=3.0, tau=1.0, seed=s)),
            ('OU 10±3', lambda t, s: CUR.ou_current(t, mean=10.0, sigma=3.0, tau=1.0, seed=s)),
            ('OU 7±5 (tau 5ms)', lambda t, s: CUR.ou_current(t, mean=7.0, sigma=5.0, tau=5.0, seed=s)),
            ('random step -5~20/20ms', lambda t, s: CUR.random_pulse_current(t, i_min=-5.0, i_max=20.0,
                                                                             interval=20.0, seed=s)),
            ('random pulse 5ms', lambda t, s: CUR.random_timing_pulse_current(t, i_min=-5.0, i_max=20.0, width=5.0,
                                                                              rate=0.05, seed=s)),
        ]
        return [('{} seed {}'.format(n, s), (lambda t, f=f, s=s: f(t, s)), 500.0)
                for n, f in fams for s in range(100, 110)]
    if name == 'long':
        out = [('OU 7±5 (tau 5ms) seed {}'.format(s),
                (lambda t, s=s: CUR.ou_current(t, mean=7.0, sigma=5.0, tau=5.0, seed=s)), LONG_MS)
               for s in range(200, 204)]
        out += [('level+OU seed {}'.format(s), (lambda t, s=s: CUR.level_ou_current(t, seed=s)), LONG_MS)
                for s in (300, 301)]
        return out
    if name == 'ood':
        return [
            ('step -20 (50-250ms)', lambda t: CUR.step_current(t, amplitude=-20.0, t_start=50.0, t_end=250.0), 400.0),
            ('const 60', lambda t: CUR.constant_current(t, amplitude=60.0), 300.0),
            ('step 60 (50-150ms)', lambda t: CUR.step_current(t, amplitude=60.0, t_start=50.0, t_end=150.0), 300.0),
            ('chirp 0.5-200Hz 10±10', lambda t: _chirp(t), 2000.0),
            ('slow ramp 0-10 (2 s)', lambda t: _ramp(t, 0.0, 10.0), 2000.0),
            ('slow ramp 0-15 (3 s)', lambda t: _ramp(t, 0.0, 15.0), 3000.0),
            ('down ramp 20-0', lambda t: _ramp(t, 20.0, 0.0), 500.0),
            ('fast OU 7±10 (tau 0.1ms)', lambda t: CUR.ou_current(t, mean=7.0, sigma=10.0, tau=0.1, seed=500), 500.0),
        ]
    if name == 'final':
        return _final_defs()
    raise ValueError(name)


SUITES = ['fi', 'seeds', 'long', 'ood']


# ---------------------------------------------------------------------------------------------
# 最終テスト（final）。2026-11-22 に作って固定した。試行錯誤には使わず、最終候補の比較で 1 回だけ使う。
#   - それまでのテスト 26 本・検証 17 本・追加 125 本は、結果を見て学習の電流（hard v1 / v2）を直したので
#     「開発用」として扱う。final はそれらとは別の乱数（FINAL_SEED）で、種類ごとにパラメータを引いて作る
#   - 名前は「グループ: 電流」。グループごとに Γ を集計する（score_suite の gamma_by_group）
#   - in / out は学習に使った電流（hh_improve.py の gen_current, gen_hard, gen_hard_v2）の範囲の内か外か
#       学習の範囲: 水準 −5〜30（v2 のランプは 40 まで）、OU の τ 1〜10 ms、正弦波 2〜40 Hz、
#                   0 から上がるランプの傾き 0.008〜0.2 µA/cm²/ms、1 本の長さ 3 s 以下
# 電流は 50 ms 休んで（I = 0）から入れる。静止状態から刺激が始まる場面を必ず含めるため
# ---------------------------------------------------------------------------------------------
FINAL_SEED = 20261122
FINAL_GROUPS = ['threshold', 'transient', 'tracking', 'fluct', 'ood', 'long']
FINAL_REST = 50.0


def _after_rest(f, rest=FINAL_REST):
    # 最初の rest [ms] は 0、その後は f(t - rest)
    def g(t):
        I = np.zeros_like(t)
        m = t >= rest
        I[m] = f(t[m] - rest) if np.any(m) else I[m]
        return I
    return g


def _ramp_hold(a, b, L):
    # a から b まで L [ms] かけて直線で上げ、その後は b を保つ
    return lambda t: a + (b - a) * np.clip(t / L, 0.0, 1.0)


def _final_defs():
    """
    Returns:
        list of (名前, 関数 f(t) -> I, 長さ [ms])。乱数は FINAL_SEED だけで決まる（引く順番も固定）
    """
    rng = np.random.default_rng(FINAL_SEED)
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    LU = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))   # 対数一様
    S = lambda: int(rng.integers(10 ** 6, 10 ** 7))                          # 電流関数に渡す seed
    out = []

    # 1. threshold（in）: 静止から一定電流。半分は反復発火が始まる付近（HH は約 6.3 から）
    lv = [U(5.0, 10.0) for _ in range(5)] + [U(1.0, 5.0) for _ in range(2)] + [U(10.0, 25.0) for _ in range(3)]
    for a in sorted(lv):
        out.append(('threshold: const {:.2f}'.format(a), _after_rest(lambda t, a=a: np.full_like(t, a)), 1000.0))

    # 2. transient（in）: ステップの on/off、短いパルス、負のステップ（終わった後のリバウンド発火）
    for _ in range(4):
        a, d = U(3.0, 25.0), U(20.0, 200.0)
        out.append(('transient: step {:.1f} for {:.0f}ms'.format(a, d),
                    _after_rest(lambda t, a=a, d=d: np.where(t < d, a, 0.0)), 400.0))
    for _ in range(3):
        a, w = U(10.0, 40.0), U(0.5, 2.0)
        out.append(('transient: pulse {:.1f} {:.2f}ms'.format(a, w),
                    _after_rest(lambda t, a=a, w=w: np.where(t < w, a, 0.0)), 200.0))
    for _ in range(3):
        a, d = U(-5.0, -2.0), U(20.0, 200.0)
        out.append(('transient: step {:.1f} for {:.0f}ms'.format(a, d),
                    _after_rest(lambda t, a=a, d=d: np.where(t < d, a, 0.0)), 400.0))

    # 3. tracking（in）: 正弦波、パルス列、0 からのランプ（傾き 0.01〜0.1）
    for _ in range(4):
        f, off = LU(2.0, 40.0), U(2.0, 12.0)
        amp = off if rng.random() < 0.5 else U(2.0, 10.0)
        out.append(('tracking: sin {:.1f}Hz {:.1f}±{:.1f}'.format(f, off, amp),
                    _after_rest(lambda t, f=f, o=off, a=amp: CUR.sin_current(t, amplitude=a, freq=f, offset=o)),
                    1000.0))
    for _ in range(3):
        a, w, itv = U(5.0, 30.0), U(0.5, 5.0), U(10.0, 60.0)
        out.append(('tracking: pulse {:.1f}/{:.1f}ms/{:.0f}ms'.format(a, w, itv),
                    _after_rest(lambda t, a=a, w=w, i=itv: CUR.pulse_train_current(t, amplitude=a, width=w,
                                                                                    interval=i)), 1000.0))
    for _ in range(3):
        b, L = U(10.0, 30.0), U(300.0, 1000.0)
        out.append(('tracking: ramp 0-{:.1f} in {:.0f}ms'.format(b, L), _after_rest(_ramp_hold(0.0, b, L)), 1000.0))

    # 4. fluct（in）: OU とランダム段階電流
    for _ in range(6):
        m, s, tau, sd = U(3.0, 12.0), U(1.0, 5.0), LU(1.0, 10.0), S()
        out.append(('fluct: OU {:.1f}±{:.1f} tau {:.1f}ms'.format(m, s, tau),
                    _after_rest(lambda t, m=m, s=s, tau=tau, sd=sd: CUR.ou_current(t, mean=m, sigma=s, tau=tau,
                                                                                   seed=sd)), 1000.0))
    for _ in range(4):
        itv, sd = U(10.0, 50.0), S()
        out.append(('fluct: random step -5~20/{:.0f}ms'.format(itv),
                    _after_rest(lambda t, i=itv, sd=sd: CUR.random_pulse_current(t, i_min=-5.0, i_max=20.0,
                                                                                  interval=i, seed=sd)), 1000.0))

    # 5. ood（out）: 学習の範囲の外。強い入力、強い抑制、速い正弦波、遅い・速い OU、とても遅いランプ
    for _ in range(2):
        a = U(45.0, 70.0)
        out.append(('ood: const {:.1f}'.format(a), _after_rest(lambda t, a=a: np.full_like(t, a)), 500.0))
    for _ in range(2):
        a, d = U(-20.0, -10.0), U(50.0, 200.0)
        out.append(('ood: step {:.1f} for {:.0f}ms'.format(a, d),
                    _after_rest(lambda t, a=a, d=d: np.where(t < d, a, 0.0)), 400.0))
    for _ in range(2):
        f, off, amp = LU(60.0, 150.0), U(5.0, 15.0), U(5.0, 15.0)
        out.append(('ood: sin {:.0f}Hz {:.1f}±{:.1f}'.format(f, off, amp),
                    _after_rest(lambda t, f=f, o=off, a=amp: CUR.sin_current(t, amplitude=a, freq=f, offset=o)),
                    1000.0))
    for _ in range(2):
        m, s, tau, sd = U(5.0, 10.0), U(2.0, 5.0), U(20.0, 50.0), S()
        out.append(('ood: slow OU {:.1f}±{:.1f} tau {:.0f}ms'.format(m, s, tau),
                    _after_rest(lambda t, m=m, s=s, tau=tau, sd=sd: CUR.ou_current(t, mean=m, sigma=s, tau=tau,
                                                                                   seed=sd)), 2000.0))
    m, s, tau, sd = U(5.0, 10.0), U(5.0, 10.0), U(0.2, 0.5), S()
    out.append(('ood: fast OU {:.1f}±{:.1f} tau {:.2f}ms'.format(m, s, tau),
                _after_rest(lambda t, m=m, s=s, tau=tau, sd=sd: CUR.ou_current(t, mean=m, sigma=s, tau=tau, seed=sd)),
                1000.0))
    b, L = U(12.0, 18.0), U(4000.0, 6000.0)
    out.append(('ood: very slow ramp 0-{:.1f} in {:.0f}ms'.format(b, L), _after_rest(_ramp_hold(0.0, b, L)),
                FINAL_REST + L))

    # 6. long（電流の種類は in、長さが out）: OU を 20 s
    for _ in range(4):
        m, s, tau, sd = U(5.0, 10.0), U(2.0, 5.0), LU(1.0, 10.0), S()
        out.append(('long: OU {:.1f}±{:.1f} tau {:.1f}ms 20s'.format(m, s, tau),
                    _after_rest(lambda t, m=m, s=s, tau=tau, sd=sd: CUR.ou_current(t, mean=m, sigma=s, tau=tau,
                                                                                   seed=sd)), LONG_MS))
    return out


def reference(name):
    """
    Returns:
        names (list[str]), I (n, N) float64（短い電流は末尾を 0 で埋める）, V_true (list of (len_i - 1,)), lens (list[int])
    """
    defs = suite_defs(name)
    names = [n for n, _, _ in defs]
    path = os.path.join(CACHE_DIR, name + '.npz')
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if list(d['names']) == names:
            return names, d['I'], [d['V{}'.format(i)] for i in range(len(names))], list(d['lens'])
    os.makedirs(CACHE_DIR, exist_ok=True)
    N = int(round(max(T for *_, T in defs) / DT))
    I = np.zeros((len(defs), N))
    Vt, lens = [], []
    t0 = time.time()
    for i, (n, f, T) in enumerate(defs):
        t = np.arange(0, T, DT)
        I[i, :len(t)] = f(t)
        Vt.append(simulate(t, I[i, :len(t)], V0=V_REST)[0][1:])
        lens.append(len(t))
    print('正解の HH を計算: {} ({} 本, {:.0f} s)'.format(name, len(defs), time.time() - t0), flush=True)
    np.savez(path, names=np.array(names), I=I, lens=np.array(lens), **{'V{}'.format(i): v for i, v in enumerate(Vt)})
    return names, I, Vt, lens


def fingerprint(I):
    # 電流の配列の SHA-256 の先頭 16 文字。final が作った日から変わっていないことの確認に使う
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.round(I, 9)).tobytes()).hexdigest()[:16]


SHAPE_KEYS = ['peak', 'width', 'ahp']   # AP のピーク [mV]、0 mV より上にいる時間 [ms]、直後の谷 [mV]


def spike_shape(V, dt, ahp_ms=20.0):
    """
    スパイクの形の特徴量を、スパイクごとに出す。スパイクは 0 mV を下から上に横切ってから上から下に横切るまで。
        peak : その間の V の最大値 [mV]
        width: その間の長さ [ms]（0 mV での幅）
        ahp  : 下に横切ってから、次のスパイクか ahp_ms [ms] 後までの V の最小値 [mV]（後過分極の深さ）
    Returns:
        dict: 各キー -> (スパイク数,) の配列。最後まで下に横切らないスパイクは除く
    """
    above = V >= 0.0
    up = np.flatnonzero(~above[:-1] & above[1:]) + 1
    down = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    out = {k: [] for k in SHAPE_KEYS}
    n_ahp = int(round(ahp_ms / dt))
    for j, u in enumerate(up):
        d = down[np.searchsorted(down, u)] if np.searchsorted(down, u) < len(down) else None
        if d is None:
            break
        nxt = up[j + 1] if j + 1 < len(up) else len(V)
        out['peak'].append(float(V[u:d].max()))
        out['width'].append((d - u) * dt)
        out['ahp'].append(float(V[d:min(nxt, d + n_ahp)].min()))
    return {k: np.array(v) for k, v in out.items()}


def shape_error(v_true, v_pred, dt):
    # 各特徴量の電流内の平均を比べた誤差（予測 − 正解）。どちらかにスパイクがなければ None
    a, b = spike_shape(v_true, dt), spike_shape(v_pred, dt)
    if len(a['peak']) == 0 or len(b['peak']) == 0:
        return None
    return {k: float(b[k].mean() - a[k].mean()) for k in SHAPE_KEYS}


def _shape_summary(rows):
    # 特徴量ごとの平均絶対誤差（両方にスパイクがある電流だけ）と、使った電流の本数
    ok = [r['shape'] for r in rows if r['shape'] is not None]
    if not ok:
        return {'n': 0}
    return {'n': len(ok), **{k: float(np.mean([abs(s[k]) for s in ok])) for k in SHAPE_KEYS}}


def _range_summary(rows):
    # 状態が範囲の外に出た割合の平均、はみ出した最大量、外に出た電流の本数（記録がないモデルは空）
    names = sorted({n for r in rows for n in r['range']})
    return {n: {'frac_mean': float(np.mean([r['range'][n]['frac'] for r in rows])),
                'max': float(np.max([r['range'][n]['max'] for r in rows])),
                'n_rows_out': int(sum(r['range'][n]['frac'] > 0 for r in rows))} for n in names}


def _rate(st, t_from, t_to):
    # [t_from, t_to) ms のスパイク数から発火率 [Hz]
    return float(np.sum((st >= t_from) & (st < t_to))) / ((t_to - t_from) / 1000.0)


def score_suite(run, name, label=None):
    """
    Returns:
        dict: rows（電流ごと: spikes, match, gamma, rmse, diverged と、セットごとの追加の値）と、セット全体の要約 summary
    """
    names, I, Vt, lens = reference(name)
    t0 = time.time()
    ct.start([n - 1 for n in lens])         # 閉ループの中で状態が範囲の外に出た量を記録する（clamp_track.py）
    try:
        Vp = run(I)
    finally:
        rng_stats = ct.stop()
    sec = time.time() - t0
    rows = []
    for i, n in enumerate(names):
        vp = np.asarray(Vp[i, :lens[i] - 1], dtype=float)
        div = bool((not np.all(np.isfinite(vp))) or np.max(np.abs(np.nan_to_num(vp, nan=1e9))) > 1e3)
        vp = np.nan_to_num(vp, nan=1e3, posinf=1e3, neginf=-1e3)
        T = (lens[i] - 1) * DT
        st, sp = tc.spike_times(Vt[i], DT), tc.spike_times(vp, DT)
        r = {'name': n, 'spikes': (len(st), len(sp)), 'match': tc.match_rate(st, sp),
             'gamma': hb.coincidence_factor(st, sp, T), 'rmse': float(np.sqrt(np.mean((vp - Vt[i]) ** 2))),
             'diverged': div, 'shape': shape_error(Vt[i], vp, DT),
             'range': {k: {'frac': v['frac'][i], 'max': v['max'][i]} for k, v in rng_stats.items()}}
        if name == 'fi':
            r['rate'] = (_rate(st, 500.0, 1000.0), _rate(sp, 500.0, 1000.0))
        if name == 'long':
            first = (st < 5000.0, sp < 5000.0)
            last = (st >= T - 5000.0, sp >= T - 5000.0)
            r['gamma_first5s'] = hb.coincidence_factor(st[first[0]], sp[first[1]], 5000.0)
            r['gamma_last5s'] = hb.coincidence_factor(st[last[0]] - (T - 5000.0), sp[last[1]] - (T - 5000.0), 5000.0)
        rows.append(r)
    summ = {'n': len(rows), 'n100': int(sum(r['match'] == 1.0 for r in rows)),
            'mean': float(np.mean([r['match'] for r in rows])), 'gamma': float(np.nanmean([r['gamma'] for r in rows])),
            'n_diverged': int(sum(r['diverged'] for r in rows)), 'sec': sec,
            'shape': _shape_summary(rows), 'range': _range_summary(rows)}
    if name == 'fi':
        amps = FI_AMPS
        rt = np.array([r['rate'][0] for r in rows])
        rp = np.array([r['rate'][1] for r in rows])
        rep = lambda rr: float(amps[np.argmax(rr >= 4.0)]) if np.any(rr >= 4.0) else float('nan')   # 後半 500 ms に 2 発以上
        summ.update({'rate_rmse_hz': float(np.sqrt(np.mean((rp - rt) ** 2))),
                     'rate_exact': int(np.sum(np.abs(rp - rt) < 1e-9)),
                     'rheobase_rep': (rep(rt), rep(rp))})
    if name == 'long':
        summ.update({'gamma_first5s': float(np.nanmean([r['gamma_first5s'] for r in rows])),
                     'gamma_last5s': float(np.nanmean([r['gamma_last5s'] for r in rows]))})
    if name == 'final':
        grp = {g: [] for g in FINAL_GROUPS}
        for r in rows:
            grp[r['name'].split(':', 1)[0]].append(r)
        summ['gamma_by_group'] = {g: (float(np.nanmean([r['gamma'] for r in v])),
                                      int(sum(r['match'] == 1.0 for r in v)), len(v)) for g, v in grp.items()}
        summ['shape_by_group'] = {g: _shape_summary(v) for g, v in grp.items()}
        summ['range_by_group'] = {g: _range_summary(v) for g, v in grp.items()}
        summ['fingerprint'] = fingerprint(I)
    if name == 'seeds':
        fam = {}
        for r in rows:
            fam.setdefault(r['name'].rsplit(' seed ', 1)[0], []).append(r['gamma'])
        summ['gamma_by_family'] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in fam.items()}
    if label:
        extra = ''
        if name == 'fi':
            extra = '  発火率 RMSE {:.1f} Hz  反復発火の開始 正解 {} / 予測 {}'.format(
                summ['rate_rmse_hz'], *summ['rheobase_rep'])
        if name == 'final':
            extra = '\n    ' + '  '.join('{} {:.3f}({}/{})'.format(g, *v) for g, v in summ['gamma_by_group'].items())
        sh = summ['shape']
        if sh['n']:
            extra += '\n    形（{} 本）: ピーク {:.2f} mV  幅 {:.3f} ms  谷 {:.2f} mV'.format(
                sh['n'], sh['peak'], sh['width'], sh['ahp'])
        for k, v in summ['range'].items():
            # gate の [0, 1] は物理的な範囲。V の [-100, 60] mV は閉ループの安全装置で、HH 自身も強い抑制で
            # −120 mV まで下がるので「ありえない状態」ではない（切り詰めが効いた = 出力が歪んだ、と読む）
            head = 'ゲートが [0, 1] の外' if k == 'gate' else 'V が切り詰め [-100, 60] mV に当たった'
            extra += '\n    {}: 平均 {:.2e}  最大 {:.3g}  {} 本'.format(head, v['frac_mean'], v['max'], v['n_rows_out'])
        if name == 'long':
            extra = '  Γ 前半 5 s {:.3f} / 後半 5 s {:.3f}'.format(summ['gamma_first5s'], summ['gamma_last5s'])
        print('[{} {}] 100%: {}/{}  平均 {:.3f}  Γ {:.3f}  発散 {}{}  ({:.0f} s)'.format(
            label, name, summ['n100'], summ['n'], summ['mean'], summ['gamma'], summ['n_diverged'], extra, sec),
            flush=True)
    return {'suite': name, 'label': label, 'summary': summ, 'rows': rows}


if __name__ == '__main__':
    # 正解の HH を計算してキャッシュするだけ
    for s in (sys.argv[1:] or SUITES):
        n, I, Vt, lens = reference(s)
        if s == 'final':
            print('final の指紋:', fingerprint(I))
            for nm, ln, v in zip(n, lens, Vt):
                print('  {:48s} {:6.0f} ms  HH {:3d} 発'.format(nm, ln * DT, len(tc.spike_times(v, DT))))
        print(s, len(n), '本', '最長 {:.0f} ms'.format(max(lens) * DT),
              'スパイク数', [len(tc.spike_times(v, DT)) for v in Vt][:8], '...')
