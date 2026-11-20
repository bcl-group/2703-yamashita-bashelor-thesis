"""
26 本のテスト電流を補う追加のテスト（HH 代理モデル用）。4 つのセットに分ける。

    fi    : f–I 曲線。定常電流 0〜30 µA/cm² を 0.5 刻み（61 本）、各 1000 ms。
            後半 500 ms のスパイク数から発火率 [Hz] を出し、反復発火が始まる電流（後半に 2 発以上）を比べる
    seeds : 系統ごとに seed を 10 本ずつ（5 系統 × 10 = 50 本、各 500 ms）。26 本・検証 17 本とは別の seed
            OU 5±3、OU 10±3、OU 7±5（τ 5 ms）、ランダムステップ −5〜20 / 20 ms、ランダムなタイミングのパルス 5 ms
    long  : 長時間（各 20 s）。OU 7±5（τ 5 ms）を 4 本、段階 + OU を 2 本。発散と、前半 5 s と後半 5 s の Γ の差を見る
    ood   : 学習の範囲の外。強い抑制（−20）、強い入力（60）、チャープ（0.5→200 Hz）、とても遅いランプ（0→10 を 2 s、
            0→15 を 3 s）、下りランプ（20→0）、白色雑音に近い速い OU（τ 0.1 ms、σ 10）

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
    raise ValueError(name)


SUITES = ['fi', 'seeds', 'long', 'ood']


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
    Vp = run(I)
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
             'diverged': div}
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
            'n_diverged': int(sum(r['diverged'] for r in rows)), 'sec': sec}
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
        print(s, len(n), '本', '最長 {:.0f} ms'.format(max(lens) * DT),
              'スパイク数', [len(tc.spike_times(v, DT)) for v in Vt][:8], '...')
