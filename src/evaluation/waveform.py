"""
原系（HH）と代理モデル（RNN など）の膜電位波形を比べる指標。

棟近先輩の SINDyNeuroSurrogate（neurosurrogate/sim/waveform.py、core/diverge.py）と
同じ定義・同じ名前にしてある。スパイクの検出と1発ごとの特徴量は、先輩と同じく
eFEL（https://github.com/openbraininstitute/eFEL）に任せる。

単位:
    時刻 t、dt: ms
    膜電位 V: mV
"""
import warnings

import efel
import numpy as np


# スパイク1発ごとに取り出す eFEL の特徴量（先輩の _MEDIAN_FEATURES と同じ並び）
SPIKE_FEATURES = [
    'peak_voltage',            # ピーク時の膜電位 [mV]
    'AP_amplitude',            # スパイク開始点からピークまでの高さ [mV]
    'AP_begin_voltage',        # スパイク開始点の膜電位（実質的な閾値）[mV]
    'AP_rise_rate',            # 立ち上がりの平均の傾き [mV/ms]
    'AP_fall_rate',            # 下がるときの平均の傾き [mV/ms]（負の値）
    'AP_duration_half_width',  # 半値幅 [ms]
    'AP_rise_time',            # スパイク開始点からピークまでの時間 [ms]
    'AP_fall_time',            # ピークからスパイク終了点までの時間 [ms]
    'AHP_depth',               # 後過分極の深さ（基準電位からの差）[mV]
    'AHP_time_from_peak',      # ピークから後過分極の最小値までの時間 [ms]
]

_EFEL_FEATURES = ['peak_indices', 'peak_time', 'ISI_values', 'time_to_first_spike',
                  *SPIKE_FEATURES]

# 原系と代理モデルの両方に値がある指標（表では HH / 代理モデル の2列で並べる）
ROW_METRICS = ['spike_count', 'latency', 'mean_isi', 'std_isi']
# 両者を比べて1つの値になる指標
SCALAR_METRICS = ['rmse', 'mae', 'periodicity_gap', 'spike_shape_corr']

# |V| がこれを超えたら発散とみなす [mV]（生理的な範囲はせいぜい ±200 mV）
DIVERGE_V = 1e3

# 平均スパイク波形を切り出す幅（ピークの前後それぞれ）[ms]
SPIKE_HALF_WINDOW_MS = 2.0

_NAN = float('nan')


def diverged(V):
    """
    膜電位が NaN・inf を含むか、生理的にあり得ない大きさになったかを判定する。

    Args:
        V (np.ndarray): 膜電位 [mV]。shape (N,)。

    Returns:
        bool: 発散していれば True。
    """
    return (not bool(np.all(np.isfinite(V)))) or float(np.max(np.abs(V))) > DIVERGE_V


def efel_features(V, dt):
    """
    1本の膜電位波形から eFEL の特徴量を求める。

    刺激区間は波形全体とする（先輩と同じく、AHP の基準電位を求めるために
    開始時刻だけ1サンプル後ろにずらす）。

    Args:
        V (np.ndarray): 膜電位 [mV]。shape (N,)。
        dt (float): 時間刻み [ms]。

    Returns:
        dict[str, np.ndarray | None]: 特徴量名 → 値の配列。スパイクがないなどで
            求まらない特徴量は None。
    """
    time = np.arange(len(V), dtype=float) * dt
    trace = {
        'T': time,
        'V': np.asarray(V, dtype=float),
        'stim_start': [time[min(1, len(time) - 1)]],
        'stim_end': [time[-1]],
    }
    with warnings.catch_warnings():
        # スパイクがないと eFEL が RuntimeWarning を出すが、値は None になるだけなので無視する
        warnings.filterwarnings('ignore', category=RuntimeWarning, module=r'efel\.*')
        return efel.get_feature_values([trace], _EFEL_FEATURES)[0]


def _first_or_nan(a):
    return _NAN if a is None or len(a) == 0 else float(a[0])


def _stat_or_nan(fn, a):
    return _NAN if a is None or len(a) == 0 else float(fn(a))


def _at_or_nan(a, i):
    return _NAN if a is None or i >= len(a) else float(a[i])


def _diff_or_nan(a, b):
    return _NAN if np.isnan(a) or np.isnan(b) else a - b


def _mean_spike(V, peaks, half_win):
    # ピークの前後 half_win 点を切り出して平均する。端で切り出せないスパイクは除く
    snippets = [V[p - half_win:p + half_win + 1]
                for p in peaks
                if p - half_win >= 0 and p + half_win + 1 <= len(V)]
    return np.mean(snippets, axis=0) if snippets else None


def compare_waveforms(V_true, V_pred, dt, spike_true=0, spike_pred=0):
    """
    原系と代理モデルの膜電位を比べ、先輩と同じ指標をすべて求める。

    Args:
        V_true (np.ndarray): 原系（HH）の膜電位 [mV]。shape (N,)。
        V_pred (np.ndarray): 代理モデルの膜電位 [mV]。shape (N,)。同じ時刻に揃えておく。
        dt (float): 時間刻み [ms]。
        spike_true (int): 1発ごとの特徴量を比べる原系のスパイク番号（0 始まり）。
        spike_pred (int): 同じく代理モデル側のスパイク番号。

    Returns:
        dict: 次のキーを持つ。
            'rmse', 'mae' (float): 波形の誤差 [mV]。
            'spike_count', 'latency', 'mean_isi', 'std_isi'
                (tuple[float, float]): (原系, 代理モデル)。latency は最初のスパイクまでの
                時間 [ms]、ISI はスパイク間隔 [ms]。スパイクがなければ nan。
            'periodicity_gap' (float): |平均 ISI の差| [ms]。
            'spike_shape_corr' (float): 平均スパイク波形どうしの Pearson 相関。
            'spike_features' (dict[str, tuple[float, float]]): SPIKE_FEATURES の
                各特徴量の (原系, 代理モデル)。
            'diverged' (bool): 代理モデルの膜電位が発散したか。
    """
    V_true = np.asarray(V_true, dtype=float)
    V_pred = np.asarray(V_pred, dtype=float)
    if V_true.shape != V_pred.shape:
        raise ValueError(f'shape が違う: {V_true.shape} と {V_pred.shape}')

    is_diverged = diverged(V_pred)
    if is_diverged:
        # 発散した波形は eFEL に渡せないので、誤差以外は nan にする
        feat_true, feat_pred = efel_features(V_true, dt), {}
    else:
        feat_true, feat_pred = efel_features(V_true, dt), efel_features(V_pred, dt)

    def pair(fn):
        return fn(feat_true), fn(feat_pred)

    def peaks(f):
        # eFEL は波形を interp_step（既定 0.1 ms）刻みに補間してから処理するので、
        # peak_indices は補間後の番号で、dt 刻みの元の配列の番号ではない。
        # 元の配列の番号はピーク時刻 peak_time [ms] を dt で割って求める。
        # （先輩の spike_shape_corr は peak_indices をそのまま使っているため、
        #   dt = 0.01 ms では本来の 1/10 の位置を切り出してしまう）
        p = f.get('peak_time')
        return [] if p is None else [int(round(tp / dt)) for tp in p]

    peaks_true, peaks_pred = pair(peaks)
    mean_isi = pair(lambda f: _stat_or_nan(np.mean, f.get('ISI_values')))
    std_isi = pair(lambda f: _stat_or_nan(np.std, f.get('ISI_values')))

    half_win = int(SPIKE_HALF_WINDOW_MS / dt)
    tmpl_true = _mean_spike(V_true, peaks_true, half_win)
    tmpl_pred = None if is_diverged else _mean_spike(V_pred, peaks_pred, half_win)
    if tmpl_true is None or tmpl_pred is None:
        shape_corr = _NAN
    else:
        shape_corr = float(np.corrcoef(tmpl_true, tmpl_pred)[0, 1])

    err = V_pred - V_true
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'spike_count': (float(len(peaks_true)), float(len(peaks_pred))),
        'latency': pair(lambda f: _first_or_nan(f.get('time_to_first_spike'))),
        'mean_isi': mean_isi,
        'std_isi': std_isi,
        'periodicity_gap': abs(_diff_or_nan(*mean_isi)),
        'spike_shape_corr': shape_corr,
        'spike_features': {
            name: (_at_or_nan(feat_true.get(name), spike_true),
                   _at_or_nan(feat_pred.get(name), spike_pred))
            for name in SPIKE_FEATURES
        },
        'diverged': is_diverged,
    }


def format_metrics(name, m):
    """
    compare_waveforms の結果を、人が読む表の文字列にする。

    Args:
        name (str): 評価条件の名前（表の見出しに使う）。
        m (dict): compare_waveforms の返り値。

    Returns:
        str: 複数行の文字列。
    """
    lines = ['[{}]{}'.format(name, '  ※代理モデルが発散' if m['diverged'] else '')]
    lines.append('  {:24s} {:>10s} {:>10s}'.format('', 'HH', 'surrogate'))
    for key in ROW_METRICS:
        o, s = m[key]
        lines.append('  {:24s} {:10.3f} {:10.3f}'.format(key, o, s))
    for key, (o, s) in m['spike_features'].items():
        lines.append('  {:24s} {:10.3f} {:10.3f}'.format(key, o, s))
    for key in SCALAR_METRICS:
        lines.append('  {:24s} {:>10s} {:10.3f}'.format(key, '', m[key]))
    return '\n'.join(lines)
