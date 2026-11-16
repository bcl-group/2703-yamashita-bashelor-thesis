"""
HH モデルに入れる入力電流 I_ext(t) を作る関数集。

すべての関数は時刻の配列 t を受け取り、同じ shape の電流配列を返す。
返り値の I[i]mA が時刻 t[i]ms の電流となり、neuron.hh.simulate のループで
i 番目のステップに使われる。

単位:
    時刻 t: ms
    電流 I: uA/cm^2
"""
import numpy as np


def step_current(t, amplitude=10.0, t_start=10.0, t_end=40.0):
    """
    区間 [t_start, t_end] だけ一定値を流すステップ電流を作る。
    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、float。
        amplitude (float): 電流を流している間の値 [uA/cm^2]。
        t_start (float): 電流を流し始める時刻 [ms]（この時刻を含む）。
        t_end (float): 電流を止める時刻 [ms]（この時刻を含む）。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)、t と同じ dtype
    """
    I = np.zeros_like(t)
    I[(t >= t_start) & (t <= t_end)] = amplitude
    return I


def constant_current(t, amplitude=10.0):
    """
    全時刻で同じ値を流す一定電流を作る。

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、float。
        amplitude (float): 電流の値 [uA/cm^2]。

    Returns:
        np.ndarray: 全要素が amplitude の配列 [uA/cm^2]。shape (N,)。
    """
    return np.full_like(t, amplitude)



def ramp_current(t, i_start=0.0, i_end=20.0):
    """
    i_start から i_end まで直線的に増える電流を作る。

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)。値は使わず長さ N だけを使う。
        i_start (float): 最初の時刻 t[0] での電流 [uA/cm^2]。
        i_end (float): 最後の時刻 t[-1] での電流 [uA/cm^2]。

    Returns:
        np.ndarray: i_start から i_end までを両端を含めて N 等分した配列
            [uA/cm^2]。shape (N,)。
    """
    return np.linspace(i_start, i_end, len(t))


def sin_current(t, amplitude=10.0, freq=50.0, offset=0.0):
    """
    正弦波の電流を作る。
    I(t) = offset + amplitude * sin(2π * freq * t / 1000)
    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)。
        amplitude (float): 振幅 [uA/cm^2]。
        freq (float): 周波数 [Hz]。
        offset (float): 波の中心の値 [uA/cm^2]。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)。
            値は offset - amplitude から offset + amplitude の範囲。
    """
    return offset + amplitude * np.sin(2.0 * np.pi * freq * t / 1000.0)


def pulse_train_current(t, amplitude=10.0, width=1.0, interval=10.0, t_start=0.0):
    """
    一定の間隔で同じ形のパルスを出すパルス列電流を作る。
    t_start 以降、interval [ms] ごとに、幅 width [ms]、高さ amplitude の
    電流を出力。

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、float。
        amplitude (float): パルスの高さ [uA/cm^2]。
        width (float): 1つのパルスの幅 [ms]。
        interval (float): パルスの周期（パルスの始まりから次の始まりまで）[ms]。
        t_start (float): 最初のパルスを出す時刻 [ms]。これより前は 0。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)。
    """
    I = np.zeros_like(t)
    phase = np.mod(t - t_start, interval)
    I[(t >= t_start) & (phase < width)] = amplitude
    return I


def random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=None):
    """
    interval [ms] ごとに値がランダムに切り替わる電流を作る

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        i_min (float): 電流の下限 [uA/cm^2]（この値を含む）。
        i_max (float): 電流の上限 [uA/cm^2]（この値を含まない）。
        interval (float): 電流値を切り替える間隔 [ms]。
        seed (int | None): 乱数のシード。同じ値なら毎回同じ電流になる。
            None なら実行ごとに変わる。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)、float64
    """
    rng = np.random.default_rng(seed)
    # 全体の長さ (T = t[-1] - t[0] + dt ) / interval をし，切り上げて区間の数 n_segments を求める。
    n_segments = int(np.ceil((t[-1] - t[0] + (t[1] - t[0])) / interval))

    # 区間ごとの電流値 amplitudes を乱数で作る
    amplitudes = rng.uniform(i_min, i_max, size=n_segments)
    # 各時刻が何番目の区間かを floor((t - t[0]) / interval) で求める
    # 各時刻に電流の大きさを割り振る
    segment_index = ((t - t[0]) / interval).astype(int)
    return amplitudes[segment_index]


def _unit_ou(n, dt, tau, rng):
    """
    平均 0・標準偏差 1 の OU 過程（Ornstein–Uhlenbeck 過程）を n 点作る。

    厳密な離散化  z_{k+1} = a z_k + sqrt(1 - a^2) ξ_k,  a = exp(-dt/τ),  ξ_k ~ N(0, 1)
    を使うので、dt によらず標準偏差は 1、相関は exp(-|Δt|/τ) になる。z_0 も定常分布 N(0, 1) から引く。
    """
    a = np.exp(-dt / tau)
    b = np.sqrt(1.0 - a * a)
    xi = rng.standard_normal(n)
    z = np.empty(n)
    z[0] = xi[0]
    for k in range(1, n):
        z[k] = a * z[k - 1] + b * xi[k]
    return z


def ou_current(t, mean=10.0, sigma=3.0, tau=1.0, seed=None):
    """
    平均 mean のまわりを OU 過程で揺らぐ電流（Jolivet et al., 2008 のモデル予測コンテストと同じ形）。

        I(t) = mean + sigma * z(t)、z は平均 0・標準偏差 1・相関時間 tau の OU 過程

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        mean (float): 平均 [uA/cm^2]。
        sigma (float): 標準偏差 [uA/cm^2]。
        tau (float): 相関時間 [ms]。これより離れた時刻の値はほぼ無関係になる。
        seed (int | None): 乱数のシード。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)、float64
    """
    rng = np.random.default_rng(seed)
    return mean + sigma * _unit_ou(len(t), t[1] - t[0], tau, rng)


def level_ou_current(t, level_min=0.0, level_max=20.0, hold_min=100.0, hold_max=1000.0,
                     sigma_max=4.0, tau=1.0, i_min=-5.0, i_max=30.0, seed=None):
    """
    ゆっくり変わる基準レベルに、速い OU の揺らぎを足した電流（学習用）。

        I(t) = clip( level(t) + sigma(t) * z(t), i_min, i_max )

    - level(t) : U[level_min, level_max] の値を、U[hold_min, hold_max] ms の長さだけ保つ
    - sigma(t) : 基準レベルと同じ区間ごとに U[0, sigma_max] から選ぶ（0 に近い区間はほぼ定常電流）
    - z(t)     : 平均 0・標準偏差 1・相関時間 tau の OU 過程

    Traub の教師データ（基準レベル + 10 ms ごとの揺らぎ）と同じ組み立てで、揺らぎだけ OU にしたもの。

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        level_min, level_max (float): 基準レベルの範囲 [uA/cm^2]。
        hold_min, hold_max (float): 基準レベルを保つ長さの範囲 [ms]。
        sigma_max (float): 揺らぎの標準偏差の上限 [uA/cm^2]。
        tau (float): OU の相関時間 [ms]。
        i_min, i_max (float): 電流を収める範囲 [uA/cm^2]。
        seed (int | None): 乱数のシード。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)、float64
    """
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    n = len(t)
    level = np.empty(n)
    sigma = np.empty(n)
    k = 0
    while k < n:
        length = int(round(rng.uniform(hold_min, hold_max) / dt))
        level[k:k + length] = rng.uniform(level_min, level_max)
        sigma[k:k + length] = rng.uniform(0.0, sigma_max)
        k += length
    z = _unit_ou(n, dt, tau, rng)
    return np.clip(level + sigma * z, i_min, i_max)


def knot_ou_current(t, ranges=((0.0, 20.0, 1.0),), knot_min=20.0, knot_max=500.0,
                    sigma_max=1.5, tau=1.0, i_min=-5.0, i_max=30.0, seed=None):
    """
    折れ線でゆっくり変わる基準レベルに、OU の揺らぎを足した電流（学習用）。

        I(t) = clip( level(t) + sigma(t) * z(t), i_min, i_max )

    - level(t): U[knot_min, knot_max] ms おきに節点を置き、節点の値を ranges から選んで直線でつなぐ。
                節点の間隔が長いとゆっくりしたランプ、短いと速い変化になる
    - ranges  : (下限, 上限, 重み) の組。節点ごとに重みに比例した確率で範囲を1つ選び、その中の一様乱数にする。
                発火の閾値付近の範囲に大きな重みを付けると、閾値付近の入力を多く含められる
    - sigma(t): 節点の区間ごとに U[0, sigma_max] から選ぶ
    - z(t)    : 平均 0・標準偏差 1・相関時間 tau の OU 過程

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        ranges (tuple): 節点の値の範囲と重み [uA/cm^2]。
        knot_min, knot_max (float): 節点の間隔の範囲 [ms]。
        sigma_max (float): 揺らぎの標準偏差の上限 [uA/cm^2]。
        tau (float): OU の相関時間 [ms]。
        i_min, i_max (float): 電流を収める範囲 [uA/cm^2]。
        seed (int | None): 乱数のシード。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)、float64
    """
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    n = len(t)
    w = np.array([r[2] for r in ranges], dtype=float)
    w /= w.sum()

    def draw():
        lo, hi, _ = ranges[rng.choice(len(ranges), p=w)]
        return rng.uniform(lo, hi)

    # 節点の位置（ステップ番号）と値。最後の節点は系列の端より後ろに置く
    pos, val = [0], [draw()]
    while pos[-1] < n - 1:
        pos.append(pos[-1] + max(1, int(round(rng.uniform(knot_min, knot_max) / dt))))
        val.append(draw())
    level = np.interp(np.arange(n), pos, val)
    sigma = np.empty(n)
    for a, b in zip(pos[:-1], pos[1:]):
        sigma[a:b] = rng.uniform(0.0, sigma_max)
    z = _unit_ou(n, dt, tau, rng)
    return np.clip(level + sigma * z, i_min, i_max)


def random_timing_pulse_current(t, i_min=-5.0, i_max=20.0, width=5.0, rate=0.05, seed=None):
    """
    ランダムな時刻に、ランダムな高さのパルスを出す電流を作る。


    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        i_min (float): パルスの高さの下限 [uA/cm^2]（この値を含む）。
        i_max (float): パルスの高さの上限 [uA/cm^2]（この値を含まない）。
        width (float): 1 つのパルスの幅 [ms]。dt より小さくても最低 1 ステップ。
        rate (float): 単位時間あたりにパルスが始まる頻度 [1/ms]。
        seed (int | None): 乱数のシード。同じ値なら毎回同じ電流になる。
            None なら実行ごとに変わる。

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)。パルスがない時刻は 0。
    """
    rng = np.random.default_rng(seed)

    dt = t[1] - t[0]
    n_width = max(1, int(width / dt))
    trigger_prob = rate * dt

    I = np.zeros_like(t)

    i = 0
    while i < len(t) - n_width:
        if rng.random() < trigger_prob:
            I[i:i + n_width] = rng.uniform(i_min, i_max)
            i += n_width
        else:
            i += 1

    return I


def munechika_train_current(t, options=(-5.0, 1.3, 6.3, 20.0), weights=(0.3, 1.0, 1.0, 1.0),
                            sigma=1.0, interval=20.0, silence=80.0, seed=991927697):
    """
    出典: https://github.com/MunechikaHaruki/SINDyNeuroSurrogate
    棟近さんの学習用電流（SINDyNeuroSurrogate の train()）と同じ電流を作る。

    最初と最後の silence [ms] は 0 にし、その間を interval [ms] ごとの区間に分ける。
    区間ごとに options から weights の比で 1 つ選び、標準偏差 sigma の正規乱数を足す。
    既定値で t = np.arange(0, 9000.0, 0.01) を渡すと、元のコードと完全に同じ配列になる。

    Args:
        t (np.ndarray): 時刻の配列 [ms]。shape (N,)、等間隔で N >= 2。
        options (tuple[float, ...]): 区間ごとに選ぶ電流の候補値 [uA/cm^2]。
        weights (tuple[float, ...]): options の各値を選ぶ重み。合計で割って確率にする。
        sigma (float): 選んだ値に足す正規乱数の標準偏差 [uA/cm^2]。
        interval (float): 電流値を切り替える間隔 [ms]。
        silence (float): 最初と最後に電流を 0 にする時間 [ms]。
        seed (int | None): 乱数のシード

    Returns:
        np.ndarray: 各時刻の電流 [uA/cm^2]。shape (N,)。
            両端の silence と、区間に収まらない末尾の余りは 0。
    """
    rng = np.random.default_rng(seed)

    dt = t[1] - t[0]
    n_interval = int(round(interval / dt))
    n_silence = int(round(silence / dt))
    p = np.array(weights) / np.sum(weights)

    I = np.zeros_like(t)
    active = I[n_silence:len(t) - n_silence]

    for k in range(len(active) // n_interval):
        value = rng.choice(options, p=p) + rng.normal(0.0, sigma)
        active[k * n_interval:(k + 1) * n_interval] = value

    return I


# このファイルを直接実行したときだけ動く
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dt = 0.01
    t = np.arange(0, 200.0, dt)

    currents = [
        ('step',                step_current(t, amplitude=10.0, t_start=50.0, t_end=150.0)),
        ('constant',            constant_current(t, amplitude=10.0)),
        ('ramp',                ramp_current(t, i_start=0.0, i_end=20.0)),
        ('sin',                 sin_current(t, amplitude=10.0, freq=50.0, offset=10.0)),
        ('pulse train',         pulse_train_current(t, amplitude=20.0, width=1.0, interval=10.0)),
        ('random pulse',        random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=0)),
        ('random timing pulse', random_timing_pulse_current(t, seed=0)),
        ('munechika train',     munechika_train_current(t, silence=0.0)),
    ]

    fig, axes = plt.subplots(len(currents), 1, figsize=(9, 11), sharex=True)

    for ax, (name, I) in zip(axes, currents):
        ax.plot(t, I, linewidth=1.0)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Input currents defined in current.py')
    fig.tight_layout()
    plt.show()
