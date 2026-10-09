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
