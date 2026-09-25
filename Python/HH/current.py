"""
ニューロンモデルに与える外部電流（入力電流）を生成するモジュール。

このファイルは電流を作る関数だけをまとめたもので、HH.py などから

    from current import step_current, random_pulse_current

のように読み込んで使う。

共通の約束
----------
- 時刻の配列 t の単位は ms（ミリ秒）。
- 返す電流の単位は uA/cm^2。
- どの関数も t と同じ長さの numpy 配列を返す。
  そのため、シミュレーション側は I_ext[i] を読むだけでよく、
  どの電流を使っても本体のコードを変える必要がない。

このファイル単体で実行すると、各電流の波形をまとめて表示する。

    python current.py
"""

import numpy as np


# ==================================================================
# 決まった形の電流
# ==================================================================

def step_current(t, amplitude=10.0, t_start=10.0, t_end=40.0):
    """
    ある区間だけ一定の電流を流す（ステップ電流）。

    スパイクが発射される様子を確認する、最も基本的な入力。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    amplitude : float
        電流の大きさ (uA/cm^2)。
    t_start, t_end : float
        電流を流し始める時刻と止める時刻 (ms)。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape
    """
    I = np.zeros_like(t)
    I[(t >= t_start) & (t <= t_end)] = amplitude
    return I


def constant_current(t, amplitude=10.0):
    """
    最初から最後まで同じ大きさの電流を流す（定常電流）。

    振幅を変えながらスパイク数の変化を調べるとき（発火頻度の評価）に使う。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    amplitude : float
        電流の大きさ (uA/cm^2)。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape
    """
    return np.full_like(t, amplitude)


def ramp_current(t, i_start=0.0, i_end=20.0):
    """
    時間とともに直線的に増えていく電流（ramp 電流）。

    どの電流値で発火が始まるか（発火閾値）を調べるときに使う。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    i_start, i_end : float
        シミュレーション開始時と終了時の電流の大きさ (uA/cm^2)。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape
    """
    return np.linspace(i_start, i_end, len(t))


def sin_current(t, amplitude=10.0, freq=50.0, offset=0.0):
    """
    sin 波の電流。

    定常電流ともパルスとも違う入力に対して正しく応答できるか
    （学習データに無い入力への外挿性能）を調べるときに使う。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    amplitude : float
        振幅 (uA/cm^2)。
    freq : float
        周波数 (Hz)。t の単位が ms なので、1000 で割って秒に直してから使う。
    offset : float
        電流のオフセット (uA/cm^2)。0 のままだと電流が正負に振れる。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape
    """
    return offset + amplitude * np.sin(2.0 * np.pi * freq * t / 1000.0)


def pulse_train_current(t, amplitude=10.0, width=1.0, interval=10.0, t_start=0.0):
    """
    等間隔に並んだパルス電流。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    amplitude : float
        パルスの高さ (uA/cm^2)。
    width : float
        1 本のパルスの幅 (ms)。
    interval : float
        パルスの間隔（パルスの立ち上がりから次の立ち上がりまで）(ms)。
    t_start : float
        最初のパルスが立ち上がる時刻 (ms)。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape
    """
    I = np.zeros_like(t)
    # 各時刻が「パルスが始まってから width 以内か」を、周期 interval の余りで判定する
    phase = np.mod(t - t_start, interval)
    I[(t >= t_start) & (phase < width)] = amplitude
    return I


# ==================================================================
# ランダムな電流（学習データ用）
# ==================================================================

def random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=None):
    """
    一定時間ごとに大きさがランダムに変わる電流（棟近先輩の教師電流の再現）。

    RNN の教師データを作るための入力。様々な入力とその応答を含んだデータに
    することで、特定の入力に依存しないモデルの学習が期待できる。

    棟近先輩の進捗報告スライド p.7 の「教師電流」と同じ作り方をしている。
    区間ごとに一様乱数で決めた大きさの電流を流し続ける形であり、
    各区間の間で 0 に戻ることはない（階段状の波形になる）。
    卒業論文の Algorithm 2 も同じ考え方で、パラメータだけが異なる。

        スライド p.7（最新）: i_min=-5, i_max=20, interval=20 ms, 総時間 9000 ms, dt=0.01 ms
        卒業論文 Algorithm 2 : i_min= 0, i_max=20, interval=10 ms, 総時間  900 ms, dt=0.05 ms

    なお、先輩が使った乱数の種までは分からないため、波形そのものが1点ずつ
    一致するわけではない。同じ規則・同じ範囲で生成している、という意味での再現である。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    i_min, i_max : float
        電流の大きさの範囲 (uA/cm^2)。区間ごとに一様分布 U(i_min, i_max) から選ぶ。
    interval : float
        電流の大きさを切り替える間隔 (ms)。
    seed : int or None
        乱数の種。同じ値を指定すると毎回同じ電流が得られる。
        教師データを作り直すときに再現できるよう、指定しておくのが望ましい。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape
    """
    rng = np.random.default_rng(seed)

    # 全区間を interval ごとに区切ったときの区間数
    n_segments = int(np.ceil((t[-1] - t[0] + (t[1] - t[0])) / interval))

    # 区間ごとに1つずつ、一様乱数で電流の大きさを決める
    amplitudes = rng.uniform(i_min, i_max, size=n_segments)

    # 各時刻がどの区間に属するかを求め、その区間の電流値を割り当てる
    segment_index = ((t - t[0]) / interval).astype(int)
    return amplitudes[segment_index]


def random_timing_pulse_current(t, i_min=5.0, i_max=30.0, width=5.0,
                                rate=0.02, seed=None):
    """
    発生するタイミングも大きさもランダムなパルス電流。

    random_pulse_current との違いは、パルスとパルスの間は電流が 0 になる点。
    実際のニューロンが受け取るような、まばらな入力を模したいときに使う。

    Parameters
    ----------
    t : numpy.ndarray
        時刻の配列 (ms)。
    i_min, i_max : float
        パルスの高さの範囲 (uA/cm^2)。1本ごとに一様分布から選ぶ。
    width : float
        1 本のパルスの幅 (ms)。
    rate : float
        1 ms あたりのパルスの発生率。例えば 0.02 なら平均して 50 ms に 1 本。
        （1 秒あたり 20 本に相当する）
    seed : int or None
        乱数の種。

    Returns
    -------
    numpy.ndarray
        各時刻の電流。shape = t.shape

    Notes
    -----
    もともとこのファイルにあった実装を関数にしたもの。
    元の実装は単位が秒・A（幅 0.05 秒、平均 2.5 本/秒、振幅 0.2〜1.5 A）だったが、
    HH モデルに合わせて単位を ms・uA/cm^2 に直し、
    デフォルト値もスパイクが発射される大きさに変更している。
    """
    rng = np.random.default_rng(seed)

    dt = t[1] - t[0]
    n_width = max(1, int(width / dt))    # パルス幅に相当するデータ点数
    trigger_prob = rate * dt             # 1 ステップあたりにパルスが始まる確率

    I = np.zeros_like(t)

    i = 0
    while i < len(t) - n_width:
        if rng.random() < trigger_prob:
            # パルスの高さをランダムに決めて書き込む
            I[i:i + n_width] = rng.uniform(i_min, i_max)
            # パルスが重ならないよう、パルス幅のぶんだけ先に進める
            i += n_width
        else:
            i += 1

    return I


# ==================================================================
# 動作確認用：このファイルを直接実行すると各電流の波形を表示する
# ==================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 確認用に 200 ms 分を dt = 0.01 ms で作る
    dt = 0.01
    t = np.arange(0, 200.0, dt)

    # グラフのラベルは、日本語フォントが無い環境での文字化けを避けるため英語にする
    currents = [
        ('step',                step_current(t, amplitude=10.0, t_start=50.0, t_end=150.0)),
        ('constant',            constant_current(t, amplitude=10.0)),
        ('ramp',                ramp_current(t, i_start=0.0, i_end=20.0)),
        ('sin',                 sin_current(t, amplitude=10.0, freq=50.0, offset=10.0)),
        ('pulse train',         pulse_train_current(t, amplitude=20.0, width=1.0, interval=10.0)),
        ('random pulse',        random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=0)),
        ('random timing pulse', random_timing_pulse_current(t, seed=0)),
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
