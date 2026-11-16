"""
段階 A の教師データを作る：Traub モデル（19 コンパートメント）の細胞体だけに電流を入れ、
膜電位の応答を記録する。

学習タスク（段階 A）:
    入力 u(t): 細胞体への注入電流 [nA]（1 次元）。DELTA [ms] ごとに値が変わる階段状の電流
    出力 y(t): 細胞体の膜電位 V_soma(t) [mV]。樹状突起の電位も粗い間隔で保存しておく

入力電流の作り方:
    u_j = clip(level_j + jitter_j, i_min, i_max)       （j は DELTA ごとの区間）
        level_j : ゆっくり変わる基準レベル。一様乱数 [i_min, i_max] で決め、
                  level_min〜level_max [ms] のランダムな長さだけ同じ値を保つ
        jitter_j: 区間ごとの揺らぎ。一様乱数 [-jitter, jitter]
    Traub モデルは 0.1〜0.3 nA の一定電流で約 1 Hz のバースト、0.8〜1.2 nA で反復発火と、
    発火の様子が「電流の平均的な大きさ」で決まる。10 ms ごとの一様乱数だけだと平均が
    ほぼ (i_min + i_max)/2 に張り付き、遅いバーストが教師データに現れない。そこで、
    基準レベルを数百 ms〜数秒保つことで、各発火モードを含める。
    level_min = level_max = DELTA、jitter = 0 にすると、ただの 10 ms ごとの一様乱数になる。

処理の流れ:
    1. 初期状態を作る：静止状態から、軌道ごとにランダムな長さ（0〜warmup_max ms）だけ
       同じ作り方の電流で計算を進め、その終わりの状態を x0 にする。initial_state() の状態は
       厳密な静止状態ではなく（電流 0 でも約 230 ms 後にバーストする）、Ca や K_AHP の
       ゆっくりした変数もばらつかせたいので、助走を入れる
    2. x0 から本番の電流で T ms 計算する。batch 本ずつまとめて計算する
    3. 細胞体の膜電位を save_every ms ごと、全コンパートメントの膜電位を save_every_all ms
       ごとに間引いて保存する。x0 の全状態変数・入力電流と一緒に npz にする
    4. 確認用に、最初の数本の軌道を図にする

src/neuron/traub.py の simulate() は全ステップの全状態を保持するので、長い軌道や本数が多い
ときはメモリが足りない。ここでは step() を直接呼び、間引いた時刻だけ記録する。

保存するファイル（data/traub/{name}.npz）:
    t        : 細胞体の電位の時刻 [ms]。shape (N_out,)
    V_soma   : 細胞体の膜電位 [mV]。shape (N_traj, N_out)、float32
    I_soma   : t での細胞体電流 [nA]。shape (N_traj, N_out)、float32
    t_all    : 全コンパートメントの電位の時刻 [ms]。shape (N_all,)
    V_all    : 全コンパートメントの膜電位 [mV]。shape (N_traj, N_all, NC)、float32
    u        : 区間ごとの電流 [nA]。shape (N_traj, N_seg)。u[k, j] は [j·DELTA, (j+1)·DELTA) の値
    x0_*     : 初期状態（V, xi, m, h, s, r, n, a, b, c, q）。それぞれ shape (N_traj, NC)、float64
    warmup_ms: 初期状態を作るのに使った助走の長さ [ms]。shape (N_traj,)
    split    : 'train' / 'val' / 'test'。shape (N_traj,)。論文と同じ 60/20/20
    meta     : 条件をまとめた JSON 文字列
    時刻 t[i] の値は、その時刻の更新前の状態（traub.simulate() と同じ）。

実行方法（リポジトリのルートから）:
    uv run python experiments/traub/make_dataset.py               # DATASETS の全条件
    uv run python experiments/traub/make_dataset.py traub_a_test  # 名前を指定して 1 つだけ
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from neuron.traub import step, initial_state, compartment_conductance, NC, SOMA, area

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'traub')

# 1 nA = 1e-3 uA。traub.step() が受け取るのは電流密度 [uA/cm^2] なので、細胞体の面積で割る
NA_TO_DENSITY = 1e-3 / area[SOMA]

# 図に描く樹状突起（基底側 3、細胞体の両隣 7・9、尖端側 13・16）
PLOT_DENDS = [3, 7, 9, 13, 16]

#   name           : 保存するファイル名（拡張子なし）
#   note           : 条件の説明。meta に残す
#   n_traj         : 軌道の本数
#   T              : 1 本の長さ [ms]
#   dt             : オイラー法の時間刻み [ms]
#   save_every     : 細胞体の電位を保存する間隔 [ms]
#   save_every_all : 全コンパートメントの電位を保存する間隔 [ms]
#   delta          : 電流の値を切り替える間隔 [ms]（論文の Δ）
#   i_min, i_max   : 電流の範囲 [nA]
#   level_min/max  : 基準レベルを保つ長さの範囲 [ms]
#   jitter         : 区間ごとの揺らぎの幅 [nA]
#   warmup_max     : 助走の最大長 [ms]。0 なら全軌道が initial_state() から始まる
#   batch          : 一度にまとめて計算する軌道の本数
#   seed           : 乱数のシード
# 時間の長さはすべて dt の整数倍、T と warmup_max と level_min/max は delta の整数倍にする。
COMMON = {
    'dt': 0.025, 'save_every': 0.1, 'save_every_all': 1.0, 'delta': 10.0,
    'i_min': 0.0, 'i_max': 1.2, 'level_min': 200.0, 'level_max': 2000.0, 'jitter': 0.2,
}
DATASETS = [
    dict(COMMON, name='traub_a_test', note='動作確認用の小さいデータ',
         n_traj=4, T=1000.0, warmup_max=200.0, batch=4, seed=1),
    dict(COMMON, name='traub_a', note='段階 A の本番データ。細胞体だけに電流を入れる',
         n_traj=300, T=3000.0, warmup_max=1000.0, batch=150, seed=0),
]


def n_steps_of(length, dt):
    """長さ length [ms] が dt の何個分かを返す。割り切れなければエラーにする。"""
    n = int(round(length / dt))
    if not np.isclose(n * dt, length):
        raise ValueError(f'{length} ms は {dt} ms の整数倍でない')
    return n


def make_current(rng, n_traj, n_seg, cfg):
    """
    基準レベル＋揺らぎの階段状電流を作る（式はモジュールの説明を参照）。

    Args:
        rng (np.random.Generator): 乱数生成器。
        n_traj (int): 軌道の本数。
        n_seg (int): 区間の数（1 区間 = cfg['delta'] ms）。
        cfg (dict): DATASETS の要素。

    Returns:
        np.ndarray: shape (n_traj, n_seg)。u[k, j] が軌道 k の区間 j の電流 [nA]。
    """
    lv_min = n_steps_of(cfg['level_min'], cfg['delta'])
    lv_max = n_steps_of(cfg['level_max'], cfg['delta'])
    level = np.zeros((n_traj, n_seg))
    for k in range(n_traj):
        j = 0
        while j < n_seg:
            length = rng.integers(lv_min, lv_max + 1)
            level[k, j:j + length] = rng.uniform(cfg['i_min'], cfg['i_max'])
            j += length
    jitter = rng.uniform(-cfg['jitter'], cfg['jitter'], size=(n_traj, n_seg))
    return np.clip(level + jitter, cfg['i_min'], cfg['i_max'])


def run(state, u, n_per_seg, dt, g_comp, active=None, records=()):
    """
    階段状の電流 u を細胞体に入れて状態を進める。

    Args:
        state (dict[str, np.ndarray]): 初期状態。値は shape (B, NC)。書き換えない。
        u (np.ndarray): 区間ごとの電流 [nA]。shape (B, n_seg)。
        n_per_seg (int): 1 区間のステップ数（delta / dt）。
        dt (float): 時間刻み [ms]。
        g_comp (np.ndarray): compartment_conductance() の返り値。shape (NC-1,)。
        active (np.ndarray | None): shape (B, n_seg) の bool。False の区間では、その軌道の状態を
            進めずに止めておく（助走の長さを軌道ごとに変えるために使う）。None なら全部 True。
        records (tuple[tuple[np.ndarray, int, slice | int], ...]): 記録先の組
            (配列, 何ステップごとに記録するか, 記録するコンパートメント)。
            配列の shape は (B, N_rec, ...) で、[:, i] に i·間隔 ステップ目の更新前の V を入れる。

    Returns:
        dict[str, np.ndarray]: 最後の状態。値は shape (B, NC)。
    """
    B, n_seg = u.shape
    I_inj = np.zeros((B, NC))
    k = 0  # 通しのステップ番号
    for j in range(n_seg):
        I_inj[:, SOMA] = u[:, j] * NA_TO_DENSITY
        for _ in range(n_per_seg):
            for arr, every, comps in records:
                if k % every == 0:
                    arr[:, k // every] = state['V'][:, comps]
            new = step(state, I_inj, dt, g_comp)
            if active is not None:
                keep = ~active[:, j:j + 1]  # shape (B, 1)
                new = {name: np.where(keep, state[name], v) for name, v in new.items()}
            state = new
            k += 1
    return state


def make_one(cfg):
    """
    1 つの条件でデータを作り、npz と確認用の図を保存する。

    Returns:
        dict: 保存先と、表示用の要約。

    Raises:
        RuntimeError: 膜電位が nan や inf になった（計算が発散した）とき。
    """
    name, dt, delta, n_traj = cfg['name'], cfg['dt'], cfg['delta'], cfg['n_traj']
    rng = np.random.default_rng(cfg['seed'])
    g_comp = compartment_conductance()

    n_per_seg = n_steps_of(delta, dt)
    n_seg = n_steps_of(cfg['T'], delta)
    n_total = n_seg * n_per_seg
    every = n_steps_of(cfg['save_every'], dt)
    every_all = n_steps_of(cfg['save_every_all'], dt)
    t = np.arange(n_total // every) * cfg['save_every']
    t_all = np.arange(n_total // every_all) * cfg['save_every_all']

    # 助走：軌道 k は最後の n_warm[k] 区間だけ動かし、それより前は initial_state() のまま止める
    n_warm_max = n_steps_of(cfg['warmup_max'], delta)
    n_warm = rng.integers(0, n_warm_max + 1, size=n_traj)
    u_warm = make_current(rng, n_traj, n_warm_max, cfg)
    warm_active = np.arange(n_warm_max)[None, :] >= (n_warm_max - n_warm)[:, None]
    u = make_current(rng, n_traj, n_seg, cfg)

    V_soma = np.zeros((n_traj, len(t)), dtype=np.float32)
    V_all = np.zeros((n_traj, len(t_all), NC), dtype=np.float32)
    x0 = {k: np.zeros((n_traj, NC)) for k in initial_state()}

    start = time.time()
    for b0 in range(0, n_traj, cfg['batch']):
        idx = np.arange(b0, min(b0 + cfg['batch'], n_traj))
        state = initial_state((len(idx),))
        if n_warm_max > 0:
            state = run(state, u_warm[idx], n_per_seg, dt, g_comp, active=warm_active[idx])
        for k in state:
            x0[k][idx] = state[k]

        rec_soma = np.zeros((len(idx), len(t)))
        rec_all = np.zeros((len(idx), len(t_all), NC))
        run(state, u[idx], n_per_seg, dt, g_comp,
            records=((rec_soma, every, SOMA), (rec_all, every_all, slice(None))))
        if not (np.all(np.isfinite(rec_soma)) and np.all(np.isfinite(rec_all))):
            raise RuntimeError(f'{name}: 膜電位が発散した（dt が大きすぎる可能性）')
        V_soma[idx] = rec_soma
        V_all[idx] = rec_all
        print(f'  {idx[-1] + 1}/{n_traj} 本 ({time.time() - start:.0f} 秒)', flush=True)
    elapsed = time.time() - start

    # t[i] が入る区間は floor(t[i] / delta)。浮動小数点の誤差で 1 つ前にならないよう整数で計算する
    I_soma = u[:, (np.arange(len(t)) * every) // n_per_seg].astype(np.float32)

    # 細胞体のスパイク数（-20 mV を下から上に横切った回数）
    n_spikes = np.sum((V_soma[:, :-1] < -20.0) & (V_soma[:, 1:] >= -20.0), axis=1)

    n_tr, n_va = int(0.6 * n_traj), int(0.2 * n_traj)
    split = np.array(['train'] * n_tr + ['val'] * n_va + ['test'] * (n_traj - n_tr - n_va))

    meta = {
        'note': cfg['note'],
        'config': cfg,
        'model': 'Traub et al. (1991) CA3, 19 compartments (src/neuron/traub.py), explicit Euler',
        'soma_index': SOMA,
        'n_spikes_per_traj': {'mean': float(n_spikes.mean()),
                              'min': int(n_spikes.min()), 'max': int(n_spikes.max())},
        'columns': {
            't': '細胞体の電位の時刻 [ms]。shape (N_out,)',
            'V_soma': '細胞体の膜電位 [mV]。shape (N_traj, N_out)',
            'I_soma': 't での細胞体電流 [nA]。shape (N_traj, N_out)',
            't_all': '全コンパートメントの電位の時刻 [ms]。shape (N_all,)',
            'V_all': '全コンパートメントの膜電位 [mV]。shape (N_traj, N_all, 19)',
            'u': '区間ごとの細胞体電流 [nA]。shape (N_traj, N_seg)',
            'x0_*': '初期状態。shape (N_traj, 19)',
            'warmup_ms': '助走の長さ [ms]。shape (N_traj,)',
            'split': 'train / val / test。shape (N_traj,)',
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    npz_path = os.path.join(DATA_DIR, f'{name}.npz')
    np.savez_compressed(
        npz_path,
        t=t, V_soma=V_soma, I_soma=I_soma, t_all=t_all, V_all=V_all, u=u,
        warmup_ms=n_warm * delta, split=split,
        **{f'x0_{k}': v for k, v in x0.items()},
        meta=json.dumps(meta, ensure_ascii=False),
    )

    fig_path = os.path.join(DATA_DIR, f'{name}.png')
    plot(t, I_soma, V_soma, t_all, V_all, n_show=min(3, n_traj), path=fig_path, title=name)

    return {
        'npz': npz_path, 'png': fig_path, 'elapsed': elapsed, 'n_spikes': n_spikes,
        'size_mb': os.path.getsize(npz_path) / 1024 ** 2,
        'V_range': (float(V_soma.min()), float(V_soma.max())),
    }


def plot(t, I_soma, V_soma, t_all, V_all, n_show, path, title):
    """最初の n_show 本について、電流・細胞体の電位・樹状突起の電位を縦に並べて描く。"""
    fig, axes = plt.subplots(3 * n_show, 1, figsize=(12, 6.5 * n_show), sharex=True)
    for k in range(n_show):
        ax_i, ax_s, ax_d = axes[3 * k:3 * k + 3]
        ax_i.plot(t, I_soma[k], color='red', linewidth=0.8)
        ax_i.set_ylabel(f'#{k} I [nA]')
        ax_s.plot(t, V_soma[k], color='blue', linewidth=0.5)
        ax_s.set_ylabel('V soma [mV]')
        for c in PLOT_DENDS:
            ax_d.plot(t_all, V_all[k, :, c], linewidth=0.6, label=f'comp {c}')
        ax_d.set_ylabel('V dend [mV]')
        ax_d.legend(loc='upper right', fontsize=7, ncol=len(PLOT_DENDS))
        for ax in (ax_i, ax_s, ax_d):
            ax.grid(True, linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('t [ms]')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    names = sys.argv[1:]
    for cfg in DATASETS:
        if names and cfg['name'] not in names:
            continue
        print(f"--- {cfg['name']} を生成中 ---")
        r = make_one(cfg)
        print(f"  細胞体のスパイク数（1 本あたり）: 平均 {r['n_spikes'].mean():.1f}, "
              f"最小 {r['n_spikes'].min()}, 最大 {r['n_spikes'].max()}")
        print(f"  細胞体の膜電位の範囲: {r['V_range'][0]:.1f} 〜 {r['V_range'][1]:.1f} mV")
        print(f"  計算時間   : {r['elapsed']:.1f} 秒")
        print(f"  保存先     : {r['npz']} ({r['size_mb']:.1f} MB)")
        print(f"  確認用の図 : {r['png']}")
        print()
