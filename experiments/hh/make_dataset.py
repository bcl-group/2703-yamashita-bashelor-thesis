import inspect
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# src/（neuron, rnn）を import できるようにする
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from neuron.hh import simulate, V_REST
from neuron.current import random_pulse_current, munechika_train_current

# このファイルがあるフォルダ（experiments/hh/）と、データの保存先（data/hh/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'hh')
# 実際に呼ぶ電流関数
CURRENT_FUNCS = {
    'random_pulse_current': random_pulse_current,
    'munechika_train_current': munechika_train_current,
}

#   name    : 保存するファイル名（拡張子なし）
#   note    : 条件の説明。meta に残す
#   dt, T   : 時間刻みとシミュレーションの長さ [ms]
#   kind    : 使う電流関数の名前
#   current : 電流関数に渡す引数。書かなかった引数は関数の既定値を使う
DATASETS = [
    {
        'name': 'teacher_slide',
        'note': '棟近先輩のスライド p.7 と範囲（-5〜20）・区間幅（20 ms）だけ合わせた一様乱数の電流。'
                '値の分布は棟近先輩と違う（同じ電流は munechika_train）',
        'dt': 0.01,
        'T': 9000.0,
        'kind': 'random_pulse_current',
        'current': {'i_min': -5.0, 'i_max': 20.0, 'interval': 20.0, 'seed': 0},
    },
    {
        'name': 'teacher_thesis',
        'note': '棟近先輩の卒業論文 Algorithm 2 の条件',
        'dt': 0.01,
        'T': 900.0,
        'kind': 'random_pulse_current',
        'current': {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0, 'seed': 0},
    },
    {
        'name': 'munechika_train',
        'note': '棟近先輩の SINDyNeuroSurrogate の train() と同じ電流',
        'dt': 0.01,
        'T': 9000.0,
        'kind': 'munechika_train_current',
        'current': {},
    },
]


def make_one(config):
    """
    1 つの条件でデータを作り、npz と確認用の図を保存する。

    処理の流れ:
        1. 電流関数の引数を決める（既定値 + config['current'] で上書き）
        2. 入力電流 I_ext を作る
        3. HH モデルを計算する
        4. 発散していないか確かめ、スパイク数を数える
        5. 時系列と条件（meta）を npz に保存する
        6. 最初の 500 ms を図にして png で保存する

    Args:
        config (dict): DATASETS の要素 1 つ。

    Returns:
        dict: 保存先のパスと、表示用の要約（ステップ数、スパイク数、計算時間など）。

    Raises:
        RuntimeError: 膜電位が nan や inf になった（計算が発散した）とき。
    """
    name = config['name']
    dt = config['dt']
    T = config['T']

    func = CURRENT_FUNCS[config['kind']]
    params = {k: p.default for k, p in inspect.signature(func).parameters.items() if k != 't'}
    params.update(config['current'])

    t = np.arange(0, T, dt)
    I_ext = func(t, **params)

    start = time.time()
    V, m, h, n = simulate(t, I_ext, V0=V_REST)
    elapsed = time.time() - start

    if not np.all(np.isfinite(V)):
        raise RuntimeError(f'{name}: 膜電位が発散した（dt が大きすぎる可能性）')

    n_spikes = int(np.sum((V[:-1] < 0.0) & (V[1:] >= 0.0)))

    os.makedirs(DATA_DIR, exist_ok=True)

    meta = {
        'note': config['note'],
        'dt_ms': dt,
        'T_ms': T,
        'n_steps': len(t),
        'V_rest_mV': V_REST,
        'current': dict(params, kind=config['kind']),
        'n_spikes': n_spikes,
        'columns': {
            't': '時刻 (ms)',
            'I_ext': '入力電流 (uA/cm^2)',
            'V': '膜電位 (mV)',
            'm': 'Na チャネル活性化ゲート',
            'h': 'Na チャネル不活性化ゲート',
            'n': 'K チャネル活性化ゲート',
        },
    }

    npz_path = os.path.join(DATA_DIR, f'{name}.npz')
    np.savez_compressed(
        npz_path,
        t=t, I_ext=I_ext, V=V, m=m, h=h, n=n,
        meta=json.dumps(meta, ensure_ascii=False),
    )

    width = max(10.0, T / 50.0)

    fig, axes = plt.subplots(3, 1, figsize=(width, 7), sharex=True)

    axes[0].plot(t, I_ext, color='red', linewidth=0.8)
    axes[0].set_ylabel(r'$I_{ext}$ [$\mu$A/cm$^2$]')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(t, V, color='blue', linewidth=0.8)
    axes[1].set_ylabel('V [mV]')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].plot(t, m, color='green', linewidth=0.8, label='m')
    axes[2].plot(t, h, color='green', linewidth=0.8, linestyle='--', label='h')
    axes[2].plot(t, n, color='magenta', linewidth=0.8, label='n')
    axes[2].set_ylabel('gate')
    axes[2].set_xlabel('Time [ms]')
    axes[2].set_xlim(0, T)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'{name}  ({int(T)} ms)')
    fig.tight_layout()
    fig_path = os.path.join(DATA_DIR, f'{name}.png')
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)

    return {
        'name': name,
        'npz': npz_path,
        'png': fig_path,
        'n_steps': len(t),
        'n_spikes': n_spikes,
        'elapsed': elapsed,
        'size_mb': os.path.getsize(npz_path) / 1024 ** 2,
        'V_range': (V.min(), V.max()),
    }


if __name__ == '__main__':
    for config in DATASETS:
        print(f"--- {config['name']} を生成中 ---")
        r = make_one(config)
        print(f"  ステップ数 : {r['n_steps']:,}")
        print(f"  スパイク数 : {r['n_spikes']}")
        print(f"  膜電位の範囲: {r['V_range'][0]:.1f} 〜 {r['V_range'][1]:.1f} mV")
        print(f"  計算時間   : {r['elapsed']:.1f} 秒")
        print(f"  保存先     : {r['npz']} ({r['size_mb']:.1f} MB)")
        print(f"  確認用の図 : {r['png']}")
        print()
