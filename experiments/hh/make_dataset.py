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
from neuron.current import random_pulse_current


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'hh')


DATASETS = [
    {
        'name': 'teacher_slide',
        'note': '棟近先輩の進捗報告スライド p.7 の教師電流の条件',
        'dt': 0.01,
        'T': 9000.0,
        'current': {'i_min': -5.0, 'i_max': 20.0, 'interval': 20.0, 'seed': 0},
    },
    {
        'name': 'teacher_thesis',
        'note': '棟近先輩の卒業論文 Algorithm 2 の条件',
        'dt': 0.05,
        'T': 900.0,
        'current': {'i_min': 0.0, 'i_max': 20.0, 'interval': 10.0, 'seed': 0},
    },
]


def make_one(config):
    name = config['name']
    dt = config['dt']
    T = config['T']

    t = np.arange(0, T, dt)
    I_ext = random_pulse_current(t, **config['current'])

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
        'current': dict(config['current'], kind='random_pulse_current'),
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

    n_show = min(len(t), int(500.0 / dt))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t[:n_show], I_ext[:n_show], color='red', linewidth=0.8)
    axes[0].set_ylabel(r'$I_{ext}$ [$\mu$A/cm$^2$]')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(t[:n_show], V[:n_show], color='blue', linewidth=0.8)
    axes[1].set_ylabel('V [mV]')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].plot(t[:n_show], m[:n_show], color='green', linewidth=0.8, label='m')
    axes[2].plot(t[:n_show], h[:n_show], color='green', linewidth=0.8,
                 linestyle='--', label='h')
    axes[2].plot(t[:n_show], n[:n_show], color='magenta', linewidth=0.8, label='n')
    axes[2].set_ylabel('gate')
    axes[2].set_xlabel('Time [ms]')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'{name}  (first {int(n_show * dt)} ms of {int(T)} ms)')
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
