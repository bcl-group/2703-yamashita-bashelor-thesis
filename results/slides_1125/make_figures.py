"""
11/25 の進捗発表のスライド用の図を描く（PNG、白背景）。
    uv run python -X utf8 results/slides_1125/make_figures.py
入力: results/slides_1125/val_traces.npz（検証 17 本での閉ループ波形）、results/ranking/accuracy.csv、results/ranking/speed_cpu.csv
出力: results/slides_1125/fig_*.png

図の約束（研究室の図表の掟と、AI っぽい装飾を避けるため）:
    - 色はモデルの区別だけに使う（A〜D の 4 色は plot_final.py と同じ）。HH は黒か灰色、文字は黒と灰色
    - 影・グラデーション・角丸の枠・アイコンは使わない。軸には名前と単位を付ける
    - 凡例の箱は使わず、線や棒の横に直接名前を書く
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COL = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'D': '#eda100'}
LBL = {'A': 'A  FFNN', 'B': 'B  RNN', 'C': 'C  FFNN 3 個', 'D': 'D  RNN 3 個'}
TAG = {'A': 'r8_mlp_hd_w256', 'B': 'sp_r_init', 'C': 'm3_hdf_s0', 'D': 'rnn3/p_h64_cl1'}
SPEED = {'A': 'Markov MLP 256×3', 'B': 'RNN 1 個（W_hh）128×2', 'C': 'MLP 3 個 128×2',
         'D': 'RNN 3 個 H64（並列・直列）'}
INK, INK2, GRID, RED = '#1a1a1a', '#5c5c5c', '#e2e2e2', '#b0372c'
plt.rcParams.update({'font.family': ['Yu Gothic', 'Meiryo', 'sans-serif'], 'font.size': 16,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix',
                     'savefig.dpi': 200, 'savefig.facecolor': 'white'})


def wave():
    d = np.load(os.path.join(HERE, 'val_traces.npz'))
    names = list(d['names'])
    i = names.index('v ramp 0-25')
    n = int(d['lens'][i]) - 1
    t = np.arange(n) * 0.01
    fig, axs = plt.subplots(5, 1, figsize=(11.5, 6.2), sharex=True,
                            gridspec_kw={'height_ratios': [0.7, 1, 1, 1, 1], 'hspace': 0.2})
    axs[0].plot(t, d['I'][i, :n], color=INK, lw=1.6)
    axs[0].set_ylabel('$I$ [µA/cm²]', fontsize=13)
    axs[0].set_yticks([0, 25])
    for ax, k in zip(axs[1:], 'ABCD'):
        ax.plot(t, d[f'hh{i}'], color='#bdbdbd', lw=3.4)
        ax.plot(t, d[f'{k}{i}'], color=COL[k], lw=1.3)
        ax.set_ylim(-90, 55)
        ax.set_yticks([-60, 0])
        ax.text(1.01, 0.5, LBL[k], transform=ax.transAxes, color=INK, fontsize=15, va='center')
    axs[2].set_ylabel('$V$ [mV]', fontsize=14)
    axs[0].text(1.01, 0.5, '灰色の太線：HH', transform=axs[0].transAxes, color=INK2, fontsize=13, va='center')
    axs[-1].set_xlabel('時刻 $t$ [ms]')
    axs[-1].set_xlim(0, t[-1])
    fig.subplots_adjust(left=0.09, right=0.84, top=0.97, bottom=0.11)
    fig.savefig(os.path.join(HERE, 'fig_wave_ramp.png'))
    plt.close(fig)


def accuracy():
    rows = {r['key']: r for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'ranking', 'accuracy.csv'),
                                                        encoding='utf-8-sig'))}
    cols = list(next(iter(rows.values())).keys())
    # 列: 順位, model, arch, テスト Γ, テスト 100% 本数, テスト平均一致率, 検証 100% 本数, 検証平均一致率, src
    g = [float(rows[TAG[k]][cols[3]]) for k in 'ABCD']
    n = [int(rows[TAG[k]][cols[4]]) for k in 'ABCD']
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.8), gridspec_kw={'wspace': 0.32})
    x = np.arange(4)
    spec = ((axs[0], g, '平均 Γ', (0.85, 1.0), '{:.3f}'), (axs[1], n, '全スパイクが一致した本数', (0, 26), '{}'))
    for ax, y, yl, lim, fmt in spec:
        ax.bar(x, y, width=0.6, color=[COL[k] for k in 'ABCD'])
        for xi, yi in zip(x, y):
            ax.text(xi, yi + (lim[1] - lim[0]) * 0.015, fmt.format(yi), ha='center', va='bottom',
                    fontsize=15, color=INK)
        ax.set_xticks(x, [LBL[k].replace('  ', '\n') for k in 'ABCD'], fontsize=14)
        ax.set_ylim(*lim)
        ax.set_ylabel(yl)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axs[0].set_yticks([0.85, 0.9, 0.95, 1.0])
    axs[1].set_yticks([0, 13, 26])
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.13)
    fig.savefig(os.path.join(HERE, 'fig_accuracy.png'))
    plt.close(fig)


def speed():
    # GPU（学内サーバの Tesla V100、fp32、CUDA Graph）での 1 ステップの時間。results/ranking/speed_gpu_all.csv
    rows = {r['arch']: r for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'ranking', 'speed_gpu_all.csv'),
                                                       encoding='utf-8-sig'))}
    B = np.array([1, 32, 1024, 8192, 65536])
    cols = [c for c in next(iter(rows.values())) if c.startswith('cudagraph B=')]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    hh = np.array([float(rows['HH 陽的 Euler（基準）'][c]) for c in cols])
    ax.plot(B, hh, color=INK, lw=2.8, marker='o', ms=8)
    ends = {'HH': hh[-1]}
    for k in 'ABCD':
        y = np.array([float(rows[SPEED[k]][c]) for c in cols])
        ax.plot(B, y, color=COL[k], lw=2, marker='o', ms=7)
        ends[k] = y[-1]
    last = 0
    for k, y in sorted(ends.items(), key=lambda kv: kv[1]):
        yy = max(y, last * 1.4)
        ax.text(B[-1] * 1.4, yy, 'HH（陽的 Euler）' if k == 'HH' else LBL[k], color=INK, va='center', fontsize=15)
        last = yy
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.6, 65536 * 30)
    ax.set_ylim(25, max(ends.values()) * 2.6)   # 右端の名前が上に切れないように
    ax.set_xticks(B, ['1', '32', '1,024', '8,192', '65,536'])
    ax.set_xlabel('同時に計算する細胞の数 $B$')
    ax.set_ylabel('1 ステップの実行時間 [µs]')
    ax.grid(True, which='major', color=GRID)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.15)
    fig.savefig(os.path.join(HERE, 'fig_speed_gpu.png'))
    plt.close(fig)


def gamma_demo():
    # Γ の考え方：HH のスパイク時刻の ±2 ms に代理モデルのスパイクが入れば一致
    hh = np.array([8, 20, 33, 47, 60])
    sm = np.array([8.6, 21.5, 36.4, 46.5, 59.2])
    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    for s in hh:
        ax.axvspan(s - 2, s + 2, ymin=0.1, ymax=0.85, color='#e8e8e8', lw=0)
        ax.vlines(s, 0.55, 0.85, color=INK, lw=2.4)
    for s, ok in zip(sm, abs(sm - hh) <= 2):
        ax.vlines(s, 0.15, 0.45, color=COL['A'] if ok else RED, lw=2.4)
    ax.text(-1, 0.70, 'HH', ha='right', va='center', fontsize=15)
    ax.text(-1, 0.30, '代理モデル', ha='right', va='center', fontsize=15)
    ax.annotate('', xy=(30.6, 0.97), xytext=(35.4, 0.97), arrowprops=dict(arrowstyle='<->', color=INK2, lw=1.2))
    ax.text(33, 1.02, '±2 ms', ha='center', va='bottom', fontsize=13, color=INK2)
    ax.text(36.4, 0.07, '外れ', ha='center', va='top', fontsize=13, color=RED)
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.12, 1.2)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('時刻 [ms]', fontsize=14)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.26)
    fig.savefig(os.path.join(HERE, 'fig_gamma.png'))
    plt.close(fig)


def perturb_demo():
    # 学習の考え方：HH の軌道（V–n 平面の周期軌道）の周りに、雑音でずらした状態を置き、その状態での HH の 1 ステップを正解にする
    dt = 0.01
    t = np.arange(0, 60, dt)
    V, m, h, n = _sim(t)
    sel = t > 25
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(V[sel], n[sel], color='#9a9a9a', lw=2.4)
    rng = np.random.default_rng(3)
    # 軌道の長さ（V を 100 mV で割って n とそろえた座標）で等間隔に 14 点取る
    vi, ni = V[sel] / 100, n[sel]
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(vi), np.diff(ni)))])
    idx = np.where(sel)[0][np.searchsorted(arc, np.linspace(0, arc[-1], 15)[:-1])]
    for k in idx:
        v0, n0 = V[k] + rng.normal(0, 4.0), n[k] + rng.normal(0, 0.03)
        dv, dn = _step(v0, m[k], h[k], n0)
        u = np.array([dv / 100, dn])
        u = u / np.linalg.norm(u) * 0.06       # 向きだけを見せる（長さはそろえる）
        ax.plot(v0, n0, 'o', color=COL['A'], ms=6)
        ax.annotate('', xy=(v0 + 100 * u[0], n0 + u[1]), xytext=(v0, n0),
                    arrowprops=dict(arrowstyle='->', color=COL['A'], lw=1.5))
    ax.text(0.98, 0.04, '灰色：HH の軌道\n点：雑音でずらした状態\n矢印：その状態での HH の変化の向き', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=13, color=INK2)
    ax.set_xlabel('$V$ [mV]')
    ax.set_ylabel('$n$')
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.15)
    fig.savefig(os.path.join(HERE, 'fig_perturb.png'))
    plt.close(fig)


def _rates(V):
    am = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    bm = 4 * np.exp(-(V + 65) / 18)
    ah = 0.07 * np.exp(-(V + 65) / 20)
    bh = 1 / (1 + np.exp(-(V + 35) / 10))
    an = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    bn = 0.125 * np.exp(-(V + 65) / 80)
    return am, bm, ah, bh, an, bn


def _step(V, m, h, n, I=10.0, dt=0.01):
    am, bm, ah, bh, an, bn = _rates(V)
    dV = -120 * m ** 3 * h * (V - 50) - 36 * n ** 4 * (V + 77) - 0.3 * (V + 54.4) + I
    return dt * dV, dt * (an * (1 - n) - bn * n)


def _sim(t, I=10.0, dt=0.01):
    V = -65.0
    am, bm, ah, bh, an, bn = _rates(V)
    m, h, n = am / (am + bm), ah / (ah + bh), an / (an + bn)
    out = np.zeros((4, len(t)))
    for k in range(len(t)):
        out[:, k] = V, m, h, n
        am, bm, ah, bh, an, bn = _rates(V)
        dV = -120 * m ** 3 * h * (V - 50) - 36 * n ** 4 * (V + 77) - 0.3 * (V + 54.4) + I
        m, h, n = m + dt * (am * (1 - m) - bm * m), h + dt * (ah * (1 - h) - bh * h), n + dt * (an * (1 - n) - bn * n)
        V = V + dt * dV
    return out


RAW_SRC = {'A': 'hh_improve/r8_mlp_hd_w256', 'B': 'hh_improve/sp_r_init', 'C': 'hh_improve/m3_hdf_s0',
           'D': 'hh_rnn3/p_h64_cl1'}


def raw_grid():
    # 4 モデルの生データ（開発用テスト 26 本）。セルの数字 = 予測のスパイク数。Γ < 1 のセルだけ色を付けて Γ も書く
    import json
    from matplotlib.patches import Rectangle
    R = {k: json.load(open(os.path.join(ROOT, 'results', v + '.json'), encoding='utf-8'))['test']['rows']
         for k, v in RAW_SRC.items()}
    rows = R['A']
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.set_xlim(0, 2)
    ax.set_ylim(14.2, -1.2)
    ax.axis('off')
    half = 13
    # 列の位置（各ブロックの中で）: 電流名, HH, A, B, C, D
    xs = [0.0, 0.40, 0.52, 0.64, 0.76, 0.88]
    ws_ = [0.40, 0.12, 0.12, 0.12, 0.12, 0.12]

    def shade(g):
        if g >= 0.999:
            return None
        if g >= 0.9:
            return '#fbe0cf'
        if g >= 0.5:
            return '#f4b183'
        return '#e8836a'

    for b in range(2):
        x0 = b * 1.02
        heads = ['電流', 'HH', 'A', 'B', 'C', 'D']
        for h, x, w in zip(heads, xs, ws_):
            ax.text(x0 + x + (0.005 if h == '電流' else w / 2), -0.55, h, ha='left' if h == '電流' else 'center',
                    va='center', fontsize=14, fontweight='bold',
                    color=COL[h] if h in COL else INK)
        ax.plot([x0, x0 + 1.0], [-0.05, -0.05], color=INK2, lw=1)
        for i in range(half):
            n = b * half + i
            if n >= len(rows):
                break
            y = i + 0.45
            ax.text(x0 + 0.005, y, rows[n]['name'], ha='left', va='center', fontsize=12.5, color=INK)
            ax.text(x0 + xs[1] + ws_[1] / 2, y, str(rows[n]['spikes'][0]), ha='center', va='center',
                    fontsize=13, color=INK)
            for j, k in enumerate('ABCD'):
                r = R[k][n]
                x = x0 + xs[2 + j]
                f = shade(r['gamma'])
                if f:
                    ax.add_patch(Rectangle((x + 0.004, i + 0.05), ws_[2 + j] - 0.008, 0.8, color=f, lw=0))
                    ax.text(x + ws_[2 + j] / 2, y - 0.17, str(r['spikes'][1]), ha='center', va='center',
                            fontsize=12.5, color=INK, fontweight='bold')
                    ax.text(x + ws_[2 + j] / 2, y + 0.22, f"Γ {r['gamma']:.2f}", ha='center', va='center',
                            fontsize=10, color=INK)
                else:
                    ax.text(x + ws_[2 + j] / 2, y, str(r['spikes'][1]), ha='center', va='center',
                            fontsize=12.5, color=INK2)
            ax.plot([x0, x0 + 1.0], [i + 0.9, i + 0.9], color=GRID, lw=0.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(HERE, 'fig_raw26.png'))
    plt.close(fig)


def fails():
    # 外した電流の波形（4 種類）。灰色の太線が HH、色の線が外したモデル
    d = np.load(os.path.join(HERE, 'test_traces.npz'))
    names = list(d['names'])
    panels = [
        ('ramp 0-20', 'ABCD', (380, 500), 'ramp 0→20：HH は最後に 1 回だけ発火'),
        ('ramp 0-30', 'BC', (300, 420), 'ramp 0→30：B は毎回 3 ms 早い（C は一致）'),
        ('sin 10Hz 7.5±7.5', 'CD', (80, 260), 'sin 10 Hz：C, D は 2 周期目が約 10 ms 遅れる'),
        ('OU 5±3', 'AD', (0, 120), 'OU 5±3：A, D は 38 ms 付近で余分に 1 回発火'),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 5.8), gridspec_kw={'hspace': 0.55, 'wspace': 0.12})
    for ax, (name, ks, (t0, t1), title) in zip(axs.flat, panels):
        j = names.index(name)
        v = d[f'hh{j}']
        t = np.arange(len(v)) * 0.01
        s = (t >= t0) & (t <= t1)
        ax.plot(t[s], v[s], color='#bdbdbd', lw=3.6)
        for k in ks:
            ax.plot(t[s], d[f'{k}{j}'][s], color=COL[k], lw=1.3)
        ax.set_xlim(t0, t1)
        ax.set_ylim(-85, 55)
        ax.set_yticks([-60, 0])
        ax.set_title(title, fontsize=14, loc='left', color=INK)
        for m, k in enumerate(reversed(ks)):   # 線の色と同じ色でモデル名を書く（凡例の代わり）
            ax.text(1.0 - 0.045 * m, 1.02, k, transform=ax.transAxes, ha='right', va='bottom', fontsize=14,
                    color=COL[k], fontweight='bold')
        ax.set_xlabel('時刻 $t$ [ms]', fontsize=13)
    for ax in axs[:, 0]:
        ax.set_ylabel('$V$ [mV]', fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.1)
    fig.savefig(os.path.join(HERE, 'fig_fails.png'))
    plt.close(fig)


if __name__ == '__main__':
    wave()
    accuracy()
    speed()
    gamma_demo()
    perturb_demo()
    raw_grid()
    fails()
