"""
学習用の電流をその場で作る。検証・テストの電流（eval_sets.py）そのものは作らない（水準・長さ・傾きは乱数）。

    normal(B, N, rng)             通常の電流 (B, N)：切り替わる水準 + OU の揺らぎ + 短いパルス
    hard(B, N, rng, kind='v1')    苦手な場面を多く含む電流 (B, N)
        v1: 長い一定・ゆっくりしたランプ・閾値付近の揺らぎ（第 8 ラウンドまで、m3_hdf）
        v2: v1 に加えて、静止からの一定（閾値付近）・0 からのランプ・正弦波を重点的に（m3_v2f05d）

乱数を引く順番は experiments/hh_surrogate/hh_improve.py の gen_current / gen_hard_set と同じ。
同じ seed なら同じ電流になる（旧コードの学習を同じ条件で作り直せる）。

OU 過程の計算が 2 通りある（_ou と _ou_fast）のは、旧コードがそうなっていたため。数式は同じだが
丸めの順番が違うので、同じ電流を作り直すには v1 と通常の電流は _ou、v2 は _ou_fast を使う必要がある。
"""
import math

import numpy as np
import torch

from hh import DT


def _ou(N, tau, sig, rng):
    # 平均 0 の OU 過程 (N,)：x_0 = 0、x_k = a x_{k-1} + e_k、a = exp(-dt/τ)、e_k ~ N(0, σ²(1 - a²))
    a = math.exp(-DT / tau)
    e = rng.standard_normal(N) * sig * math.sqrt(1 - a * a)
    x = np.zeros(N)
    for k in range(1, N):
        x[k] = a * x[k - 1] + e[k]
    return x


def _ou_fast(N, tau, sig, rng):
    # _ou と同じ式を numpy の累積で計算する: x_k = a^k Σ_{j<=k} a^{-j} e_j。a^{-j} があふれないよう 2000 点ごとに区切る
    a = math.exp(-DT / tau)
    e = rng.standard_normal(N) * sig * math.sqrt(1 - a * a)
    S = 2000
    k = np.arange(S)
    pk, pk1, pinv = a ** k, a ** (k + 1), a ** (-k)
    E = np.concatenate([e, np.zeros(-N % S)]).reshape(-1, S)     # (区切りの数, S)
    Z = pk * np.cumsum(E * pinv, axis=1)                          # 区切りの頭を 0 とした解
    x = np.empty_like(Z)
    prev = 0.0
    for r in range(Z.shape[0]):                                   # 前の区切りの最後の値を引き継ぐ
        x[r] = pk1 * prev + Z[r]
        prev = x[r, -1]
    return x.reshape(-1)[:N]


def _normal_one(N, rng):
    """
    基準レベル: 2〜50 ms ごとに切り替える。0（30%）か U[-5, 25]。区間ごとに「一定」か「直線で次へ」（30%）
    揺らぎ: OU（τ = U[1, 5] ms、σ = U[0, 3]）
    短いパルス: 平均 50 ms に 1 回、幅 0.5〜2 ms、高さ U[10, 40]
    """
    I = np.zeros(N)
    t = 0
    lvl = 0.0 if rng.random() < 0.3 else rng.uniform(-5, 25)
    while t < N:
        L = max(1, int(rng.uniform(2, 50) / DT))
        nxt = 0.0 if rng.random() < 0.3 else rng.uniform(-5, 25)
        seg = np.linspace(lvl, nxt, L) if rng.random() < 0.3 else np.full(L, lvl)
        I[t:t + L] = seg[:N - t]
        t += L
        lvl = nxt
    tau, sig = rng.uniform(1, 5), rng.uniform(0, 3)
    I += _ou(N, tau, sig, rng)
    for k in rng.integers(0, N, size=rng.poisson(N * DT / 50)):
        I[k:k + int(rng.uniform(0.5, 2) / DT)] += rng.uniform(10, 40)
    return I


def _hard_levels(kind, N, rng):
    """
    苦手な電流の水準（OU を足す前）(N,):
      hold  : 50〜500 ms の一定区間を続ける。水準の半分は閾値付近 U[4, 12]、残りは U[-5, 30]
      ramp  : 100〜3000 ms かけて直線で次の水準へ（ゆっくり閾値をまたぐ）
      nearou: 閾値付近の平均 U[4, 10]（100〜1000 ms ごとに変える）
    """
    lvl_draw = lambda: rng.uniform(4, 12) if rng.random() < 0.5 else rng.uniform(-5, 30)
    I = np.zeros(N)
    t, lvl = 0, lvl_draw()
    while t < N:
        if kind == 'hold':
            L = max(1, int(rng.uniform(50, 500) / DT))
            I[t:t + L] = lvl
            lvl = lvl_draw()
        elif kind == 'ramp':
            L = max(1, int(rng.uniform(100, 3000) / DT))
            nxt = lvl_draw()
            I[t:t + L] = np.linspace(lvl, nxt, L)[:N - t]
            lvl = nxt
        else:
            L = max(1, int(rng.uniform(100, 1000) / DT))
            I[t:t + L] = rng.uniform(4, 10)
        t += L
    return I


def _hard_ou(kind, rng):
    # 重ねる OU の (τ, σ)。nearou は τ 1〜10 ms・σ 0〜3、hold と ramp は弱い揺らぎ（τ 1〜5 ms、σ 0〜1）
    tau = rng.uniform(1, 10) if kind == 'nearou' else rng.uniform(1, 5)
    sig = rng.uniform(0, 3) if kind == 'nearou' else rng.uniform(0, 1)
    return tau, sig


def _hard_v1_one(N, rng):
    kind = rng.choice(['hold', 'ramp', 'nearou'])
    I = _hard_levels(kind, N, rng)
    return I + _ou(N, *_hard_ou(kind, rng), rng)


# v2 の種類と割合。評価で崩れやすい「静止から始まる一定（閾値付近）」を最も多くする
HARD_V2 = {'const': 0.4, 'ramp0': 0.2, 'sin': 0.15, 'hold': 0.1, 'ramp': 0.05, 'nearou': 0.1}


def _const_level(rng):
    # 70% が反復発火の閾値付近 U[5.5, 10]（HH は約 6.3 から反復発火、〜9.8 まで静止と共存）、10% が U[2, 5.5]、20% が U[-5, 30]
    u = rng.random()
    if u < 0.7:
        return rng.uniform(5.5, 10)
    if u < 0.8:
        return rng.uniform(2, 5.5)
    return rng.uniform(-5, 30)


def _hard_v2_one(N, rng):
    """
    種類を HARD_V2 の割合で選ぶ:
      const : [休み 30〜150 ms] + [一定 150〜700 ms] をくり返す（最初の休みは半分の確率でなし）。水準は _const_level
      ramp0 : [休み 30〜150 ms] + [U[-1, 1] から U[8, 40] へ 200〜1000 ms かけて直線（20% は下り）] + [終点で 0〜200 ms 保持]
      sin   : 200〜800 ms ごとに offset U[2, 12]、振幅（半分は offset と同じ、残りは U[2, 10]）、
              周波数 2〜40 Hz（対数一様）、位相（半分は 0）を選び直す正弦波
      hold, ramp, nearou : v1 と同じ（OU は _ou_fast）
    hold, ramp, nearou 以外は、半分の本数に弱い OU（τ 1〜5 ms、σ 0〜1）を重ねる
    """
    kinds = list(HARD_V2)
    p = np.array([HARD_V2[k] for k in kinds])
    kind = kinds[rng.choice(len(kinds), p=p / p.sum())]
    if kind in ('hold', 'ramp', 'nearou'):
        I = _hard_levels(kind, N, rng)
        return I + _ou_fast(N, *_hard_ou(kind, rng), rng)
    I = np.zeros(N)
    t, first = 0, True
    while t < N:
        if kind == 'const':
            if not (first and rng.random() < 0.5):
                t += max(1, int(rng.uniform(30, 150) / DT))
            L = max(1, int(rng.uniform(150, 700) / DT))
            I[t:t + L] = _const_level(rng)
            t += L
        elif kind == 'ramp0':
            t += max(1, int(rng.uniform(30, 150) / DT))
            L = max(1, int(rng.uniform(200, 1000) / DT))
            lo, hi = rng.uniform(-1, 1), rng.uniform(8, 40)
            a0, a1 = (hi, lo) if rng.random() < 0.2 else (lo, hi)
            n = len(I[t:t + L])
            I[t:t + L] = np.linspace(a0, a1, L)[:n]
            t += L
            Lh = int(rng.uniform(0, 200) / DT)
            I[t:t + Lh] = a1
            t += Lh
        else:
            L = max(1, int(rng.uniform(200, 800) / DT))
            off = rng.uniform(2, 12)
            amp = off if rng.random() < 0.5 else rng.uniform(2, 10)
            f = math.exp(rng.uniform(math.log(2), math.log(40))) / 1000.0      # [1/ms]
            ph = 0.0 if rng.random() < 0.5 else rng.uniform(0, 2 * math.pi)
            tt = np.arange(L) * DT          # 旧コードと同じ順番で掛ける（丸めをそろえるため）
            n = len(I[t:t + L])
            I[t:t + L] = (off + amp * np.sin(2 * math.pi * f * tt + ph))[:n]
            t += L
        first = False
    if rng.random() < 0.5:
        I += _ou_fast(N, rng.uniform(1, 5), rng.uniform(0, 1), rng)
    return I


def normal(B, N, rng):
    return torch.from_numpy(np.stack([_normal_one(N, rng) for _ in range(B)]))


def hard(B, N, rng, kind='v1'):
    one = {'v1': _hard_v1_one, 'v2': _hard_v2_one}[kind]
    return torch.from_numpy(np.stack([one(N, rng) for _ in range(B)]))
