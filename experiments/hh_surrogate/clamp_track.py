"""
閉ループで状態を範囲に切り詰める（clamp する）前に、範囲の外へ出た回数と量を記録する。

代理モデルの閉ループ（hh_improve.run3 / run_state / m3_step、rescore_gamma.ensemble）は、
ゲート変数を [0, 1]、V を [-100, 60] mV に切り詰めている（採点中は V の下限を EVAL_V_LO = -150 mV に広げる）。
切り詰めた後の値は必ず範囲内なので、
「物理的にありえない状態を出していないか」は、切り詰める前の値で見る必要がある。

使い方:
    import clamp_track as ct
    ct.start(lens)              # lens: 各電流で採点するステップ数 (B,)。これより後（末尾の 0 埋め）は数えない
    V = run(I)                  # 閉ループの中で ct.clamp(x, lo, hi, 'gate') が呼ばれる
    stats = ct.stop()           # {'gate': {'frac': (B,), 'max': (B,)}, 'V': {...}}。記録がなければ {}
        frac: 範囲の外に出たステップの割合、max: はみ出した最大量（ゲートは無次元、V は mV）

読み方の注意: ゲート変数の [0, 1] は物理的な範囲なので、外に出たら「ありえない状態」である。
一方 V の範囲は閉ループが発散しないための安全装置で、HH 自身も強い抑制（−20 µA/cm²）で
約 −121 mV まで下がる。そのため採点中は下限を −150 mV に広げる。V が切り詰めに当たったことは
「出力が切り詰めで歪んだ」ことを表す。

記録していないとき（学習中など）は x.clamp(lo, hi) と同じで、追加の計算はしない。
"""
import numpy as np
import torch

_S = {'on': False}

# 採点中（start〜stop の間）だけ、V の下限をここまで広げる。学習中は呼び出し側の値（-100 mV）のまま。
# HH 自身が強い抑制で約 −121 mV まで下がるため、−100 mV で止めると代理モデルだけが不利になる（2026-11-23 に変更）
EVAL_V_LO = -150.0


def start(lens):
    _S.clear()
    _S.update({'on': True, 'lens': np.asarray(lens)})


def stop():
    out = {}
    for name, st in _S.items():
        if not isinstance(st, dict):
            continue
        n_used = np.minimum(_S['lens'], st['k'])
        out[name] = {'frac': (st['n'].cpu().numpy() / np.maximum(n_used, 1)).tolist(),
                     'max': st['max'].cpu().numpy().tolist()}
    _S.clear()
    _S['on'] = False
    return out


def clamp(x, lo, hi, name):
    """
    x を [lo, hi] に切り詰める。記録中なら、切り詰める前に範囲の外へ出た量を電流ごとに記録する。
    x: (B,) または (B, k)。name ごとに、1 ステップに 1 回だけ呼ぶこと（呼ばれた回数をステップ数とみなす）
    """
    if _S.get('on'):
        if name == 'V':
            lo = min(lo, EVAL_V_LO)
        ex =(lo - x).clamp(min=0) + (x - hi).clamp(min=0)
        if ex.dim() > 1:
            ex = ex.amax(1)
        ex = ex.detach().double()
        st = _S.get(name)
        if st is None:
            st = _S[name] = {'n': torch.zeros_like(ex), 'max': torch.zeros_like(ex), 'k': 0,
                             'lim': torch.as_tensor(_S['lens'], device=ex.device)}
        live = st['k'] < st['lim']
        st['n'] += ((ex > 0) & live).double()
        st['max'] = torch.where(live, torch.maximum(st['max'], ex), st['max'])
        st['k'] += 1
    return x.clamp(lo, hi)
