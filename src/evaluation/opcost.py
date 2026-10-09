"""
1ステップあたりの演算回数（OpCost）を数える。

棟近先輩の SINDyNeuroSurrogate（neurosurrogate/core/opcost.py、neurons/hh.py、
surrogate/parts/preprocessor/autoencoder.py）と同じ数え方にしてある。

数え方の決まり:
    - exp: 指数関数、div: 除算、pm: 加算・減算、mul: 乗算 の回数を数える。
    - 行列積とバイアス x @ W + b（x: a 次元、W: a×b）は mul = a*b、pm = a*b と数える
      （各出力の a-1 回の加算 + バイアスの 1 回 = a 回が b 個）。
    - tanh(x) = 1 - 2 / (exp(2x) + 1) として exp=1, div=1, pm=2, mul=1 と数える。
    - HH は右辺（dV/dt と dm/dt, dh/dt, dn/dt）の評価だけを数え、Euler 法の更新
      x += dx * dt は数えない（先輩と同じ）。
"""
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class OpCost:
    """演算の種類ごとの回数。足し算と整数倍ができる。"""
    exp: int = 0
    div: int = 0
    pm: int = 0
    mul: int = 0

    def __add__(self, other):
        return OpCost(**{f.name: getattr(self, f.name) + getattr(other, f.name)
                         for f in fields(self)})

    def __mul__(self, n):
        return OpCost(**{f.name: getattr(self, f.name) * n for f in fields(self)})

    def total(self):
        """全演算の合計回数。"""
        return self.exp + self.div + self.pm + self.mul

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


# tanh(x) = 1 - 2 / (exp(2x) + 1)
TANH_COST = OpCost(exp=1, div=1, pm=2, mul=1)


# --- HH（先輩の neurons/hh.py の HH_RATE_COST_MAP, HH_DV_COST, _HH_OPCOST と同じ値）---

HH_RATE_COST = {
    'alpha_m': OpCost(exp=1, div=1, pm=2, mul=2),
    'beta_m': OpCost(exp=1, div=1, pm=1, mul=1),
    'alpha_h': OpCost(exp=1, div=1, pm=1, mul=1),
    'beta_h': OpCost(exp=1, div=1, pm=2, mul=1),
    'alpha_n': OpCost(exp=1, div=1, pm=2, mul=2),
    'beta_n': OpCost(exp=1, div=1, pm=1, mul=1),
}

HH_DV_COST = (
    OpCost(pm=1)                    # 静止電位からの相対電位
    + OpCost(pm=3, mul=5) * 2       # Na 電流、K 電流
    + OpCost(pm=1, mul=1)           # リーク電流
    + OpCost(pm=3, div=1)           # dV/dt = (I - I_Na - I_K - I_L) / C
)

HH_COST = (
    sum(HH_RATE_COST.values(), OpCost())   # α, β の計算
    + HH_DV_COST
    + OpCost(pm=6, mul=6)                  # dm/dt, dh/dt, dn/dt
)


def rnn_cost(input_dim=1, hidden_dim=50, output_dim=1, standardize=True):
    """
    SimpleRNN（src/rnn/model.py）で1時刻進めるときの演算回数を数える。

    h(t) = tanh(x(t) W_xh + h(t-1) W_hh + b_h)
    y(t) = h(t) W_hy + b_y

    Args:
        input_dim (int): 入力の次元。
        hidden_dim (int): 隠れ状態の次元。
        output_dim (int): 出力の次元。
        standardize (bool): 入力の標準化 (x - mean) / std と、出力の逆変換
            y * std + mean も数えるか（hh_surrogate.py では行っている）。

    Returns:
        OpCost: 1ステップあたりの演算回数。
    """
    # x W_xh + h W_hh + b_h: 各隠れユニットで input_dim + hidden_dim 個の積を足し、バイアスを足す
    cost = OpCost(mul=(input_dim + hidden_dim) * hidden_dim,
                  pm=(input_dim + hidden_dim) * hidden_dim)
    cost = cost + TANH_COST * hidden_dim
    # h W_hy + b_y
    cost = cost + OpCost(mul=hidden_dim * output_dim, pm=hidden_dim * output_dim)
    if standardize:
        cost = cost + OpCost(pm=input_dim, div=input_dim)     # (x - mean) / std
        cost = cost + OpCost(mul=output_dim, pm=output_dim)   # y * std + mean
    return cost


def format_cost(name_a, cost_a, name_b, cost_b):
    """
    2つの OpCost を並べた表の文字列を作る（右端の列は b - a）。

    Returns:
        str: 複数行の文字列。
    """
    da, db = cost_a.to_dict(), cost_b.to_dict()
    lines = ['  {:6s} {:>10s} {:>10s} {:>10s}'.format('', name_a, name_b, 'diff')]
    for k in da:
        lines.append('  {:6s} {:10d} {:10d} {:+10d}'.format(k, da[k], db[k], db[k] - da[k]))
    lines.append('  {:6s} {:10d} {:10d} {:+10d}'.format(
        'total', cost_a.total(), cost_b.total(), cost_b.total() - cost_a.total()))
    return '\n'.join(lines)
