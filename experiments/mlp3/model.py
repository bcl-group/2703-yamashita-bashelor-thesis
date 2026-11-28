"""
MLP 3 個（C）の代理モデル。HH の陽的オイラー法 1 ステップを 3 つのブロックに分け、それぞれを MLP で置き換える。

    ブロック 1: V                        -> αβ (6)           = α_m, β_m, α_h, β_h, α_n, β_n
    ブロック 2: αβ (6), g = (m, h, n)    -> Δg (3)
    ブロック 3: g (3), V, I (+ 積の特徴 4) -> ΔV (1)          積の特徴 = m^3 h, n^4, m^3 h V, n^4 V（--sfeat）

閉ループの 1 ステップ（hh.py と同じ順番。ブロック 3 には更新前の g を渡す）:
    αβ = max(Block1(V), 0)
    g' = clamp(g + Block2(αβ, g), 0, 1)
    V' = clamp(V + Block3(g, V, I), v_lo, 60)

重みの保存形式は experiments/hh_surrogate/hh_improve.py の Net(cell='mlp') と同じ（state_dict のキーも同じ）なので、
results/hh_improve/m3_*_b{1,2,3}.pt をそのまま読める。
"""
import torch
from torch import nn

import hh

V_HI = 60.0
V_LO_TRAIN = -100.0     # 学習・検証・テスト（26 本）で使う V の下限（安全装置）
V_LO_FINAL = -150.0     # 最終テストの採点だけ広げる。HH 自身が強い抑制で約 -121 mV まで下がるため（2026-11-23）


class Block(nn.Module):
    """
    標準化 -> [Linear -> 活性化] × layers -> Linear -> 標準化を戻す。入出力は物理単位。
    buffers: im, is_ (入力の平均・標準偏差), om, os_ (出力の平均・標準偏差)
    """

    def __init__(self, d_in, H, layers, d_out, feat=0, act='silu', **_):
        super().__init__()
        # cfg は旧 Net と同じキーで持つ（旧形式で保存し直せるように）
        self.cfg = dict(d_in=d_in, H=H, layers=layers, d_out=d_out, d_aux=0, feat=feat, leak=0.0,
                        cell='mlp', phys=False, act=act)
        d = d_in + (4 if feat == 3 else 0)
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(d, H), {'silu': nn.SiLU(), 'relu': nn.ReLU()}[act]]
            d = H
        self.mlp = nn.Sequential(*mods)
        self.out = nn.Linear(H, d_out)
        d_x = d_in + (4 if feat == 3 else 0)
        for k, dim in [('im', d_x), ('is_', d_x), ('om', d_out), ('os_', d_out)]:
            self.register_buffer(k, torch.zeros(dim) if k.endswith('m') else torch.ones(dim))

    def expand(self, x):
        # feat=3（ブロック 3）: 入力 (m, h, n, V, I) に Na・K 電流の形 m^3 h, n^4, m^3 h V, n^4 V を足す
        if self.cfg['feat'] != 3:
            return x
        m, h, n, V = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        a, b = m ** 3 * h, n ** 4
        return torch.cat([x, torch.stack([a, b, a * V, b * V], -1)], -1)

    def set_scalers(self, X, Y):
        X = self.expand(X)
        self.im.copy_(X.mean(0)); self.is_.copy_(X.std(0) + 1e-12)
        self.om.copy_(Y.mean(0)); self.os_.copy_(Y.std(0) + 1e-12)

    def forward(self, x):
        # x: (B, d_in) 物理単位 -> (B, d_out) 物理単位
        u = (self.expand(x) - self.im) / self.is_
        return self.out(self.mlp(u)) * self.os_ + self.om


class MLP3(nn.Module):

    def __init__(self, H=128, layers=2, act='silu', sfeat=True):
        super().__init__()
        self.b1 = Block(1, H, layers, 6, act=act)
        self.b2 = Block(9, H, layers, 3, act=act)
        self.b3 = Block(5, H, layers, 1, act=act, feat=3 if sfeat else 0)

    @property
    def blocks(self):
        return [self.b1, self.b2, self.b3]

    @property
    def dtype(self):
        return self.b1.out.weight.dtype

    def step(self, V, g, I_k, v_lo=V_LO_TRAIN):
        """V (B,), g (B, 3), I_k (B,) -> V' (B,), g' (B, 3), αβ (B, 6)（ゲートの更新に使った値）"""
        ab = self.b1(V[:, None]).clamp(min=0.0)
        dg = self.b2(torch.cat([ab, g], 1))
        dv = self.b3(torch.cat([g, V[:, None], I_k[:, None]], 1))
        return (V + dv[:, 0]).clamp(v_lo, V_HI), (g + dg).clamp(0.0, 1.0), ab

    @torch.no_grad()
    def run(self, I, v_lo=V_LO_TRAIN, record=False):
        """
        静止状態から閉ループで回す。I: (B, N) torch か numpy。
        Returns:
            record=False: V (B, N-1) float64 numpy。V[:, k] は k+1 ステップ目（旧 run_m3 と同じ約束）
            record=True : dict  V (B, N) float64, g (B, N, 3) float32（初期値込み）, ab (B, N-1, 6) float32
        """
        I = torch.as_tensor(I)
        B, N = I.shape
        V, g = hh.rest_state(B, self.dtype)
        I = I.to(V.device)
        Vs = torch.empty(B, N, dtype=torch.float64)
        Vs[:, 0] = V.double()
        if record:
            Gs = torch.empty(B, N, 3, dtype=torch.float32)
            ABs = torch.empty(B, N - 1, 6, dtype=torch.float32)
            Gs[:, 0] = g.float()
        for k in range(N - 1):
            V, g, ab = self.step(V, g, I[:, k].to(self.dtype), v_lo)
            Vs[:, k + 1] = V.double()
            if record:
                Gs[:, k + 1], ABs[:, k] = g.float(), ab.float()
        if not record:
            return Vs[:, 1:].cpu().numpy()
        return {'V': Vs.cpu().numpy(), 'g': Gs.cpu().numpy(), 'ab': ABs.cpu().numpy()}

    @torch.no_grad()
    def visited_states(self, I, stride):
        # DAgger 用: float32 で閉ループを回し、通った状態 [V, m, h, n, I] を stride ごとに集める (·, 5) float64
        B, N = I.shape
        V, g = hh.rest_state(B, torch.float32)
        out = []
        for k in range(N):
            if k % stride == 0:
                out.append(torch.cat([V[:, None], g, I[:, k, None].float()], 1).double())
            V, g, _ = self.step(V, g, I[:, k].float())
        return torch.stack(out, 1).reshape(-1, 5)

    # ---------------- 保存と読み込み ----------------

    def save(self, prefix, info=None):
        # 旧形式と同じく <prefix>_b1.pt, _b2.pt, _b3.pt に分けて保存する
        for k, b in enumerate(self.blocks, 1):
            torch.save({'cfg': b.cfg, 'state': b.state_dict(), 'info': info or {}}, f'{prefix}_b{k}.pt')

    @classmethod
    def load(cls, prefix, dtype=torch.float64):
        # prefix: 例 results/hh_improve/m3_hdf_s0（_b1.pt などは付けない）
        ds = [torch.load(f'{prefix}_b{k}.pt', weights_only=False, map_location='cpu') for k in (1, 2, 3)]
        c1, c3 = ds[0]['cfg'], ds[2]['cfg']
        net = cls(H=c1['H'], layers=c1['layers'], act=c1.get('act', 'silu'), sfeat=c3['feat'] == 3)
        for b, d in zip(net.blocks, ds):
            b.load_state_dict(d['state'])
        return net.to(dtype)
