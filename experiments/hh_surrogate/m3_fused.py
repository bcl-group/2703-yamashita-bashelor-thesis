"""
C（MLP 3 個、m3_hdf_*）を、1 ステップが行列の積だけで進むように書き直した版（FusedM3）。
式は hh_improve.m3_step と同じ（書き換えで式は変えない）。hh_fused.py（RNN 3 個）の MLP 版。

元の 1 ステップ（hh_improve.m3_step）:
    ab = max(0, MLP1(V))                       V (B,1)          -> α, β (B,6)
    dg = MLP2([ab, g])                         (B,9)            -> Δm, Δh, Δn (B,3)
    dv = MLP3([g, V, I, m³h, n⁴, m³hV, n⁴V])   (B,9)            -> ΔV (B,1)
    V <- clamp(V + dv, -100, 60)、g <- clamp(g + dg, 0, 1)
    各 MLP は 標準化 -> Linear -> act -> Linear -> act -> Linear -> 標準化の逆（行列積 3 回）で、計 9 回。

書き換え:
    1. 標準化を重みに畳み込む（行列積の外の割り算・引き算をなくす）
         入力:  W ((x - μ)/σ) + b = (W diag(1/σ)) x + (b - W (μ/σ))
         出力:  s ⊙ (W z + b) + m = (diag(s) W) z + (s ⊙ b + m)
    2. MLP1 と MLP3 を 1 つにまとめる。MLP1 は V だけ、MLP3 は更新前の g, V, I だけを使い、どちらも MLP2 の結果を
       待たないので同時に計算できる。入力はどちらも x = [m, h, n, V, I, m³h, n⁴, m³hV, n⁴V] (B,9) にそろえる
       （MLP1 の 1 層目の重みは V の列だけ値を持ち、他の列は 0）。
         1 層目: x (B,9) @ W1 (2,9,128)            -> (2,B,128)   （2 個の MLP をまとめた 1 回の積）
         2 層目: z (2,B,128) @ W2 (2,128,128)       -> (2,B,128)   （バッチ行列積 bmm 1 回）
         出力  : z (2,B,128) @ Wo (2,128,6)         -> (2,B,6)     （MLP3 の出力 1 個は 6 列に 0 で埋める）
    3. MLP2 は MLP1 の α, β を待つので、その後に行列積 3 回。
    → 1 ステップは「行列積 6 回 + 要素ごとの演算」（元は行列積 9 回）。バイアスは addmm / baddbmm で積と同時に足す。
    4. B 個のニューロン（電流）はバッチの行に並べて同時に回す。

行列積でない部分（要素ごと。行列積に直せない、または直すと遅い）:
    活性化関数（SiLU / ReLU）、積の特徴量 m³h, n⁴、α, β ≥ 0 の切り捨て、V と g の値域の切り捨て、状態の足し込み。

使い方:
    import m3_fused as mf
    f = mf.FusedM3.from_tag('m3_hdf_s0', dev, dtype)
    V, g = f.rest(B)
    V, g = f.step(V, g, I_k)
    確認・速度は hh_speed.py のモデル名 'm3f:m3_hdf_s0' で（check / bench / gamma）。
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_improve as hi   # noqa: E402

IMP = os.path.join(hi.hb.ROOT, 'results', 'hh_improve')
X_COLS = ['m', 'h', 'n', 'V', 'I', 'm3h', 'n4', 'm3hV', 'n4V']   # まとめた入力 x の列（MLP3 の expand と同じ順）
IV = X_COLS.index('V')


def _fold(net):
    """
    Net（cell='mlp'、2 層）の重みを、標準化を畳み込んだ (W, b) のリストにする（W は x @ W の向き = (入力, 出力)）。
    戻り値: [(W1 (d,H), b1 (H,)), (W2 (H,H), b2 (H,)), (Wo (H,d_out), bo (d_out,))]
    """
    assert net.cfg['cell'] == 'mlp' and net.cfg['layers'] == 2
    L1, L2, Lo = net.mlp[0], net.mlp[2], net.out
    mu, sd, om, os_ = net.im, net.is_, net.om, net.os_
    W1 = L1.weight / sd                         # (H, d): W diag(1/σ)
    b1 = L1.bias - W1 @ mu                      # b - W (μ/σ) = b - (W/σ) μ
    Wo = Lo.weight * os_[:, None]               # (d_out, H): diag(s) W
    bo = Lo.bias * os_ + om                     # s ⊙ b + m
    return [(W1.T, b1), (L2.weight.T, L2.bias), (Wo.T, bo)]


def _mm32(a, b):
    # fp16 (M,K) @ fp16 (K,N) -> fp32 (M,N)。cuBLAS の fp16 入力・fp32 和・fp32 出力（V100 では Tensor Core）。
    # CPU には fp16 入力・fp32 出力の mm がないので、同じ丸めを fp32 で再現する（fp16 どうしの積は fp32 で正確）
    if a.device.type != 'cuda':
        return a.float() @ b.float()
    return torch.mm(a, b, out_dtype=torch.float32)


def _bmm16(bias, x, W):
    # fp16 の (2,B,K) @ (2,K,N) + bias -> fp16。GPU は fp32 で足して最後に fp16 に丸める（cuBLAS の既定）。CPU は同じ丸めを再現
    if x.device.type != 'cuda':
        return torch.baddbmm(bias.float(), x.float(), W.float()).half()
    return torch.baddbmm(bias, x, W)


def _mm16(bias, x, W):
    if x.device.type != 'cuda':
        return torch.addmm(bias.float(), x.float(), W.float()).half()
    return torch.addmm(bias, x, W)


def _act16(f, z):
    # 活性化関数は fp32 で計算して fp16 に丸める（GPU の fp16 の SiLU と同じ丸め）
    return f(z.float()).half()


def _pad(t, dim, size):
    # t の dim 次元を size まで 0 で埋める（Tensor Core の 8・16 単位にそろえる。0 の行・列なので結果は変わらない）
    p = [0, 0] * (t.dim() - 1 - dim) + [0, size - t.shape[dim]]
    return F.pad(t, p)


class FusedM3Half:
    """
    FusedM3 の Tensor Core 版（V100: fp16 入力・fp32 和）。Codex との相談（2026-11-25）で決めた精度の分け方:
        fp32: 状態 V, m, h, n、電流 I、積の特徴量、入力の標準化 (x - μ)/σ、出力の標準化の逆、clamp、状態の足し込み
        fp16: 行列積の入力（標準化した入力・隠れ層の活性値）と重み。行列積の和は fp32
    標準化の平均はバイアスに畳み込まない。生の V（-65 mV 付近）を fp16 に丸めると刻みが 0.0625 mV になり、
    畳み込んだバイアスとの引き算で桁落ちするため。標準化してから fp16 に丸める（|u| ~ 1 なら刻みは約 0.001）。
    形（Tensor Core の単位にそろえる）: 入力 K = 1 / 9 -> 16、隠れ層 128、出力 N = 6 / 1 / 3 -> 8。
        段 A: MLP1 と MLP3 を (2, B, 16) にまとめ、bmm で 1 層目・2 層目。出力層は MLP ごとに mm（fp32 出力）
        段 B: MLP2 の 3 層
    """

    def __init__(self, n1, n2, n3):
        act = {n.cfg['act'] for n in (n1, n2, n3)}
        assert len(act) == 1
        self.act = {'silu': F.silu, 'relu': F.relu}[act.pop()]
        assert n1.cfg['feat'] == 0 and n2.cfg['feat'] == 0 and n3.cfg['feat'] == 3
        dev = n1.im.device
        f32 = dict(device=dev, dtype=torch.float32)

        def parts(net):
            L1, L2, Lo = net.mlp[0], net.mlp[2], net.out
            return dict(
                mu=net.im.to(**f32), inv=(1.0 / net.is_).to(**f32),                      # 入力の標準化（fp32）
                W1=_pad(L1.weight.T, 0, 16).half(), b1=L1.bias.half(),                   # (16, H)
                W2=L2.weight.T.contiguous().half(), b2=L2.bias.half(),                   # (H, H)
                Wo=_pad(Lo.weight.T, 1, 8).contiguous().half(),                           # (H, 8)
                bo=Lo.bias.to(**f32), os=net.os_.to(**f32), om=net.om.to(**f32), d_out=Lo.weight.shape[0])

        with torch.no_grad():
            p1, p2, p3 = parts(n1), parts(n2), parts(n3)
            self.W1 = torch.stack([p1['W1'], p3['W1']])                    # (2, 16, H)
            self.b1 = torch.stack([p1['b1'], p3['b1']])[:, None, :]        # (2, 1, H)
            self.W2 = torch.stack([p1['W2'], p3['W2']])                    # (2, H, H)
            self.b2 = torch.stack([p1['b2'], p3['b2']])[:, None, :]
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.dev, self.dtype = dev, torch.float32

    def rest(self, B):
        V = torch.full((B,), hi.H_.V_REST, device=self.dev, dtype=torch.float32)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=self.dev, dtype=torch.float32).repeat(B, 1)
        return V, g

    @staticmethod
    def _in16(x, p):
        # fp32 で標準化してから 0 埋めして fp16 に（(B, d) -> (B, 16)）
        return _pad((x - p['mu']) * p['inv'], 1, 16).half()

    @staticmethod
    def _out32(z, p):
        # 出力層: fp16 の積 -> fp32、標準化の逆も fp32（(B, H) -> (B, d_out)）
        y = _mm32(z, p['Wo'])[:, :p['d_out']]
        return (y + p['bo']) * p['os'] + p['om']

    def step(self, V, g, I_k):
        V, g, I_k = V.float(), g.float(), I_k.float()
        m, h, n = g[:, 0], g[:, 1], g[:, 2]
        a, b = m ** 3 * h, n ** 4
        x3 = torch.stack([m, h, n, V, I_k, a, b, a * V, b * V], 1)               # (B, 9) fp32
        x = torch.stack([self._in16(V[:, None], self.p1), self._in16(x3, self.p3)])   # (2, B, 16) fp16
        z = _act16(self.act, _bmm16(self.b1, x, self.W1))                        # (2, B, H) fp16
        z = _act16(self.act, _bmm16(self.b2, z, self.W2))
        ab = self._out32(z[0], self.p1).clamp(min=0.0)                           # (B, 6) fp32
        dv = self._out32(z[1], self.p3)[:, 0]                                    # (B,)
        p = self.p2
        u = self._in16(torch.cat([ab, g], 1), p)                                  # (B, 16) fp16
        u = _act16(self.act, _mm16(p['b1'], u, p['W1']))
        u = _act16(self.act, _mm16(p['b2'], u, p['W2']))
        dg = self._out32(u, p)                                                   # (B, 3) fp32
        return (V + dv).clamp(-100.0, 60.0), (g + dg).clamp(0.0, 1.0)


class FusedM3:
    def __init__(self, n1, n2, n3):
        act = {n.cfg['act'] for n in (n1, n2, n3)}
        assert len(act) == 1, '3 個の MLP の活性化関数がそろっていない'
        self.act = {'silu': F.silu, 'relu': F.relu}[act.pop()]
        assert n1.cfg['feat'] == 0 and n2.cfg['feat'] == 0 and n3.cfg['feat'] == 3
        assert n1.cfg['H'] == n3.cfg['H']
        dev, dt = n1.im.device, n1.im.dtype
        with torch.no_grad():
            (a1, c1), (a2, c2), (ao, co) = _fold(n1)            # MLP1: 入力 1 (V)、出力 6
            (e1, f1), (e2, f2), (eo, fo) = _fold(n3)            # MLP3: 入力 9 (x)、出力 1
            H = a1.shape[1]
            # 1 層目: MLP1 の重みを x の V の列に置き、他の列は 0
            A1 = torch.zeros(len(X_COLS), H, device=dev, dtype=dt)
            A1[IV] = a1[0]
            self.W1 = torch.stack([A1, e1])                      # (2, 9, H)
            self.b1 = torch.stack([c1, f1])[:, None, :]          # (2, 1, H)
            self.W2 = torch.stack([a2, e2])                      # (2, H, H)
            self.b2 = torch.stack([c2, f2])[:, None, :]          # (2, 1, H)
            Eo = torch.zeros(H, 6, device=dev, dtype=dt)
            Eo[:, 0] = eo[:, 0]                                  # MLP3 の出力 ΔV は 0 列目、残りは 0
            Fo = torch.zeros(6, device=dev, dtype=dt)
            Fo[0] = fo[0]
            self.Wo = torch.stack([ao, Eo])                      # (2, H, 6)
            self.bo = torch.stack([co, Fo])[:, None, :]          # (2, 1, 6)
            # MLP2（α, β, g -> Δg）
            self.m2 = [(W.contiguous(), b.contiguous()) for W, b in _fold(n2)]
        self.dev, self.dtype = dev, dt

    @classmethod
    def from_tag(cls, tag, dev, dtype):
        nets = [hi.load_net(os.path.join(IMP, f'{tag}_b{k}.pt')).to(device=dev, dtype=dtype).eval() for k in (1, 2, 3)]
        return cls(*nets)


    def rest(self, B):
        V = torch.full((B,), hi.H_.V_REST, device=self.dev, dtype=self.dtype)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=self.dev, dtype=self.dtype).repeat(B, 1)
        return V, g

    def step(self, V, g, I_k):
        """V: (B,)、g: (B,3) = [m, h, n]、I_k: (B,) -> V_next (B,), g_next (B,3)"""
        m, h, n = g[:, 0], g[:, 1], g[:, 2]
        a, b = m ** 3 * h, n ** 4                                            # Na・K のコンダクタンスの形
        x = torch.stack([m, h, n, V, I_k, a, b, a * V, b * V], 1)            # (B, 9)
        # MLP1 と MLP3 を同時に（行列積 3 回）
        z = self.act(torch.matmul(x, self.W1) + self.b1)                     # (2, B, H)
        z = self.act(torch.baddbmm(self.b2, z, self.W2))                     # (2, B, H)
        y = torch.baddbmm(self.bo, z, self.Wo)                               # (2, B, 6)
        ab, dv = y[0].clamp(min=0.0), y[1, :, 0]                             # α, β (B,6)、ΔV (B,)
        # MLP2（行列積 3 回）
        (W1, b1), (W2, b2), (Wo, bo) = self.m2
        u = self.act(torch.addmm(b1, torch.cat([ab, g], 1), W1))             # (B, H)
        u = self.act(torch.addmm(b2, u, W2))                                 # (B, H)
        dg = torch.addmm(bo, u, Wo)                                          # (B, 3)
        return (V + dv).clamp(-100.0, 60.0), (g + dg).clamp(0.0, 1.0)


def from_tag(tag, dev, dtype, mm='fp32'):
    """mm='fp32': FusedM3（dtype のまま）。mm='fp16': FusedM3Half（Tensor Core 版、状態は fp32）"""
    if mm == 'fp16':
        nets = [hi.load_net(os.path.join(IMP, f'{tag}_b{k}.pt')).to(device=dev, dtype=torch.float32).eval()
                for k in (1, 2, 3)]
        return FusedM3Half(*nets)
    assert mm == 'fp32', f'm3_fused は fp32 / fp16 だけ（{mm}）'
    return FusedM3.from_tag(tag, dev, dtype)


# ======================================================================================================
# 1 本のネットワークにした版（FusedM3One）: 1 ステップ = 「行列積 1 回 + 列ごとの活性化関数」を 6 層くり返すだけ
# ======================================================================================================
#
#   入力  u0 = (X - μ3) / σ3    X = [m, h, n, V, I, m³h, n⁴, m³hV, n⁴V]（MLP3 の標準化。fp32 の要素ごとの演算）
#   第 l 層  h_l = σ_l(h_{l-1} W_l + 1 b_l^T)   σ_l は列ごとに SiLU / 下限の切り捨て / 恒等 のどれか
#   出力  h_6 = [ΔV, Δm, Δh, Δn]
#
#   層   入力                           出力の列                                         活性化
#   1    u0 (9)                         [MLP1 (128) | MLP3 (128) | u_G (3)]       = 259    SiLU, SiLU, 恒等
#   2    259                            [MLP1 (128) | MLP3 (128) | u_G (3)]       = 259    SiLU, SiLU, 恒等
#   3    259                            [u_ab (6) | s_V (1) | u2_G (3)]           = 10     max(·, -μ2/σ2), 恒等
#   4    10                             [MLP2 (128) | s_V (1)]                    = 129    SiLU, 恒等
#   5    129                            [MLP2 (128) | s_V (1)]                    = 129    SiLU, 恒等
#   6    129                            [ΔV (1) | Δm, Δh, Δn (3)]                 = 4      恒等
#
#   u_G : MLP3 の標準化での m, h, n。単位行列のブロックで 3 層目まで運ぶ（4 層目の MLP2 が使う）
#   u_ab: MLP2 の標準化での α, β。元の max(0, ab) は、標準化した後では max(u_ab, -μ2/σ2) になる（下限の切り捨て）
#   u2_G: MLP2 の標準化での m, h, n（MLP3 の標準化からの 1 次変換を 3 層目の重みに入れる）
#   s_V : MLP3 の出力（標準化したままの ΔV）。単位行列で 6 層目まで運び、6 層目で ΔV = s_V os3 + om3 に戻す
#   重みはブロック対角（0 のブロックも計算するので、MLP を分けたときより演算量は大きい）
#   Tensor Core 用に、列の数を 16 の倍数（出力 4 は 8）に 0 で埋める: 9->16, 259->272, 10->16, 129->144, 4->8

def _ceil(n, k):
    return (n + k - 1) // k * k


class FusedM3One:
    def __init__(self, n1, n2, n3, mm='fp32'):
        act = {n.cfg['act'] for n in (n1, n2, n3)}
        assert len(act) == 1
        self.act = {'silu': F.silu, 'relu': F.relu}[act.pop()]
        assert n1.cfg['feat'] == 0 and n2.cfg['feat'] == 0 and n3.cfg['feat'] == 3
        self.mm = mm
        dev, dt = n1.im.device, n1.im.dtype
        H = n1.cfg['H']

        def z(r, c):
            return torch.zeros(r, c, device=dev, dtype=dt)

        def eye(k):
            return torch.eye(k, device=dev, dtype=dt)

        with torch.no_grad():
            mu3, sd3 = n3.im, n3.is_                 # (9,) X の標準化
            mu1, sd1 = n1.im[0], n1.is_[0]           # MLP1 の V の標準化
            mu2, sd2 = n2.im, n2.is_                 # (9,) MLP2 の入力 [ab (6), g (3)] の標準化
            P = {}
            # ---- 1 層目: u0 (9) -> 259
            W, b = z(9, 2 * H + 3), z(1, 2 * H + 3)[0]
            w1 = n1.mlp[0].weight[:, 0]
            # MLP1: u1 = (V - μ1)/σ1 = (σ3_V u_V + μ3_V - μ1)/σ1 -> 重み (σ3_V/σ1) w1、バイアス b1 + w1 (μ3_V - μ1)/σ1
            W[IV, :H] = w1 * sd3[IV] / sd1
            b[:H] = n1.mlp[0].bias + w1 * (mu3[IV] - mu1) / sd1
            W[:, H:2 * H] = n3.mlp[0].weight.T
            b[H:2 * H] = n3.mlp[0].bias
            W[0:3, 2 * H:] = eye(3)                  # u_G を運ぶ
            P[1] = (W, b)
            # ---- 2 層目: 259 -> 259（ブロック対角）
            W, b = z(2 * H + 3, 2 * H + 3), z(1, 2 * H + 3)[0]
            W[:H, :H] = n1.mlp[2].weight.T
            W[H:2 * H, H:2 * H] = n3.mlp[2].weight.T
            W[2 * H:, 2 * H:] = eye(3)
            b[:H], b[H:2 * H] = n1.mlp[2].bias, n3.mlp[2].bias
            P[2] = (W, b)
            # ---- 3 層目: 259 -> 10
            W, b = z(2 * H + 3, 10), z(1, 10)[0]
            # u_ab = (ab - μ2)/σ2、ab = (z Wo1^T + bo1) os1 + om1
            W[:H, 0:6] = n1.out.weight.T * (n1.os_ / sd2[:6])
            b[0:6] = (n1.out.bias * n1.os_ + n1.om - mu2[:6]) / sd2[:6]
            # s_V = z wo3 + bo3（標準化したままの ΔV）
            W[H:2 * H, 6] = n3.out.weight[0]
            b[6] = n3.out.bias[0]
            # u2_G = (σ3_G u_G + μ3_G - μ2_G)/σ2_G
            W[2 * H:, 7:10] = torch.diag(sd3[0:3] / sd2[6:9])
            b[7:10] = (mu3[0:3] - mu2[6:9]) / sd2[6:9]
            P[3] = (W, b)
            # ---- 4 層目: 10 -> 129（入力の並び [u_ab (6), s_V (1), u2_G (3)]）
            W, b = z(10, H + 1), z(1, H + 1)[0]
            W2a = n2.mlp[0].weight.T                 # (9, H): 行 0:6 が ab、6:9 が g
            W[0:6, :H], W[7:10, :H] = W2a[0:6], W2a[6:9]
            W[6, H] = 1.0                            # s_V を運ぶ
            b[:H] = n2.mlp[0].bias
            P[4] = (W, b)
            # ---- 5 層目: 129 -> 129
            W, b = z(H + 1, H + 1), z(1, H + 1)[0]
            W[:H, :H] = n2.mlp[2].weight.T
            W[H, H] = 1.0
            b[:H] = n2.mlp[2].bias
            P[5] = (W, b)
            # ---- 6 層目: 129 -> 4 = [ΔV, Δm, Δh, Δn]
            W, b = z(H + 1, 4), z(1, 4)[0]
            W[H, 0] = n3.os_[0]
            b[0] = n3.om[0]
            W[:H, 1:4] = n2.out.weight.T * n2.os_
            b[1:4] = n2.out.bias * n2.os_ + n2.om
            P[6] = (W, b)
            # 0 埋め（Tensor Core の単位）。埋めた行・列は 0 なので結果は変わらない。埋めた列の活性化は恒等
            K = {1: 16, 2: _ceil(2 * H + 3, 16), 3: _ceil(2 * H + 3, 16), 4: 16, 5: _ceil(H + 1, 16), 6: _ceil(H + 1, 16)}
            N = {1: K[2], 2: K[3], 3: 16, 4: K[5], 5: K[6], 6: 8}
            act_cols = {1: 2 * H, 2: 2 * H, 4: H, 5: H}    # 先頭からこの列数に SiLU（ReLU 版なら ReLU）、残りは恒等
            self.W, self.b, self.mask = {}, {}, {}
            for l in range(1, 7):
                W, b = P[l]
                Wp = z(K[l], N[l])
                Wp[:W.shape[0], :W.shape[1]] = W
                bp = z(1, N[l])[0]
                bp[:b.shape[0]] = b
                self.W[l], self.b[l] = Wp, bp
                if l in act_cols:
                    mk = torch.zeros(N[l], dtype=torch.bool, device=dev)
                    mk[:act_cols[l]] = True
                    self.mask[l] = mk
            # 3 層目の下限: α, β >= 0 は u_ab >= -μ2/σ2。他の列は下限なし
            lo = torch.full((N[3],), -float('inf'), device=dev, dtype=dt)
            lo[0:6] = -mu2[:6] / sd2[:6]
            self.lo3 = lo
            self.mu0, self.inv0 = mu3.clone(), 1.0 / sd3
            if mm == 'fp16':
                # 重みは fp16。中間層のバイアスも fp16（GPU では積と一緒に足す）。3・6 層目（出力が fp32）のバイアスは fp32
                for l in range(1, 7):
                    self.W[l] = self.W[l].half()
                    self.b[l] = self.b[l].float() if l in (3, 6) else self.b[l].half()
                self.lo3 = self.lo3.float()
                self.mu0, self.inv0 = self.mu0.float(), self.inv0.float()
        self.dev, self.dtype = dev, (torch.float32 if mm == 'fp16' else dt)
        self.K1 = K[1]

    def rest(self, B):
        V = torch.full((B,), hi.H_.V_REST, device=self.dev, dtype=self.dtype)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=self.dev, dtype=self.dtype).repeat(B, 1)
        return V, g

    def _hidden(self, l, h):
        # 中間層（1, 2, 4, 5 層目）: 積 1 回 + 列ごとの活性化
        if self.mm == 'fp16':
            z = _mm16(self.b[l], h, self.W[l])                                   # fp16
            return torch.where(self.mask[l], _act16(self.act, z), z)
        z = torch.addmm(self.b[l], h, self.W[l])
        return torch.where(self.mask[l], self.act(z), z)

    def _readout(self, l, h):
        # 3・6 層目: 積 1 回、出力は fp32（fp32 版では dtype のまま）
        if self.mm == 'fp16':
            return _mm32(h, self.W[l]) + self.b[l]
        return torch.addmm(self.b[l], h, self.W[l])

    def step(self, V, g, I_k):
        if self.mm == 'fp16':
            V, g, I_k = V.float(), g.float(), I_k.float()
        m, h, n = g[:, 0], g[:, 1], g[:, 2]
        a, b = m ** 3 * h, n ** 4
        X = torch.stack([m, h, n, V, I_k, a, b, a * V, b * V], 1)                 # (B, 9)
        u = _pad((X - self.mu0) * self.inv0, 1, self.K1)                         # (B, 16) 標準化（要素ごと）
        if self.mm == 'fp16':
            u = u.half()
        h1 = self._hidden(1, u)                                                  # (B, 272)
        h2 = self._hidden(2, h1)                                                 # (B, 272)
        h3 = torch.maximum(self._readout(3, h2), self.lo3)                       # (B, 16) 下限の切り捨て
        if self.mm == 'fp16':
            h3 = h3.half()
        h4 = self._hidden(4, h3)                                                 # (B, 144)
        h5 = self._hidden(5, h4)                                                 # (B, 144)
        d = self._readout(6, h5)                                                 # (B, 8) = [ΔV, Δm, Δh, Δn, 0...]
        return (V + d[:, 0]).clamp(-100.0, 60.0), (g + d[:, 1:4]).clamp(0.0, 1.0)


def from_tag_one(tag, dev, dtype, mm='fp32'):
    """1 本のネットワークにした版。mm='fp32' は dtype のまま、mm='fp16' は Tensor Core 版（状態は fp32）"""
    dt = torch.float32 if mm == 'fp16' else dtype
    nets = [hi.load_net(os.path.join(IMP, f'{tag}_b{k}.pt')).to(device=dev, dtype=dt).eval() for k in (1, 2, 3)]
    return FusedM3One(*nets, mm=mm)
