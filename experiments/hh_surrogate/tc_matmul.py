"""
代理モデルの行列積を Tensor Core（fp16 の行列積器）で計算するための切り替え。状態の更新（V + ΔV など）は fp32 のまま。

方式（mode）:
    fp32    : 今までどおり x @ W.T（fp32、CUDA コア）
    fp16    : 入力 x と重み W を fp16 に丸め、Tensor Core で積を取る。和と出力は fp32
              （torch.mm(..., out_dtype=torch.float32)。行列積の有効数字は約 3 桁）
    fp16x3  : fp16 を上位・残りに分けて 3 回の積で誤差を補正する（Markidis ら 2018）。有効数字は fp32 に近い
              x = x_h + x_l / S,  W = W_h + W_l / S   （x_h = fp16(x)、x_l = fp16((x - x_h) S)、S = 2^11）
              x W^T ≈ x_h W_h^T + (x_l W_h^T + x_h W_l^T) / S      （x_l W_l^T / S^2 は捨てる）
              残りを S 倍してから fp16 にするのは、残りが fp16 の非正規数（< 6e-5）に落ちて桁を失わないため。
              2 つ目の括弧は [x_l | x_h] @ [W_h ; W_l]^T の 1 回の行列積にまとめる（行列積は計 2 回）

使い方:
    import tc_matmul as tc
    tc.convert(net, 'fp16')   # hi.Net / NetR の全ての行列積（nn.Linear と RNN・GRU の W_ih, W_hh）を置き換える
"""
import types

import torch
import torch.nn as nn

S = 2.0 ** 11
MODES = ('fp32', 'fp16', 'fp16x3')


def _mm32(a, b):
    # fp16 (M, K) @ fp16 (K, N) -> fp32 (M, N)。cuBLAS の fp16 入力・fp32 和・fp32 出力（V100 では Tensor Core）
    if a.device.type != 'cuda':
        # CPU には fp16 入力・fp32 出力の mm がないので、同じ丸めを fp32 で再現する（fp16 どうしの積は fp32 で正確）。動作確認用
        return a.float() @ b.float()
    return torch.mm(a, b, out_dtype=torch.float32)


class TCMatmul(nn.Module):
    """y = x @ W.T (+ b) を mode の方式で計算する。W: (N, K) fp32。重みの fp16 版は作るときに 1 回だけ用意する。"""

    def __init__(self, W, b, mode):
        super().__init__()
        assert mode in ('fp16', 'fp16x3')
        self.mode = mode
        W = W.detach().float()
        Wh = W.half()
        self.register_buffer('Wh_t', Wh.t().contiguous())                                  # (K, N)
        if mode == 'fp16x3':
            Wl = ((W - Wh.float()) * S).half()
            self.register_buffer('WhWl_t', torch.cat([Wh, Wl], 1).t().contiguous())       # (2K, N)
        self.register_buffer('b', None if b is None else b.detach().float().clone())

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1]).float()
        xh = x.half()
        if self.mode == 'fp16':
            y = _mm32(xh, self.Wh_t)
        else:
            xl = ((x - xh.float()) * S).half()
            y = _mm32(xh, self.Wh_t) + _mm32(torch.cat([xl, xh], 1), self.WhWl_t) / S
        if self.b is not None:
            y = y + self.b
        return y.reshape(*shp[:-1], y.shape[-1])


def _tc_step(self, u, hs):
    """hi.Net._step と同じ式（rnn / gru）。u @ W_ih.T と h @ W_hh.T を TCMatmul で計算する。"""
    new = []
    for l in range(self.cfg['layers']):
        if self.cfg.get('cell', 'rnn') == 'gru':
            gi, gh = self._tc_ih[l](u), self._tc_hh[l](hs[l])
            ir, iz, i_n = gi.chunk(3, -1)
            hr, hz, h_n = gh.chunk(3, -1)
            r, zg = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            z = (1 - zg) * torch.tanh(i_n + r * h_n) + zg * hs[l]
        else:
            z = torch.tanh(self._tc_ih[l](u) + self._tc_hh[l](hs[l]))
            if self.cfg['leak'] > 0:
                z = hs[l] + self.cfg['leak'] * (z - hs[l])
        new.append(z)
        u = z
    return new


def _swap_linears(mod, mode):
    for name, ch in list(mod.named_children()):
        if isinstance(ch, nn.Linear):
            setattr(mod, name, TCMatmul(ch.weight, ch.bias, mode).to(ch.weight.device))
        else:
            _swap_linears(ch, mode)


@torch.no_grad()
def convert(net, mode):
    """net（hi.Net / hh_mlp_whh.NetR、fp32 で目的のデバイスに置いたもの）の行列積を mode の方式にする。fp32 なら何もしない。"""
    if mode == 'fp32':
        return net
    assert mode in MODES, mode
    cell = net.cfg.get('cell', 'rnn')
    if cell in ('rnn', 'gru'):
        # RNN・GRU は _step で weight_ih / weight_hh を直接掛けているので、その 2 つを TCMatmul に置き換えた _step にする。
        # tanh RNN のバイアスは b_ih + b_hh を ih 側にまとめる（GRU は n ゲートで b_hh を r 倍するので別々に持つ）
        ih, hh = nn.ModuleList(), nn.ModuleList()
        for l in range(net.cfg['layers']):
            Wi, Wh = getattr(net.rnn, f'weight_ih_l{l}'), getattr(net.rnn, f'weight_hh_l{l}')
            bi, bh = getattr(net.rnn, f'bias_ih_l{l}'), getattr(net.rnn, f'bias_hh_l{l}')
            if cell == 'gru':
                ih.append(TCMatmul(Wi, bi, mode)); hh.append(TCMatmul(Wh, bh, mode))
            else:
                ih.append(TCMatmul(Wi, bi + bh, mode)); hh.append(TCMatmul(Wh, None, mode))
        net._tc_ih, net._tc_hh = ih.to(Wi.device), hh.to(Wi.device)
        net._step = types.MethodType(_tc_step, net)
    # mlp / mlpr の各段と、読み出し（out, aux）の nn.Linear を置き換える（mlpr の _step は self.mlp[i] を呼ぶのでそのまま動く）
    _swap_linears(net, mode)
    return net
