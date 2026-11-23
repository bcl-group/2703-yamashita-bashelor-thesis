"""
保存済みの HH 代理モデルを、26 本のテスト電流で採点し直す（スパイク時刻の一致率に加えて coincidence factor Γ）。
結果は results/gamma/<タグ>.json（hb.score の出力そのまま: n100, mean, gamma, rows）。

使い方（リポジトリのルートで）:
    uv run python -X utf8 results/rescore_gamma.py --list            # タグの一覧
    uv run python -X utf8 results/rescore_gamma.py mm_rel k_3_w3     # 指定したタグだけ
    並列: uv run python -X utf8 results/rescore_gamma.py --list | xargs -P 6 -n 1 uv run python -X utf8 results/rescore_gamma.py
"""
import json
import os
import sys

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
import hh_bench as hb        # noqa: E402
import hh_improve as hi      # noqa: E402
import hh_serial as hs       # noqa: E402
import clamp_track as ct     # noqa: E402

OUT = os.path.join(ROOT, 'results', 'gamma')
IMP = os.path.join(ROOT, 'results', 'hh_improve')
SER = os.path.join(ROOT, 'results', 'hh_serial')
BASE = [os.path.join(ROOT, 'results', 'hh_modular_ou', f'block{k}.npz') for k in (1, 2, 3)]


def rnn3(b1=None, b2=None, b3=None):
    # 分割型（hh_improve.run3）。指定しないブロックは基準（hh_modular_ou）のまま
    paths = [b or BASE[i] for i, b in enumerate((b1, b2, b3))]
    nets = [hi.load_net(p) for p in paths]
    return lambda I: hi.run3(nets, I)


def imp(name):
    return os.path.join(IMP, name)


def serial(b1, b2, b3):
    nets = [hs.load_net(os.path.join(SER, t + '.pt')) for t in (b1, b2, b3)]
    return lambda I: hs.run(nets, I)


def single(tag):
    # RNN 1 個型・Markov MLP・式組込み MLP（入力が状態を含むかで閉ループの回し方を選ぶ）
    d = torch.load(imp(tag + '.pt'), weights_only=False, map_location='cpu')
    if d['cfg'].get('cell') == 'mlpr':
        # RNN 1 個（W_hh）: Markov MLP の 1 段目に z1(t-1) を戻す型（hh_mlp_whh.NetR）。回し方は Markov MLP と同じ
        import hh_mlp_whh as hw
        net = hw.NetR(**d['cfg'])
        net.load_state_dict(d['state'])
        net = net.double()
        return lambda I: hi.run_state(net, I)
    net = hi.Net(**d['cfg'])
    net.load_state_dict(d['state'])
    net = net.double()
    st = d['cfg'].get('phys') or d['cfg']['d_in'] == 5
    return lambda I: (hi.run_state if st else hi.run_single)(net, I)


MODELS = {
    # RNN 3 個（分割型）
    'hh_modular_ou': lambda: rnn3(),
    **{t: (lambda t=t: rnn3(b3=imp(t + '.pt'))) for t in
       ['a3_h64x2', 'a3_h128', 'a3_h64_feat', 'a3_h64', 'a3_h64_noise001', 'a3_h64_noise01', 'a3_h64_wsub10',
        'a3_h64_nomem']},
    'c123_lr1e-6': lambda: rnn3(*[imp(f'c123_lr1e-6_b{k}.pt') for k in (1, 2, 3)]),
    'k_3_w3': lambda: rnn3(b3=imp('k_3_w3_b3.pt')),
    'k_123_w3': lambda: rnn3(*[imp(f'k_123_w3_b{k}.pt') for k in (1, 2, 3)]),
    'k_123_w1': lambda: rnn3(*[imp(f'k_123_w1_b{k}.pt') for k in (1, 2, 3)]),
    'k_3_w3_K2000': lambda: rnn3(b3=imp('k_3_w3_K2000_b3.pt')),
    # 直列型（ホワイトボードのやつ）
    'e_s': lambda: serial('b1', 'b2', 'b3_s'),
    'e_s_i': lambda: serial('b1_i', 'b2', 'b3_s'),
    'e_s_g': lambda: serial('b1', 'b2', 'b3_sg'),
    'e_s_ig': lambda: serial('b1_i', 'b2', 'b3_sg'),
    'e_p': lambda: serial('b1', 'b2', 'b3_p'),
    'c_s': lambda: serial('c_s_b1', 'c_s_b2', 'c_s_b3'),
    'c_s_i': lambda: serial('c_s_i_b1', 'c_s_i_b2', 'c_s_i_b3'),
    'c_s_g': lambda: serial('c_s_g_b1', 'c_s_g_b2', 'c_s_g_b3'),
    'c_s_ig': lambda: serial('c_s_ig_b1', 'c_s_ig_b2', 'c_s_ig_b3'),
    'c_p': lambda: serial('c_p_b1', 'c_p_b2', 'c_p_b3'),
    # RNN 1 個、Markov MLP、式組込み MLP
    **{t: (lambda t=t: single(t)) for t in
       ['s_h64', 's_h64_aux', 's_h64_aux_leak', 's_h64_ro200', 's_mix_h64_aux_r', 's_mix_h128_aux_r',
        'cl_v_rnn', 'cl_v_gru', 'cl_s_rnn', 'cl_s_rnn_h128', 'cl_s_gru', 'cl_s_rnn_noonpol',
        'cl_m_mlp', 'mm_base', 'mm_rel', 'mm_sv05', 'mm_sv5', 'mm_w256', 'mm_w256_long',
        'mm_rel_s1', 'mm_rel_relu', 'mm_rel_relu_s1', 'mm_w256_relu', 'cl_p_mlp']},
}


def m3(tag):
    # MLP 3 個（hh_improve.py mlp3）: results/hh_improve/<tag>_b1〜3.pt
    nets = [hi.load_net(imp(f'{tag}_b{k}.pt')).double() for k in (1, 2, 3)]
    return lambda I: hi.run_m3(nets, I)


def ensemble(tags):
    """
    状態を戻す型（Markov MLP）のアンサンブル: 各ステップで、全モデルの出力（ΔV, Δm, Δh, Δn）を平均してから状態を進める。
    tags: 同じ入出力のモデルのタグのリスト
    """
    nets = []
    for t in tags:
        d = torch.load(imp(t + '.pt'), weights_only=False, map_location='cpu')
        n = hi.Net(**d['cfg'])
        n.load_state_dict(d['state'])
        nets.append(n.double())

    @torch.no_grad()
    def run(I_np):
        B, N = I_np.shape
        I = torch.from_numpy(I_np).to(hi.DEV)
        V = torch.full((B,), hi.H_.V_REST, dtype=torch.float64)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), dtype=torch.float64).repeat(B, 1)
        hs = [n.zeros(B, torch.float64) for n in nets]
        out = torch.empty(B, N - 1, dtype=torch.float64)
        for k in range(N - 1):
            x = torch.cat([V[:, None], g, I[:, k, None]], 1)
            ys = []
            for j, n in enumerate(nets):
                y, hs[j], _ = hi.state_step(n, x, hs[j])
                ys.append(y)
            y = torch.stack(ys).mean(0)
            V = ct.clamp(V + y[:, 0], -100.0, 60.0, "V")
            g = ct.clamp(g + y[:, 1:], 0.0, 1.0, "gate")
            out[:, k] = V
        return out.cpu().numpy()
    return run


def get_model(tag):
    """
    タグから閉ループの関数 run(I) を作る。MODELS にないタグは名前で判断する:
        ens:<タグ1>+<タグ2>+...   状態を戻す型のアンサンブル
        m3_...                     MLP 3 個（hh_improve.py mlp3）
        それ以外                   results/hh_improve/<タグ>.pt の単一ネット（RNN 1 個・Markov MLP・式組込み）
    """
    if tag in MODELS:
        return MODELS[tag]()
    if tag.startswith('ens:'):
        return ensemble(tag[4:].split('+'))
    if tag.startswith('m3_'):
        return m3(tag)
    if tag.startswith('rnn3:'):
        # RNN 3 個（hh_rnn3.py）: results/hh_rnn3/<tag>_b1〜3.pt。並列・直列は保存した info['conn'] で決める
        import hh_rnn3 as h3
        t = tag[5:]
        conn = torch.load(os.path.join(ROOT, 'results', 'hh_rnn3', f'{t}_b1.pt'), weights_only=False,
                          map_location='cpu')['info']['conn']
        nets = h3.load_nets(t)
        return lambda I: h3.run(nets, conn, I)
    return single(tag)


def main():
    args = sys.argv[1:]
    if args == ['--list']:
        print('\n'.join(MODELS))
        return
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    for tag in args or list(MODELS):
        hb.score(get_model(tag), tag, out_json=os.path.join(OUT, tag.replace(":", "_").replace("+", "_") + ".json"))


if __name__ == '__main__':
    main()
