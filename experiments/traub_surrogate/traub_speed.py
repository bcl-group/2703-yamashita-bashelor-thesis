"""
Traub（19 区画）の ODE と、代理モデルの形（乱数の重み）の速度を、同じ GPU・同じ PyTorch で比べる。
速度は重みの値に関係なく、ネットの形と同時に計算する細胞の数 B で決まるので、学習の前に測れる。

比べるもの:
    traub_euler : Traub の陽的 Euler（traub_torch.step、dt = 0.01 ms）。基準
    cell_W×L    : 細胞まるごとの Markov MLP。入力 210 = [V 19, χ 19, ゲート 171, I] -> 出力 209 の増分
    comp_W×L    : 区画ごとの Markov MLP（全区画で重みを共有）。行 = B×19、
                  入力 15 = [V, χ, ゲート 9, V_左, V_右, I_区画, 区画の番号の埋め込みの代わりの g_Na] -> 出力 11 の増分
                  （HH の Markov MLP をそのまま区画ごとに使う形。隣との結合は入力の V_左, V_右 で受け取る）
各モデルは行列積の方式（--mm fp32 / fp16 / fp16x3、tc_matmul）も変えて測る。

損益分岐の刻み幅: Δt_model > Δt_ODE × c_model / c_ODE（c は同じ B・同じ回し方の 1 ステップの時間）

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/traub_surrogate/traub_speed.py --device cpu --B 1 64 --steps 200 --tag smoke
    uv run python -X utf8 experiments/traub_surrogate/traub_speed.py --device cuda --runner eager cudagraph --tag gpu
結果: results/traub_speed/<tag>.json と <tag>.md（表）
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'hh_surrogate'))
import traub_torch as tt    # noqa: E402
import hh_speed as hs       # noqa: E402  （回し方・時間の測り方は HH と共通）
import tc_matmul as tcm     # noqa: E402

OUT = os.path.join(ROOT, 'results', 'traub_speed')
NC = tt.NC
D_STATE = NC * 11          # 209


def mlp(d_in, d_out, W, L, dev, dtype, seed=0):
    torch.manual_seed(seed)
    layers, d = [], d_in
    for _ in range(L):
        layers += [nn.Linear(d, W), nn.SiLU()]
        d = W
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers).to(device=dev, dtype=dtype).eval()


def with_mm(net, mm):
    if mm != 'fp32':
        tcm._swap_linears(net, mm)
    return net


def make_euler(dev, dtype):
    P = tt.Params(dev, dtype)
    return (lambda B: tt.init_state(B, P)), (lambda s, I: tt.step(s, I, P))


def make_cell(W, L, mm, dev, dtype):
    # 細胞まるごと: 状態 209 個と電流を 1 本のベクトルにして MLP に通し、増分を足す
    net = with_mm(mlp(D_STATE + 1, D_STATE, W, L, dev, dtype), mm)
    P = tt.Params(dev, dtype)

    def step(s, I):
        V, xi, G = s
        B = V.shape[0]
        x = torch.cat([V, xi * 1e-3, G.reshape(B, -1), I[:, None]], 1)
        y = net(x) * 1e-2
        V = (V + y[:, :NC]).clamp(-100.0, 60.0)
        xi = (xi + y[:, NC:2 * NC] * 1e3).clamp(min=0.0)
        G = (G + y[:, 2 * NC:].reshape(B, NC, 9)).clamp(0.0, 1.0)
        return V, xi, G
    return (lambda B: tt.init_state(B, P)), step


def make_comp(W, L, mm, dev, dtype):
    # 区画ごと（重みを共有）: 行を B×19 に並べ、HH の Markov MLP と同じ大きさの MLP を 1 回通す
    net = with_mm(mlp(15, 11, W, L, dev, dtype), mm)
    P = tt.Params(dev, dtype)
    gna = (P.gNa / 30.0)

    def step(s, I):
        V, xi, G = s
        B = V.shape[0]
        VL = torch.cat([V[:, :1], V[:, :-1]], 1)
        VR = torch.cat([V[:, 1:], V[:, -1:]], 1)
        Ic = I[:, None] * P.inj
        x = torch.cat([V[..., None], (xi * 1e-3)[..., None], G, VL[..., None], VR[..., None], Ic[..., None],
                       gna.expand(B, NC)[..., None]], -1).reshape(B * NC, 15)
        y = (net(x) * 1e-2).reshape(B, NC, 11)
        V = (V + y[..., 0]).clamp(-100.0, 60.0)
        xi = (xi + y[..., 1] * 1e3).clamp(min=0.0)
        G = (G + y[..., 2:]).clamp(0.0, 1.0)
        return V, xi, G
    return (lambda B: tt.init_state(B, P)), step


def make(name, mm, dev, dtype):
    if name == 'traub_euler':
        return make_euler(dev, dtype)
    kind, shape = name.split('_')
    W, L = map(int, shape.split('x'))
    return (make_cell if kind == 'cell' else make_comp)(W, L, mm, dev, dtype)


def gen_current(B, S, dev, dtype, seed=0):
    # 10 ms ごとに -0.2〜1.2 nA の一様乱数で切り替わる階段（教師データの電流の範囲）
    gen = torch.Generator(device=dev).manual_seed(seed)
    n_seg = S // 1000 + 1
    lv = -0.2 + 1.4 * torch.rand(B, n_seg, generator=gen, device=dev, dtype=dtype)
    return lv.repeat_interleave(1000, dim=1)[:, :S].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--models', nargs='+',
                    default=['traub_euler', 'comp_64x2', 'comp_128x2', 'cell_256x2', 'cell_512x2', 'cell_1024x3'])
    ap.add_argument('--mm', nargs='+', default=['fp32'], choices=['fp32', 'fp16', 'fp16x3'])
    ap.add_argument('--runner', nargs='+', default=['eager'], choices=['eager', 'cudagraph'])
    ap.add_argument('--B', nargs='+', type=int, default=[1, 32, 1024, 8192])
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--tag', default='smoke')
    a = ap.parse_args()
    a.compile = False
    dev = torch.device(a.device)
    dtype = torch.float32
    if a.threads:
        torch.set_num_threads(a.threads)
    assert a.steps % a.chunk == 0 and a.warmup % a.chunk == 0
    hs.gen_current = gen_current          # time_one が使う電流を Traub の範囲にする
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.tag + '.json')
    res = {'env': hs.env_info(dev), 'args': vars(a), 'dt_ms': 0.01, 'flops': {}, 'rows': []}
    for name in a.models:
        for mm in (['fp32'] if name == 'traub_euler' else a.mm):
            si = make(name, mm, dev, dtype)
            if mm == 'fp32':
                res['flops'][name] = hs.flops_per_neuron_step(si, dev, dtype)
            for runner in a.runner:
                if runner == 'cudagraph' and dev.type != 'cuda':
                    continue
                for B in a.B:
                    if dev.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    try:
                        r = hs.time_one(name, si, B, a, dev, dtype, runner, False)
                    except torch.OutOfMemoryError:
                        print(f'{name} {mm} B={B}: メモリ不足', flush=True)
                        torch.cuda.empty_cache()
                        continue
                    r['mm'] = mm
                    res['rows'].append(r)
                    print('{:12s} {:9s} {:6s} B={:6d}  1 ステップ {:10.2f} µs  {:12.2f} cell-s/s'.format(
                        name, runner, mm, B, r['per_step_s'] * 1e6, r['neuron_s_per_s']), flush=True)
                    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    base = {(r['B'], r['runner']): r['per_step_s'] for r in res['rows'] if r['model'] == 'traub_euler'}
    for r in res['rows']:
        c0 = base.get((r['B'], r['runner']))
        r['ratio_vs_ode'] = r['per_step_s'] / c0 if c0 else None
        r['breakeven_dt_ms'] = 0.01 * r['ratio_vs_ode'] if c0 else None
    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    write_md(res, path.replace('.json', '.md'))
    print('保存:', path)


def write_md(res, md):
    Bs = sorted({r['B'] for r in res['rows']})
    lines = [f"# Traub の速度（{res['env'].get('gpu', res['env'].get('cpu'))}、1 ステップの時間 [µs]、括弧は ODE 比）", '']
    for runner in sorted({r['runner'] for r in res['rows']}):
        lines += [f'## 回し方 {runner}', '', '| モデル | 行列積 | FLOP/細胞・ステップ | ' + ' | '.join(f'B={B}' for B in Bs) + ' |',
                  '|---|---|---|' + '---|' * len(Bs)]
        keys = []
        for r in res['rows']:
            if r['runner'] == runner and (r['model'], r['mm']) not in keys:
                keys.append((r['model'], r['mm']))
        for m, mm in keys:
            cells = []
            for B in Bs:
                r = next((r for r in res['rows'] if (r['model'], r['mm'], r['runner'], r['B']) == (m, mm, runner, B)), None)
                cells.append('—' if r is None else '{:.1f}'.format(r['per_step_s'] * 1e6)
                             + ('' if not r.get('ratio_vs_ode') else ' ({:.2f})'.format(r['ratio_vs_ode'])))
            lines.append(f"| {m} | {mm} | {res['flops'].get(m, 0):,.0f} | " + ' | '.join(cells) + ' |')
        lines.append('')
    open(md, 'w', encoding='utf-8').write('\n'.join(lines))


if __name__ == '__main__':
    main()
