"""
HH の代理モデル（Markov MLP と RNN 3 個（並列））と HH の陽的 Euler 法の速度を、同じ GPU・同じ PyTorch で比べる。
評価手法（計画）.md の 4 章（4.1、4.2）の「B を変えた実行時間」と「損益分岐の刻み幅」を測る。

全体の流れ:
    1. 各方式の 1 ステップを、状態のタプル → 状態のタプル の関数 step にする（式は hh_improve の run_state / run3 / hh_sim と同じ）。
    2. B 個のニューロンをバッチの行に並べ、S ステップ回す時間を測る（ウォームアップ後、GPU を同期して repeats 回）。
    3. 1 ステップの時間 c [s] と neuron-seconds/s（B × 模擬時間 ÷ 実時間）を出し、
       損益分岐の刻み幅  Δt_model > Δt_ODE × c_model / c_ODE  を B ごとに計算する。

実行の方式（--runner）:
    eager      : Python の for ループで 1 ステップずつカーネルを起動する（今の評価コードと同じ回し方）
    cudagraph  : chunk ステップ分のカーネル列を CUDA Graph に記録し、まとめて再生する（起動の手間を除く。GPU のみ）
    --compile を付けると step を torch.compile する（カーネルの融合。Linux + Triton が必要）

使い方（リポジトリのルートで）:
    uv run python -X utf8 experiments/hh_surrogate/hh_speed.py check                       # 既存の閉ループと一致するか（CPU 可）
    uv run python -X utf8 experiments/hh_surrogate/hh_speed.py bench --device cpu --B 1 64 --steps 200   # 動作確認
    uv run python -X utf8 experiments/hh_surrogate/hh_speed.py bench --device cuda --runner eager cudagraph --tag gpu
    uv run python -X utf8 experiments/hh_surrogate/hh_speed.py plot --tag gpu
結果: results/hh_speed/<tag>.json と <tag>.png
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hh_improve as hi   # noqa: E402  （hh_modular 経由で src も sys.path に入る）
import tc_matmul as tcm   # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
IMP = os.path.join(ROOT, 'results', 'hh_improve')
BASE = [os.path.join(ROOT, 'results', 'hh_modular_ou', f'block{k}.npz') for k in (1, 2, 3)]
OUT = os.path.join(ROOT, 'results', 'hh_speed')
DT = hi.DT   # 0.01 ms

# 比べるモデル（README の 26 本・追加テストの上位）。mm_sv05 は mm_rel と同じ形（128×2）なので速度は同じ
MODELS = {
    'hh_euler': 'HH 陽的 Euler（基準）',
    'mm_rel': 'Markov MLP 128×2（mm_rel）',
    'mm_w256': 'Markov MLP 256×3（mm_w256）',
    'k_3_w3': 'RNN 3 個・並列 H=50（k_3_w3）',
}
# 速度はネットの形だけで決まるので、形ごとに 1 本ずつ測る（--models all で全部）
ALL_MODELS = {
    **MODELS,
    'r8_mlp_hdf_s2': 'Markov MLP 128×2 + コンダクタンスの項（r8_mlp_hdf_s2）',
    'm3_v2f05d_s1': 'MLP 3 個 128×2（m3_v2f05d_s1）',
    'rnn3:p_h64_cl05': 'RNN 3 個・並列 H=64（p_h64_cl05）',
    'rnn3:s_h128_ft05': 'RNN 3 個・直列 H=128（s_h128_ft05）',
    'rnn3:s_gru64': 'RNN 3 個・直列 GRU 64（s_gru64）',
    'rnn1:sp_r_init': 'RNN 1 個（W_hh）128×2（sp_r_init）',
    'ens:mm_rel+mm_rel_s1': 'Markov MLP 2 本のアンサンブル',
}


# ---------------- 1 ステップの関数 ----------------
# どれも state（テンソルのタプル）と I_k (B,) を受け取り、次の state を返す。state[0] が V (B,)

MM = 'fp32'   # 行列積の方式（tc_matmul: fp32 / fp16 / fp16x3）。--mm で切り替える


def load(path, dev, dtype):
    return tcm.convert(hi.load_net(path).to(device=dev, dtype=dtype).eval(), MM)


def make_hh(dev, dtype):
    def init(B):
        V = torch.full((B,), hi.H_.V_REST, device=dev, dtype=dtype)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=dev, dtype=dtype).repeat(B, 1)
        return (V, g)

    def step(s, I_k):
        V, g = s
        return (V + hi.d_V(g, V, I_k), g + hi.d_gates(hi.rates(V), g))
    return init, step


def make_mlp(tag, dev, dtype):
    # hi.run_state と同じ: 入力 [V, m, h, n, I] -> [ΔV, Δm, Δh, Δn]、V と g は値域で切る
    net = load(os.path.join(IMP, tag + '.pt'), dev, dtype)

    def init(B):
        V = torch.full((B,), hi.H_.V_REST, device=dev, dtype=dtype)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=dev, dtype=dtype).repeat(B, 1)
        return (V, g)

    def step(s, I_k):
        V, g = s
        y, _ = net.cell(torch.cat([V[:, None], g, I_k[:, None]], 1), None)
        return ((V + y[:, 0]).clamp(-100.0, 60.0), (g + y[:, 1:]).clamp(0.0, 1.0))
    return init, step


def make_rnn3(b3, dev, dtype):
    # hi.run3 と同じ: RNN1（V -> α, β）→ RNN2（α, β, g -> Δg）、RNN3（g, V, I -> ΔV）は RNN1 と独立
    n1, n2, n3 = (load(p, dev, dtype) for p in (BASE[0], BASE[1], b3))
    H = [n.cfg['H'] for n in (n1, n2, n3)]

    def init(B):
        V = torch.full((B,), hi.H_.V_REST, device=dev, dtype=dtype)
        g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=dev, dtype=dtype).repeat(B, 1)
        return (V, g) + tuple(torch.zeros(B, h, device=dev, dtype=dtype) for h in H)

    def step(s, I_k):
        V, g, h1, h2, h3 = s
        ab, (h1,) = n1.cell(V[:, None], [h1])
        ab = ab.clamp(min=0.0)
        dg, (h2,) = n2.cell(torch.cat([ab, g], 1), [h2])
        dv, (h3,) = n3.cell(torch.cat([g, V[:, None], I_k[:, None]], 1), [h3])
        return (V + dv[:, 0], (g + dg).clamp(0.0, 1.0), h1, h2, h3)
    return init, step


def _rest(B, dev, dtype):
    V = torch.full((B,), hi.H_.V_REST, device=dev, dtype=dtype)
    g = torch.tensor(hi.H_.steady_state_gates(hi.H_.V_REST), device=dev, dtype=dtype).repeat(B, 1)
    return V, g


def make_m3(tag, dev, dtype):
    # MLP 3 個（hi.m3_step と同じ）: 各ブロックは再帰のない MLP。RNN3 には更新前の g を渡す
    n1, n2, n3 = (load(os.path.join(IMP, f'{tag}_b{k}.pt'), dev, dtype) for k in (1, 2, 3))

    def init(B):
        return _rest(B, dev, dtype)

    def step(s, I_k):
        V, g = s
        ab, _ = n1.cell(V[:, None], None)
        dg, _ = n2.cell(torch.cat([ab.clamp(min=0.0), g], 1), None)
        dv, _ = n3.cell(torch.cat([g, V[:, None], I_k[:, None]], 1), None)
        return ((V + dv[:, 0]).clamp(-100.0, 60.0), (g + dg).clamp(0.0, 1.0))
    return init, step


def make_rnn3_new(tag, dev, dtype):
    # RNN 3 個（hh_rnn3.step と同じ）。conn は保存した重みの info から読む（serial なら RNN3 に g(t+dt)）
    import hh_rnn3 as h3
    paths = [os.path.join(ROOT, 'results', 'hh_rnn3', f'{tag}_b{k}.pt') for k in (1, 2, 3)]
    conn = torch.load(paths[0], weights_only=False, map_location='cpu')['info'].get('conn', 'parallel')
    nets = [load(p, dev, dtype) for p in paths]
    nl = [len(n.zeros(1, dtype)) for n in nets]
    Hs = [n.cfg['H'] for n in nets]

    def init(B):
        return _rest(B, dev, dtype) + tuple(torch.zeros(B, h, device=dev, dtype=dtype)
                                            for h, n in zip(Hs, nl) for _ in range(n))

    def step(s, I_k):
        V, g, rest = s[0], s[1], list(s[2:])
        hs, i = [], 0
        for n in nl:
            hs.append(rest[i:i + n])
            i += n
        V, g, hs = h3.step(nets, conn, V, g, I_k, hs)
        return (V, g) + tuple(x for h in hs for x in h)
    return init, step


def make_rnn1(tag, dev, dtype):
    # RNN 1 個（hh_mlp_whh.NetR、cell='mlpr'）: Markov MLP の 1 段目に前の時刻の z1 を戻す。重みがなければ同じ形の乱数で測る
    import hh_mlp_whh as hw
    path = os.path.join(IMP, tag + '.pt')
    if os.path.exists(path):
        d = torch.load(path, weights_only=False, map_location='cpu')
        net = hw.NetR(**d['cfg'])
        net.load_state_dict(d['state'])
    else:
        print(f'{tag}.pt がないので、同じ形（mlpr 128×2）の乱数の重みで測る', flush=True)
        net = hw.new_net('mlpr')
    net = tcm.convert(net.to(device=dev, dtype=dtype).eval(), MM)
    H = net.cfg['H']

    def init(B):
        return _rest(B, dev, dtype) + (torch.zeros(B, H, device=dev, dtype=dtype),
                                       torch.zeros(B, H, device=dev, dtype=dtype))

    def step(s, I_k):
        V, g, z1, z2 = s
        y, (z1, z2) = net.cell(torch.cat([V[:, None], g, I_k[:, None]], 1), [z1, z2])
        return ((V + y[:, 0]).clamp(-100.0, 60.0), (g + y[:, 1:]).clamp(0.0, 1.0), z1, z2)
    return init, step


def make_ens(tags, dev, dtype):
    # Markov MLP のアンサンブル: 毎ステップ、全モデルの出力を平均してから進める
    nets = [load(os.path.join(IMP, t + '.pt'), dev, dtype) for t in tags]

    def init(B):
        return _rest(B, dev, dtype)

    def step(s, I_k):
        V, g = s
        x = torch.cat([V[:, None], g, I_k[:, None]], 1)
        y = torch.stack([n.cell(x, None)[0] for n in nets]).mean(0)
        return ((V + y[:, 0]).clamp(-100.0, 60.0), (g + y[:, 1:]).clamp(0.0, 1.0))
    return init, step


def make(name, dev, dtype):
    if name == 'hh_euler':
        return make_hh(dev, dtype)
    if name == 'k_3_w3':
        return make_rnn3(os.path.join(IMP, 'k_3_w3_b3.pt'), dev, dtype)
    if name.startswith('m3_'):
        return make_m3(name, dev, dtype)
    if name.startswith('rnn3:'):
        return make_rnn3_new(name[5:], dev, dtype)
    if name.startswith('rnn1:'):
        return make_rnn1(name[5:], dev, dtype)
    if name.startswith('ens:'):
        return make_ens(name[4:].split('+'), dev, dtype)
    return make_mlp(name, dev, dtype)


# ---------------- 回し方 ----------------

def gen_current(B, S, dev, dtype, seed=0):
    # 教師と同じ形の電流: 10 ms（1000 ステップ）ごとに 0〜20 µA/cm² の一様乱数で切り替わる階段。(B, S)
    gen = torch.Generator(device=dev).manual_seed(seed)
    n_seg = S // 1000 + 1
    lv = 20.0 * torch.rand(B, n_seg, generator=gen, device=dev, dtype=dtype)
    return lv.repeat_interleave(1000, dim=1)[:, :S].contiguous()


def run_eager(step, s, I, record):
    # I: (B, S)。record=True なら V を GPU 上の (B, S) に書く（転送はしない）
    out = torch.empty_like(I) if record else None
    for k in range(I.shape[1]):
        s = step(s, I[:, k])
        if record:
            out[:, k] = s[0]
    return s, out


class GraphRunner:
    """chunk ステップを 1 つの CUDA Graph に記録し、再生で回す。状態と電流は固定のバッファを使い回す。"""

    def __init__(self, step, s0, B, chunk, record, dev, dtype):
        self.chunk, self.record = chunk, record
        self.s = tuple(x.clone() for x in s0)
        self.I = torch.zeros(B, chunk, device=dev, dtype=dtype)
        self.out = torch.zeros(B, chunk, device=dev, dtype=dtype) if record else None

        def body():
            s = self.s
            for k in range(chunk):
                s = step(s, self.I[:, k])
                if record:
                    self.out[:, k] = s[0]
            for dst, src in zip(self.s, s):
                dst.copy_(src)

        # 記録の前に別ストリームで数回回す（PyTorch の CUDA Graph の手順）
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            body()

    def run(self, I):
        # I: (B, S)、S は chunk の倍数
        for j in range(0, I.shape[1], self.chunk):
            self.I.copy_(I[:, j:j + self.chunk])
            self.g.replay()
            # 記録込みのときも、ここでは GPU 上のバッファに書くだけ（転送は測らない）


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def time_one(name, step_init, B, a, dev, dtype, runner, record):
    init, step = step_init
    S = a.steps
    I = gen_current(B, S + a.warmup, dev, dtype)
    I_w, I_m = I[:, :a.warmup], I[:, a.warmup:]
    times = []
    with torch.no_grad():
        if runner == 'cudagraph':
            gr = GraphRunner(step, init(B), B, a.chunk, record, dev, dtype)
            gr.run(I_w)   # ウォームアップ（静止状態から少し進めた状態で測る）
            sync(dev)
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                gr.run(I_m)
                sync(dev)
                times.append(time.perf_counter() - t0)
        else:
            s, _ = run_eager(step, init(B), I_w, record)
            sync(dev)
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                run_eager(step, s, I_m, record)
                sync(dev)
                times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    sim_s = S * DT / 1000.0   # 模擬時間 [s]
    return {'model': name, 'B': B, 'runner': runner + ('+compile' if a.compile else ''), 'record': record,
            'steps': S, 'times': times, 'median_s': t,
            'per_step_s': t / S,
            'neuron_s_per_s': B * sim_s / t,
            'mem_MB': (torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == 'cuda' else None}


def flops_per_neuron_step(step_init, dev, dtype):
    # 行列積の FLOP（FlopCounterMode は要素ごとの演算を数えない。HH は 0 と出る）
    from torch.utils.flop_counter import FlopCounterMode
    init, step = step_init
    B = 64
    s = init(B)
    I_k = torch.full((B,), 10.0, device=dev, dtype=dtype)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        step(s, I_k)
    return fc.get_total_flops() / B


def env_info(dev):
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'os': platform.platform(),
            'cpu': platform.processor(), 'threads': torch.get_num_threads()}
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(dev)
        info.update(gpu=p.name, gpu_mem_GB=round(p.total_memory / 2 ** 30, 1), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return info


def bench(a):
    dev = torch.device(a.device)
    dtype = {'fp32': torch.float32, 'fp64': torch.float64}[a.dtype]
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.tag + '.json')
    global MM
    res = {'env': env_info(dev), 'args': vars(a), 'dt_ms': DT, 'flops': {}, 'rows': []}
    for name, mm in [(n, m) for n in a.models for m in (['fp32'] if n == 'hh_euler' else a.mm)]:
        MM = mm
        si = make(name, dev, dtype)
        if mm == 'fp32' or name not in res['flops']:
            res['flops'][name] = flops_per_neuron_step(si, dev, dtype)
        if a.compile:
            si = (si[0], torch.compile(si[1]))
        for runner in a.runner:
            if runner == 'cudagraph' and dev.type != 'cuda':
                continue
            for B in a.B:
                for record in ([False, True] if a.record else [False]):
                    if dev.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    try:
                        r = time_one(name, si, B, a, dev, dtype, runner, record)
                        r['mm'] = mm
                    except torch.OutOfMemoryError:
                        print(f'{name:9s} {mm} B={B:6d}: メモリ不足', flush=True)
                        torch.cuda.empty_cache()
                        continue
                    res['rows'].append(r)
                    print('{:9s} {:16s} {:6s} rec={:d} B={:6d}  1 ステップ {:9.2f} µs  {:12.1f} neuron-s/s'.format(
                        name, r['runner'], mm, record, B, r['per_step_s'] * 1e6, r['neuron_s_per_s']), flush=True)
                    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    add_breakeven(res)
    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('保存:', path)
    plot(res, os.path.join(OUT, a.tag + '.png'))


def add_breakeven(res):
    """
    損益分岐の刻み幅（評価手法（計画）.md 4.2 節）:
        t_ODE = (T/Δt_ODE) c_ODE、t_model = (T/Δt_model) c_model、t_model < t_ODE
        ⇔ Δt_model > Δt_ODE × c_model / c_ODE
    c は同じ B・同じ回し方・同じ記録の有無で測った 1 ステップの時間。Δt_ODE = 0.01 ms（今の教師と同じ）。
    """
    base = {(r['B'], r['runner'], r['record']): r['per_step_s'] for r in res['rows'] if r['model'] == 'hh_euler'}
    for r in res['rows']:
        c0 = base.get((r['B'], r['runner'], r['record']))
        r['ratio_vs_hh'] = r['per_step_s'] / c0 if c0 else None
        r['breakeven_dt_ms'] = DT * r['ratio_vs_hh'] if c0 else None


def plot(res, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # 見つからない候補フォントの警告を消す
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'IPAexGothic', 'Noto Sans CJK JP', 'sans-serif']
    rows = [r for r in res['rows'] if not r['record'] and r.get('mm', 'fp32') == 'fp32']
    runners = sorted({r['runner'] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = dict(zip(MODELS, ['k', 'tab:blue', 'tab:cyan', 'tab:red']))
    styles = dict(zip(runners, ['-', '--', ':', '-.']))
    for name in MODELS:
        for rn in runners:
            rs = sorted([r for r in rows if r['model'] == name and r['runner'] == rn], key=lambda r: r['B'])
            if not rs:
                continue
            Bs = [r['B'] for r in rs]
            lab = f'{ALL_MODELS.get(name, name)} [{rn}]'
            kw = dict(color=colors.get(name), ls=styles[rn], marker='o', ms=3)
            axes[0].loglog(Bs, [r['median_s'] for r in rs], label=lab, **kw)
            axes[1].loglog(Bs, [r['neuron_s_per_s'] for r in rs], label=lab, **kw)
            if name != 'hh_euler' and rs[0].get('breakeven_dt_ms'):
                axes[2].loglog(Bs, [r['breakeven_dt_ms'] for r in rs], label=lab, **kw)
    S = rows[0]['steps'] if rows else 0
    axes[0].set(xlabel='バッチ数 B（同時に回すニューロン数）', ylabel=f'実行時間 [s]（{S} ステップ = {S * DT:.0f} ms、中央値）',
                title='(a) B と実行時間')
    axes[1].set(xlabel='バッチ数 B', ylabel='neuron-seconds/s（大きいほど速い）', title='(b) スループット')
    axes[2].axhline(DT, color='k', lw=0.8)
    axes[2].set(xlabel='バッチ数 B', ylabel='損益分岐の刻み幅 Δt [ms]',
                title='(c) HH（Δt = 0.01 ms）に追いつくのに必要な刻み幅')
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=7)
    env = res['env']
    fig.suptitle('HH 代理モデルの速度（{}、{}、torch {}）'.format(env.get('gpu', 'CPU'), res['args']['dtype'], env['torch']))
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print('図:', png)


# ---------------- 一致の確認 ----------------

def check(a):
    """新しい step 関数（fp64、CPU）が既存の閉ループ（hi.run_state / hi.run3 / hi.hh_sim）と同じ V を出すか。"""
    dev, dtype = torch.device('cpu'), torch.float64
    t = np.arange(0, 50.0, DT)
    I_np = np.stack([np.full_like(t, 10.0), np.full_like(t, 7.0)])
    I = torch.from_numpy(I_np)
    for name in a.models:
        init, step = make(name, dev, dtype)
        s = init(2)
        with torch.no_grad():
            # run_state の最初の 20 ms の前走行は隠れ状態だけを進め、V, g は更新しない（MLP では何もしない）ので省く
            _, out = run_eager(step, s, I[:, :-1], True)
            if name == 'hh_euler':
                ref = hi.hh_sim(I[:, :-1])[0][:, 1:].numpy()
            elif name == 'k_3_w3':
                ref = hi.run3([hi.load_net(p) for p in (BASE[0], BASE[1], os.path.join(IMP, 'k_3_w3_b3.pt'))], I_np)
            else:
                ref = hi.run_state(hi.load_net(os.path.join(IMP, name + '.pt')), I_np)
        err = np.abs(out.numpy() - ref).max()
        print(f'{name:9s} 既存の閉ループとの最大差 {err:.2e} mV（50 ms、const 10 / const 7）')


def gamma(a):
    """
    行列積の方式（--mm）ごとに、26 本のテスト電流で閉ループを回して Γ を採点する（hh_bench.score と同じ採点）。
    fp64 は行列積も状態も fp64（今までの CPU の採点と同じ精度。GPU で回すときの基準）。
    結果: results/hh_speed/gamma_<tag>.json（{mm: {model: {n100, mean, gamma, sec}}}）と、gamma_<tag>/ に電流ごとの rows
    """
    import hh_bench as hb
    global MM
    dev = torch.device(a.device)
    path = os.path.join(OUT, f'gamma_{a.tag}.json')
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    res['env'] = env_info(dev)
    for mm in a.mm:
        dtype = torch.float64 if mm == 'fp64' else torch.float32
        MM = 'fp32' if mm == 'fp64' else mm
        for name in a.models:
            if name == 'hh_euler':
                continue
            if name.startswith('rnn1:') and not os.path.exists(os.path.join(IMP, name[5:] + '.pt')):
                print(f'{name}: 重みがないので精度は測らない（乱数の重みでは意味がない）', flush=True)
                continue
            init, step = make(name, dev, dtype)

            @torch.no_grad()
            def run(I_np, init=init, step=step, dtype=dtype):
                I = torch.as_tensor(I_np).to(device=dev, dtype=dtype)
                _, out = run_eager(step, init(I.shape[0]), I[:, :-1], True)
                return out.double().cpu().numpy()
            fn = '{}_{}.json'.format(name.replace(':', '_').replace('+', '_'), mm)
            r = hb.score(run, f'{name} [{mm}]', out_json=os.path.join(OUT, f'gamma_{a.tag}', fn))
            res.setdefault(mm, {})[name] = {k: r[k] for k in ('n100', 'mean', 'gamma', 'sec')}
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('保存:', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'bench', 'plot', 'gamma'])
    ap.add_argument('--mm', nargs='+', default=['fp32'], choices=['fp64', 'fp32', 'fp16', 'fp16x3'],
                    help='行列積の方式（tc_matmul）。bench は fp32 / fp16 / fp16x3、gamma は fp64 も可')
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', choices=['fp32', 'fp64'], default='fp32')
    ap.add_argument('--tf32', action='store_true', help='行列積に TF32 を許す（Tensor Core）')
    ap.add_argument('--B', type=int, nargs='+', default=[1, 8, 32, 128, 512, 2048, 8192, 32768, 131072])
    ap.add_argument('--steps', type=int, default=2000, help='測るステップ数（2000 = 20 ms）')
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=100, help='CUDA Graph 1 個に入れるステップ数')
    ap.add_argument('--runner', nargs='+', choices=['eager', 'cudagraph'], default=['eager'])
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--record', action='store_true', help='V を毎ステップ記録する場合も測る')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--tag', default='run')
    a = ap.parse_args()
    if a.models == ['all']:
        a.models = list(ALL_MODELS)
    if a.cmd == 'check':
        check(a)
    elif a.cmd == 'gamma':
        gamma(a)
    elif a.cmd == 'bench':
        assert a.steps % a.chunk == 0 and a.warmup % a.chunk == 0
        assert 'fp64' not in a.mm, 'bench の --mm は fp32 / fp16 / fp16x3（fp64 は --dtype fp64 で）'
        bench(a)
    else:
        res = json.load(open(os.path.join(OUT, a.tag + '.json'), encoding='utf-8'))
        plot(res, os.path.join(OUT, a.tag + '.png'))


if __name__ == '__main__':
    main()
