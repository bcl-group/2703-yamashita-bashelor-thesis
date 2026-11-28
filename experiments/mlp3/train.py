"""
MLP 3 個（C）の学習。状態 [V, m, h, n, I] の集まりから 3 つのブロックを同時に学習する（教師強制）。
どのブロックの正解も「その時刻の入力の厳密な関数」なので、状態に雑音を足しても正解を HH の式で計算し直せる。

    損失 = L1 + L2 + L3
        L1 = mean(((Block1(V) - αβ) / σ_1)^2)                         σ: 正解の標準偏差（標準化）
        L2 = mean(((Block2(αβ, g) - Δg) / σ_2)^2)                     入力の αβ は式の値
        L3 = Huber((Block3(g, V, I) - ΔV) / max(|ΔV|, pre_rel))       相対誤差（pre_rel = 0 なら L1, L2 と同じ形）

学習の流れ:
    1. 状態を集める: 通常の電流 bank 本 × ep_ms の HH の軌道 + 苦手な電流 hard 本 × hard_ms（hard_stride ごと）
    2. 事前学習 pre_iters 回（ミニバッチの半分の状態に V ± pre_sv mV、ゲート ± pre_sg の雑音を足す）
       hard_frac > 0 なら、ミニバッチのその割合を苦手な電流の状態から取る
    3. DAgger を dagger 回: モデルを閉ループで回して通った状態を足し、dagger_iters 回ずつ学習を続ける
    4. 各段階の始めと最後に検証 32 本で採点し、(100% 一致の本数, 平均一致率) が最良の重みを残す
    学習率は全体で余弦減衰（lr → lr/100、最初の 2% は線形に上げる）

保存するもの（results/mlp3/<tag>/。学習が最後まで進むと必ずそろう）:
    config.json                         引数
    train.log                           画面に出した記録
    model_b1.pt, model_b2.pt, model_b3.pt   検証で最良の重み（旧 hh_improve.py と同じ形式）
    loss.csv, val_history.csv, loss.png 損失の推移（100 回ごとの平均）と、検証の採点の推移
    responses_val.npz, responses_test.npz   検証 32 本・テスト 26 本の V・ゲート・速度定数の応答（outputs.py）
    score_val.json, score_test.json     電流ごとの採点
    figs/val/*.png, figs/test/*.png     電流ごとの図（V、ゲート、速度定数）
    metrics.json                        要約（最良の時点、検証・テストの成績、時間）

使い方（リポジトリのルートで）。下は m3_hdf と同じ条件（GPU で約 1.5 時間）:
    uv run python -X utf8 experiments/mlp3/train.py --tag m3_hdf_s0 --seed 0 --device cuda
動作確認（CPU で数分）:
    uv run python -X utf8 experiments/mlp3/train.py --tag smoke --bank 16 --hard 4 --hard-ms 200 \\
        --pre-iters 300 --dagger 1 --dagger-iters 200 --dagger-n 8 --dagger-ms 100
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import currents         # noqa: E402
import eval_sets as es  # noqa: E402
import hh               # noqa: E402
import outputs          # noqa: E402
from evaluate import evaluate   # noqa: E402
from model import MLP3          # noqa: E402

OUT_ROOT = os.path.join(es.ROOT, 'results', 'mlp3')
LOSS_EVERY = 100        # 損失をこの回数ごとに平均して記録する
PRINT_EVERY = 500


class Tee:
    # print の出力を画面とファイルの両方に書く
    def __init__(self, path):
        self.f, self.out = open(path, 'a', encoding='utf-8'), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def lr_at(it, iters, lr):
    w = max(1, iters // 50)
    if it < w:
        return lr * (it + 1) / w
    return lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - w) / max(1, iters - w))))


def collect_states(a, rng, xdt):
    """
    1. の状態 X (n, 5) = [V, m, h, n, I] と、そのうち苦手な電流の状態の範囲 [lo, hi)（なければ None）。
    乱数は 通常の電流 -> 苦手な電流（256 本ずつ）の順に引く
    """
    t0 = time.time()
    I = currents.normal(a.bank, int(a.ep_ms / hh.DT), rng).to(torch.get_default_device())
    V, G = hh.simulate(I)
    X = torch.cat([V[:, :-1, None], G[:, :-1], I[..., None]], -1).reshape(-1, 5).to(xdt)
    del V, G
    if a.hard > 0:
        Nh = int(a.hard_ms / hh.DT)
        Xh = []
        for c0 in range(0, a.hard, 256):
            Ih = currents.hard(min(256, a.hard - c0), Nh, rng, a.hard_set).to(torch.get_default_device())
            Xh.append(hh.states(Ih, a.hard_stride).to(xdt))
        Xh = torch.cat(Xh)
        print('苦手な電流の状態: {} 本 × {:.0f} ms → {} 個（通常 {} 個）  ({:.0f} s)'.format(
            a.hard, a.hard_ms, Xh.shape[0], X.shape[0], time.time() - t0), flush=True)
        X = torch.cat([X, Xh])
        return X, (X.shape[0] - Xh.shape[0], X.shape[0])
    return X, None


def set_scalers(model, X, rng):
    # 標準化の平均・標準偏差を、状態 20 万個とその正解から決める
    s = X[torch.from_numpy(rng.integers(0, X.shape[0], size=200000))].double()
    V, g, I = s[:, 0], s[:, 1:4], s[:, 4]
    ab = hh.rates(V)
    model.b1.set_scalers(V[:, None].float(), ab.float())
    model.b2.set_scalers(torch.cat([ab, g], 1).float(), hh.d_gates(ab, g).float())
    model.b3.set_scalers(torch.cat([g, V[:, None], I[:, None]], 1).float(), hh.d_V(g, V, I)[:, None].float())


def losses(model, x, a):
    # x: (B, 5) float64 の状態（雑音を足した後）-> (L1, L2, L3)。正解は float64 で式から計算する
    V, g, I = x[:, 0], x[:, 1:4], x[:, 4]
    ab = hh.rates(V)
    y1 = model.b1(V[:, None].float())
    y2 = model.b2(torch.cat([ab, g], 1).float())
    y3 = model.b3(torch.cat([g, V[:, None], I[:, None]], 1).float())
    l1 = (((y1 - ab.float()) / model.b1.os_) ** 2).mean()
    l2 = (((y2 - hh.d_gates(ab, g).float()) / model.b2.os_) ** 2).mean()
    t3 = hh.d_V(g, V, I).float()
    if a.pre_rel > 0:
        e3 = (y3[:, 0] - t3) / t3.abs().clamp(min=a.pre_rel)
        l3 = nn.functional.huber_loss(e3, torch.zeros_like(e3), delta=1.0)
    else:
        l3 = (((y3[:, 0] - t3) / model.b3.os_[0]) ** 2).mean()
    return l1, l2, l3


def sample_idx(n_all, a, rng, hard_rng):
    # ミニバッチの添字 (B,)。hard_frac > 0 なら B × hard_frac 個を苦手な電流の状態から、残りを全体から一様に取る
    B = a.pre_batch
    if hard_rng is None or a.hard_frac <= 0:
        return torch.from_numpy(rng.integers(0, n_all, size=B))
    nh = int(round(B * a.hard_frac))
    return torch.from_numpy(np.concatenate([rng.integers(hard_rng[0], hard_rng[1], size=nh),
                                            rng.integers(0, n_all, size=B - nh)]))


def perturb(x, a):
    # ミニバッチの半分の状態に雑音を足す（閉ループでずれた状態でも正しい 1 ステップを学ぶため）
    B = x.shape[0]
    mask = (torch.rand(B, 1, dtype=x.dtype) < 0.5).to(x.dtype)
    x[:, :1] += mask * a.pre_sv * torch.randn(B, 1, dtype=x.dtype)
    x[:, 1:4] = (x[:, 1:4] + mask * a.pre_sg * torch.randn(B, 3, dtype=x.dtype)).clamp(0, 1)
    return x


def train(a, run_dir):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    xdt = torch.float32 if a.x32 else torch.float64
    X, hard_rng = collect_states(a, rng, xdt)
    model = MLP3(a.H, a.layers, a.act, a.sfeat)
    set_scalers(model, X, rng)
    model = model.float()
    opt = torch.optim.Adam(model.parameters(), lr=a.pre_lr)
    val_ref = es.reference('val')
    prefix = os.path.join(run_dir, 'model')
    loss_rows, val_rows, dagger_its = [], [], []
    best = (-1, -1.0, 0)

    def validate(it):
        # float32 のまま検証 32 本で採点し、最良なら保存する。損失の記録もここで書き出す（途中で止まっても残る）
        nonlocal best
        r = es.score(val_ref, model.run(val_ref.I), label='{} it {}'.format(a.tag, it))
        val_rows.append((it, r['n100'], r['mean'], r['gamma']))
        if (r['n100'], r['mean']) > best[:2]:
            best = (r['n100'], r['mean'], it)
            model.save(prefix, {'it': it, 'm3': True})
        outputs.save_loss(run_dir, loss_rows, val_rows, dagger_its)

    total = a.pre_iters + a.dagger * a.dagger_iters
    stages = [(0, a.pre_iters)] + [(r + 1, a.dagger_iters) for r in range(a.dagger)]
    it = 0
    acc = 0.0           # LOSS_EVERY 回分の (L1, L2, L3) の和（GPU のまま足し、記録するときだけ CPU に移す）
    for rnd, n_it in stages:
        if rnd > 0:
            Nd = int(a.dagger_ms / hh.DT)
            nd = a.dagger_n
            Id = torch.cat([currents.normal(nd // 2, Nd, rng), currents.hard(nd - nd // 2, Nd, rng, a.hard_set)])
            Xd = model.visited_states(Id.to(torch.get_default_device()), a.dagger_stride)
            X = torch.cat([X, Xd.to(xdt)])
            dagger_its.append(it)
            print('DAgger {}: 閉ループの状態 {} 個を追加 → 計 {} 個  ({:.0f} s)'.format(
                rnd, Xd.shape[0], X.shape[0], time.time() - t0), flush=True)
            validate(it)
        for _ in range(n_it):
            lr = lr_at(it, total, a.pre_lr)
            for gr in opt.param_groups:
                gr['lr'] = lr
            it += 1
            x = perturb(X[sample_idx(X.shape[0], a, rng, hard_rng)].double(), a)
            l1, l2, l3 = losses(model, x, a)
            opt.zero_grad()
            (l1 + l2 + l3).backward()
            opt.step()
            acc = acc + torch.stack([l1, l2, l3]).detach()
            if it % LOSS_EVERY == 0:
                loss_rows.append((it, lr, *(acc / LOSS_EVERY).tolist()))
                acc = 0.0
            if it % PRINT_EVERY == 0:
                print('it {:6d} loss ブロック1 {:.2e} ブロック2 {:.2e} ブロック3 {:.2e}  ({:.0f} s)'.format(
                    it, l1.item(), l2.item(), l3.item(), time.time() - t0), flush=True)
    validate(it)
    return best, t0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True, help='結果のフォルダ名 results/mlp3/<tag>/')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu', help='cpu か cuda（cuda:0 など）')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--force', action='store_true', help='学習済み（metrics.json がある）でもやり直す')
    # モデル（既定値は m3_hdf）
    p.add_argument('--H', type=int, default=128, help='各ブロックの隠れ層の幅')
    p.add_argument('--layers', type=int, default=2, help='各ブロックの隠れ層の数')
    p.add_argument('--act', default='silu', choices=['silu', 'relu'])
    p.add_argument('--no-sfeat', dest='sfeat', action='store_false', help='ブロック 3 の積の特徴を使わない')
    # 状態の集め方
    p.add_argument('--bank', type=int, default=2048, help='通常の電流の本数')
    p.add_argument('--ep-ms', type=float, default=100.0, help='通常の電流 1 本の長さ [ms]')
    p.add_argument('--hard', type=int, default=256, help='苦手な電流の本数')
    p.add_argument('--hard-ms', type=float, default=3000.0, help='苦手な電流 1 本の長さ [ms]')
    p.add_argument('--hard-stride', type=int, default=10, help='苦手な電流の状態を何ステップごとに取るか')
    p.add_argument('--hard-set', default='v1', choices=['v1', 'v2'], help='苦手な電流の作り方（currents.py）')
    p.add_argument('--hard-frac', type=float, default=0.0,
                   help='ミニバッチのうち苦手な電流の状態から取る割合（0 なら全体から一様）')
    p.add_argument('--x32', action='store_true', help='状態を float32 で持つ（メモリ半分。正解は float64 で計算）')
    # 学習
    p.add_argument('--pre-iters', type=int, default=50000)
    p.add_argument('--pre-lr', type=float, default=1e-3)
    p.add_argument('--pre-batch', type=int, default=1024)
    p.add_argument('--pre-sv', type=float, default=2.0, help='V に足す雑音の標準偏差 [mV]')
    p.add_argument('--pre-sg', type=float, default=0.02, help='ゲートに足す雑音の標準偏差')
    p.add_argument('--pre-rel', type=float, default=0.02, help='>0 なら ΔV を相対誤差の Huber で学習（値は下限 [mV]）')
    p.add_argument('--dagger', type=int, default=3, help='DAgger の回数')
    p.add_argument('--dagger-iters', type=int, default=20000)
    p.add_argument('--dagger-n', type=int, default=256, help='DAgger で回す電流の本数（半分は通常、半分は苦手な電流）')
    p.add_argument('--dagger-ms', type=float, default=1000.0)
    p.add_argument('--dagger-stride', type=int, default=10)
    a = p.parse_args()

    run_dir = os.path.join(OUT_ROOT, a.tag)
    if os.path.exists(os.path.join(run_dir, 'metrics.json')) and not a.force:
        print('スキップ（学習済み）:', run_dir)
        return
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(run_dir, 'train.log'))
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(a), f, ensure_ascii=False, indent=1)
    torch.set_num_threads(a.threads)
    if a.device != 'cpu':
        torch.set_default_device(a.device)
        torch.backends.cuda.matmul.allow_tf32 = False      # 精度を CPU とそろえる
        torch.backends.cudnn.allow_tf32 = False

    best, t0 = train(a, run_dir)
    sec_train = time.time() - t0
    # 検証で最良の重みを float64 で読み直し、検証・テストの全電流の応答を保存して採点する
    model = MLP3.load(os.path.join(run_dir, 'model')).to(torch.get_default_device())
    res = evaluate(model, run_dir, ('val', 'test'), label=a.tag + ' (best)')
    summ = lambda r: {k: r[k] for k in ('n', 'n100', 'mean', 'gamma')}
    metrics = {'best_it': best[2], 'val_at_selection': {'n100': best[0], 'mean': best[1], 'dtype': 'float32'},
               'val': summ(res['val']), 'test': summ(res['test']),
               'sec_train': sec_train, 'sec_total': time.time() - t0}
    with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print('完了:', run_dir, flush=True)


if __name__ == '__main__':
    main()
