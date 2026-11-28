"""
C の 1 ステップで、どの GPU カーネルが動き、Tensor Core を使っているかを torch.profiler で調べる。
Tensor Core を使う V100 の行列積のカーネルは、名前に h884 / hmma / tensorop / volta_fp16 などが入る（cuBLAS の版で変わる）。
    uv run python -X utf8 experiments/hh_surrogate/m3_prof.py --B 65536 --mm fp32 fp16
出力: カーネルごとの GPU 時間の表（上位 25）と、Tensor Core らしい名前のカーネルが占める時間の割合
"""
import argparse
import os
import sys

import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_fused as mf   # noqa: E402

TC_KEYS = ('h884', 'hmma', 'tensorop', 's884', 'h1688', 'wmma', 'xmma', 'volta_fp16', 'volta_h')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='m3_hdf_s0')
    ap.add_argument('--B', type=int, default=65536)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--mm', nargs='+', default=['fp32', 'fp16'])
    a = ap.parse_args()
    dev = torch.device('cuda')
    for kind, mm in [(k, m) for k in ('m3f', 'm3one') for m in a.mm]:
        # m3f: MLP を分けた版（bmm + MLP2）、m3one: 1 本の 6 層ネットワーク（ブロック対角の重み）
        f = (mf.from_tag if kind == 'm3f' else mf.from_tag_one)(a.tag, dev, torch.float32, mm)
        V, g = f.rest(a.B)
        I = torch.full((a.B,), 10.0, device=dev)
        with torch.no_grad():
            for _ in range(10):
                V, g = f.step(V, g, I)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(a.steps):
                    V, g = f.step(V, g, I)
                torch.cuda.synchronize()
        ev = [e for e in prof.key_averages() if e.device_type.name == 'CUDA' or getattr(e, 'self_device_time_total', 0) > 0]
        tot = sum(e.self_device_time_total for e in ev)
        tc = sum(e.self_device_time_total for e in ev if any(k in e.key.lower() for k in TC_KEYS))
        print(f'===== {kind}:{a.tag} {mm}  B={a.B}  {a.steps} ステップ  GPU 時間 {tot / a.steps:.1f} µs/step  '
              f'Tensor Core らしいカーネル {100 * tc / max(tot, 1):.1f}%')
        print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=25, max_name_column_width=70))


if __name__ == '__main__':
    main()
