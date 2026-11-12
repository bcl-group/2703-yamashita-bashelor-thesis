"""
GPU サーバで torch が GPU を使えるかを確かめる。

確かめること
1. torch のバージョンと、ビルドに使われた CUDA・cuDNN のバージョン
2. 見えている GPU ごとの名前・compute capability・空きメモリ
3. 実際に GPU 上で LSTM の順伝播と逆伝播が動くか（V100 非対応のビルドだとここで落ちる）

使い方（リポジトリのルートで）:
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/check_gpu.py
"""
import os
import time

import torch


def main():
    print('torch       ', torch.__version__)
    print('CUDA (build)', torch.version.cuda)
    print('cuDNN       ', torch.backends.cudnn.version())
    print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES', '(未設定: 全 GPU が見える)'))

    if not torch.cuda.is_available():
        raise SystemExit('GPU が使えない（CPU 版 torch か、ドライバと CUDA ビルドが合っていない）')

    # 見えている GPU の一覧。free は他の人のプロセスが使っている分を引いた残り
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}  '
              f'CC {torch.cuda.get_device_capability(i)}  '
              f'free {free / 2**30:.1f} / {total / 2**30:.1f} GiB')

    # LSTM の順伝播・逆伝播を GPU 0（= CUDA_VISIBLE_DEVICES で選んだ GPU）で動かす
    # x: (batch=256, 系列長=100, 入力次元=2) -> y: (256, 100, 32)
    device = torch.device('cuda:0')
    lstm = torch.nn.LSTM(input_size=2, hidden_size=32, batch_first=True).to(device)
    x = torch.randn(256, 100, 2, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        y, _ = lstm(x)
        y.square().mean().backward()
    torch.cuda.synchronize()
    print(f'GPU LSTM test: OK  y.shape={tuple(y.shape)}  20 回で {time.perf_counter() - t0:.2f} s')


if __name__ == '__main__':
    main()
