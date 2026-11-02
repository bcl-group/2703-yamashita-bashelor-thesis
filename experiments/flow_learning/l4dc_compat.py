"""
論文コード（flow-learning の 2407-l4dc ブランチ）を torch 2.14 で動かすための起動用ラッパー。

論文コードは 2024 年の torch（2.3 頃）向けに書かれていて、torch 2.14 では次の 2 か所で落ちる。
論文コードのファイルは書き換えず、実行前にこの 2 か所だけ torch 側を旧来の動きに戻す。
どちらも学習の計算には関わらないので、結果は torch 2.3 で動かした場合と変わらない。

1. torch.load の既定値
   torch 2.6 から weights_only=True が既定になり、論文コードが保存する Python オブジェクト
   （ExperimentData・Experiment）を読み込めない。既定値を旧来の weights_only=False に戻す。
2. ReduceLROnPlateau の verbose 引数
   学習率を変えたときに表示するだけの引数で、torch 2.7 頃に削除された。渡されても無視する。

使い方（論文コードのフォルダで。引数はそのまま論文コードに渡る）:
    python l4dc_compat.py experiments/hhfs_flow.py hhfs.pth --control_delta 0.1 ...
    python l4dc_compat.py experiments/load_data.py outputs/hhfs.pth --control_rnn_size 24 ...
"""
import os
import runpy
import sys

import torch

_torch_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load(*args, **kwargs)


torch.load = _load

_Plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
_plateau_init = _Plateau.__init__


def _init(self, *args, verbose=None, **kwargs):
    _plateau_init(self, *args, **kwargs)


_Plateau.__init__ = _init

if __name__ == '__main__':
    script = os.path.abspath(sys.argv[1])
    sys.argv = sys.argv[1:]
    # 直接実行したときと同じく、sys.path[0] をスクリプトのフォルダにする
    # （論文コードは sys.path[0] を基準に import 先を足している）
    sys.path[0] = os.path.dirname(script)
    runpy.run_path(script, run_name='__main__')
