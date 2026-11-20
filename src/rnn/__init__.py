"""
NumPy で自作した RNN のパッケージ。sin 波・モデル1・モデル2（hh_modular.py）・traub_modular.py が使う。
torch_backend.py だけは PyTorch を使い、同じ式の RNN の学習を速く回す。
改良実験（experiments/hh_surrogate/hh_improve.py 以降）のモデル本体は PyTorch で直接組んでいる。

このファイルで各モジュールの関数・クラスを読み込んでおくと、使う側は
from rnn.model import SimpleRNN ではなく from rnn import SimpleRNN と短く書ける。

    model.py      : SimpleRNN（順伝播と BPTT）
    optimizers.py : SGD、Adam
    losses.py     : 平均二乗誤差
    data.py       : 時系列から（問題, 答え）の組を作る、前から順に分ける
    train.py      : ミニバッチ学習のループ
    callbacks.py  : EarlyStopping
    gradcheck.py  : 勾配の検算
    torch_backend.py : 学習だけを PyTorch で回し、重みを SimpleRNN に書き戻す
"""

from .model import SimpleRNN
from .optimizers import SGD, Adam
from .losses import mean_squared_error
from .data import create_dataset, split_in_order
from .train import train
from .callbacks import EarlyStopping
from .gradcheck import numerical_gradient_check
