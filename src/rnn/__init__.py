"""
NumPy だけで書いた RNN のパッケージ。

このファイルで各モジュールの関数・クラスを読み込んでおくと、使う側は
from rnn.model import SimpleRNN ではなく from rnn import SimpleRNN と短く書ける。

    model.py      : SimpleRNN（順伝播と BPTT）
    optimizers.py : SGD、Adam
    losses.py     : 平均二乗誤差
    data.py       : 時系列から（問題, 答え）の組を作る、前から順に分ける
    train.py      : ミニバッチ学習のループ
    callbacks.py  : EarlyStopping
    gradcheck.py  : 勾配の検算
"""

from .model import SimpleRNN
from .optimizers import SGD, Adam
from .losses import mean_squared_error
from .data import create_dataset, split_in_order
from .train import train
from .callbacks import EarlyStopping
from .gradcheck import numerical_gradient_check
