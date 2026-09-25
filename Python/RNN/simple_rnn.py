"""
NumPyだけで実装するシンプルなRNN

このファイルは卒業研究用の学習教材コードであり、PyTorch / Keras / TensorFlow を
一切使わずに、RNNの順伝播・逆伝播（BPTT）・パラメータ更新を自分で実装している。


------------
    h(t) = f( W_xh x(t) + W_hh h(t-1) + b_h )     … 資料 式(5.4)
    活性化関数f：tanh
    y    = W_hy h(T-1) + b_y                          … 資料 式(5.5),(5.15)

  入力 x(t) を時刻 t=0,1,...,T-1 まで順に読み込み、
  最後の時刻の隠れ状態 h(T-1) だけを使って次の1点y(T)を予測する。
"""

import numpy as np
import matplotlib.pyplot as plt



# データ生成  [資料参照]
def sin(x, T=100):
    """周期 T の sin 波を計算する。

    Args:
        x (numpy.ndarray): 時刻を表す配列。
        T (int): sin 波の周期。デフォルト値は 100。

    Returns:
        numpy.ndarray: sin 波の値。
    """
    return np.sin(2.0 * np.pi * x / T)


def toy_problem(T=100, ampl=0.05):
    """ノイズ入りのsin波（トイ・プロブレム）を生成する。

    Args:
        T (int): sin波の周期。デフォルト値は 100。
        ampl (float): 加えるノイズの振幅。0.0 にするとノイズなしの純粋なsin波になる。
    
    Returns:
        numpy.ndarray: ノイズ入りsin波。shape = (2*T + 1,)  ＝ 2周期分＋1点
    """

    # x = [0, 1, 2, ..., 2T] の整数時刻。2周期分のデータを作る
    x = np.arange(0, 2 * T + 1)                      # shape = (2T+1,)

    # 一様分布 U(-1, 1) のノイズに振幅 ampl を掛ける
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))

    return sin(x, T) + noise                          # shape = (2T+1,)


def create_dataset(f, maxlen):
    """1本の時系列データを、RNNの学習に使う入力と正解に変換する。

    maxlen （P267ではτ）個の連続した値を入力とし、その直後の1個の値を正解とする。
    τ=25 の場合の例：
        [f0, f1, ..., f24]  ->  f25
        [f1, f2, ..., f25]  ->  f26
        ...
        [f(2T-25), f(2T-24), ..., f(2T-1)]  ->  f(2T)

    Args:
        f(numpy.ndarray): 元となる時系列データ。shape = (系列長,1)
        maxlen(int): 1つの入力系列に使う時刻数（何ステップ分を見て次を予測するかを与えている。資料ではτ=25を採用）
    
    Returns:
        x(numpy.ndarray): RNNへの入力データ。shape = (サンプル数, maxlen, 1)
        t(numpy.ndarray): 入力系列の次に来る正解値。shape = (サンプル数, 1)
    """


    length_of_sequences = len(f) #2T+1

    x = []
    t = []

    # 取り出せる窓の数だけ繰り返す。
    # i が 0 から (系列長 - maxlen - 1) まで動くので、サンプル数は 系列長 - maxlen
    for i in range(length_of_sequences - maxlen):
        x.append(f[i:i + maxlen])   # 入力: i 番目から maxlen 個
        t.append(f[i + maxlen])     # 正解: その次の1点

    # reshape の -1 は「残りは自動計算」の意味。
    # (サンプル数, maxlen) -> (サンプル数, maxlen, 1) に次元を1つ増やす。
    # これは「各時刻の入力が input_dim=1 次元のベクトルである」ことを明示するため。
    x = np.array(x).reshape(-1, maxlen, 1)   # x.shape = (N, maxlen, 1)
    t = np.array(t).reshape(-1, 1)           # t.shape = (N, 1)

    return x, t


def split_in_order(x, t, test_size=0.2):
    """
    時系列データを、順番を保ったまま訓練用と検証用に分割する。  [補助処理]

    PyTorch版では sklearn の train_test_split(shuffle=False) を使っているが、
    今回は外部ライブラリを使わないので同じ挙動を自分で実装する。
    時系列データなのでシャッフルはしない（未来のデータで学習してしまうため）。

    Parameters
    ----------
    x : numpy.ndarray
        入力データ。shape = (N, maxlen, 1)
    t : numpy.ndarray
        正解データ。shape = (N, 1)
    test_size : float
        検証用に回す割合。

    Returns
    -------
    x_train, x_val, t_train, t_val : numpy.ndarray
        先頭 (1-test_size) 割が訓練データ、残りが検証データ。
    """
    n_train = int(len(x) * (1.0 - test_size))
    return x[:n_train], x[n_train:], t[:n_train], t[n_train:]


# ==================================================================
# 2. 損失関数  [資料参照] 式(5.16)
# ==================================================================

def mean_squared_error(y_true, y_pred):
    """
    平均二乗誤差（MSE）を計算する。

    資料 p.261 式(5.16) では E = (1/2) * Σ ||y - t||^2 と定義されているが、
    ここでは PyTorch版の nn.MSELoss(reduction='mean') に合わせて
    「全要素の二乗誤差の平均」とする。
    （係数の違いは学習率の違いに吸収されるだけで、本質は変わらない）

    Parameters
    ----------
    y_true : numpy.ndarray
        正解値。shape = (batch_size, output_dim)
    y_pred : numpy.ndarray
        予測値。shape = (batch_size, output_dim)

    Returns
    -------
    float
        平均二乗誤差（スカラー）。
    """
    return np.mean((y_pred - y_true) ** 2)


# ==================================================================
# 3. RNN本体  [PyTorch処理をNumPyで再実装]
# ==================================================================
# PyTorch版の
#     nn.RNN(1, hidden_dim, nonlinearity='tanh', batch_first=True)
#     nn.Linear(hidden_dim, 1)
# が内部で行っている計算を、すべて NumPy の行列演算で書き下す。

class SimpleRNN:
    """
    NumPyだけで実装した基本的な（Elman型）RNN。

    資料の式との対応
    ----------------
        h(t) = tanh( W_xh x(t) + W_hh h(t-1) + b_h )   資料の W, U, b  … 式(5.4)
        y    = W_hy h(T-1) + b_y                        資料の V, c     … 式(5.5)

    資料は縦ベクトル（W x(t)）で式を書いているが、NumPyでは
    「バッチを行方向に並べる」実装にするため x(t) @ W_xh のように
    行列の左右が入れ替わる（＝資料の式を転置した形）点に注意する。

    Attributes
    ----------
    W_xh : numpy.ndarray
        入力 -> 隠れ層 の重み（資料の W）。shape = (input_dim, hidden_dim)
    W_hh : numpy.ndarray
        前時刻の隠れ層 -> 隠れ層 の重み（資料の U）。shape = (hidden_dim, hidden_dim)
        正方行列になるのは、隠れ状態 h を同じ次元の h に写すため。
    b_h : numpy.ndarray
        隠れ層のバイアス（資料の b）。shape = (hidden_dim,)
    W_hy : numpy.ndarray
        隠れ層 -> 出力層 の重み（資料の V）。shape = (hidden_dim, output_dim)
    b_y : numpy.ndarray
        出力層のバイアス（資料の c）。shape = (output_dim,)
    """

    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1,
                 init='simple'):
        """
        重みとバイアスを初期化する。

        Parameters
        ----------
        input_dim : int
            各時刻の入力の次元。sin波は1時刻あたり1つの値なので 1。
        hidden_dim : int
            隠れ状態の次元（＝RNNの記憶の大きさ）。PyTorch版は 50。
        output_dim : int
            出力の次元。次の1点を予測するので 1。
        init : {'simple', 'orthogonal'}
            重みの初期化方法。
            'simple'     : 小さな乱数。まず動かして理解するための最も単純な方法。[補助処理]
            'orthogonal' : W_hh を直交行列にする。資料 5.1.4（p.265）で
                           「再帰計算では同じ重みが繰り返し掛かるため、
                             通常の初期化だとオーバーフローしうる」と説明されている
                           対策にあたる。[資料参照]
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if init == 'orthogonal':
            # --------------------------------------------------
            # 資料 5.1.4（p.264-265）に基づく初期化  [資料参照]
            # --------------------------------------------------
            # W_xh は Xavier の初期化：標準偏差 sqrt(1/n_in) の正規乱数。
            self.W_xh = np.random.normal(
                loc=0.0, scale=np.sqrt(1.0 / input_dim),
                size=(input_dim, hidden_dim))

            # W_hh は直交行列で初期化する。
            # 直交行列 W は W W^T = I を満たすので、何度掛けても値が発散しない。
            # QR分解を使うと、ランダム行列から直交行列 Q を取り出せる。
            # （QR分解を使う実装方法自体は資料には書かれていない）[補助処理]
            a = np.random.normal(size=(hidden_dim, hidden_dim))
            q, r = np.linalg.qr(a)
            # 符号を揃えて、毎回同じ規則で直交行列が得られるようにする
            self.W_hh = q * np.sign(np.diag(r))

            self.W_hy = np.random.normal(
                loc=0.0, scale=np.sqrt(1.0 / hidden_dim),
                size=(hidden_dim, output_dim))
        else:
            # --------------------------------------------------
            # 最も単純な初期化  [補助処理]
            # --------------------------------------------------
            # 小さな乱数にするのは、tanh の入力が大きすぎると
            # 勾配 (1 - tanh^2) がほぼ0になって学習が進まなくなるため。
            self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.01
            self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
            self.W_hy = np.random.randn(hidden_dim, output_dim) * 0.01

        # バイアスは0で初期化する（対称性の問題は重みの乱数で既に解けているため）
        self.b_h = np.zeros(hidden_dim)      # shape = (hidden_dim,)
        self.b_y = np.zeros(output_dim)      # shape = (output_dim,)

        # 順伝播の途中経過を逆伝播で使うため、ここに保存する
        self.cache = None

    # --------------------------------------------------------------
    # 順伝播
    # --------------------------------------------------------------
    def forward(self, x):
        """
        時系列を1ステップずつ処理して、最後の時刻から出力を求める。

        PyTorch版の
            h, _ = self.l1(x)   # nn.RNN
            y = self.l2(h[:, -1])
        に相当する処理を、for ループで書き下したもの。

        Parameters
        ----------
        x : numpy.ndarray
            入力系列。shape = (batch_size, maxlen, input_dim)

        Returns
        -------
        y : numpy.ndarray
            予測値。shape = (batch_size, output_dim)
        """
        batch_size, maxlen, _ = x.shape

        # --------------------------------------------------
        # 隠れ状態の初期値を0にする
        # --------------------------------------------------
        # 最初の時刻 t=0 には「1つ前の時刻」が存在しないので、
        # h(-1) を 0 ベクトルとして扱う。
        # （PyTorchの nn.RNN も h0 を省略すると 0 で初期化する）
        h = np.zeros((batch_size, self.hidden_dim))   # h.shape = (batch, hidden_dim)

        # 逆伝播（BPTT）では各時刻の隠れ状態が必要になるので、全部保存しておく。
        # h_list[t] が「時刻 t に入る直前の隠れ状態」＝ h(t-1) になるように並べる。
        h_list = [h]

        # --------------------------------------------------
        # 時刻方向のループ（RNNの本体）
        # --------------------------------------------------
        for t in range(maxlen):
            # 時刻tの入力を取り出す
            x_t = x[:, t, :]                     # x_t.shape = (batch, input_dim)

            # 活性化前の値 p(t) を計算する（資料 式(5.6)）
            #   第1項: 今の入力からの寄与        x(t) と W_xh の行列積
            #   第2項: 1つ前の時刻の隠れ状態からの寄与  h(t-1) と W_hh の行列積
            #   第3項: バイアス
            p = x_t @ self.W_xh + h @ self.W_hh + self.b_h
            #  (batch, input_dim)@(input_dim, hidden) + (batch,hidden)@(hidden,hidden)
            #  -> p.shape = (batch, hidden_dim)

            # 活性化関数 tanh を通して、新しい隠れ状態にする（資料 式(5.4)）
            # tanh を使うのは、出力が -1〜1 に収まり再帰計算が安定するため。
            h = np.tanh(p)                       # h.shape = (batch, hidden_dim)

            # 次の時刻のために保存する（このhが次ループの h(t-1) になる）
            h_list.append(h)

        # --------------------------------------------------
        # 出力層
        # --------------------------------------------------
        # 最後の時刻の隠れ状態 h(maxlen-1) だけを使って予測する。
        # ここまでの全時刻の情報が h に畳み込まれている、という考え方。
        h_last = h_list[-1]                      # shape = (batch, hidden_dim)

        # 出力は線形変換のみ（資料 p.261 式(5.15)：sin波の回帰なので活性化関数なし）
        y = h_last @ self.W_hy + self.b_y        # y.shape = (batch, output_dim)

        # 逆伝播で使う値を保存
        self.cache = (x, h_list, y)

        return y

    # --------------------------------------------------------------
    # 逆伝播（BPTT）
    # --------------------------------------------------------------
    def backward(self, t_true):
        """
        BPTT（Backpropagation Through Time）で各パラメータの勾配を計算する。

        PyTorch版の `loss.backward()` に相当する処理を、
        資料 5.1.3（p.262-264）の式にしたがって自分で計算する。

        forward() を呼んだ直後に呼ぶこと（self.cache を使うため）。

        Parameters
        ----------
        t_true : numpy.ndarray
            正解値。shape = (batch_size, output_dim)

        Returns
        -------
        grads : dict of numpy.ndarray
            各パラメータの勾配。キーは 'W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y'。
            shapeはそれぞれ対応するパラメータと同じ。
        """
        x, h_list, y = self.cache
        batch_size, maxlen, _ = x.shape

        # 勾配を入れる箱を0で用意する。
        # W_xh, W_hh, b_h は「全時刻の寄与を足し合わせる」ので += で蓄積していく
        # （資料 式(5.24)(5.25)(5.27) の Σ にあたる）
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        # --------------------------------------------------
        # (1) 出力層の誤差項 e_o   資料 式(5.18)
        # --------------------------------------------------
        #   e_o(t) = g'(q(t)) * (y - t)
        # 出力層の活性化 g は恒等関数なので g' = 1。つまり e_o = (y - t) に比例する。
        # 損失を「平均」二乗誤差にしたので、要素数 (batch_size * output_dim) で割り、
        # 二乗の微分から出てくる係数 2 を掛ける。
        e_o = 2.0 * (y - t_true) / (batch_size * self.output_dim)
        #  e_o.shape = (batch, output_dim)

        # --------------------------------------------------
        # (2) 出力層のパラメータの勾配   資料 式(5.11)(5.14)
        # --------------------------------------------------
        h_last = h_list[-1]                       # shape = (batch, hidden_dim)

        # 資料の ∂E/∂V = e_o(t) h(t)^T を、行ベクトル表現に直したもの
        dW_hy = h_last.T @ e_o                    # shape = (hidden_dim, output_dim)

        # 資料の ∂E/∂c = e_o(t)。バッチ方向は合計する
        db_y = np.sum(e_o, axis=0)                # shape = (output_dim,)

        # --------------------------------------------------
        # (3) 最後の時刻の隠れ層の誤差項 e_h   資料 式(5.17)
        # --------------------------------------------------
        #   e_h(t) = f'(p(t)) * V^T e_o(t)
        # tanh の微分は f'(p) = 1 - tanh(p)^2 = 1 - h^2 なので、
        # 保存しておいた h をそのまま使える（p を保存する必要がない）。
        e_h = (e_o @ self.W_hy.T) * (1.0 - h_last ** 2)
        #  e_h.shape = (batch, hidden_dim)

        # --------------------------------------------------
        # (4) 時間を遡るループ（ここがBPTTの核心）
        # --------------------------------------------------
        # 時刻 maxlen-1 から 0 へ、逆向きに進みながら
        #   ・その時刻での重みの勾配を足し込む
        #   ・誤差項を1つ前の時刻へ伝える
        # を繰り返す。
        for t in reversed(range(maxlen)):
            x_t = x[:, t, :]                      # shape = (batch, input_dim)
            h_prev = h_list[t]                    # h(t-1)。shape = (batch, hidden_dim)

            # 資料 式(5.10): ∂E/∂W = Σ e_h(t-z) x(t-z)^T
            dW_xh += x_t.T @ e_h                  # shape = (input_dim, hidden_dim)

            # 資料 式(5.12): ∂E/∂U = Σ e_h(t-z) h(t-z-1)^T
            dW_hh += h_prev.T @ e_h               # shape = (hidden_dim, hidden_dim)

            # 資料 式(5.13): ∂E/∂b = Σ e_h(t-z)
            db_h += np.sum(e_h, axis=0)           # shape = (hidden_dim,)

            # ----------------------------------------------
            # 誤差項を1つ前の時刻へ伝播させる  資料 式(5.22)(5.23)
            #   e_h(t-1) = e_h(t) * ( U f'(p(t-1)) )
            # ----------------------------------------------
            # (1 - h_prev^2) が f'(p(t-1)) にあたる。
            # t=0 のとき h_prev は初期値0なのでここで計算した値は以降使われない。
            e_h = (e_h @ self.W_hh.T) * (1.0 - h_prev ** 2)

        grads = {
            'W_xh': dW_xh,
            'W_hh': dW_hh,
            'b_h': db_h,
            'W_hy': dW_hy,
            'b_y': db_y,
        }
        return grads

    # --------------------------------------------------------------
    # パラメータ更新（最も基本的な勾配降下法）
    # --------------------------------------------------------------
    def param_names(self):
        """更新対象のパラメータ名を返す。"""
        return ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']


# ==================================================================
# 4. 最適化アルゴリズム  [PyTorch処理をNumPyで再実装]
# ==================================================================
# PyTorch版の `optimizer.step()` にあたる処理。
# まず最も基本的な勾配降下法(SGD)を実装し、そのうえでAdamを実装する。

class SGD:
    """
    最も基本的な勾配降下法（Stochastic Gradient Descent）。

    資料 式(5.24)-(5.28) の
        パラメータ <- パラメータ - 学習率 * 勾配
    をそのまま実装したもの。  [資料参照]
    """

    def __init__(self, lr=0.05):
        """
        Parameters
        ----------
        lr : float
            学習率 η。1回の更新でどれだけ勾配の方向に進むか。
        """
        self.lr = lr

    def step(self, model, grads):
        """
        パラメータを1回更新する。

        Parameters
        ----------
        model : SimpleRNN
            更新対象のモデル。
        grads : dict of numpy.ndarray
            backward() が返した勾配。
        """
        for name, grad in grads.items():
            param = getattr(model, name)
            # param -= lr * grad （インプレース演算なのでモデルの中身が直接書き換わる）
            param -= self.lr * grad


class Adam:
    """
    Adam（+ AMSGrad）による最適化。

    GitHub版の
        optimizers.Adam(model.parameters(), lr=0.001,
                        betas=(0.9, 0.999), amsgrad=True)
    と同じ設定を、NumPyで自分で実装したもの。
    アルゴリズム自体は資料 4.6.6 Adam（p.236）、4.6.7 AMSGrad（p.238）に対応する。

    考え方（SGDとの違い）
    ---------------------
    - m : 勾配の「移動平均」。過去の勾配の向きを覚えておき、更新をなめらかにする。
    - v : 勾配の2乗の移動平均。よく動くパラメータほど更新幅を小さくする。
    - AMSGrad : v の最大値を保持して使うことで、学習が不安定になるのを防ぐ。

    このsin波の問題では、SGDだと「次の1点の予測」はできても、
    予測値を入力に戻して波形を生成させると形が崩れてしまった。
    Adamにすると生成もうまくいくことを確認している。
    """

    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, amsgrad=True):
        """
        Parameters
        ----------
        model : SimpleRNN
            更新対象のモデル（パラメータのshapeを知るために受け取る）。
        lr : float
            学習率。
        beta1, beta2 : float
            移動平均の減衰率。1に近いほど過去を長く覚える。
        eps : float
            0除算を防ぐための小さな値。
        amsgrad : bool
            True のとき AMSGrad を使う。
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad

        # パラメータごとに、同じshapeの「状態」を0で用意する
        names = model.param_names()
        self.m = {k: np.zeros_like(getattr(model, k)) for k in names}
        self.v = {k: np.zeros_like(getattr(model, k)) for k in names}
        self.v_max = {k: np.zeros_like(getattr(model, k)) for k in names}

        # 何回更新したかのカウンタ（バイアス補正に使う）
        self.t = 0

    def step(self, model, grads):
        """
        パラメータを1回更新する。

        Parameters
        ----------
        model : SimpleRNN
            更新対象のモデル。
        grads : dict of numpy.ndarray
            backward() が返した勾配。
        """
        self.t += 1

        for name, grad in grads.items():
            param = getattr(model, name)

            # 1次モーメント（勾配そのものの移動平均）
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # 2次モーメント（勾配の2乗の移動平均）
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad ** 2

            # 初期値が0なので最初のうちは値が小さく出る。それを補正する
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            if self.amsgrad:
                # これまでの v_hat の最大値を使う（更新幅が急に大きくならないようにする）
                self.v_max[name] = np.maximum(self.v_max[name], v_hat)
                v_hat = self.v_max[name]

            # 勾配の大きさで割ることで、パラメータごとに適切な歩幅にする
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==================================================================
# 5. 数値微分による勾配の検算  [補助処理]
# ==================================================================
# BPTT（解析的な微分）が正しく実装できているかを確かめるために、
# 「パラメータを少しずらして損失の変化を見る」という定義どおりの微分と比較する。
# 学習そのものには使わない（非常に遅いため）。

def numerical_gradient_check(model, x, t_true, n_check=3, eps=1e-5):
    """
    BPTTで求めた勾配と、数値微分で求めた勾配を比較する。

    数値微分（中心差分）:
        df/dw ≒ ( f(w + eps) - f(w - eps) ) / (2 * eps)

    Parameters
    ----------
    model : SimpleRNN
        検査するモデル。
    x : numpy.ndarray
        入力データ。shape = (batch, maxlen, input_dim)
    t_true : numpy.ndarray
        正解データ。shape = (batch, output_dim)
    n_check : int
        各パラメータについて、ランダムに何要素を調べるか。
    eps : float
        ずらす幅。小さすぎると桁落ちし、大きすぎると近似が悪くなる。

    Returns
    -------
    None
        結果を標準出力に表示する。
    """
    # まず解析的な勾配（BPTT）を求める
    model.forward(x)
    grads = model.backward(t_true)

    print('--- 勾配の検算（BPTT vs 数値微分） ---')
    for name in model.param_names():
        param = getattr(model, name)
        flat = param.ravel()          # 1次元表示（元の配列とメモリを共有する）
        grad_flat = grads[name].ravel()

        # 調べる要素をランダムに選ぶ
        idx_list = np.random.choice(flat.size, min(n_check, flat.size),
                                    replace=False)

        max_rel_err = 0.0
        for idx in idx_list:
            original = flat[idx]

            # w + eps のときの損失
            flat[idx] = original + eps
            loss_plus = mean_squared_error(t_true, model.forward(x))

            # w - eps のときの損失
            flat[idx] = original - eps
            loss_minus = mean_squared_error(t_true, model.forward(x))

            # 値を元に戻す（戻し忘れるとモデルが壊れる）
            flat[idx] = original

            numerical = (loss_plus - loss_minus) / (2.0 * eps)
            analytic = grad_flat[idx]

            # 相対誤差。1e-6 程度以下なら実装は正しいと考えてよい
            denom = max(1e-12, abs(numerical) + abs(analytic))
            rel_err = abs(numerical - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)

        print('  {:5s} : 最大相対誤差 = {:.3e}'.format(name, max_rel_err))
    print()


# ==================================================================
# 6. 学習処理
# ==================================================================

def train(model, optimizer, x_train, t_train, x_val, t_val,
          epochs=1000, batch_size=100, verbose_every=50):
    """
    ミニバッチ学習でモデルを学習させる。

    PyTorch版の学習ループ（train_step / val_step）に相当するが、
    loss.backward() と optimizer.step() は自作の backward() / optimizer.step() で行う。

    Parameters
    ----------
    model : SimpleRNN
        学習させるモデル。
    optimizer : SGD or Adam
        パラメータ更新の方法。
    x_train, t_train : numpy.ndarray
        訓練データ。shape = (N_train, maxlen, 1), (N_train, 1)
    x_val, t_val : numpy.ndarray
        検証データ。学習には使わず、損失の確認だけに使う。
    epochs : int
        全データを何周するか。
    batch_size : int
        1回の更新に使うサンプル数。
    verbose_every : int
        何エポックごとに損失を表示するか。

    Returns
    -------
    hist : dict of list
        'loss'（訓練損失）と 'val_loss'（検証損失）の履歴。
    """
    n_train = len(x_train)
    hist = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --------------------------------------------------
        # エポックごとにデータの順番をシャッフルする
        # --------------------------------------------------
        # 毎回同じ順番で学習すると、その順番に依存した更新になってしまうため。
        # （PyTorch版の sklearn.utils.shuffle に相当）[PyTorch処理をNumPyで再実装]
        perm = np.random.permutation(n_train)
        x_ = x_train[perm]
        t_ = t_train[perm]

        train_loss = 0.0
        n_batches = 0

        # --------------------------------------------------
        # ミニバッチごとに「順伝播 -> 損失 -> 勾配 -> 更新」を行う
        # --------------------------------------------------
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            x_batch = x_[start:end]     # shape = (batch, maxlen, 1)
            t_batch = t_[start:end]     # shape = (batch, 1)

            # 1. 順伝播
            y = model.forward(x_batch)

            # 2. 損失
            loss = mean_squared_error(t_batch, y)

            # 3. 勾配計算（BPTT）
            grads = model.backward(t_batch)

            # 4. パラメータ更新（PyTorch版の optimizer.step() にあたる）
            optimizer.step(model, grads)

            train_loss += loss
            n_batches += 1

        train_loss /= n_batches

        # --------------------------------------------------
        # 検証データでの損失（学習はしない＝更新を呼ばない）
        # --------------------------------------------------
        val_loss = mean_squared_error(t_val, model.forward(x_val))

        hist['loss'].append(train_loss)
        hist['val_loss'].append(val_loss)

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            print('epoch: {:4d}, loss: {:.6f}, val_loss: {:.6f}'.format(
                epoch + 1, train_loss, val_loss))

    return hist


# ==================================================================
# 7. 予測（学習済みモデルでsin波を生成する）  [資料参照]
# ==================================================================

def generate_sequence(model, x_seed, n_steps, maxlen):
    """
    予測値を入力に戻しながら、sin波を逐次生成する。

    最初の maxlen 点だけを与え、あとはモデル自身の予測を次の入力として
    使い続ける。モデルが波の形を本当に学習できていれば、
    予測だけでsin波が続いていくはずである。

    Parameters
    ----------
    model : SimpleRNN
        学習済みモデル。
    x_seed : numpy.ndarray
        生成の起点となる最初の1系列。shape = (1, maxlen, 1)
    n_steps : int
        何点を生成するか。
    maxlen : int
        入力系列の長さ。

    Returns
    -------
    gen : list
        生成された値のリスト。先頭 maxlen 個は None（起点なので予測値が無い）。
    """
    # 先頭 maxlen 点は「予測していない区間」なので None で埋める
    gen = [None for _ in range(maxlen)]

    z = x_seed.copy()               # z.shape = (1, maxlen, 1)

    for _ in range(n_steps):
        # 今の系列から次の1点を予測する
        pred = model.forward(z)     # pred.shape = (1, 1)

        # 予測値を末尾に足し、先頭を1つ捨てて、窓を1つ右にずらす
        z = np.append(z, pred)[1:]          # いったん1次元になる。shape = (maxlen,)
        z = z.reshape(-1, maxlen, 1)        # 形を戻す。shape = (1, maxlen, 1)

        gen.append(pred[0, 0])

    return gen


# ==================================================================
# 8. 可視化  [補助処理]（RNNそのものではなく、評価・可視化の処理）
# ==================================================================

def plot_results(hist, f, sin_true, gen, T, save_dir='.'):
    """
    学習曲線と、生成したsin波の比較グラフを表示・保存する。

    グラフのラベルは、日本語フォントが無い環境での文字化けを避けるため英語にしている。

    Parameters
    ----------
    hist : dict of list
        train() が返した損失の履歴。
    f : numpy.ndarray
        学習に使ったノイズ入りsin波。
    sin_true : numpy.ndarray
        ノイズなしの正解sin波。
    gen : list
        モデルが生成したsin波。
    T : int
        sin波の周期（グラフの横軸範囲に使う）。
    save_dir : str
        画像の保存先ディレクトリ（相対パス）。
    """
    # --- 学習曲線 ---
    fig1 = plt.figure()
    plt.plot(hist['loss'], color='black', linewidth=1, label='train loss')
    plt.plot(hist['val_loss'], color='gray', linewidth=1, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE)')
    plt.yscale('log')          # 損失は桁で変化するので対数軸が見やすい
    plt.legend()
    plt.title('Learning curve')
    fig1.savefig('{}/learning_curve.png'.format(save_dir), dpi=120)

    # --- 正解sin波と生成sin波の比較 ---
    # 描画の設定（figure〜plot）は資料 p.271 の実装と同じにしてある。[資料参照]
    # 軸ラベル・凡例・タイトルだけは、見てすぐ分かるように追加した。[補助処理]
    fig2 = plt.figure()
    plt.rc('font', family='serif')
    plt.xlim([0, 2 * T])
    plt.ylim([-1.5, 1.5])
    plt.plot(range(len(f)), sin_true,
             color='gray', linestyle='--', linewidth=0.5,
             label='true sin')
    plt.plot(range(len(f)), gen,
             color='black', linewidth=1,
             marker='o', markersize=1,
             markerfacecolor='black', markeredgecolor='black',
             label='predicted')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.title('Sin wave prediction by NumPy RNN')
    fig2.savefig('{}/prediction.png'.format(save_dir), dpi=120)

    plt.show()


# ==================================================================
# 9. メイン処理
# ==================================================================

def main():
    """データ生成から可視化までを一通り実行する。"""
    # 乱数の種を固定して、実行するたびに同じ結果が出るようにする
    np.random.seed(123)

    # ==================================================
    # 1. データの準備  [資料参照] 資料 5.1.5.1（p.266-268）
    # ==================================================
    T = 100                 # sin波の周期
    maxlen = 25             # 25個の値から次の1個を予測する

    f = toy_problem(T)      # ノイズ入りsin波。f.shape = (2T+1,) = (201,)
    length_of_sequences = len(f)

    x, t = create_dataset(f, maxlen)
    # x.shape = (176, 25, 1)  … (サンプル数, 系列長, 入力次元)
    # t.shape = (176, 1)      … (サンプル数, 出力次元)
    print('x.shape =', x.shape)
    print('t.shape =', t.shape)

    x_train, x_val, t_train, t_val = split_in_order(x, t, test_size=0.2)
    print('train: {}, val: {}'.format(len(x_train), len(x_val)))
    print()

    # ==================================================
    # 2. モデルの構築
    # ==================================================
    # hidden_dim=50 は資料 p.269 / GitHub版と同じ。
    model = SimpleRNN(input_dim=1, hidden_dim=50, output_dim=1,
                      init='simple')

    # --------------------------------------------------
    # BPTTの実装が正しいかを数値微分で検算する  [補助処理]
    # --------------------------------------------------
    # 学習を始める前に確認しておく。相対誤差が十分小さければ実装は正しい。
    numerical_gradient_check(model, x_train[:8], t_train[:8])

    # ==================================================
    # 3. モデルの学習  [資料参照] 資料 p.270
    # ==================================================
    # 最適化は、資料 p.270 / GitHub版と同じ設定のAdamを使う。
    #   Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    #
    # SGD(lr=0.05) でも「次の1点の予測」自体は学習できる（lossは下がる）が、
    # 予測値を入力に戻して波形を生成させると形が崩れてしまった。
    #   ・SGD を試す場合:  optimizer = SGD(lr=0.05)
    optimizer = Adam(model, lr=0.001, beta1=0.9, beta2=0.999, amsgrad=True)

    # epochs=1000, batch_size=100 は資料 p.270 / GitHub版と同じ。
    # ただし資料は EarlyStopping(patience=10) を併用しており、実際には
    # 95エポック程度で学習を打ち切っている（資料 p.271 の実行結果）。
    # 今回は EarlyStopping を実装しない方針なので、1000エポック回し切る。
    hist = train(model, optimizer, x_train, t_train, x_val, t_val,
                 epochs=1000, batch_size=100, verbose_every=100)
    print()

    # ==================================================
    # 4. モデルの評価  [資料参照] 資料 p.270-271
    # ==================================================
    # 最初の25点だけを与え、以降はモデル自身の予測値を入力に戻して
    # sin波を生成させる。比較のためノイズなしのsin波も描く。
    sin_true = toy_problem(T, ampl=0.0)

    gen = generate_sequence(model, x[:1], length_of_sequences - maxlen, maxlen)

    # 可視化（学習曲線は資料にはない追加のグラフ）  [補助処理]
    plot_results(hist, f, sin_true, gen, T)


if __name__ == '__main__':
    main()
