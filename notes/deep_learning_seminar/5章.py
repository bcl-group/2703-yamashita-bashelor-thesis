import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class RNN:
    """Simple Recurrent Neural Network (RNN).

    NumPyのみを用いて隠れ層1個のシンプルなRNNを実装する．
    入力層，隠れ層，出力層の重みとバイアスを保持し，
    順伝播・BPTT（Backpropagation Through Time）・パラメータ更新を行う．

    Attributes:
        W (np.ndarray):
            入力層から隠れ層への重み行列．
            形状は (hidden_size, input_size)
        U (np.ndarray):
            隠れ層から隠れ層への再帰重み行列。
            形状は (hidden_size, hidden_size)
            QR分解を用いた直交行列で初期化する
        V (np.ndarray):
            隠れ層から出力層への重み行列。
            形状は (output_size, hidden_size)
        b (np.ndarray):
            隠れ層のバイアスベクトル
            形状は (hidden_size, 1)
        c (np.ndarray):
            出力層のバイアスベクトル
            形状は (output_size, 1)
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size):
        """RNNのパラメータを初期化する．

        重み行列 W と V は Xavier初期化，
        再帰重み U は QR分解による直交初期化を用いる．

        隠れ層の活性化関数：tanh
        出力層の活性化関数：恒等関数

        Args:
            input_size:
                入力層のユニット数 D
            hidden_size:
                隠れ層のユニット数 J
            output_size:
                出力層のユニット数 K
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 入力→隠れ層
        # 平均0の正規分布から生成
        # 分散：1/D
        # サイズ：D×J
        self.W = np.random.normal(
            0.0,
            np.sqrt(1 / input_size),
            (hidden_size, input_size)
        )

        # 再帰重み（直交初期化）
        # J×Jのランダムな正方行列
        A = np.random.randn(hidden_size, hidden_size)
        # AをQR分解してQに
        Q, _ = np.linalg.qr(A)
        self.U = Q.astype(np.float32)

        # 隠れ層→出力層
        # 平均0の正規分布から生成
        # 分散：1/H
        # サイズ：K×J
        self.V = np.random.normal(
            0.0,
            np.sqrt(1 / hidden_size),
            (output_size, hidden_size)
        )


        self.b = np.zeros((hidden_size, 1), dtype=np.float32)
        self.c = np.zeros((output_size, 1), dtype=np.float32)

        # p(t)=Wx(t)+Uh(t−1)+b
        self.p = [] 

        # h(t)=f(p(t))
        self.h = []

        # q(t)=Vh(t)+c
        self.q = []

        # y(t)=g(q(t))
        self.y = []

        self.x = []

        # 勾配
        self.dW = np.zeros_like(self.W)
        self.dU = np.zeros_like(self.U)
        self.dV = np.zeros_like(self.V)
        self.db = np.zeros_like(self.b)
        self.dc = np.zeros_like(self.c)

    def tanh(x):
        """
        双曲線正接関数を計算する。

        Args:
            x (np.ndarray): 入力ベクトルまたは行列。

        Returns:
            np.ndarray: tanh(x)
        """
        return np.tanh(x)
    def dtanh(x):
        """
        tanh関数の導関数()を計算する。

        Args:
            x (np.ndarray): 入力ベクトルまたは行列。

        Returns:
            np.ndarray: tanh'(x)
        """
        return 1.0 - np.tanh(x) ** 2

    def identity(x):
        """恒等関数を計算する。

        Args:
            x (np.ndarray): 入力ベクトルまたは行列。

        Returns:
            np.ndarray: 入力をそのまま返す。
        """
        return x
    
    def didentity(x):
        """恒等関数の導関数を計算する。

        Args:
            x (np.ndarray): 入力ベクトルまたは行列。

        Returns:
            np.ndarray: 全要素が1の配列。
        """
        return np.ones_like(x)

    def forward(self, x):
        """順伝播を行う。

        入力
            x(t)∈R^D×1

        隠れ層
            h(t)∈R^H×1

        出力
            y(t)∈R^K×1

        重み
            W∈R^H×D
            U∈R^H×H
            V∈R^K×H

        Args:
            x (np.ndarray):
                入力時系列データ

        Returns:
            list[np.ndarray]:
                各時刻の出力 y(t)
        """

        self.p = []
        self.h = []
        self.q = []
        self.y = []
        self.x = []

        # h(t)∈R^H×1
        h_prev = np.zeros((self.hidden_size, 1), dtype=np.float32)

        for xt in x:

            # 自動計算して1列の行列に
            xt = xt.reshape(-1, 1)

            self.x.append(xt)

            # p(t)=Wx(t)+Uh(t−1)+b
            p = self.W @ xt + self.U @ h_prev + self.b

            # hi​(t)=tanh(pi​(t))
            h = np.tanh(p)

            # q(t)=Vh(t)+c
            q = self.V @ h + self.c

            # 出力は恒等関数 y(t)=q(t)​
            y = q

            self.p.append(p)
            self.h.append(h)
            self.q.append(q)
            self.y.append(y)

            # h(t−1)←h(t)
            h_prev = h

        return np.stack(self.y, axis=0)

    def loss(self, target):
        """損失関数を計算する。
            教師データと出力self.yとの差の二乗誤差を計算する。
        Args:
            target (numpy.ndarray or list of numpy.ndarray):
                教師データ（正解ラベル）
                各要素は `self.y` の各要素に対応する
                形状の配列である必要がある

        Returns:
            float: 計算された総損失値（スカラー値）
        """

        loss = 0.0

        for y, d in zip(self.y, target):

            d = d.reshape(-1, 1)

            # L = 1/2 Σ(d-y)^2
            loss += 0.5 * np.sum((d - y) ** 2)

        return loss

    def backward(self, target):
        """BPTTによる勾配計算
        
            RNNでは現在時刻の誤差だけではなく、
            未来時刻から伝播してくる誤差も考慮する必要がある。

            時刻Tから0へ逆向きに計算することで、
            以下の勾配を求める。

            dW:
                入力→隠れ層の重み勾配

            dU:
                隠れ状態→隠れ状態の再帰重み勾配

            dV:
                隠れ層→出力層の重み勾配

            db:
                隠れ層バイアス勾配

            dc:
                出力層バイアス勾配


            Args:
                target(np.ndarray):
                    教師データ。
                    shape=(time_steps, output_size)
        """

        # 初期化
        self.dW.fill(0)
        self.dU.fill(0)
        self.dV.fill(0)
        self.db.fill(0)
        self.dc.fill(0)


        # 時刻数
        T = len(target)


        # h(t)への誤差
        dh_next = np.zeros(
            (self.hidden_size,1),
            dtype=np.float32
        )


        # 後ろから時間を戻る
        for t in reversed(range(T)):


            y = self.y[t]
            q = self.q[t]
            p = self.p[t]


            d = target[t].reshape(-1,1)


            # 出力誤差
            # L=1/2(d-y)^2
            # dy = y-d
            dy = y - d


            # q -> y 恒等関数
            dq = dy


            # Vの勾配
            self.dV += dq @ self.h[t].T


            # 出力バイアス
            self.dc += dq



            # hへの誤差
            dh = self.V.T @ dq + dh_next



            # tanhの微分
            dp = dh * (1 - np.tanh(p)**2)



            # W
            xt = target[t].reshape(-1,1)


            # 入力を取得
            # xはforward時に保存していないので後で修正
            self.dW += dp @ self.x[t].reshape(1,-1)



            # U
            if t > 0:
                h_prev = self.h[t-1]
            else:
                h_prev = np.zeros_like(self.h[0])


            self.dU += dp @ h_prev.T


            # b
            self.db += dp


            # 次の時刻へ伝播
            dh_next = self.U.T @ dp


        return
    
    def update(self, lr):
        """勾配降下法によるパラメータ更新"""

        self.W -= lr * self.dW
        self.U -= lr * self.dU
        self.V -= lr * self.dV

        self.b -= lr * self.db
        self.c -= lr * self.dc

    def predict(self, x):
        """予測値を返す"""

        y = self.forward(x)

        return y


def sin(x, T=100):
    """正弦波を生成する

    Args:
        x:
            時刻を表す配列
        T:
            正弦波の周期。

    Returns:
        各時刻に対応する正弦波の値
    """
    return np.sin(2.0 * np.pi * x / T)

def toy_problem(T=100, ampl=0.05):
    """ノイズ付き正弦波データを生成する。

    Args:
        T:
            正弦波の周期。
        ampl:
            ノイズ振幅。

    Returns:
        ノイズ付き正弦波データ。
    """
    x = np.arange(0, 2*T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0,
                                         size=len(x))
    return sin(x,T) + noise

T = 100
f = toy_problem(T).astype(np.float32)
length_of_sequences = len(f)

# 訓練データの準備
# t=0 から t=N-1 を入力データ、t=1 から t=N を正解データとする
# 配列 f の先頭（インデックス 0）から最後から2番目の要素までを取得
train_x = f[:-1].reshape(-1, 1)
# 配列 f の2番目（インデックス 1）から一番最後の要素までを取得
train_y = f[1:].reshape(-1, 1)

# 2. ハイパーパラメータの設定
epochs = 800        # 学習回数
lr = 0.001          # 学習率（発散を防ぐため小さめに設定）
input_size = 1      # 1時点での入力次元数
hidden_size = 30    # 隠れ層のノード数
output_size = 1     # 出力次元数

# 3. RNNモデルのインスタンス化
rnn_model = RNN(input_size, hidden_size, output_size)

# 損失の履歴を保存するリスト
loss_history = []

print("=== 学習を開始します ===")
for epoch in range(epochs):
    # 順伝播
    rnn_model.forward(train_x)
    
    # 損失の計算
    current_loss = rnn_model.loss(train_y)
    loss_history.append(current_loss)
    
    # 逆伝播（BPTTによる勾配計算）
    rnn_model.backward(train_y)
    
    # パラメータの更新
    rnn_model.update(lr)
    
    # 100エポックごとに進行状況を出力
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1:04d} | Loss: {current_loss:.4f}")

print("=== 学習が完了しました ===")

# ==========================================
# 結果の可視化
# ==========================================

# 学習したモデルを用いて予測
pred_y = rnn_model.predict(train_x)

plt.figure(figsize=(12, 5))

# --- 損失の推移をプロット ---
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue')
plt.title("Training Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# --- 予測結果と正解データの比較をプロット ---
plt.subplot(1, 2, 2)
time_steps = np.arange(1, len(f))  # 予測している時刻（1以降）
plt.plot(time_steps, train_y.flatten(), label="True Data (Target)", color='gray', linestyle='dashed')
plt.plot(time_steps, pred_y.flatten(), label="Predicted", color='red', alpha=0.7)
plt.title("Time Series Prediction (1 Step Ahead)")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
