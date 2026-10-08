import numpy as np
import random as ra

def step(x):
    '''
    カッコ内がtrueなら1を、Falseなら0を返す
    '''
    return 1 * (x > 0)

class SimplePerceptron(object):
    '''
    単純パーセプトロン
    入力次元：n
    パラメータ：重みw,バイアスb
    入力xを受け取り、出力yを返す
    学習時はΔwとΔbが使われる
    '''
    def __init__(self,input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,))
        '''
        重みは正規分布乱数を用いて初期化
        '''
        self.b = 0.

    def foward (self,x):
        '''
        wとxの内積計算
        '''
        y = step(np.matmul(self.w, x) + self.b)
        return y

    def compute_deltas(self,x,t):
        y = self.foward(x)
        delta = y - t
        dw = delta * x
        db = delta
        return dw,db

if __name__ == '__main__':
    np.random.seed(123)
    '''
    1.データの準備
    正規分布に従うデータ（2種）を分類する
    ニューロン数：２
    発火しないデータ：平均値が0
    発火するデータ：平均値が5
    '''
    d = 2
    N = 20

    mean = 5
    '''
    np.random.randn(N//2, d):
    標準正規分布（平均 0、標準偏差 1）に従う乱数を、
    行数 = N//2、列数 = d（次元数）の行列（2次元配列）として生成
    // は切り捨て除算
    全体のデータ数 N を 2 で割った整数値を計算
    x1: 原点 [0, 0] を中心とするデータ群を生成する
    x2: 点 [mean, mean] を中心とするデータ群を生成する

    '''
    x1 = np.random.randn(N//2, d) + np.array([0, 0])
    x2 = np.random.randn(N//2, d) + np.array([mean, mean])

    t1 = np.zeros(N//2)
    t2 = np.ones(N//2)

    x = np.concatenate((x1, x2), axis=0)
    t = np.concatenate((t1, t2))

    '''
    2.モデルの構築
    今回は入力次元2の単純パーセプトロンを構築する
    '''
    model = SimplePerceptron(input_dim=d)

    '''
    3.モデルの学習
    単純パーセプトロンの学習にじゃ、誤り訂正学習法が用いられる
    更新式はk回目の学習において、重みwとバイアスbを以下のように更新する
    w(k+1) = w(k) - η * Δw
    b(k+1) = b(k) - η * Δb
    ここで、ηは学習率、
    ΔwとΔbは誤り訂正学習法における重みとバイアスの更新量
    簡単のため、学習率は1とする
    終了判定：すべてのデータが正しく分類されるまで繰り返す
                →ΔwとΔbが0になるまで繰り返す
    '''
    def compute_loss(dw, db):
        '''
        終了判定
        all 0ならTrueを返す
        '''
        return all(dw == 0) * (db == 0)

    def train_step(x, t):
        '''
        def compute_deltas(self,x,t):
            y = self.foward(x)
            delta = y - t
            dw = delta * x
            db = delta
            return dw,db
        '''
        dw, db = model.compute_deltas(x, t)
        loss = compute_loss(dw, db)
        model.w -= dw
        model.b -= db
        return loss
    
    while True:
        classified = True
        for i in range(N):
            loss = train_step(x[i], t[i])
            classified *= loss
        if classified:
            break

    
    '''
    4.モデルの評価
    '''
    print('w:', model.w)
    print('b:', model.b)