import numpy as np
import random as ra

def step(x):
    '''
    カッコ内がtrueなら1を、Falseなら0を返す
    return 1 * (x>0)
    '''

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
        self.w = np.random.nomal(size=(input_dim,))
        '''
        重みは正規分布乱数を用いて初期化
        '''
        self.b = 0.

    def foward(self.x):
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
    '''


    '''
    2.モデルの構築
    '''


    '''
    3.モデルの学習
    '''


    '''
    4.モデルの評価
    '''