# RNN Neuro Surrogate
### RNNを用いた神経細胞モデル（Multi-Compartmentモデル）の代理モデル構築による高速シミュレーション
ニューロンの膜電位応答（外部からの入力電流に対する膜電位の変化）を既存の微分方程式によるモデルから，RNNを用いた行列積計算によって構成される代理モデルを作成し，GPUを用いた高速シミュレーションを実現する研究です。



## 週の報告

### 10/9(金)〜10/16(金)
- やったこと
  - HH の代理モデルを2種類学習させ、学習に使っていない6種類の電流で評価した（隠れ層 50、Adam lr = 0.001、1000 エポック、seed = 123）
  - モデル1（RNN 1個）：スパイクを1発も出さなかった。学習電流での RMSE は 22.3 mV（HH は 53 発）（[図](results/hh_surrogate/prediction_eval.png)）
  - 論文読み
    - A Critical Review of Recurrent Neural Networks（[原稿](references/reviews/論文紹介10月.md)）
    - Learning flow functions of spiking systems（[原稿](references/reviews/論文紹介_learning_flow_functions.md)）
- 困っていること
  - なし。論文を読んで実装します。
- 来週のTODO
  - モデル２として、Learning flow functions of spiking systems（[原稿](references/reviews/論文紹介_learning_flow_functions.md)）を参考に HH の代理モデルを完成させる
    - 入力電流が 10 ms ごとの区分定数なので、10 ms を1ステップとして状態を進める RNN（論文では LSTM）にする。符号化器で初期状態を隠れ状態に変換し、復号器で V を出力する
    - スパイクの時刻を多く残す棄却サンプリングと、短い区間ごとの窓付き損失（L1 誤差）を取り入れる
    - モデル1と同じ6種類の評価電流で比べる
  - Multi-Compartment（traubモデル）の実装

## 全体の進捗
- 生理学的基礎知識（済, 〜6月）
  - 先行研究調査：棟近さんの SINDy による代理モデル、Learning flow functions of spiking systems
- HHモデルのオイラー法シミュレーション（済, 〜6月）
  - FitzHugh-Nagumo 方程式と HH モデルを数値計算した
- RNNの基礎知識（済, 7月〜10/2）
  - 詳解ディープラーニング 5章まで
  - NumPy で順伝播・BPTT・Adam を自作し、sin 波を学習できることを確認した
- HHモデルの代理モデルをRNNで作成（**いまここ**, 期限 10/18）
  - 教師データ作成（済, 9/25〜10/9）
    - 陽的オイラー法、dt = 0.01 ms、900 ms（90,000 ステップ）
    - 初期状態：V(0) = −65 mV、m, h, n はその電位での定常値
    - 入力電流：10 ms ごとに 0〜20 µA/cm² の一様乱数で変わる階段状の電流（seed = 0）
  - モデル1：RNN 1個で I → V（済, 10/9）
    - どの電流でもスパイクを出さず、電流に応じた平均的な膜電位だけを出力した
  - モデル2：Learning flow functions of spiking systems の方法で作成（未着手）
- Multi-Compartmentモデルの実装（未着手, 〜10月末）
- Multi-Compartment × RNN（未着手, 〜10月末）
- 卒論執筆（初稿 〜2027/1/20、提出 〜2027年3月）
