### RNNを用いた神経細胞モデル（Multi-Compertmentモデル）の代理モデル構築による高速シミュレーション
微分方程式で記述される。外部からの入力電流に対する膜電位の応答を計算するニューロンモデルをRNNを用いた代理モデルで近似し、高速シミュレーションを実現する研究です。

# 直近でやったこと
- スライド作成
- RNNの理解進める
- 教師電流と正解の組を作成

# 近々やること
- HH教師データの作成
  - テスト用のパルス電流をHHに入力として与える
- HH教師データを自作RNNに学習させる
- いい感じのMulti Compertmentモデルを見つける
  - 小林先生から教えて頂いた論文を読む
- 学内GPUサーバのマウント（gitとも連携したい）


# 長期計画
- HH 実装（済）
- RNN実装（済）
- HH × RNN　
  - パルス電流をHHに入力
  - パルス入力電流 $I$ に対するHHの出力 $V$ をRNNに学習させる
  -  $I$ と $V$ の組だけでは学習が難しい場合
      -  $I$ と $V$ だけでなく，パラメータ $g,m,n,h,\alpha,\beta$なども学習させる（詳しくはREAD MEに）
- Multi Compertment(Troub model)実装 ～2026年 11月中旬
- Multi Compertment × RNN 〜2026年末
  - RNNに何を学習させるかを決める
  - 検証・評価
  - できれば大規模シミュレーションしてみたい
- 卒論執筆 〜2027年3月
- より複雑なMulti Compertmentを
    
# 困っていること
### 院試
### 研究は今のところ楽しいです（タスク遂行中）

# スケジュール
https://calendar.google.com/calendar/embed?src=f0b22906c034bb6e3753b4dadf50a59ff6fc9b7d10a15e9a6db9abfab723b298%40group.calendar.google.com&ctz=Asia%2FTokyo


