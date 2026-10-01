# 直近でやったこと
## 〜9/25
- 教育実習（高校物理 2週間）
- RNNの勉強と実装（numpyのみ使用ほぼ完成、sin波を学習させるtaskが終わり次第HH学習に入ります）
- スライドのひな形作成

# 近々やること
- HH教師データの作成
  - テスト用のパルス電流をHHに入力として与える ～10/2
- HH教師データを自作RNNに学習させる  ～10/2
- いい感じのMulti Compertmentモデルを見つける
  - 小林先生から教えて頂いた論文を読む
- 学内GPUサーバのマウント（gitとも連携したい）


# 長期計画
- HH 実装（済）
- RNN実装（済）
- HH × RNN　←今ここ ～8/10
  - パルス電流をHHに入力
  - パルス入力電流 $I$ に対するHHの出力 $V$ をRNNに学習させる
  -  $I$ と $V$ の組だけでは学習が難しい場合
      - 一部HH，一部RNNのハイブリッド
      -  $I$ と $V$ だけでなく，パラメータ $g,m,n,h,\alpha,\beta$なども学習させる（詳しくはREAD MEに）
  - 同時進行で簡単なMulti Compertmentモデルを実装
- Multi Compertment実装 ～2026年 10/10
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


