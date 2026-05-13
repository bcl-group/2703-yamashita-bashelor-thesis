**テーマ**  
RNNを用いて神経細胞モデル（Hodgkin–Huxleyモデル等）の代理モデル（Surrogate Model）を構築し、シミュレーション高速化を目指す

---

# 全体進捗状況
**現在地：0%（基礎知識整理フェーズ）**

---

# 進捗ロードマップ

## Phase 0：研究理解・環境構築（現在）
### 目的
- 卒論テーマの全体像理解
- HHモデルの生理学・数理モデル理解
- Python/C/GitHub環境整備

### ToDo
- [ ] Hodgkin-Huxleyモデルの生理学的理解
  - [ ] 膜電位とは何か
  - [ ] Na/Kチャネル開閉メカニズム
  - [ ] 活動電位発生の流れ
- [ ] HHモデルの数式理解
  - [ ] 膜電位微分方程式
  - [ ] m, h, nゲート方程式
  - [ ] 数値積分法（Euler法 / Runge-Kutta法）
- [ ] PythonでHHモデル実装
- [ ] CでHHモデル実装
- [ ] GitHub卒論用リポジトリ作成
- [ ] README作成
- [ ] .gitignore整備

### 成果物
- `hh_model_python.py`
- `hh_model_c.c`
- `README.md`

---

# Phase 1：先行研究調査
### 目的
- SINDyNeuroSurrogateとの差別化
- RNNを使う意義明確化

### ToDo
- [ ] 棟近さん（SINDyNeuroSurrogate）の理解
- [ ] SINDyとは何か整理
- [ ] Operator Learning（DeepONet/FNO等）理解
- [ ] FPGA高速化研究理解
- [ ] RNN（LSTM/GRU）の時系列モデリング理解
- [ ] 比較表作成
  - [ ] 精度
  - [ ] 計算速度
  - [ ] 解釈性
  - [ ] 実装難易度

### 成果物
- `literature_review.md`
- `comparison_table.md`

---

# Phase 2：データ生成
### 目的
- RNN学習用データセット作成

### ToDo
- [ ] HHモデルから教師データ生成
- [ ] 入力電流条件変更
  - [ ] 定常入力
  - [ ] パルス入力
  - [ ] ランダム入力
- [ ] 出力データ保存
  - [ ] 膜電位V
  - [ ] m, h, n
- [ ] train/val/test分割

### 成果物
- `dataset_generator.py`
- `data/train.csv`
- `data/test.csv`

---

# Phase 3：RNN代理モデル構築
### 目的
- HHモデルを近似するRNN作成

### ToDo
- [ ] ベースラインSimpleRNN
- [ ] LSTM実装
- [ ] GRU実装
- [ ] 損失関数設計（MSE）
- [ ] 学習曲線可視化
- [ ] 推論速度測定

### 評価項目
- [ ] RMSE
- [ ] 発火タイミング一致率
- [ ] 波形再現性
- [ ] 計算速度倍率（HH比）

### 成果物
- `train_rnn.py`
- `train_lstm.py`
- `evaluate.py`

---

# Phase 4：比較・考察
### 目的
- 「RNNはなぜ有効か」を卒論として成立させる

### ToDo
- [ ] HH vs RNN 精度比較
- [ ] HH vs RNN 計算時間比較
- [ ] SINDy vs RNN 比較（可能なら）
- [ ] 生理学的妥当性考察
- [ ] 教育応用可能性整理

### 考察候補
- [ ] RNNは生理詳細を省略しつつ時系列本質を保持できるか
- [ ] 教育用途で高速神経シミュレータとして有用か
- [ ] 他神経モデルへ拡張可能か

### 成果物
- `results.md`
- `discussion.md`

---

# Phase 5：卒論執筆
### ToDo
- [ ] 背景
- [ ] 目的
- [ ] 手法
- [ ] 実験
- [ ] 結果
- [ ] 考察
- [ ] 結論
- [ ] 発表資料

### 成果物
- `thesis.tex`
- `slides.pptx`

---

# 直近優先タスク（最優先）
## 今週やること
- [ ] GitHubリポジトリ作成
- [ ] README.md作成
- [ ] progress.md作成
- [ ] HHモデルの仕組み完全理解
- [ ] Python版HHモデル実装開始
- [ ] C版HHモデル実装開始

---

# 現時点の課題
## 不足しているもの
- 生理学理解
- 微分方程式理解
- 数値解析理解
- RNN構造理解
- PyTorch/TensorFlow習熟

---

# メモ
## 仮説
- RNNはHHの「時系列ダイナミクス」を学習し、高速近似できる可能性が高い
- 生理学的厳密性はHHに劣るが、教育・大規模計算で有利
- 「生理学モデルの高速代理」として価値あり

---

# 最終目標
## 卒論として目指す姿
**「HHモデルの高計算コスト問題に対し、RNN代理モデルを構築し、高速かつ十分な精度で神経活動を再現できることを示す」**

---
