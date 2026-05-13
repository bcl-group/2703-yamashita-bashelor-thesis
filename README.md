# RNN Neuro Surrogate
### RNNを用いた神経細胞モデル（Hodgkin–Huxleyモデル）の代理モデル構築による高速シミュレーション

---

# 1. 研究概要

## 背景
神経細胞の活動電位を高精度に再現する代表的な数理モデルとして、Hodgkin–Huxley（HH）モデルがある。  
HHモデルは生理学的解釈性に優れる一方、複数の非線形微分方程式を数値的に解く必要があるため、計算コストが高い。

特に以下の場面では計算量が問題となる。

- 長時間シミュレーション
- 多数のニューラルネットワーク
- （リアルタイム制御）

---

## 問題点
HHモデルは高精度だが、

- 微分方程式が複雑
- 時間刻みが細かい
- パラメータ探索が重い

という理由から、簡易利用や大規模計算に不向きである。

---

## 研究目的
本研究では、

**RNN（Recurrent Neural Network）を用いてHHモデルの時系列挙動を学習し、  
HHモデルの代理モデル（Surrogate Model）を構築することで、  
生理学的特徴を保持しつつ高速な膜電位予測を実現すること**

を目的とする。

---

# 2. 研究の新規性

## 既存研究
先行研究では、

- ライブラリ法（SINDy）
- Operator Learning
- Reduced Order Model

などが存在する。

### 限界
- 微分方程式構造への依存
- ノイズ耐性の問題
- 長期予測の不安定性
- 時系列依存性の弱さ

---

## 本研究の特徴
RNN（LSTM / GRUを含む可能性）を用いることで、

↓わからない
- 時系列特化
- 入出力関係を直接学習
- 微分方程式を逐次解かず予測
- 長期ダイナミクス再現の可能性

---

# 3. 仮説
↓わからない

## 主仮説
RNNはHHモデルの膜電位時系列を十分な精度で近似でき、  
従来の数値計算より高速である。

---

## 副仮説
- スパイクタイミング再現可能
- 発火周期保持可能
- 外部入力変化への追従可能
- 教育・リアルタイム応用に有効

---
↑わからない

# 4. 研究フロー

## Step 1: HHモデル理解・実装
- Hodgkin–Huxley方程式理解
- C実装（高速基準）
- Python実装（可視化）

---

## Step 2: データ生成
HHモデルから以下を生成：

### 入力
- 外部電流 \( I_{ext}(t) \)

### 出力
- 膜電位 \( V(t) \)

### 必要に応じて
- m, h, nゲート変数

---

## Step 3: 前処理
- 正規化
- 時系列窓分割
- 学習/検証/テスト分割

---

## Step 4: RNN構築
候補：
- Simple RNN
- LSTM
- GRU

評価：
- MSE
- MAE
- 発火タイミング誤差
- 計算速度比較

---

## Step 5: 比較
### 比較対象
- HH元モデル
- SINDy surrogate（棟近氏）
- RNN surrogate（本研究）

---

# 5. 使用技術

## 言語
- Python
- C

---

## ライブラリ
- NumPy
- SciPy
- Matplotlib
- PyTorch / TensorFlow

---

## 開発環境
- GitHub
- VSCode
- Google Colab

---

# 6. ディレクトリ構成（予定）

```bash
thesis-rnn-hh/
│
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/              # HHシミュレーション元データ
│   └── processed/        # 学習用整形データ
│
├── src/
│   ├── hh_model/
│   │   ├── hh_model.c
│   │   └── hh_model.py
│   │
│   ├── rnn_model/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── evaluate.py
│   │
│   └── utils/
│
├── notebooks/
│
├── results/
│   ├── figures/
│   └── logs/
│
└── references/
