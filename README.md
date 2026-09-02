# RNN Neuro Surrogate
### RNNを用いた神経細胞モデル（Multi-Compertmentモデル）の代理モデル構築による高速シミュレーション
微分方程式で記述されるニューロンモデルをRNNを用いた代理モデルで近似し、高速シミュレーションを実現する研究です。

## 背景
### 脳
- 脳の情報処理は，ニューロン間で伝達される電気信号（スパイク）によって実現される[1]。
- ニューロンの活動をシミュレーションすることで，脳の情報処理機構を解析する研究が行われている。
- 実際の脳には約1000億個のニューロンが存在するため，全ニューロンを直接観測することは困難である。
- そのため，ニューロンを数理モデルとして表現し，シミュレーションによって解析する手法が広く用いられている
    
    
### Multi-Compartmentモデル
- 高精度なニューロンモデルとして，Multi-Compartmentモデルが利用されている。
- ニューロンを複数のコンパートメントに分割し，各コンパートメントの膜電位や状態変数を計算することで，形状を考慮した電位伝播を再現できる。
- 単一コンパートメントモデルでは表現できない
  - 樹状突起での入力統合
  - 時間的・空間的な刺激シーケンスの弁別
  - 非線形演算（XOR演算など）
  を再現できる。
- 一方，各コンパートメントごとに状態変数を保持し，連立微分方程式を数値積分する必要があるため，
  - 計算量
  - メモリ使用量
  が非常に大きい。
- 大規模脳シミュレーションでは，この計算コストが大きな課題となっている．

### RNN
#### 時系列データ
- RNNでは、並びに規則性・パターンがある（または、ありそうに見える）データを学習することで未知の時系列データが与えられたとき、そのデータの未来の状態を予測する。
#### 過去の隠れ層
- 時系列データを保持するためには、過去の状態をモデル内で保持しておく必要
- 現在に対する過去からの目に見えない影響を把握しておく必要
- これらを過去の隠れ層として定義
- 一般的なNN:入力層$\mathbb{x}(t)$-隠れ層$\mathbb{h}(t)$-出力層$\mathbb{y}(t)$
- RNN:時刻$t-1$における隠れ層の値$\mathbb{h}(t-1)$を保持しておき、それも$\mathbb{h}(t)$に伝える
- 隠れ層に過去の状態がすべて反映されている
- 隠れ層に過去の状態がすべて反映されている
![alt text](image.png)

#### RNNを用いる理由

#### ① 内部状態を保持できる

- Multi-Compartmentモデルでは，現在の膜電位は現在の入力だけでなく，過去の膜電位や各コンパートメントの状態にも依存する。
- RNNは隠れ状態（Hidden State）として過去の情報を保持できる。
- 動的システムの状態遷移を自然に学習できるため，サロゲートモデルとして適している。

---

#### ② 時系列データとの親和性

- 膜電位は時間とともに変化する時系列データである。
- RNNは時系列データの時間相関を学習するニューラルネットワークである。
- Multi-Compartmentモデルの時間発展を近似するモデルとして適している。

---

#### ③ GPUによる高速化

- RNNの学習・推論は主に行列演算で構成される。
    - $\mathbb{h}(t)=f(W\mathbb{x}(t)+U\mathbb{h}(t-1))+\mathbb{b}$
    - $\mathbb{y}(t)=g(V\mathbb{h}(t))+\mathbb{c}$
- GPUは行列演算を高速に実行できるため，CPUによる数値積分より高速な推論が期待できる。
- サロゲートモデル化することで，大規模脳シミュレーションの高速化が期待される。

### Hodkin-Hukslayモデル
1952年 イギリスケンブリッジ大学 A.L.Hodkin ＆ A.F.Hukslayが，イカの巨大軸索の活動電位と，$Na^+$チャネル、$K^{+}$チャネルの開閉を電位固定法を用いた実験によって測定
- ニューロンを空間上の1点として表現
- 多入力を受け，和をとり，閾値を超えるかどうかを判定し，スパイクを発生させるという機構は実現可能
- 入力電流に対するイオンチャネルの開閉（膜のコンダクタンス）と膜電位の上下動を非線形連立微分方程式によって表現
$$

\begin{aligned}
C\frac{dV}{dt}
&=
-g_{\mathrm{leak}}(V(t)-E_{\mathrm{leak}})
-g_{\mathrm{Na}}(V,t)(V(t)-E_{\mathrm{Na}})
-g_{\mathrm{K}}(V,t)(V(t)-E_{\mathrm{K}})
+I_{\mathrm{ext}}(t)
\\[6pt]
\end{aligned}
$$
$$
\begin{cases}
g_{\mathrm{Na}}(V,t)
&=
\bar{g}_{\mathrm{Na}}\,m^{3}(V,t)h(V,t)
\\[6pt]
g_{\mathrm{K}}(V,t)
&=
\bar{g}_{\mathrm{K}}\,n^{4}(V,t)
\end{cases}
$$


$$
\begin{cases}
\frac{d}{dt} m(V,t) &= \alpha_m(V)(1 - m(V,t)) - \beta_m(V)m(V,t) \\
\frac{d}{dt} h(V,t) &= \alpha_h(V)(1 - h(V,t)) - \beta_h(V)h(V,t) \\
\frac{d}{dt} n(V,t) &= \alpha_n(V)(1 - n(V,t)) - \beta_n(V)n(V,t)
\end{cases}
$$
$$
\begin{cases}
\alpha_m(V) &= \frac{2.5 - 0.1V}{\exp(2.5 - 0.1V) - 1} \\[1.5ex]
\beta_m(V) &= 4 \exp\left(-\frac{V}{18}\right) \\[1.5ex]
\alpha_h(V) &= 0.07 \exp\left(-\frac{V}{20}\right) \\[1.5ex]
\beta_h(V) &= \frac{1}{\exp(3 - 0.1V) + 1} \\[1.5ex]
\alpha_n(V) &= \frac{0.1 - 0.001V}{\exp(1 - 0.1V) - 1} \\[1.5ex]
\beta_n(V) &= 0.125 \exp\left(-\frac{V}{80}\right)
\end{cases}
$$

## 研究の現在位置
- Hodkin-Hukslayモデルの数値シミュレーション（済）
    - まずは空間形状をもたない単一ニューロンの発火を確認した
    - 詳しくは「猿でもわかるニューロン発火 by Hodgkin,Huxley and Yamashita(2026/07/06)」を参照
    ![alt text](image-4.png)
- RNN実装
    - Pytouchを使った簡単なRNNを実装した
    - sin関数の学習に成功
    - 25ステップ分の過去の波形の塊を、時間を1ステップずつずらしながら
    ![alt text](image-5.png)
- HH × RNN　←今ここ
  - パルス電流をHHに入力
  - パルス入力電流 $I$ に対するHHの出力 $V$ をRNNに学習させる
  -  $I$ と $V$ の組だけでは学習が難しい場合
      - 一部HH，一部RNNのハイブリッド
      -  $I$ と $V$ だけでなく，パラメータ $g,m,n,h,\alpha,\beta$なども学習させる（詳しくはREAD MEに）
  - 同時進行で簡単なMulti Compertmentモデルを実装
- Multi Compertment実装 ～2026年 8/10
- Multi Compertment × RNN 〜2026年末
  - RNNに何を学習させるかを決める
  - 検証・評価
  - できれば大規模シミュレーションしてみたい
- 卒論執筆 〜2027年3月
- 修士過程からはより複雑なMulti CompertmentをRNNに学習させる
    




# 参考文献
[^1]：山﨑 匡 and 五十嵐 潤. はじめての神経回路シミュレーション:1ニューロンからヒト全脳モ
デルまで. 森北出版株式会社,2021年12月22日, pp. 56–61.


[^2] Eric R. Kandel et al. カンデル神経科学. 第2版. メディカル・サイエンス・インターナショ
ナル, 2022, p. 57.

[^3] Gidon Albert. Dendritic Action Potentials and Computation in Human Layer 2/3 Corti
cal Neurons | Science. https://www.science.org/doi/10.1126/science.aax6239. Jan. 2020.
(Visited on 09/05/2024).

[^4] Tiago Branco, Beverley A. Clark, and Michael Häusser. “Dendritic Discrimination of
Temporal Input Sequences in Cortical Neurons”. In: Science (New York, N.Y.) 329.5999
(Sept. 2010), p. 1671. doi: 10.1126/science.1189664. (Visited on 09/05/2024).

[^5] Kaaya, Tamura Akira, and Rin Kuriyama. “Development of a lightweight and cus
tomizable biophysical neuron simulator”. 2024. url: https : / / researchmap . jp /
tairakobayashi/presentations/48836360.

[^6] A. L. Hodgkin and A. F. Huxley. “A Quantitative Description of Membrane Current
and Its Application to Conduction and Excitation in Nerve”. In: The Journal of Physi
ology 117.4 (Aug. 1952), pp. 500–544. issn: 0022-3751. doi: 10.1113/jphysiol.1952.
sp004764.

[^7] 棟近先輩の卒論
