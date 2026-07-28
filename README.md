# RNN Neuro Surrogate
### RNNを用いた神経細胞モデル（Hodgkin–Huxleyモデル）の代理モデル構築による高速シミュレーション
### 脳
- 脳は生物の思考や行動などの情報処理を司る
    - ニューロンのネットワークを介したスパイク伝播によってなされる[1]．
        - スパイク：ニューロンの膜電位がスパイク状に急上昇すること
            - 隣接するニューロンがスパイクを起こす
            - イオン電流を刺激として受けることで膜電位が上昇
            - 膜電位が閾値を超えると，イオンチャネルが一気に開いて膜電位が急上昇（スパイク）する．
            - ここで流入チャネルが閉じ，流出チャネルが開くため，膜電位がそれ以上上がることはない．
            - 発生した電気信号は軸索を伝わり，隣接ニューロンへと送り出される
            - HHモデルのシミュレーションを用いた詳しいニューロン発火の流れの説明はHHモデルの章で
- 脳の情報処理の全過程を観測したい
    - 脳内の膨大な数（約$10^{11}$個）のニューロンを同時に観測する必要があり，実現は困難である[2]．
        - 一方で、個々のニューロンの挙動はHodgkin-Huxleyモデルなど数理モデルによって表現可能である[1]．
        - ニューロンの個々の振る舞いを記述する数理モデルを組み合わせた脳モデルを
作り、脳シミュレーションを行うことができる．

### Hodkin-Hukslayモデル
1952年 イギリスケンブリッジ大学 A.L.Hodkin ＆ A.F.Hukslayが
#### A QUANTITATIVE DESCRIPTION OF MEMBRANE CURRENT AND ITS APPLICATION TO CONDUCTION AND EXCITATION IN NERVE（神経における膜電流の定量的記述とその伝導および興奮への応用）
と題して発表
- イカの巨大軸索から得たデータを基に，

### Multi-Compartmentモデル
- Hodkin-Hukslayモデルではニューロンの空間的な形状は無視して，単に空間上の1点として表現した．
- しかし，実際のニューロンは空間形状を持っており，膜電位のイオンチャネルの種類や膜電位の値は位置に応じて異なる．

- こういった形状を考慮すると，従来考えられていたよりもはるかに高度な情報処理を単一ニューロンで実施できることが期待される．
- 弁別や排他的論理和演算などの複雑なニューロンの特性を再現できる[3][4]．
- ニューロンを複数のコンパートメントで表現し、各コンパートメントは膜電位と複数の隠れ変数を保持するため、多くのメモリを必要とする[1]．
- マウス脳規模（約10^8～10^9個のニューロン）のシミュレーションでも、スーパーコンピュータ富岳でも扱いきれないほど空間計算量が増大する[5]．
- 隠れ変数はイオン輸送を制御する重要な変数であるが、脳シミュレーションでは直接利用しないため、メモリ消費の要因となっている[1]．
- 大規模脳シミュレーションの実現には、ニューロンモデルの空間計算量削減が重要である．

## 研究目的
- 脳の大規模シミュレーションを行うにあたって，以下の2つを満たしつつ，膜電位応答を再現できるMulti-Compartmentモデルの代理モデル（surrogate model）の構築を目指す．
    - 空間計算量を削減
    - GPUよって高速処理が可能
- 本研究では、RNNを用いてMulti-Compartmentモデルに対する代理モデルを構築する

## 参考文献
- [1]
- [2]
- [3]
- [4]
- [5]
- [6]
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