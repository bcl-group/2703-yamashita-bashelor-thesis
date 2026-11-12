# ドリル：フロー関数学習モデルのアーキテクチャ

対象の図：[figures/flow_architecture.svg](figures/flow_architecture.svg)
対象のコード：`external/flow-learning` の `origin/2407-l4dc` ブランチ `flow_model/model.py`（`git -C external/flow-learning show origin/2407-l4dc:flow_model/model.py` で表示できる）

設定は論文§4.1の HH-FS（`run_paper.sh fs`）とする。

フロー関数・半群性・LSTM などの前提があやふやなときは、先に [準備編 drill_prerequisites.md](drill_prerequisites.md) を解く。

| 記号 | 意味 | 値 |
|---|---|---|
| $x$ | 初期状態 $(V/100, n, m, h)$ | 4次元 |
| $\Delta$ | 入力電流が一定の区間の長さ | 10 ms |
| $\omega_j$ | 区間 $j$ の電流値 | スカラー |
| $\tau_j$ | 区間 $j$ のうち進める割合 | $0 \le \tau_j \le 1$ |
| $(h_j, c_j)$ | $j$ 回更新した後の LSTM の隠れ状態・セル状態 | 各24次元 |
| $E$, $D$ | 符号化器・復号器（隠れ3層 tanh、出力層は線形） | 幅24 |

各問の答えは「答え」を開くと見える。まず自分で紙に書いてから開くこと。

---

## レベル1：時間軸と $\tau$ の計算

### 問1
$t = 35$ ms のとき、$k_t$、LSTM セルの数、$(\tau_0, \tau_1, \tau_2, \tau_3)$ を求めよ。

<details><summary>答え</summary>

$$
\begin{aligned}
k_t &= \lfloor t / \Delta \rfloor \\
    &= \lfloor 35 / 10 \rfloor \\
    &= \lfloor 3.5 \rfloor \\
    &= 3
\end{aligned}
$$

セルの数は $k_t + 1 = 4$ 個。完全な区間3つ（0–10, 10–20, 20–30 ms）と、端数の区間1つ（30–35 ms）を1セルずつ処理するため。

$$
\begin{aligned}
\tau_{k_t} &= (t - k_t \Delta) / \Delta \\
           &= (35 - 3 \cdot 10) / 10 \\
           &= 5 / 10 \\
           &= 0.5
\end{aligned}
$$

途中の区間はすべて進めるので $\tau_0 = \tau_1 = \tau_2 = 1$。答えは $(1, 1, 1, 0.5)$。
</details>

### 問2
$t = 72$ ms のとき、$k_t$、セルの数、最後の $\tau$ を求めよ。

<details><summary>答え</summary>

$$
k_t = \lfloor 72 / 10 \rfloor = 7
$$

セルは $k_t + 1 = 8$ 個。

$$
\tau_7 = (72 - 7 \cdot 10) / 10 = 0.2
$$

$\tau = (1, 1, 1, 1, 1, 1, 1, 0.2)$。
</details>

### 問3
論文コードの内部では、時間を「モデル時間 1 = 100 ms」に換算している（`dynamics.py` の `time_scale = 100`）。$\Delta = 10$ ms と $t = 35$ ms はモデル時間でいくつか。`run_paper.sh` の `--control_delta 0.1` との対応も述べよ。

<details><summary>答え</summary>

$$
\Delta = 10 / 100 = 0.1, \qquad t = 35 / 100 = 0.35
$$

`--control_delta 0.1` は「入力はモデル時間 0.1（= 10 ms）ごとに一定」という意味で、$\Delta = 10$ ms と同じ。$\tau_3 = (0.35 - 3 \cdot 0.1)/0.1 = 0.5$ となり、単位を変えても $\tau$ は変わらない（$\tau$ は比なので単位が消える）。
</details>

---

## レベル2：次元（shape）を追う

### 問4
バッチサイズを $B$、系列長（セルの数）を $L$ とする。次のテンソルの shape を答えよ（`batch_first=True`）。

1. 初期状態 `x`
2. RNN の入力 `rnn_input`（パディング後）
3. 符号化器の出力を整形した `h0`
4. LSTM の出力 `h`（`pad_packed_sequence` 後）
5. `deltas`
6. 最終出力 `output`

<details><summary>答え</summary>

| テンソル | shape | 理由 |
|---|---|---|
| `x` | $(B, 4)$ | 状態 $(V/100, n, m, h)$ |
| `rnn_input` | $(B, L, 2)$ | 各時刻の $(\omega_j, \tau_j)$ |
| `h0` | $(1, B, 24)$ | LSTM の層数1が先頭に来る（`torch.stack(h0.split(24, dim=1))`） |
| `h` | $(B, L, 24)$ | 各セルの出力 $h_1, \dots, h_L$ |
| `deltas` | $(B, L, 1)$ | `rnn_input` の $\tau$ の列。24次元に放送（broadcast）される |
| `output` | $(B, 1)$ | 予測した $V/100$ |
</details>

### 問5
符号化器の出力が24次元である理由を、LSTM の設定と結びつけて説明せよ。層を2層（`control_rnn_depth=2`）にしたら、符号化器の出力は何次元になるか。

<details><summary>答え</summary>

符号化器の出力は LSTM の初期隠れ状態 $h_0$ そのものなので、LSTM の隠れ状態の次元（24）と一致させる必要がある。コードでは `x_dnn_osz = control_rnn_depth * control_rnn_size` で決まる。

2層なら $2 \times 24 = 48$ 次元を出力し、`split(24)` で層ごとの $h_0$ に分ける。
</details>

---

## レベル3：パラメータ数

### 問6
符号化器 $E$（4→24→24→24→24）のパラメータ数を求めよ。線形層 $\mathbb{R}^a \to \mathbb{R}^b$ のパラメータ数は $ab + b$（重み＋バイアス）とする。

<details><summary>答え</summary>

$$
\begin{aligned}
\text{4→24} &: 4 \cdot 24 + 24 = 120 \\
\text{24→24} &: 24 \cdot 24 + 24 = 600 \quad (\times 3) \\
\text{合計} &= 120 + 600 \cdot 3 \\
            &= 120 + 1800 \\
            &= 1920
\end{aligned}
$$
</details>

### 問7
復号器 $D$（24→24→24→24→1）のパラメータ数を求めよ。

<details><summary>答え</summary>

$$
\begin{aligned}
\text{24→24} &: 600 \quad (\times 3) = 1800 \\
\text{24→1} &: 24 \cdot 1 + 1 = 25 \\
\text{合計} &= 1800 + 25 = 1825
\end{aligned}
$$
</details>

### 問8
LSTM（入力2次元、隠れ24次元、1層）のパラメータ数を求めよ。PyTorch の LSTM は、4つのゲート（入力・忘却・セル候補・出力）それぞれに、入力用の重み・隠れ状態用の重み・バイアス2種（`bias_ih`, `bias_hh`）を持つ。最後に、モデル全体のパラメータ数を求めよ。

<details><summary>答え</summary>

隠れ次元を $H = 24$、入力次元を $I = 2$ とする。

$$
\begin{aligned}
W_{ih} &: 4H \times I = 96 \times 2 = 192 \\
W_{hh} &: 4H \times H = 96 \times 24 = 2304 \\
b_{ih} + b_{hh} &: 4H + 4H = 192 \\
\text{LSTM 合計} &= 192 + 2304 + 192 = 2688
\end{aligned}
$$

モデル全体は

$$
1920 + 2688 + 1825 = 6433
$$

約6400個。HH の微分方程式をそのまま解く代わりに、この小さなネットワークを使う。
</details>

---

## レベル4：補間 $\tilde{z}$ の意味

### 問9
ある1成分について $h_3 = 0.2$、$h_4 = 0.6$ とする。$\tau_3 = 0, 0.5, 1$ それぞれの $\tilde{z}$ を求めよ。

<details><summary>答え</summary>

$\tilde{z} = (1 - \tau_3) h_3 + \tau_3 h_4$ に代入する。

$$
\begin{aligned}
\tau_3 = 0 &: \tilde{z} = 1 \cdot 0.2 + 0 \cdot 0.6 = 0.2 \quad (= h_3) \\
\tau_3 = 0.5 &: \tilde{z} = 0.5 \cdot 0.2 + 0.5 \cdot 0.6 = 0.1 + 0.3 = 0.4 \\
\tau_3 = 1 &: \tilde{z} = 0 \cdot 0.2 + 1 \cdot 0.6 = 0.6 \quad (= h_4)
\end{aligned}
$$

ただし実際には $h_4$ 自身も $\tau_3$ を入力として計算されるので、$\tau_3$ を変えると $h_4$ も変わる。ここでは $h_4$ を固定した数値例にすぎない。
</details>

### 問10
$t = 30$ ms ちょうどのとき、$k_t$、最後の $\tau$、$\tilde{z}$ はどうなるか。この結果が「区間の切れ目で予測が連続になる」ことにどう役立つか説明せよ。

<details><summary>答え</summary>

$$
k_t = \lfloor 30/10 \rfloor = 3, \qquad \tau_3 = (30 - 30)/10 = 0
$$

$$
\tilde{z} = (1 - 0) h_3 + 0 \cdot h_4 = h_3
$$

$t = 30$ ms では、3区間を完全に進めた状態 $h_3$ がそのまま使われ、最後のセルの出力 $h_4$ は使われない。一方、$t$ が30 msの直前（$t = 30 - \epsilon$）では、$k_t = 2$、$\tau_2 \to 1$ なので $\tilde{z} \to h_3$ になる。

つまり、どちらから近づいても $\tilde{z} = h_3$ になり、区間の切れ目で予測 $\hat{y} = D(\tilde{z})$ が跳ばない。補間がないと、$\tau = 0$ のときに LSTM が「0だけ進める」ことを学習しない限り、切れ目で値が跳ぶおそれがある。
</details>

### 問11
図の $h_4$（$t = 35$ ms の例）は「$t = 40$ ms の潜在状態」と言えるか。$t = 40$ ms を予測するときの $h_4$ と比べて答えよ。

<details><summary>答え</summary>

言えない。

- $t = 35$ ms の $h_4$：入力 $(\omega_3, \tau_3 = 0.5)$ で更新した結果
- $t = 40$ ms の場合：$k_t = 4$ で、$(\omega_3, \tau_3 = 1)$ で更新して $h_4$ を作る（さらに $(\omega_4, 0)$ で $h_5$ を作り、$\tilde{z} = h_4$）

LSTM への入力の $\tau_3$ が違うので、2つの $h_4$ は別物である。$h_j$ は「$j$ 回更新した後の状態」であって「時刻 $10j$ ms の状態」ではない。
</details>

---

## レベル5：コードを読む

`origin/2407-l4dc:flow_model/model.py` の `forward` は次のとおり。

```python
def forward(self, x, rnn_input, deltas):
    h0 = self.x_dnn(x)                                   # 符号化器
    h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
    c0 = torch.zeros_like(h0)

    rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
    h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                       batch_first=True)

    h_shift = torch.roll(h, shifts=1, dims=1)
    h_shift[:, 0, :] = h0[-1]

    encoded_controls = (1 - deltas) * h_shift + deltas * h
    output = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                         h_lens - 1, :])
    return output
```

### 問12
系列長 $L = 4$ のとき、`h[:, 0]`〜`h[:, 3]` と `h_shift[:, 0]`〜`h_shift[:, 3]` は、図の記号 $h_0, \dots, h_4$ のどれに当たるか。

<details><summary>答え</summary>

`h[:, j]` は、$j$ 番目の入力 $(\omega_j, \tau_j)$ を処理した後の出力なので $h_{j+1}$。`h_shift` は時間方向に1つずらし、先頭に $h_0$ を入れたもの。

| インデックス | `h` | `h_shift` |
|---|---|---|
| 0 | $h_1$ | $h_0$ |
| 1 | $h_2$ | $h_1$ |
| 2 | $h_3$ | $h_2$ |
| 3 | $h_4$ | $h_3$ |

`torch.roll` は末尾の要素を先頭に回すので、そのままだと `h_shift[:, 0]` は $h_4$ になる。それを `h0[-1]`（最上層の $h_0$）で上書きしている。
</details>

### 問13
`encoded_controls[..., h_lens - 1, :]` が図の $\tilde{z} = (1 - \tau_3) h_3 + \tau_3 h_4$ と一致することを、問12の表を使って示せ。

<details><summary>答え</summary>

$L = 4$ なので `h_lens - 1 = 3`。インデックス3では

$$
\begin{aligned}
\texttt{encoded\_controls}[:, 3]
&= (1 - \texttt{deltas}[:, 3]) \cdot \texttt{h\_shift}[:, 3] + \texttt{deltas}[:, 3] \cdot \texttt{h}[:, 3] \\
&= (1 - \tau_3) \, h_3 + \tau_3 \, h_4 \\
&= \tilde{z}
\end{aligned}
$$

コードはすべての位置で補間を計算してから、最後の位置だけを取り出している。
</details>

### 問14
バッチ内で系列長が違う（例：$L = 4$ と $L = 2$）とき、パディングされた位置の `deltas` は0になっている。それでも短い系列の予測に影響しない理由を述べよ。

<details><summary>答え</summary>

`pack_padded_sequence` で詰めているので、LSTM はパディング部分を計算しない。さらに出力は各系列の `h_lens - 1` の位置だけを取り出すので、パディング位置の `encoded_controls` は復号器に渡らない。短い系列（$L = 2$）ではインデックス1の $(1 - \tau_1) h_1 + \tau_1 h_2$ が使われる。
</details>

---

## レベル6：設計の理由

### 問15
初期状態 $x$ を LSTM の入力系列に混ぜず、符号化器で $h_0$ に変換するのはなぜか。論文§3.1の漸化式 $x_{k+1} = \Phi(\tau_k, x_k, \omega_k)$ と対応させて説明せよ。

<details><summary>答え</summary>

漸化式は「状態 $x_k$ を、入力 $(\omega_k, \tau_k)$ で次の状態に進める」形をしている。RNN の更新 $h_{k+1} = \mathrm{LSTM}((\omega_k, \tau_k), h_k)$ も同じ形なので、

- $x_k$（物理的な状態）↔ $h_k$（潜在状態）
- $(\omega_k, \tau_k)$（入力）↔ RNN の入力

と対応させるのが自然である。初期状態 $x_0 = x$ は漸化式のスタート地点なので、RNN の入力ではなく初期隠れ状態 $h_0$ に対応させる。符号化器 $E$ は4次元の $x$ を24次元の潜在空間へ移す役割、復号器 $D$ は潜在空間から膜電位を取り出す役割を持つ。
</details>

### 問16
セル状態の初期値 $c_0$ を符号化器で作らず、常に0にしている。$h_0$ だけで初期状態の情報が十分に伝わると言える理由を、LSTM の更新式から考えよ。

<details><summary>答え</summary>

LSTM の各ゲートは $h_{k}$ と入力から計算される。

$$
\begin{aligned}
i_k &= \sigma(W_{ii} u_k + W_{hi} h_k + b_i) \\
f_k &= \sigma(W_{if} u_k + W_{hf} h_k + b_f) \\
g_k &= \tanh(W_{ig} u_k + W_{hg} h_k + b_g) \\
o_k &= \sigma(W_{io} u_k + W_{ho} h_k + b_o) \\
c_{k+1} &= f_k \odot c_k + i_k \odot g_k \\
h_{k+1} &= o_k \odot \tanh(c_{k+1})
\end{aligned}
$$

$c_0 = 0$ でも、1回目の更新で $c_1 = i_0 \odot g_0$ となり、$i_0$, $g_0$ は $h_0$ に依存する。したがって $x$ の情報は $h_0$ を通じて最初の更新でセル状態に書き込まれる。

ただし、これは「情報が伝わる経路がある」ことを示すだけで、「$c_0$ も学習させるより良い」ことの証明ではない。論文はこの選択の理由を書いていない。
</details>

### 問17
入力に $\omega_j$ だけでなく $\tau_j$ も入れるのはなぜか。$\tau_j$ を入れずに $\omega_j$ だけにすると、どんな $t$ が予測できなくなるか。

<details><summary>答え</summary>

$\tau_j$ がないと、LSTM は「1区間＝10 ms 進める」更新しか表せない。そうすると、予測できるのは $t = 0, 10, 20, \dots$ ms の区間の切れ目だけになる（離散時間モデル）。

$\tau_j$ を入れることで、1つの LSTM セルが「$\tau_j \Delta$ だけ進める」写像 $\Phi(\tau, x, \omega)$ を近似できるようになり、任意の時刻 $t$（例：35 ms、72 ms）の膜電位を出せる。これが論文の言う「連続時間のサロゲートモデル」の意味である。
</details>
