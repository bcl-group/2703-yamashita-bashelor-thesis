# snsbook コードの読み方（マルチコンパートメント → GPU 並列化）

- 対象：[numericalbrainorg/snsbook](https://github.com/numericalbrainorg/snsbook) の `code/`
- ローカルのクローン：`external/snsbook/`（git 管理外。`.git/info/exclude` で除外済み）
- 本文のまとめは [notes_hajimete_neural_circuit_sim.md](notes_hajimete_neural_circuit_sim.md)

## 現在地とゴール

| 段階 | 状態 | snsbook | このリポジトリ |
|---|---|---|---|
| 1. HH モデル（1 個） | 済 | `part1/hh/hh.c` | `src/neuron/hh.py` |
| 2. Traub モデル（1 個、19 コンパートメント） | 済 | `part1/multi/traub.c` | `src/neuron/traub.py`、図 3.20a/b |
| 3. GPU 向けのデータ配置を知る | これから | `column/soa/{aos,soa}.cu` | — |
| 4. 1 スレッド 1 ニューロンの GPU 化 | これから | `part3/random/random_gpu.cu` | — |
| 5. 和をスレッドで分担（リダクション） | これから | `part3/random/random_gpu_reduction.cu` | — |
| 6. Traub を多数並べて GPU で解く | ゴール | （本にはない。3〜5 の組み合わせ） | — |

本の GPU 化（第15章）は LIF のランダムネットワークが題材で、マルチコンパートメントの GPU 版は載っていない。なので「第15章の考え方 × `traub.c`」を自分で組み合わせることになる。下の読み順はそのためのもの。

---

## 1. `traub.c` と `traub.py` の対応（復習）

すでに移植済みなので、GPU 化で効いてくる箇所だけ確認する。

| `traub.c` | `traub.py` | GPU 化での意味 |
|---|---|---|
| `double var[N_VARS][NC]`（L176）。`enum { V, XI, M, ... }` で変数を番号付け | `dict[str, ndarray(NC,)]` | 「変数ごとに配列」＝ SoA（後述）。GPU に向いた配置 |
| `tau()` / `inf()` の `switch`（L57–109） | `VOLTAGE_GATES` と `_tau_inf` | 変数ごとに同じ式の形 → 同じ命令を全要素に流せる |
| `g_comp[NC][2]`（左右の 2 本、端は 0）（L132–135） | `compartment_conductance()` の shape `(NC-1,)` | 辺（結合）単位で持つか、頂点単位で持つかの違い |
| `i_comp` の計算（L158–159） | `I_flow` を `I_comp[:-1]` と `I_comp[1:]` に足す | **ここだけが隣のコンパートメントを読む**。並列化で唯一注意が要る |
| `dvar` を全部計算してから一括で足す（L143–171） | `new` を作って返す（入力は書き換えない） | 陽的オイラー法では全変数が「前の時刻の値」だけで決まる → 全要素を同時に計算してよい根拠 |

### なぜ並列化できるか（陽的オイラー法）

コンパートメント $i$ の膜電位の更新は

$$
V_i^{t+1} = V_i^{t} + \frac{\Delta t}{C_m}\left(-I_{\mathrm{ion},i}^{t} + I_{\mathrm{comp},i}^{t} + I_{\mathrm{inj},i}^{t}\right)
$$

$$
I_{\mathrm{comp},i}^{t} = \frac{g_{i-1,i}\,(V_{i-1}^{t} - V_i^{t}) + g_{i,i+1}\,(V_{i+1}^{t} - V_i^{t})}{\mathrm{area}_i}
$$

右辺はすべて時刻 $t$ の値なので、$V_0^{t+1}, \dots, V_{18}^{t+1}$ はどの順に計算しても、同時に計算しても同じ結果になる。条件は「時刻 $t$ の値を書き換える前に全員が読み終えていること」だけ。`traub.c` が `dvar` に一度ためてから足しているのはこのため。

### コンパートメント間の電流は行列積で書ける

$I_\mathrm{comp}$ は $V$ の線形関数なので、$19 \times 19$ の三重対角行列 $L$ を使って

$$
I_\mathrm{comp} = \mathrm{diag}(\mathrm{area})^{-1} L V, \quad
L_{i,i-1} = g_{i-1,i},\; L_{i,i+1} = g_{i,i+1},\; L_{i,i} = -(g_{i-1,i} + g_{i,i+1})
$$

と書ける（端は片側の項が 0）。ニューロンを $B$ 個並べて $V$ を shape `(B, NC)` にすると `V @ L.T / area` で全ニューロン分が一度に出る。卒論の「微分方程式を行列積で置き換える」話と、ODE 側の計算を比べるときの接点になる。

---

## 2. `column/soa/` — AoS と SoA（コラム）

### 具体例

HH ニューロン 65536 個の $(V, m, h, n)$ をメモリに置く方法が 2 通りある。

```c
// aos.cu : Array of Structures（構造体の配列）
typedef struct { double v, m, h, n; } hh_t;
hh_t neuron[N];          // メモリ上: v0 m0 h0 n0 v1 m1 h1 n1 ...

// soa.cu : Structure of Arrays（配列の構造体）
typedef struct { double *v, *m, *h, *n; } hh_t;
                         // メモリ上: v0 v1 v2 ... | m0 m1 m2 ... | ...
```

### 原理

- GPU は 32 スレッド（ワープ）が同じ命令を同時に実行する
- スレッド $i$ が `v[i]` を読むとき、SoA なら 32 個の `v` がメモリ上で連続している → 1 回のまとまった転送で読める（コアレスアクセス、coalesced access）
- AoS だと `v` が 32 バイトおきに散らばるので、使わない `m, h, n` まで転送してしまう

`traub.c` の `var[N_VARS][NC]` は 1 ニューロン内では SoA になっている。多数のニューロンを並べるときは、**ニューロン番号が一番内側（メモリ上で連続）** になるように `var[変数][コンパートメント][ニューロン]` とするのが GPU 向き。

### 読むときの注意

- `soa.cu` の `main`（L110–116）はカーネル起動のあと `cudaDeviceSynchronize()` を呼ばずにタイマーを止めている。カーネル起動は非同期なので、自分で時間を測るときは止める前に必ず同期する
- 実行には CUDA が入った NVIDIA GPU が要る（README では V100、Compute Capability 7.0 向けの設定）

---

## 3. `part3/random/random_gpu.cu` — 1 スレッド 1 ニューロン（15.2）

### 全体の処理

```
initialize : 状態 v, ge, gi, s と結合 (ELL 形式) を cudaMallocManaged で確保
loop (1000 ステップ):
    calculateSynapticInputs<<<GRID, 32>>>   # 各ニューロンが入力の和を取る
    updateCellParameters<<<GRID, 32>>>      # 各ニューロンが膜電位を更新・発火判定
    cudaDeviceSynchronize()                 # GPU の計算終わりを待つ
    outputSpike                             # CPU がスパイクを書き出す
```

### 各処理の詳細

| 箇所 | 内容 | 押さえる点 |
|---|---|---|
| L41–42 `BLOCK_SIZE`, `GRID_SIZE` | 1 ブロック 32 スレッド、ブロック数は $\lceil N/32 \rceil = 125$ | スレッド数が $N$ を超える分は L121 `if (i < N)` で何もしない |
| L119 `i = threadIdx.x + blockIdx.x * blockDim.x` | スレッド番号 = 担当ニューロン番号 | `for (i = 0; i < N; i++)` のループを「スレッドの番号」に置き換えた形 |
| L53 など `cudaMallocManaged` | ユニファイドメモリ。CPU と GPU で同じポインタを使える | コピー（`cudaMemcpy`）を書かずに済む代わりに、転送のタイミングは見えにくい |
| L84–100 ELL 形式 | 各ニューロンの入力元の番号 `wc` と重み `w` を、行の長さをそろえて詰めた表（第12章） | 密な $N \times N$ 行列の 8 割を占める 0 を飛ばす。終わりは番兵 `-1` |
| L117–130 `calculateSynapticInputs` | ニューロン $i$ が自分の入力元を for ループで全部足す | 1 スレッドが約 800 回足し算する → 15.3 でここを分担する |
| L132–141 `updateCellParameters` | LIF の更新と発火判定 | `if` を使わず `s * V_RESET + (!s) * v` で書いている。ワープ内で分岐がそろわない（divergence）のを避けるため |
| L155–157 | カーネルを 2 回に分けて起動 | **カーネルの境目が「全スレッドの同期」になる**。入力の和を全員が出し終えてから膜電位を更新する |

### Traub に当てはめると

`updateCellParameters` の中身を `traub.c` の `solve_euler` の 1 ニューロン分（19 コンパートメントのループ）に差し替えたものが、一番素直な GPU 版になる。

- 1 スレッド = 1 ニューロン。スレッドの中で 19 コンパートメントを順に計算する
- コンパートメント間の電流はスレッドの中で閉じているので、スレッド間の同期は要らない
- `dvar` 相当の一時配列はスレッドごとのローカル変数（`double dvar[N_VARS][NC]` = 11 × 19 × 8 バイト ≈ 1.7 KB）。レジスタに収まらずローカルメモリに逃げるので、速さを詰めるならここが課題

---

## 4. `random_gpu_reduction.cu` — 和をスレッドで分担（15.3）

`random_gpu.cu` との差分は `calculateSynapticInputs_b`（L133–167）と起動行（L192）だけ。

### 具体例

ニューロン $i$ への入力が 8 本 $r_0, \dots, r_7$ のとき、8 スレッドで

```
段 1 (k=4): s[0]+=s[4]  s[1]+=s[5]  s[2]+=s[6]  s[3]+=s[7]
段 2 (k=2): s[0]+=s[2]  s[1]+=s[3]
段 3 (k=1): s[0]+=s[1]          → s[0] が合計
```

とすると、8 回の足し算が $\log_2 8 = 3$ 段で終わる。

### 原理と対応するコード

| 箇所 | 内容 |
|---|---|
| L192 `<<< N, NTHSYN >>>` | 1 ブロック = 1 ニューロン、ブロック内の `NTHSYN` スレッドが入力を分担 |
| L137–138 `i = _i / NTHSYN; j = _i % NTHSYN` | ブロック番号がニューロン、ブロック内の番号が担当する入力の位置 |
| L140 `__shared__ float s_re[NTHSYN]` | 共有メモリ。同じブロックのスレッドだけが読み書きできる速いメモリ |
| L143–150 | まず各スレッドが自分の担当分を共有メモリに置く（入力が `NTHSYN` より多ければ `k += NTHSYN` で複数受け持つ） |
| L152, L159 `__syncthreads()` | ブロック内の全スレッドがここに来るまで待つ。次の段で他のスレッドの値を読むため |
| L154–160 | 上の具体例の木構造の和（`k` を半分ずつにする） |

### 読むときの注意

- `NTHSYN` は 2048 と定義されているが、CUDA の 1 ブロックあたりのスレッド数の上限は 1024 なので、このままだと `calculateSynapticInputs_b` の起動に失敗するはず。本文（リスト15.14 付近）の値と照らし合わせる
- 木構造の和は `NTHSYN` が 2 のべき乗であることを前提にしている

### Traub に当てはめると

コンパートメント方向の並列化に使える考え方。

- 1 ブロック = 1 ニューロン、ブロック内の 1 スレッド = 1 コンパートメント（19 スレッド。ワープ 32 に合わせて 32 にして余りは遊ばせる）
- 隣の $V_{i\pm1}$ を読むので、各スレッドが $V^t$ を共有メモリに書く → `__syncthreads()` → 隣を読んで $I_\mathrm{comp}$ を計算 → `__syncthreads()` → $V^{t+1}$ を書く、の順にする
- ニューロン数が少ない（GPU を埋めきれない）ときに効く。ニューロン数が多ければ 3. の 1 スレッド 1 ニューロンで十分 GPU は埋まる

---

## 5. ゴール：Traub を多数並べる 2 通りの道

| | (a) CUDA C で書く | (b) Python の配列演算で書く |
|---|---|---|
| 元にするもの | `traub.c` + `random_gpu.cu` | `traub.py`（すでに `(NC,)` でベクトル化済み） |
| やること | `solve_euler` をカーネルにし、配列を `[変数][コンパートメント][ニューロン]` にする | 状態を `(NC,)` から `(B, NC)` にし、`numpy` を `cupy` か `torch` に差し替える |
| 並列化の単位 | 自分で決める（スレッド = ニューロン / コンパートメント） | ライブラリが要素ごとに自動で割り振る |
| 本の内容との対応 | 第15章そのもの | 「全要素に同じ式を流す」という考え方だけ同じ |
| 卒論との相性 | 速さの上限を詰められる | RNN 側（PyTorch）と同じ土俵で計算時間を比べられる |

(b) では `traub.py` の書き換えは小さい。`I_flow = g_comp * (V[1:] - V[:-1])` を `V[:, 1:] - V[:, :-1]` にするなど、コンパートメントの軸を最後に回すだけで、各ニューロンに別々の注入電流を入れた $B$ 本の教師データを一度に作れる。

なお、この PC では `nvidia-smi` が見つからなかった。GPU で動かすのは研究室の GPU マシンになる想定で、まず CPU 上で (b) の形（バッチ次元つき）に直して結果が `traub.py` と一致することを確かめる、という順が安全。

---

## 本の読み順（ノートの節との対応）

1. 第2章（数値解法）と `part1/numeric/exp.c`：オイラー法の誤差が $\Delta t$ に比例すること。Traub で $\Delta t = 0.01$ ms を使う理由
2. 3.7 を本文で読み直す：`traub.c` 以外に本文で説明している点（ノート「読むときの観点」）
3. 第12章（ELL）、第13章（OpenMP）：`random_ell.c`、`random_omp.c`。ノートには未収録の章だが、15章の `wc`/`w` の意味はここで説明されている
4. コラム AoS vs. SoA：`column/soa/`
5. 15.1–15.2：`random_gpu.cu`
6. 15.3：`random_gpu_reduction.cu`

## メモ：リスト番号のずれ

`part1/random/README-ja.md` と `part1/multi/README-ja.md` の両方が自分を「リスト3.8」と書いている。3.5 の `random.c` の方が先に出てくるので、どちらかの README の誤りと思われる。本文で確認する（ノート 3.7 の項と同じ件）。
