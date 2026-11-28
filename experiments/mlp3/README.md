# MLP 3 個（C）

HH の陽的オイラー法 1 ステップ（dt = 0.01 ms）を 3 つのブロックに分け、それぞれを MLP で置き換えた代理モデル。発表の 4 分野で最良（最終テスト 54 本で 45/54、Γ = 0.953、設定 `m3_hdf`）。

```
V ──► ブロック 1 ──► αβ (6) ──► ブロック 2 ──► Δm, Δh, Δn ──► g' = clamp(g + Δg, 0, 1)
            g = (m, h, n) ──────┘
g, V, I (+ m³h, n⁴, m³hV, n⁴V) ──► ブロック 3 ──► ΔV ──► V' = clamp(V + ΔV, -100, 60)
```

旧コード（`experiments/hh_surrogate/hh_improve.py mlp3`、依存先を含め約 3,200 行）から MLP 3 個に必要な部分だけを取り出した（約 1,000 行）。同じ seed で学習すると、旧コードと**重みがビット単位で一致**する（2026-11-28 に小さい条件で確認）。旧形式の重み `results/hh_improve/m3_hdf_s0_b*.pt` もそのまま読め、テスト 26 本の成績（23/26、平均一致率 0.9301、Γ 0.9273）が旧 JSON と一致する。

## ファイル

| ファイル | 中身 |
|---|---|
| `hh.py` | HH の式（torch）: 速度定数 `rates`、ブロック 2・3 の正解 `d_gates`・`d_V`、軌道 `simulate` |
| `currents.py` | 学習用の電流: 通常の電流 `normal`、苦手な電流 `hard`（長い一定・ゆっくりしたランプ・閾値付近の揺らぎ） |
| `model.py` | `Block`（標準化 + MLP）と `MLP3`（閉ループ `run`、DAgger 用の `visited_states`、保存・読み込み） |
| `eval_sets.py` | 評価の電流（検証 32 本・テスト 26 本・最終テスト 54 本）、HH の正解、採点（一致率・Γ） |
| `outputs.py` | 保存（損失の推移、全電流の V・ゲート・速度定数の応答の数値と図） |
| `train.py` | 学習（状態を集める → 事前学習 → DAgger、検証で最良の重みを残す → 最後に必ず `outputs` で保存） |
| `evaluate.py` | 学習済みの重みを評価し直す（旧形式の重みも可）。最終テストは `--final` のときだけ |
| `run_server.sh` | 学内サーバでジョブ一覧を並列に学習し、結果を commit・push する |
| `jobs/*.txt` | ジョブ一覧（1 行 = `train.py` の引数 1 本分） |

## 使い方（リポジトリのルートで）

```powershell
uv run python -X utf8 experiments/mlp3/train.py --tag m3_hdf_s0 --seed 0 --device cuda
```

学内サーバでは、ジョブ一覧を並列に回す:

```bash
nohup bash experiments/mlp3/run_server.sh experiments/mlp3/jobs/m3_b16k.txt 4 > results/mlp3/run_server.out 2>&1 &
```

既定値は `m3_hdf`（128×2 × 3、SiLU、積の特徴あり、苦手な電流 256 本、事前学習 5 万回 + DAgger 3 回 × 2 万回）。動作確認は `train.py` の先頭の「動作確認」の引数で CPU 数分。

```powershell
uv run python -X utf8 experiments/mlp3/evaluate.py results/hh_improve/m3_hdf_s0 --out results/mlp3/m3_hdf_s0
```

## 学習で必ず保存するもの（`results/mlp3/<tag>/`）

| ファイル | 中身 |
|---|---|
| `config.json`, `train.log` | 引数、画面に出した記録 |
| `model_b1.pt`〜`model_b3.pt` | 検証で最良の重み |
| `loss.csv`, `loss.png` | ブロックごとの損失（100 回ごとの平均）と学習率 |
| `val_history.csv` | 検証 32 本の採点の推移（各段階の始めと最後） |
| `responses_val.npz`, `responses_test.npz` | 検証 32 本・テスト 26 本の予測の V・m, h, n・α, β（全点、float32。形式は `outputs.py` の先頭） |
| `score_val.json`, `score_test.json` | 電流ごとの一致率・Γ・RMSE（V、ゲート、速度定数） |
| `figs/<set>/<番号>_<電流>_{V,gates,rates}.png` | 電流ごとの図（黒 = HH、橙 = MLP 3 個）。軸の範囲はセット内で共通 |
| `metrics.json` | 要約（最良の回、検証・テストの成績、時間） |

- HH の正解は実行ごとには保存しない（`eval_sets.reference` が同じ並びで作り直す。キャッシュ `results/mlp3/_reference/` は git に入れない）
- 応答の npz は 1 実行で約 75 MB（検証 32 本で約 40 MB + テスト 34 MB）。大きいので git には入れず、学習したマシンに残す（`.gitignore`）。重みから `evaluate.py` で同じものを作り直せる
- 図は 1 実行で約 17 MB（174 枚）。git に入れる
- 最終テスト 54 本は、最終候補を決めた後に `evaluate.py --final` で 1 回だけ使う（学習では使わない）

## 旧コードとの違い

- 検証での選抜は float32、最後の採点は float64（旧コードと同じ）。そのため `metrics.json` の `val_at_selection` と `val` は少し違うことがある（m3_hdf_s0 では、検証が 17 本だったころに 16/17 と 17/17）
- 旧コードの `--data` は mlp3 では使っていなかったので入れていない。`--sfeat` は既定で有効（外すときは `--no-sfeat`）
- 最終テストだけ V の下限を -150 mV に広げる（旧 `clamp_track.EVAL_V_LO` と同じ）

## ほかのアーキテクチャ

学内サーバで学習が進んでいる間はファイルを動かさないため、今は元の場所のままにしてある。

| 分野 | コード | 結果 |
|---|---|---|
| A Markov MLP | `experiments/hh_surrogate/hh_improve.py single_cl --cell mlp` | `results/hh_improve/` |
| B RNN 1 個（W_hh） | `experiments/hh_surrogate/hh_improve.py single_cl`、`hh_mlp_whh.py` | `results/hh_improve/`、`results/hh_whh/` |
| D RNN 3 個 | `experiments/hh_surrogate/hh_rnn3.py`、`hh_serial.py`、`hh_modular.py` | `results/hh_rnn3/`、`results/hh_serial*/`、`results/hh_modular*/` |
| C MLP 3 個（旧） | `experiments/hh_surrogate/hh_improve.py mlp3` | `results/hh_improve/m3_*` |
| Traub | `experiments/traub_surrogate/` | `results/traub_*` |
